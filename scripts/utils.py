import pandas as pd
import numpy as np
import os
import re
import torch

from darts import TimeSeries
from darts.utils.model_selection import train_test_split

from pyprojroot import here
import joblib

from sklearn.preprocessing import LabelEncoder

# TODO: This feels hacky, is there a better way? 
def get_n_epochs(model):
    """
    Check logs for how many epochs were used to train the
    initial model
    """
    path = model.load_ckpt_path
    n_epochs = re.search('checkpoints/best-epoch=(.*)-val_loss', path)
    return int(n_epochs.group(1)) + 1


def preprocess(data, model):
    if model!='lgbm':
        # Process holidays
        holiday_names = pd.get_dummies(data['Calendar:Holiday_name'], prefix='Holiday_name', prefix_sep=':')
        holiday_names = holiday_names.iloc[:,1:]
        data = data.drop(columns='Calendar:Holiday_name')
        data = pd.concat([data, holiday_names], axis=1)

        # Process other calendar variables
        hour = pd.get_dummies(data['Calendar:Hour'], prefix='Hour', prefix_sep=':')
        weekday = pd.get_dummies(data['Calendar:Weekday'], prefix='Weekday', prefix_sep=':')
        month = pd.get_dummies(data['Calendar:Month'], prefix='Month', prefix_sep=':')

        data = data.drop(columns=['Calendar:Hour', 'Calendar:Month', 'Calendar:Weekday'])
        data = pd.concat([data, hour, weekday, month], axis=1)

        # Process slip
        slip = pd.get_dummies(data['Weather:Slip'], prefix='Weather_Slip', prefix_sep=':')
        data = data.drop(columns='Weather:Slip')
        data = pd.concat([data, slip], axis=1)
    elif model=='lgbm':
        le = LabelEncoder()
        # Process holidays
        data['Calendar:Holiday_name'] = le.fit_transform(data['Calendar:Holiday_name'])
    else:
        raise ValueError('Model not supported')
        
    # Fill nans
    data = data.replace([np.inf, -np.inf], np.nan).fillna(0)

    return data


def get_data(target_name, model):
    # Import data
    # return pandas DataFrame
    data = pd.read_csv(here() / "data/interim/data.csv", 
                       index_col='datetime',
                       parse_dates=True,
                       low_memory=False)

    data = data["2017":]
    
    data = data.asfreq('h')
    data = preprocess(data, model)

    return data

def get_y(data, target_name):
    cd = {
        'occ' : 'Target:Occupancy',
        'arr' : 'Target:Arrivals'
    }
    target_name = cd[target_name]
    series = TimeSeries.from_series(data[target_name])
    return series

def convert_to_tensors(data):
    """Convert TimeSeries objects to 1D PyTorch tensors"""
    y_tensor = torch.tensor(data.values(), dtype=torch.float32) if data is not None else None
    y_tensor = torch.squeeze(y_tensor)
    return y_tensor


def get_x(data, featureset):
    # Masks
    traffic = data.columns.str.startswith('Traffic')
    beds = data.columns.str.startswith('Beds')
    google = data.columns.str.startswith('Trends')
    website = data.columns.str.startswith('Website_visits')
    ta = data.columns.str.startswith('TA')

    weather = data.columns.str.startswith('Weather')
    calendar = data.columns.str.startswith('Calendar')
    public_event = data.columns.str.startswith('Events')
    
    hours = data.columns.str.startswith('Hour')
    weekdays = data.columns.str.startswith('Weekday')
    months = data.columns.str.startswith('Month')
    holidays = data.columns.str.startswith('Holiday_name')

    if featureset=='u':
        past_covariates = None
        future_covariates = None
    if featureset=='a':
        past_cov_mask =  traffic | beds | google | website | ta
        past_covariates = data.loc[:,past_cov_mask]
        future_cov_mask = months | weekdays | hours | calendar | holidays | public_event | weather
        future_covariates = data.loc[:,future_cov_mask]
    
    if past_covariates is not None:
        past_covariates = past_covariates.fillna(0)
        past_covariates = TimeSeries.from_dataframe(past_covariates)
    if future_covariates is not None:
        future_covariates = future_covariates.fillna(0)
        future_covariates = TimeSeries.from_dataframe(future_covariates)

    return past_covariates, future_covariates

def get_x_tensors(data, featureset):
    """Get covariates as PyTorch tensors instead of TimeSeries objects"""
    pc, fc = get_x(data, featureset)
    
    pc_tensor = torch.tensor(pc.values(), dtype=torch.float32) if pc is not None else None
    fc_tensor = torch.tensor(fc.values(), dtype=torch.float32) if fc is not None else None
    
    return pc_tensor, fc_tensor

def get_y_context(data, target_name, context_length, test_start, batch_size=365):
    """
    data: pandas DataFrame
    context_length: int
    returns: pandas DataFrame dimension (batch_size, context_length)
             without index
    """
    cd = {
        'occ' : 'Target:Occupancy',
        'arr' : 'Target:Arrivals'
    }
    target_name = cd[target_name]
    y = data[target_name]

    start_idx = y.index.get_loc(test_start)
    row_list = []

    for i in range(batch_size):
        row_list.append(y.iloc[start_idx-context_length:start_idx])
        start_idx += 24

    row_list = [row.reset_index(drop=True) for row in row_list]
    y_context = pd.concat(row_list, axis=1).T
    return y_context

def to_pred_matrix(ts_list, quantile=None, test_start=None):
    """
    Converts model outputs into a properly indexed prediction matrix.
    Supports:
    - List of darts TimeSeries
    - PyTorch tensors of shape (num_rows, horizon)
    - Pandas DataFrames (keeps existing DatetimeIndex if present)
    - NumPy arrays / list-of-lists of shape (num_rows, horizon)
    """
    # Helper to build a daily DatetimeIndex
    def _build_dt_index(start_value, periods):
        if isinstance(start_value, (pd.Timestamp, np.datetime64)):
            start_ts = pd.Timestamp(start_value)
        else:
            try:
                start_ts = pd.Timestamp(start_value)
            except Exception as exc:
                raise ValueError(
                    "test_start must be a datetime-like value (e.g. '2020-01-01')"
                ) from exc
        return pd.date_range(start=start_ts, periods=periods, freq='D')

    # PyTorch tensor predictions
    if isinstance(ts_list, torch.Tensor):
        if test_start is None:
            raise ValueError('test_start must be provided when ts_list is a torch.Tensor')
        tensor = ts_list.detach().cpu()
        # Ensure 2D: (num_rows, horizon)
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.dim() != 2:
            raise ValueError('Expected 2D tensor of shape (num_rows, horizon)')

        num_rows, horizon = tensor.shape
        df = pd.DataFrame(
            tensor.numpy(),
            columns=[f"t+{i+1}" for i in range(horizon)],
        )
        df.index = _build_dt_index(test_start, num_rows)
        df.index.name = 'datetime'
        return df.round(2)

    # Pandas DataFrame predictions
    if isinstance(ts_list, pd.DataFrame):
        df = ts_list.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            if test_start is None:
                raise ValueError(
                    'DataFrame has no DatetimeIndex. Provide datetime-like test_start.'
                )
            df.index = _build_dt_index(test_start, len(df))
            df.index.name = 'datetime'
        else:
            # Preserve and normalize index name
            df.index.name = df.index.name or 'datetime'
        return df.round(2)

    # 2D np.ndarrays
    if isinstance(ts_list, np.ndarray):

        matrix = pd.DataFrame(ts_list, columns=[f"t+{i+1}" for i in range(ts_list.shape[1])])
        matrix.index = _build_dt_index(test_start, len(matrix))  # daily; adjust freq if needed
        matrix.index.name = 'datetime'
        return matrix.round(2)

    matrix = list()
    for ts in ts_list:
        if quantile:
            vector = ts.quantile_df(quantile).iloc[:,0]
        else:
            vector = ts.to_series()

        vector.name = vector.index[0]
        vector.index = [f"t+{x+1}" for x in range(len(vector))]
        matrix.append(vector)

    matrix = pd.concat(matrix, axis=1).T
    matrix.index.name = 'datetime'
    matrix = matrix.round(2)

    return matrix

def save(
    model_name, 
    featureset_name, 
    target_name,
    hpo,
    y_pred, 
    model,
    study=None,
    quantiles=[.05, .50, .95],
    settings=None
    ):
    """
    Perstists prediction matrix and model binary
    """
    unique_name = f'{target_name}-{model_name}-{featureset_name.lower()}-{hpo}'

    rootpath = here('data/processed/prediction_matrices')

    if y_pred[0].is_probabilistic:
        for quantile in quantiles:
            matrix = to_pred_matrix(y_pred, quantile)
            
            outpath = rootpath / f'{int(quantile*100):02d}'
            outpath.mkdir(parents=True, exist_ok=True)
            matrix.to_csv(outpath / f"{unique_name}.csv")
    
    if y_pred[0].is_deterministic:
        matrix = to_pred_matrix(y_pred)
        outpath = rootpath / f'50'
        outpath.mkdir(parents=True, exist_ok=True)
        matrix.to_csv(outpath / f"{unique_name}.csv")

    # models
    outpath = here('data/processed/models')
    outpath.mkdir(parents=True, exist_ok=True)
    model_path = str(outpath / f'{unique_name}.pkl')
    model.save(model_path)

    if study:
        # studies
        outpath = here('data/processed/studies')
        outpath.mkdir(parents=True, exist_ok=True)
        model_path = str(outpath / f'{unique_name}.pkl')
        joblib.dump(study, model_path)

    if settings:
        outpath = here('logs/settings.pkl')
        joblib.dump(settings, outpath)


def save(
    model_name, 
    featureset_name, 
    target_name,
    context_length,
    test_start,
    y_pred, 
    ):
    """
    Saves prediction matrix from pandas DataFrame, PyTorch tensors, or TimeSeries objects.
    """
    unique_name = f'{target_name}-{model_name}-{featureset_name.lower()}-{context_length}'

    rootpath = here('data/processed/prediction_matrices')
    
    matrix = to_pred_matrix(y_pred, test_start=test_start)
    outpath = rootpath / f'50'
    outpath.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(outpath / f"{unique_name}.csv")
