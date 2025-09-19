.PHONY: all clean print_vars clear truedata diagnostics clear_results

#################################################################################
# GLOBALS                                                                       #
#################################################################################

# if these directories don't exist, create them
# mkdir -p data/interim
# mkdir -p data/processed/prediction_matrices/{05,50,95}
# mkdir -p data/processed/{models,studies}

SHELL = /bin/bash # changed from zsh to bash

mordir :=  /home/valtti/code/ed-ml-multivar/data/interim
rawdir := data/raw
interimdir := data/interim
preddir := data/processed/prediction_matrices
truedir := data/processed/true_matrices

MORDATA := $(wildcard $(mordir)/*suht.csv)
RAWDATAUP := $(patsubst $(mordir)/%.csv, $(rawdir)/%.csv, $(MORDATA))
RAWDATA = $(shell echo $(RAWDATAUP) | tr A-Z a-z)


CHRONOS = $(preddir)/50/occ-chronos-u-0.csv\

SCRIPTS = scripts/train.py\
	scripts/models/chronos.py\
	scripts/models/utils.py

PLOT1 = output/plots/beds.jpg\

PLOT2 = output/plots/horizon_mae-a-1.jpg\
	output/plots/horizon_mae-u-1.jpg\
	output/plots/performance_monthly-u-1.jpg\
	output/plots/performance_monthly-a-1.jpg\

NTB2 = notebooks/plot_horizon_mae.ipynb\
	notebooks/plot_performance_monthly.ipynb\

PLOT3 = output/plots/importance-1.jpg\
	output/plots/importance-24.jpg

TABLES = output/tables/msis.tex\
	output/tables/performance.tex\
	output/tables/studies.tex\
	output/tables/distance_hospitals.tex\
	output/tables/distance_traffic.tex\
	output/tables/monthly_performance.tex\
	output/tables/seasonality.tex\
	output/tables/runtime.tex\

TEXFILES = report/manuscript.tex\
	report/sections/abstract.tex\
	report/sections/introduction.tex\
	report/sections/materials_and_methods.tex\
	report/sections/results.tex\
	report/sections/discussion.tex\
	report/sections/appendix_a.tex\
	report/sections/appendix_b.tex\
	report/sections/appendix_c.tex\
	report/sections/appendix_d.tex\
	report/sections/declarations.tex\
	report/sections/conclusions.tex\

PREDDATA = $(CHRONOS)

TRUEDATA = data/processed/true_matrices/occ.csv

# Notebooks testing pipeline
# List of test notebooks and context lengths to run with
TEST_NOTEBOOKS := $(wildcard notebooks/test_*.ipynb)
# User-editable list of context lengths; space-separated (e.g., 128 256 512)
CONTEXTLENGTHS ?= 8 16 24 32 48 72 96 120 144 168 192 224 256 288 320 352 384 416 448 480 \
				1024 2048 4096 8192 12840

# Per-notebook virtual environments (user-editable)
# Set these to the absolute path of the venv directory (containing bin/activate)
# Example:
# NB_VENV_test_moirai2 = /home/user/.venvs/moirai2
# NB_VENV_test_chronos = /home/user/.venvs/chronos
NB_VENV_test_moirai2 ?= /home/valtti/code/ed-ml-multivar/uni2ts/venv
NB_VENV_test_chronos ?= /home/valtti/code/ed-ml-multivar/env-chronos
NB_VENV_test_tirex ?= /home/valtti/code/ed-ml-multivar/env-tirex
NB_VENV_test_sundial ?= /home/valtti/code/ed-ml-multivar/env-sundial
NB_VENV_test_timesfm ?= /home/valtti/code/ed-ml-multivar/env-timesfm

# Export venv variables so they are visible in shell recipes
export NB_VENV_test_moirai2 NB_VENV_test_chronos NB_VENV_test_tirex NB_VENV_test_sundial NB_VENV_test_timesfm

#################################################################################
# FUNCTIONS                                                                     #
#################################################################################

# $(call get_word, position_in_list, list) 
# Get nth element from {w1}-{w2}-{w3}-..-{wn}
define get_word
$(word $(1),$(subst -, ,$(basename $(notdir $(2)))))
endef

#################################################################################
# COMMANDS                                                                      #
#################################################################################

all: output/manuscript.pdf 

manuscript: output/manuscript.pdf

blind: output/manuscript_blind.pdf output/declarations.pdf

title: output/title.pdf

response: output/response.pdf

truedata: $(TRUEDATA)

preddata: $(PREDDATA)

figures: $(FIGURES)

chronos: $(CHRONOS)

output/declarations.pdf: report/declarations.tex report/sections/declarations.tex
	cd report/\
	&& pdflatex declarations\
	&& mv declarations.pdf ../output/\
	&& open ../output/declarations.pdf

output/title.pdf: report/title.tex report/sections/abstract.tex
	cd report/\
	&& pdflatex title\
	&& mv title.pdf ../output/\
	&& open ../output/title.pdf

output/manuscript_blind.pdf: $(TEXFILES) $(TABLES) $(PLOT1) $(PLOT2) $(PLOT3) report/references.bib report/manuscript_blind.tex
	cd report/\
	&& pdflatex manuscript_blind\
	&& bibtex manuscript_blind\
	&& pdflatex manuscript_blind\
	&& pdflatex manuscript_blind\
	&& mv manuscript_blind.pdf ../output/\
	&& open ../output/manuscript_blind.pdf

output/manuscript.pdf: $(TEXFILES) $(TABLES) $(PLOT1) $(PLOT2) $(PLOT3) report/references.bib
	cd report/\
	&& pdflatex manuscript\
	&& bibtex manuscript\
	&& pdflatex manuscript\
	&& pdflatex manuscript\
	&& cp manuscript.pdf ~/Desktop/ed-tft-draft.pdf\
	&& mv manuscript.pdf ../output/\
	&& open ../output/manuscript.pdf

output/presentation.pdf:
	cd presentation/\
	&& pdflatex presentation\
	&& open presentation.pdf

$(PLOT1): output/plots/%.jpg: notebooks/plot_%.ipynb notebooks/nutils.py $(TRUEDATA)
	cd notebooks\
	&& papermill $(notdir $<) /dev/null

$(PLOT2): $(NTB2) notebooks/nutils.py $(TRUEDATA)
	cd notebooks\
	&& papermill plot_$(call get_word,1,$@).ipynb /dev/null -p FS $(call get_word,2,$@) -p HPO $(call get_word,3,$@)

$(PLOT3): notebooks/plot_importance.ipynb notebooks/nutils.py $(TRUEDATA)
	cd notebooks\
	&& papermill plot_$(call get_word,1,$@).ipynb /dev/null -p H $(call get_word,2,$@)

$(TABLES): output/tables/%.tex: notebooks/tab_%.ipynb $(TRUEDATA)
	cd notebooks\
	&& papermill $(notdir $<) /dev/null

# --account $(ACCOUNT)\ add after row 194 if using SLURM ?
$(PREDDATA): $(preddir)%.csv : $(SCRIPTS) data/interim/data.csv
	if command -v sbatch &> /dev/null; then \
		export TARGET=$(call get_word,1,$@) && \
		export MODEL=$(call get_word,2,$@) && \
		export FEATURESET=$(call get_word,3,$@) && \
		export HPO=$(call get_word,4,$@) && \
		sbatch --output logs/slurm/$$TARGET-$$MODEL-$$FEATURESET-$$HPO.log \
		--job-name $$TARGET-$$MODEL-$$FEATURESET-$$HPO \
		scripts/batch_$(call get_word,2,$@).sh; \
	else \
		cd scripts && python train.py $(call get_word,1,$@) $(call get_word,2,$@) $(call get_word,3,$@) $(call get_word,4,$@); \
	fi

# FIXME: For I am probably broken
$(TRUEDATA): scripts/create_true_matrix.py
	python scripts/create_true_matrix.py $(call get_word,1,$@)

data/interim/data.csv: data/raw/data.csv notebooks/preprocess_data.ipynb
	cd notebooks\
	&& papermill preprocess_data.ipynb /dev/null

# -----------------------------------------------------------------------------
# Run all test notebooks for all configured CONTEXTLENGTHS
.PHONY: tests
tests:
	cd notebooks\
	&& for nb in $(notdir $(TEST_NOTEBOOKS)); do \
		base=$${nb%.ipynb}; \
		venv_var=NB_VENV_$$base; \
		venv_path=$${!venv_var}; \
		if [ -n "$$venv_path" ]; then \
			echo "Activating venv for $$nb: $$venv_path"; \
			. "$$venv_path/bin/activate"; \
			PYBIN="$$venv_path/bin/python"; \
		else \
			PYBIN="python"; \
		fi; \
		for cl in $(CONTEXTLENGTHS); do \
			echo "Running $$nb with CONTEXTLENGTH=$$cl"; \
			"$$PYBIN" -m papermill $$nb /dev/null -p CONTEXTLENGTH $$cl; \
		done; \
	done

#################################################################################
# Helper Commands                                                     #
#################################################################################

clean:
	rm -rf data/processed/*
	rm -r output/*

clear_report:
	rm report/manuscript.bbl
	rm report/manuscript.blg
	rm report/manuscript.log
	rm report/manuscript.out
	rm report/manuscript.aux

clear_notebooks:
	jupyter nbconvert notebooks/*.ipynb --clear-output --inplace

clear_results:
	rm -rf data/processed/prediction_matrices
	rm -rf data/processed/models
	rm -rf data/processed/studies
	rm -rf darts_logs
	rm logs/slurm/*
	rm logs/logger/*

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

print_vars:
	@echo "TARGETS"
	@echo $(TARGETS)

	@echo "INTERIMDATA"
	@echo $(INTERIMDATA)

	@echo "RAWDATA"
	@echo $(RAWDATA)

	@echo "PREDDATA"
	@echo $(PREDDATA)

	@echo "ACPLOTS"
	@echo $(ACPLOTS)