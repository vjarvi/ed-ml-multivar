TARGET=main
TEXINPUTS_PATH=.:..:

.PHONY: all clean

all: $(TARGET).pdf

$(TARGET).pdf: $(TARGET).tex tex/text.tex
	TEXINPUTS=$(TEXINPUTS_PATH) pdflatex $(TARGET).tex
	TEXINPUTS=$(TEXINPUTS_PATH) biber $(TARGET)
	TEXINPUTS=$(TEXINPUTS_PATH) pdflatex $(TARGET).tex
	TEXINPUTS=$(TEXINPUTS_PATH) pdflatex $(TARGET).tex

clean:
	rm -f $(TARGET).aux $(TARGET).bbl $(TARGET).bcf $(TARGET).blg \
	  $(TARGET).log $(TARGET).out $(TARGET).run.xml $(TARGET).toc \
	  $(TARGET).glg $(TARGET).glo $(TARGET).gls $(TARGET).ist \
	  $(TARGET).synctex.gz
	rm -f $(TARGET).pdf


