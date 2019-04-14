name=CUDA

all: build view

build:
	pdflatex -file-line-error $(name)
	pdflatex -file-line-error $(name)
	pdflatex -file-line-error $(name)
	pdflatex -file-line-error $(name)
	mv $(name).pdf ../
view:
	acroread ../$(name).pdf

clean:
	rm -f *.toc *.snm *.out *.nav *.log *.aux *.vrb *~


