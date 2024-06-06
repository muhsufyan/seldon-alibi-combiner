.ONESHELL:

all: base loanclassifier outliersdetector combiner

base:
	docker build . -t tetewpoj/seldon-core-outliers-base:0.1

loanclassifier:
	s2i build pipeline/loanclassifier tetewpoj/seldon-core-outliers-base:0.1 tetewpoj/loanclassifier:0.1

outliersdetector:
	s2i build pipeline/outliersdetector tetewpoj/seldon-core-outliers-base:0.1 tetewpoj/outliersdetector:0.1

combiner:
	s2i build pipeline/combiner tetewpoj/seldon-core-outliers-base:0.1 tetewpoj/combiner:0.1