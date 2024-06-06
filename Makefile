.ONESHELL:

all: loanclassifier outliersdetector combiner

base:
	docker build . -t seldon-core-outliers-base:0.1

loanclassifier:
    s2i build https://github.com/muhsufyan/seldon-alibi-combiner.git \
    --context-dir=pipeline/loanclassifier \
     seldonio/seldon-core-s2i-python3:latest \
    loanclassifier
outliersdetector:
    s2i build https://github.com/muhsufyan/seldon-alibi-combiner.git \
    --context-dir=pipeline/outliersdetector \
    seldonio/seldon-core-s2i-python3:latest \
    outliersdetector
combiner:
    s2i build https://github.com/muhsufyan/seldon-alibi-combiner.git \
    --context-dir=pipeline/combiner \
    seldonio/seldon-core-s2i-python3:latest \
    combiner

datatest:
	{"data":{"ndarray":[[46,5,4,2,8,4,4,0,2036,0,60,9],[52,4,0,2,8,4,2,0,0,0,60,9],[21,4,4,1,2,3,4,1,0,0,20,9]]}}