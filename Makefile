###############################
# GLOBALS
CONDA_ENV = ./env/
CONDA_ACTIVATE = eval "$$(conda shell.bash hook)"; conda activate $(CONDA_ENV); export PYTHONPATH=`pwd`:$${PYTHONPATH}; alias python="python -m sklearnex"
SHELL=bash

.PHONY: $(CONDA_ENV) clean jupyter setup test

###############################
# COMMANDS
setup: $(CONDA_ENV)
	$(CONDA_ACTIVATE) && R -e "IRkernel::installspec()"
	pre-commit install

test:
	$(CONDA_ACTIVATE); pytest test

docker_build: Dockerfile
	docker build -t dna2prot .

benchmark: results/benchmark/config.yaml
	nextflow src/benchmark.nf -params-file results/benchmark/config.yaml -resume

$(CONDA_ENV): environment.yml
	mamba env create --force --prefix $(CONDA_ENV) --file environment.yml

jupyter:
	$(CONDA_ACTIVATE); export PYTHONPATH=`pwd`:$${PYTHONPATH}; jupyter lab --notebook-dir=notebooks/

clean:
	rm -rf env/
