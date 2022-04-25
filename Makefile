###############################
# GLOBALS
CONDA_ENV = ./env/
CONDA_ACTIVATE = eval "$$(conda shell.bash hook)"; conda activate $(CONDA_ENV); export PYTHONPATH=`pwd`:$${PYTHONPATH}; alias python="python -m sklearnex"
SHELL=bash

.PHONY: $(CONDA_ENV) clean jupyter setup gpu_setup test

###############################
# COMMANDS
setup: $(CONDA_ENV)
	$(CONDA_ACTIVATE) && R -e "IRkernel::installspec()"
	pre-commit install
	pip install --upgrade --force-reinstall "jax[cpu]"

gpu_setup: setup
	$(CONDA_ACTIVATE) && mamba uninstall pytorch && mamba install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

test:
	$(CONDA_ACTIVATE); pytest test

data/vaishnav_et_al:
	mkdir data/vaishnav_et_al
	$(CONDA_ACTIVATE); cd data/vaishnav_et_al; zenodo_get 4436477

docker_build: Dockerfile
	docker build -t dna2prot .

benchmark: results/benchmark/config.yaml
	nextflow src/benchmark.nf -params-file results/benchmark/config.yaml -resume

$(CONDA_ENV): environment.yml
	mamba env create --force --prefix $(CONDA_ENV) --file environment.yml

jupyter:
	$(CONDA_ACTIVATE); export PYTHONPATH=`pwd`:$${PYTHONPATH}; jupyter lab --notebook-dir=./

clean:
	rm -rf env/
