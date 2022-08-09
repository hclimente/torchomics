###############################
# GLOBALS
CONDA_ENV = ./env/
CONDA_ACTIVATE = eval "$$(conda shell.bash hook)"; conda activate $(CONDA_ENV); export PYTHONPATH=`pwd`:$${PYTHONPATH}; alias python="python -m sklearnex"
SHELL=bash

.PHONY: $(CONDA_ENV) clean jupyter setup gpu_setup test

###############################
# COMMANDS
setup: $(CONDA_ENV)
	$(CONDA_ACTIVATE)
	pre-commit install

test:
	$(CONDA_ACTIVATE); pytest test

$(CONDA_ENV): environment.yml
	mamba env create --force --prefix $(CONDA_ENV) --file environment.yml

clean:
	rm -rf env/
