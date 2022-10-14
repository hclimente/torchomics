###############################
# GLOBALS
CONDA_ENV = ./env/
CONDA_ACTIVATE = eval "$$(conda shell.bash hook)"; conda activate $(CONDA_ENV); export PYTHONPATH=`pwd`:$${PYTHONPATH}
SHELL=bash

.PHONY: $(CONDA_ENV) clean jupyter pip setup test

###############################
# COMMANDS
setup: $(CONDA_ENV)
	$(CONDA_ACTIVATE)
	pre-commit install

pypi:
	$(CONDA_ACTIVATE); python setup.py sdist; twine upload dist/*

test_pypi:
	$(CONDA_ACTIVATE); python setup.py sdist; twine upload --repository testpypi dist/*
	# test installation
	$(CONDA_ACTIVATE); pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ torchomics
	$(CONDA_ACTIVATE); pip uninstall torchomics

test:
	$(CONDA_ACTIVATE); pytest -s test

jupyter:
	$(CONDA_ACTIVATE); jupyter lab --notebook-dir=notebooks/

$(CONDA_ENV): environment.yml
	mamba env create --force --prefix $(CONDA_ENV) --file environment.yml

clean:
	rm -rf env/
