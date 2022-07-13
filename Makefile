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
	$(CONDA_ACTIVATE); jupyter lab --notebook-dir=./

jupyter_server:
	$(CONDA_ACTIVATE); jupyter lab --notebook-dir=./ --no-browser --port 8080

jupyter_client:
	ssh -N -L localhost:8080:localhost:8080 $(SERVER)

tensorboard_server:
	$(CONDA_ACTIVATE); ulimit -n 5000; tensorboard --logdir results/models/

tensorboard_client:
	ssh -N -L localhost:6006:localhost:6006 $(SERVER)

train:
	$(CONDA_ACTIVATE); unalias python; export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7; python scripts/train.py

clean:
	rm -rf env/

prepare_tpu:
	pip3 install -r requirements.txt
	echo 'export XRT_TPU_CONFIG="localservice;0;localhost:51011"' >>~/.bashrc
	echo 'export PYTHONPATH=$${HOME}/dna2prot:$${PYTHONPATH}' >>~/.bashrc
	echo 'export GCP_PROJECT=dream' >>~/.bashrc
	gsutil cp gs://train_sequences/* data/dream/
