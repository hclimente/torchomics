#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -jc gpu-container_g8.24h
#$ -ac d=nvcr-pytorch-2204
#$ -t 3-9:2

. /fefs/opt/dgx/env_set/nvcr-pytorch-2204.sh
export PYTHONPATH=/home/hclimente/projects/dna2prot:${PYTHONPATH}

python scripts/train.py -kernel_size $SGE_TASK_ID  -nb_repeats 6 -p_dropout 0
