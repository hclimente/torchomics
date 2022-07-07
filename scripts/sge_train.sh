#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -jc gpu-container_g8.24h
#$ -ac d=nvcr-pytorch-2204
#$ -t 1-3

. /fefs/opt/dgx/env_set/nvcr-pytorch-2204.sh
export PYTHONPATH=/home/hclimente/projects/dna2prot:${PYTHONPATH}

ALPHA=`echo "print(10**($SGE_TASK_ID-2))" | python`
SEED=2

python scripts/train.py -kernel_size 3 -nb_repeats 4 -p_dropout 0 -alpha $ALPHA -seed $SEED -transforms mixup
