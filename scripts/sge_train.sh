#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -jc gpu-container_g8.24h
#$ -ac d=nvcr-pytorch-2204
#$ -t 3-5

. /fefs/opt/dgx/env_set/nvcr-pytorch-2204.sh
export PYTHONPATH=/home/hclimente/projects/dna2prot:${PYTHONPATH}

DROPOUT=`echo "print($SGE_TASK_ID/10.0)" | python`
SEED=4

python scripts/train.py -kernel_size 3 -nb_repeats 4 -p_dropout $DROPOUT -seed $SEED -transforms mixup
