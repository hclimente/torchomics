#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -jc gpu-container_g8.24h
#$ -ac d=nvcr-pytorch-2204
#$ -t 1-10

. /fefs/opt/dgx/env_set/nvcr-pytorch-2204.sh
export PYTHONPATH=/home/hclimente/projects/dna2prot:${PYTHONPATH}

DROPOUT=`echo "print($SGE_TASK_ID/10.0)" | python`
SEED=2

python scripts/train.py -kernel_size 3 -nb_repeats 4 -p_dropout $DROPOUT -seed $SEED
python scripts/train.py -kernel_size 7 -nb_repeats 8 -p_dropout $DROPOUT -seed $SEED
python scripts/train.py -kernel_size 9 -nb_repeats 8 -p_dropout $DROPOUT -seed $SEED
python scripts/train.py -kernel_size 9 -nb_repeats 9 -p_dropout $DROPOUT -seed $SEED
