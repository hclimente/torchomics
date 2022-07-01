#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -jc gpu-container_g8.24h
#$ -ac d=nvcr-pytorch-2204
#$ -t 1-10

. /fefs/opt/dgx/env_set/nvcr-pytorch-2204.sh
export PYTHONPATH=/home/hclimente/projects/dna2prot:${PYTHONPATH}

for kernel_size in {3..9..2}
do
    python scripts/train.py -kernel_size $kernel_size -nb_repeats $SGE_TASK_ID -p_dropout 0 -seed 2
done
