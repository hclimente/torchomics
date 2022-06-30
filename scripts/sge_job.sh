#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -jc gpu-container_g8.24h
#$ -ac d=nvcr-pytorch-2204

. /fefs/opt/dgx/env_set/nvcr-pytorch-2204.sh
export PYTHONPATH=/home/hclimente/projects/dna2prot:${PYTHONPATH}

for kernel_size in {3..9..2}
do
    for nb_repeats in {1..5}
    do
        python scripts/train.py -kernel_size $kernel_size -nb_repeats $nb_repeats -p_dropout 0
    done
done
