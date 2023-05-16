#!/bin/bash

#SBATCH --partition LARGE-G2
#SBATCH --gres=gpu:0
#SBATCH --output logs

CODEROOT=. PYTHONPATH=. PYTHONUNBUFFERED=1 ../pyenv/bin/python $@