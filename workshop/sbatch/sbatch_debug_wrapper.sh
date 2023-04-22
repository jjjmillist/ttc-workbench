#!/bin/bash

#SBATCH --mem 64000
#SBATCH --partition LARGE-G2
#SBATCH --output logs

ifconfig
CODEROOT=. PYTHONPATH=. PYTHONUNBUFFERED=1 python -m debugpy --listen 5000 --wait-for-client $@