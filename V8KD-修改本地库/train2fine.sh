#!/bin/bash

python 0-Clear.py

# Train
torchrun --nproc_per_node=4 --master_port 12326 1-Train.py

# Prune
python 2-Prune.py

python 3-Train2KD.py

# Finetun
python 4-Finetune.py