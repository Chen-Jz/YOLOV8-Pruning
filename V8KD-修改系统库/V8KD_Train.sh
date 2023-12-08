#!/bin/bash

python V8KD/0-Clear.py

# Train
torchrun --nproc_per_node=4 --master_port 12326 V8KD/1-Train.py

# Prune
python V8KD/2-Prune.py

python V8KD/3-Train2KD.py

# Finetun
python V8KD/4-Finetune.py