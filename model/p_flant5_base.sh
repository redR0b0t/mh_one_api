#!/bin/sh
echo "staring prediction"
# conda acti
pip install torch
pip install transformers

python /home/u131168/mh_one_api/model/p_flant5_base.py

echo "finished precition"

# sbatch mh_one_api/model/p_flant5_base.sh
# idc-beta-batch-pvc-node-08:
# idc-beta-batch-pvc-node-10