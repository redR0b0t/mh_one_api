#!/bin/sh
echo "staring prediction"
# conda acti
pip install torch
pip install transformers
pip install peft

python /home/u131168/mh_one_api/model/p_custom_pp/p_custom.py

echo "finished precition"


# sbatch -x idc-beta-batch-pvc-node-[03,20] --priority 0 --job-name pc0 p_custom.sh
# sbatch -x idc-beta-batch-pvc-node-[03,20] --priority 0 --job-name pc1 p_custom.sh
# sbatch -x idc-beta-batch-pvc-node-[03,20] --priority 0 --job-name pc2 p_custom.sh


# idc-beta-batch-pvc-node-08:
# idc-beta-batch-pvc-node-10
