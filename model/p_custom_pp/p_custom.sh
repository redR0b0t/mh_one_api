#!/bin/bash


# sbatch -x idc-beta-batch-pvc-node-[03,20,21] --priority 0 --job-name pci1 --mem=0 --exclusive p_custom.sh
export batch_script="p_custom.sh"
# -----------set new job dep--------------
echo "got current job name=$SLURM_JOB_NAME"
export cji=$(echo -n $SLURM_JOB_NAME | tail -c 1)
export nji=$(( cji + 1 ))
export njname="pci$nji"
echo "new job name=$njname"
# nodes with gpus=[19,05]
export njid=$(sbatch -x idc-beta-batch-pvc-node-[03,09,14,20,21] --priority 0 --job-name $njname --begin=now+60 --dependency=afterany:$SLURM_JOB_ID --mem=0 --exclusive $batch_script | sed -n 's/.*job //p')
echo "-----------------new job created with id: $njid"
# -------------------end------------------


echo "----------checking if gpu available on current job-----------------"
# oneapi env and checking gpu
echo "-------------------------------------------"
groups  # Key group is render, PVC access is unavailable if you do not have render group present.
source /opt/intel/oneapi/setvars.sh --force
sycl-ls
export num_gpu="$(sycl-ls |grep "GPU" |wc -l)"
echo "num_gpu=$num_gpu\n"
export num_cpu="$(sycl-ls |grep "Xeon" |wc -l)"
echo "num_cpu=$num_cpu\n"
if [ $num_gpu == 0 && $num_cpu == 1] 
then 
    echo "---GPU not available exiting--------"
    scancel $SLURM_JOB_ID
fi 
echo "-------------------------------------------"



echo "staring prediction"


#installing intel extension for pytorch for GPU
# python -m pip install torch==2.0.1a0 torchvision==0.15.2a0 intel_extension_for_pytorch==2.0.110+xpu -f https://developer.intel.com/ipex-whl-stable-xpu

# installing intel extension for transformers
# pip install intel-extension-for-transformers

pip install torch
pip install transformers
pip install peft



python /home/uc4ddc6536e59d9d8f8f5069efdb4e25/mh_one_api/model/p_custom_pp/p_custom.py

echo "finished precition"


# sbatch -x idc-beta-batch-pvc-node-[03,20] --priority 0 --job-name pc0 p_custom.sh
# sbatch -x idc-beta-batch-pvc-node-[03,20] --priority 0 --job-name pc1 p_custom.sh
# sbatch -x idc-beta-batch-pvc-node-[03,20] --priority 0 --job-name pc2 p_custom.sh


# idc-beta-batch-pvc-node-08:
# idc-beta-batch-pvc-node-10
