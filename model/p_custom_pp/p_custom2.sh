#!/bin/bash


echo "----------checking if gpu available on current job-----------------"
# setting oneapi env and checking gpu
conda init bash
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
# python -m pip install torch==2.0.1a0 intel_extension_for_pytorch==2.0.110+xpu -f https://developer.intel.com/ipex-whl-stable-xpu


# installing intel extension for transformers
# pip install intel-extension-for-transformers

pip install torch
pip install intel-extension-for-pytorch --no-cache-dir
pip install transformers
pip install peft


python /home/uc4ddc6536e59d9d8f8f5069efdb4e25/mh_one_api/model/p_custom_pp/p_custom.py # modify the directory path to the location of the repo on system

# echo "finished precition"


