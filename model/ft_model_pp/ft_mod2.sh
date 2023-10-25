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




echo "starting fine tuning model"
cd "/home/u131168/mh_one_api/model/ft_model_pp/itp" # modify the directory path to the location of the repo on system
pip install -r "requirements.txt"

# To use ccl as the distributed backend in distributed training on CPU requires to install below requirement.
python -m pip install oneccl_bind_pt -f https://developer.intel.com/ipex-whl-stable-cpu

#installing intel extension for pytorch for GPU
python -m pip install torch==2.0.1a0 torchvision==0.15.2a0 intel_extension_for_pytorch==2.0.110+xpu -f https://developer.intel.com/ipex-whl-stable-xpu

# installing intel extension for transformers
pip install intel-extension-for-transformers


export train_file="/home/u131168/mh_one_api/data/f_traind_v1.csv" # modify the directory path to the location of the repo on system

export model_path="google/flan-t5-xl"

export checkpoint_dir="/home/u131168/mh_one_api/model/ft_models/flan-t5-xl_peft_ft_v2/"
export checkpoint_name=$(ls $checkpoint_dir | grep checkpoint | tail -2 | head -n 1)
export checkpoint_path="$checkpoint_dir$checkpoint_name"
echo $checkpoint_path

# export output_dir="/home/u131168/mh_one_api/model/ft_models/flan-t5-xl_peft_ft_v1"
export output_dir="$checkpoint_dir"


python finetune_seq2seq.py \
        --model_name_or_path $model_path \
        --resume_from_checkpoint $checkpoint_path \
        --bf16 True \
        --train_file $train_file \
        --per_device_train_batch_size 2 \
        --per_device_eval_batch_size 2 \
        --gradient_accumulation_steps 1 \
        --do_train \
        --learning_rate 1.0e-5 \
        --warmup_ratio 0.03 \
        --weight_decay 0.0 \
        --num_train_epochs 1 \
        --logging_steps 10 \
        --save_steps 100 \
        --save_total_limit 2 \
        --overwrite_output_dir \
        --output_dir $output_dir \
        --peft lora

echo "finished fine tuning model"
