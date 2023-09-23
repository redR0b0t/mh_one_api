#!/bin/sh
echo "starting fine tuning model"
cd "/home/u131168/mh_one_api/model/intel-extension-for-transformers/workflows/chatbot/fine_tuning"
pip install -r "requirements.txt"
cd "/home/u131168/mh_one_api/model/intel-extension-for-transformers/workflows/chatbot/fine_tuning/instruction_tuning_pipeline"

pip install git+https://github.com/huggingface/transformers

export train_file="/home/u131168/mh_one_api/data/train_split/full_traind.csv"

export model_path="google/flan-t5-xl"
export checkpoint_path="/home/u131168/mh_one_api/model/ft_models/flan-t5-xl_peft_finetuned_model/checkpoint-26000"

export output_dir="/home/u131168/mh_one_api/model/ft_models/flan-t5-xl_peft_finetuned_model"

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
        --num_train_epochs 3 \
        --logging_steps 10 \
        --save_steps 1000 \
        --save_total_limit 2 \
        --overwrite_output_dir \
        --output_dir $output_dir \
        --peft lora

echo "finished fine tuning model"
