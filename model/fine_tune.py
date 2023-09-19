# pip install intel-extension-for-transformers
# from intel_extension_for_transformers.workflows.chatbot.finetuning.instruction_tuning_pipeline import finetune_seq2seq

import subprocess

train_path="/home/u131168/mh_one_api/data/train.csv"
ft_model=f'python finetune_seq2seq.py \
        --model_name_or_path "google/flan-t5-xl" \
        --bf16 True \
        --train_file "/home/u131168/mh_one_api/model/formatted_td.csv" \
        --per_device_train_batch_size 2 \
        --per_device_eval_batch_size 2 \
        --gradient_accumulation_steps 1 \
        --do_train \
        --learning_rate 1.0e-1 \
        --warmup_ratio 0.03 \
        --weight_decay 0.0 \
        --num_train_epochs 2 \
        --logging_steps 10 \
        --save_steps 20 \
        --save_total_limit 2 \
        --overwrite_output_dir \
        --output_dir ./flan-t5-xl_peft_finetuned_model \
        --peft lora'


subprocess.getoutput(ft_model)
