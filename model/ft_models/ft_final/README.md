---
library_name: peft
pipeline_tag: question-answering
---



## Model description

Flan-t5-xl finetuned model on dataset provided by intel|machinehack for QnA.

## Intended uses & limitations

More information needed




### Training hyperparameters

The following hyperparameters were used during training:
        --per_device_train_batch_size 2 \
        --per_device_eval_batch_size 2 \
        --gradient_accumulation_steps 1 \
        --do_train \
        --learning_rate 1.0e-5 \
        --warmup_ratio 0.03 \
        --weight_decay 0.0 \
        --num_train_epochs 1 \

### Training results

{
    "epoch": 1.08,
    "train_loss": 0.0,
    "train_runtime": 0.017,
    "train_samples": 66611,
    "train_samples_per_second": 3929048.542,
    "train_steps_per_second": 1964553.764
}



### Framework versions


- PEFT 0.5.0