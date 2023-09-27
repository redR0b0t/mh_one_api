
import numpy as np
import pandas as pd
import torch
import subprocess

mh_dir='mh_one_api'


# batch_process=2
# # specify start index for continuing...
# start_index=[0,9500,19000]
# end_index=[9500,19000,29000]
# file_name=['0_10k.csv','10_20k.csv','20_30k.csv']
pred_file_name="full_pred1.csv"
pred_file_path=f"/home/u131168/{mh_dir}/data/custom_pred/{pred_file_name}"
# start_index=subprocess.get_output("tail {} -n 1 | awk -F' ' '{print $1}'".format(file_name))
start_index=subprocess.check_output("tail "+pred_file_path+" -n 1 | awk -F' ' '{print $1}'",shell=True)
print(start_index)
start_index=int(start_index)+1
end_index=29000

print(f"---------------got start index============{start_index}")




test_path=f"/home/u131168/{mh_dir}/data/test.csv"
test_data=pd.read_csv(test_path)



# !pip install bitsandbytes
# !pip install accelerate
# !pip install scipy

# load model
from peft import AutoPeftModelForSeq2SeqLM
from transformers import AutoTokenizer
import torch

model_path=f"/home/u131168/{mh_dir}/model/ft_models/flan-t5-xl_peft_finetuned_model/checkpoint-36000"
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")
model = AutoPeftModelForSeq2SeqLM.from_pretrained(model_path, )

# ------predict--------------
for i in range(start_index,end_index,1):
    print(f"predicting {i} prompt")
    prompts = test_data.loc[i,['Story','Question']].values.tolist()
    prompts=[prompts]
    t_prompts=[]
    for p in prompts:
        context=str(p[0]).replace(r"\n",'.')
        question=p[1]
        t_prompts+=[f"paragraph: {context}\n\n Answer the following question from the above paragraph: {question}"]
        # t_prompts+=[f"input: {context}\n\ninstruction: {question}"]
    prompts=t_prompts
        
    # print(prompts)
    res=[]
    input_ids = tokenizer(prompts, return_tensors="pt" ,padding=True,truncation=True, max_length=512).input_ids
    # sample up to 30 tokens
    torch.manual_seed(0)  # doctest: +IGNORE_RESULT
    outputs = model.generate(input_ids=input_ids, do_sample=True, max_length=20)
    res+=tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
    # Writing data to a file
    with open(pred_file_path, "a+") as file1:
        file1.writelines(f"{i+i1} $$ {res[i1]}\n" for i1 in range(len(res)))
    print(f"-------------wrote {i} to {i} preds")


