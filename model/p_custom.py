
import numpy as np
import pandas as pd
import torch


batch_process=0


# specify start index for continuing...
start_index=[0,10000,20000]
end_index=[10000,20000,30000]
file_name=['0_10k.csv','10_20k.csv','20_30k.csv']




test_path="/home/u131168/mh_one_api/data/test.csv"
test_data=pd.read_csv(test_path)



# !pip install bitsandbytes
# !pip install accelerate
# !pip install scipy

# load model
from peft import AutoPeftModelForSeq2SeqLM
from transformers import AutoTokenizer
import torch

model_path="/home/u131168/mh_one_api/model/ft_models/flan-t5-xl_peft_finetuned_model/checkpoint-18000"
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")
model = AutoPeftModelForSeq2SeqLM.from_pretrained(model_path, )

# ------predict--------------
with open(f"/home/u131168/mh_one_api/data/custom_pred/{file_name[batch_process]}", "a+") as file1:

    for i in range(start_index[batch_process],end_index[batch_process],1):
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
        file1.writelines(f"{i+i1} $$ {res[i1]}\n" for i1 in range(len(res)))
        print(f"-------------wrote {i} to {i} preds")


