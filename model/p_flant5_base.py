
import numpy as np
import pandas as pd
import torch


batch_process=1


# specify start index for continuing...
start_index=[0,19380,20000]
# start_index=[0,10000,20000]
end_index=[10000,20000,28548]
file_name=['0_10k.csv','10_20k.csv','20_30k.csv']



test_path="/home/u131168/mh_one_api/data/test.csv"
test_data=pd.read_csv(test_path)


# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
model_path="google/flan-t5-xl"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)


for i in range(start_index[batch_process],end_index[batch_process],20):
    prompts = test_data.loc[i:i+19,['Story','Question']].values.tolist()
    # print(prompts)
    res=[]
    input_ids = tokenizer(prompts, return_tensors="pt" ,padding=True,truncation=True, max_length=512).input_ids
    # sample up to 30 tokens
    torch.manual_seed(0)  # doctest: +IGNORE_RESULT
    outputs = model.generate(input_ids, do_sample=True, max_length=20)
    res+=tokenizer.batch_decode(outputs, skip_special_tokens=True)
    with open(f"/home/u131168/mh_one_api/data/flant5_pred/{file_name[batch_process]}", "a+") as file1:
        # Writing data to a file
        file1.writelines(f"{i+i1} $$ {res[i1]}\n" for i1 in range(len(res)))

    print(f"-------------wrote {i} to {i+20} preds")


