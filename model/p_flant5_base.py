
import numpy as np
import pandas as pd
import torch


# batch_process=1


# specify start index for continuing...
# start_index=[0,19380,20000]
# start_index=[0,10000,20000]
# end_index=[10000,20000,28548]
# file_name=['0_10k.csv','10_20k.csv','20_30k.csv']

test_path="/home/u131168/mh_one_api/data/test.csv"
test_data=pd.read_csv(test_path)


# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
model_path="google/flan-t5-xl"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

def predict_res(li):

    # Load model directly
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    model_path="google/flan-t5-xl"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        
    si=li[0]
    ei=li[1]
    fname=li[2]
   
    print(f"--------------starting for batch {si} to {ei}, writing to file:{fname}")
    for i in range(si,ei,1):
        prompts = test_data.loc[i,['Story','Question']].values.tolist()
        prompts=[prompts]
        # print(prompts)
        res=[]
        input_ids = tokenizer(prompts, return_tensors="pt" ,padding=True,truncation=True, max_length=512).input_ids
        # sample up to 30 tokens
        torch.manual_seed(0)  # doctest: +IGNORE_RESULT
        outputs = model.generate(input_ids, do_sample=True, max_length=20)
        res+=tokenizer.batch_decode(outputs, skip_special_tokens=True)
        print("--------got res now writing to file----------")
        with open(f"/home/u131168/mh_one_api/data/flant5_pred_pp/{fname}", "a+") as file1:
            print(f"-----------------file {fname} opened")
            # Writing data to a file
            file1.writelines(f"{i+i1} $$ {res[i1]}\n" for i1 in range(len(res)))

        print(f"-------------wrote {i} to {i} preds to file:{fname}")



from multiprocessing import Pool
pool = Pool(processes=10)

batch_start=0
batch_end=10000
batch_inc=batch_end
# batch_inc=2000

lsi=[[i,i+batch_inc,f"{i}_{i+batch_inc}k.csv"] for i in range(batch_start,batch_end,batch_inc)]
# lei=[i+batch_size for i in lsi]

# res=pool.map(predict_res,lsi)
predict_res(lsi[0])

