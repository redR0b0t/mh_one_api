
import json
from flask import Flask, jsonify, request
app = Flask(__name__)






@app.route('/predict', methods=['POST'])
def predict():
    print("-------------generating prediction-----------------")
    data = json.loads(request.data)
    data.context=data.context.replace('\n','\\n')
    data.context=data.context.replace('\t','\\t')

    data.context=f'paragraph: {data.context}'
    data.question=f'Answer the following question from the paragraph: Question : {data.question}'
    res=[]
    input_ids = tokenizer(prompts, return_tensors="pt" ,padding=True,truncation=True, max_length=512).input_ids
    outputs = model.generate(input_ids=input_ids, do_sample=True, max_length=150)
    res+=tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return '', 201, { 'response': f'{res}' }

def init_model():
    print("-------------loading model------------------")
    from peft import AutoPeftModelForSeq2SeqLM
    from transformers import AutoTokenizer
    import torch
    import numpy as np
    import pandas as pd
    import torch
    import subprocess
    global model
    global tokenizer

    mh_dir="mh_one_api"
    checkpoint_dir=f"/home/u131168/{mh_dir}/model/ft_models/flan-t5-xl_peft_ft_v2/"
    checkpoint_name=subprocess.check_output(f"ls {checkpoint_dir} | grep checkpoint | tail -1",shell=True)
    checkpoint_name=str(checkpoint_name).replace("b'","").replace("\\n'","")
    checkpoint_path=checkpoint_dir+checkpoint_name
    model_path=checkpoint_path

    # model_path=f"/home/u131168/{mh_dir}/model/ft_models/flan-t5-xl_peft_finetuned_model/checkpoint-36000"
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")
    model = AutoPeftModelForSeq2SeqLM.from_pretrained(model_path, )
    print("-------------loaded custom model-----------------")



if __name__ == '__main__':
   init_model()
   app.run(port=5000)