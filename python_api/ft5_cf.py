
import datetime
import json
import threading
import time

from pandas import Timestamp
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import intel_extension_for_pytorch as ipex

use_intel_optimization=True



global uname
uname="uc4ddc6536e59d9d8f8f5069efdb4e25"

def predict(context,question):
    print("-------------generating prediction-----------------")
    # data = json.loads(request.data)
    # data.context=data.context.replace('\n','\\n')
    # data.context=data.context.replace('\t','\\t')

    # context=request.args.get('context')
    # question=request.args.get('question')
    context=context.replace('\n','\\n')
    context=context.replace('\t','\\t')



    input=f'paragraph: {context}'
    instruction=f'Answer the following question from the paragraph: Question : {question}'
    prompts=[[input,instruction]]
    res=[]
    input_ids = tokenizer(prompts, return_tensors="pt" ,padding=True,truncation=True, max_length=512).input_ids

    if use_intel_optimization==True:
        # input_ids=input_ids.to('xpu')
        start_time = time.time()
        outputs = model.generate(input_ids=input_ids, do_sample=True, max_length=150)
        pti=time.time() - start_time
        print(f"--------------time taken by optimized model={pti} seconds")
    else:
        start_time = time.time()
        outputs = model.generate(input_ids=input_ids, do_sample=True, max_length=150)
        pt=time.time() - start_time
        print(f"--------------time taken by normal model={pt} seconds")

    res+=tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return  { 'output': f'{res[0]}' }

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
    checkpoint_dir=f"/home/{uname}/{mh_dir}/model/ft_models/flan-t5-xl_peft_ft_v2/"
    checkpoint_name=subprocess.check_output(f"ls {checkpoint_dir} | grep checkpoint | tail -1",shell=True)
    checkpoint_name=str(checkpoint_name).replace("b'","").replace("\\n'","")
    checkpoint_path=checkpoint_dir+checkpoint_name
    model_path=checkpoint_path

    # model_path=f"/home/{uname}/{mh_dir}/model/ft_models/flan-t5-xl_peft_finetuned_model/checkpoint-36000"
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")
    model = AutoPeftModelForSeq2SeqLM.from_pretrained(model_path, )
    print("-------------loaded custom model-----------------")
    if use_intel_optimization==True:
        print("--------------optimizing model---------------")
        # model = model.to('xpu')
        model=ipex.optimize(model)
        print("--------------optimized model---------------")

    print("--------------ready----------------------")

def listen_msgs():
    coll_name='mhi_pred_app'
    user_uid='mhi_pred'
        
    # Use a service account.
    cred = credentials.Certificate(f'/home/{uname}/mh_one_api/python_api/mh-pred-app-21be554adb6d.json')

    app = firebase_admin.initialize_app(cred)

    db = firestore.client()
    
    # Create an Event for notifying main thread.
    callback_done = threading.Event()

    # Create a callback on_snapshot function to capture changes
    def on_snapshot(doc_snapshot, changes, read_time):
        for doc in doc_snapshot:
            print(f"Received document snapshot: {doc.id}")
            print(doc.to_dict())
            data=doc.to_dict()

            pred_res=predict(data['context'],data['message'])
            # pred_res={'output':"model response"}

            data['senderId']='chatbot@red'
            data['message']=pred_res['output']
            data['timestamp']=datetime.datetime.utcnow()
            print(f'sending response: '+data['message'])
            doc_id=str(round(time.time() * 1000))
            db.collection(coll_name).document(user_uid).collection('allMessages').document(doc_id).set(data)
            print(f"----------sent----------------with id: {doc_id} ")

        callback_done.set()

    doc_ref = db.collection(coll_name).document(user_uid).collection('userMessages').document("message")

    # Watch the document
    doc_watch = doc_ref.on_snapshot(on_snapshot)
    

    while True:
        print('', end='', flush=True)
        time.sleep(1)



if __name__ == '__main__':
    init_model()
    try:
        listen_msgs()
    except Exception as e:
        print(f'error occured:\n\n\n {e}')  