import json
import numpy as np
import pandas as pd
import joblib as jb
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm


P = json.load(open("/data/tir/projects/tir7/user_data/priyansk/m3l_avg_img/predictions/instance_predictions_test.json"))

df_train = pd.read_json("/data/tir/projects/tir7/user_data/priyansk/MMCoQA/MMCoQA/MMCoQA_train.txt", lines=True).reset_index(drop=True)
df_test = pd.read_json("/data/tir/projects/tir7/user_data/priyansk/MMCoQA/MMCoQA/MMCoQA_test.txt", lines=True).reset_index(drop=True)


df_train['answer'] = df_train.apply(lambda x: (str(x['answer'][0]['answer'])), axis=1)
df_test['answer'] = df_test.apply(lambda x: (str(x['answer'][0]['answer'])), axis=1)

image_answers = " ".join(df_train.answer.values)

del df_train

passages_dict={}
with open("/data/tir/projects/tir7/user_data/priyansk/MMCoQA/MMCoQA/multimodalqa_final_dataset_pipeline_camera_ready_MMQA_texts.jsonl",'r') as f:
    lines=f.readlines()
    for line in tqdm(lines):
        line=json.loads(line.strip())
        passages_dict[line['id']]=line['text']

tables_dict={}
with open("/data/tir/projects/tir7/user_data/priyansk/MMCoQA/MMCoQA/multimodalqa_final_dataset_pipeline_camera_ready_MMQA_tables.jsonl",'r') as f:
    lines=f.readlines()
    for line in tqdm(lines):
        line=json.loads(line.strip())
        table_context = ''
        for row_data in line['table']["table_rows"]:
            for cell in row_data:
                table_context=table_context+" "+cell['text']
        tables_dict[line['id']]=table_context


def get_context(idx):
    if idx in passages_dict:
        return passages_dict[idx]
    elif idx in tables_dict:
        return tables_dict[idx]
    else:
        return image_answers

cnt = 0

for i, r in tqdm(df_test.iterrows(), total=len(df_test)):
    qid = r['qid']
    retrieved = P[qid]
    retrieved = [x['example_id'].split("*")[1] for x in retrieved]

    f = False
    for x in retrieved:
        if r['answer'].lower() in get_context(x).lower():
            f = True
            break
    
    if f:
        cnt += 1

print("Presence rate ", cnt/len(df_test))