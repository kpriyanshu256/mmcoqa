import json
import numpy as np
import joblib as jb
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm

EMB = "/data/tir/projects/tir7/user_data/priyansk/m3l_black_img_text/dev_blocks.txt"

Q = json.load(open("/data/tir/projects/tir7/user_data/priyansk/MMCoQA/MMCoQA/qrels.txt"))
P = json.load(open("/data/tir/projects/tir7/user_data/priyansk/m3l_avg_img/predictions/instance_predictions_test.json"))

DB = {}

with open(EMB) as fin:
    for line in tqdm(fin):
        dic = json.loads(line.strip())
        DB[dic['id']] = dic['rep']


similarity_scores = []

for k, v in tqdm(P.items()):
    truth_vec = list(Q[k].keys())[0]
    truth_vec = np.array(DB[truth_vec])

    retrieved = v
    retrieved = [x['example_id'].split("*")[1] for x in retrieved]
    retrieved_vec = [np.array(DB[x]) for x in retrieved]


    scores = [cosine_similarity(truth_vec.reshape(1, -1), x.reshape(1, -1)) for x in retrieved_vec]
    similarity_scores.append(np.mean(scores))

print(f'Similarity Mean {np.mean(similarity_scores)}\tStd {np.std(similarity_scores)}')



