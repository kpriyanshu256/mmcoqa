
import numpy as np
import json
import os
import torch
import faiss
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from retriever_utils import RetrieverDataset, GenPassageRepDataset
import timeit
from tqdm import tqdm
import pytrec_eval

def gen_passage_rep(args, model, tokenizer, logger, passages_dict, tables_dict, images_dict, idx_id_list):
    # dataset, examples, features = load_and_cache_examples(args, tokenizer, evaluate=True, output_examples=True)
    DatasetClass = GenPassageRepDataset
    dataset = DatasetClass( args.gen_passage_rep_input,tokenizer,
                           args.load_small, 
                           passage_max_seq_length=args.passage_max_seq_length, passages_dict=passages_dict,
                                 tables_dict=tables_dict,
                                 images_dict=images_dict,idx_id_list=idx_id_list)

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    # eval_sampler = SequentialSampler(
    #     dataset) if args.local_rank == -1 else DistributedSampler(dataset)
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(
        dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=args.num_workers)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
        # model.to(f'cuda:{model.device_ids[0]}')

    # Eval!
    logger.info("***** Gem passage rep *****")
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    run_dict = {}
    start_time = timeit.default_timer()
    logger.info(" Writing to %s", args.gen_passage_rep_output)

    fout = open(args.gen_passage_rep_output, 'w')
    passage_ids = []
    passage_reps_list = []

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        example_ids = np.asarray(
            batch['example_id']).reshape(-1).tolist()
        passage_ids.extend(example_ids)
        batch = {k: v.to(args.device)
                 for k, v in batch.items() if k != 'example_id'}
        with torch.no_grad():
            inputs = {}
            inputs['passage_input_ids'] = batch['passage_input_ids']
            inputs['passage_attention_mask'] = batch['passage_attention_mask']
            inputs['passage_token_type_ids'] = batch['passage_token_type_ids']
            inputs['question_type'] = batch['question_type']
            inputs['image_input'] = batch['image_input'].type(torch.FloatTensor).to(args.device)
            outputs = model(**inputs)
            passage_reps = outputs[0]
            passage_reps_list.extend(to_list(passage_reps))
        
        # with open(args.gen_passage_rep_output, 'w') as fout:
        for example_id, passage_rep in zip(example_ids, to_list(passage_reps)):
            fout.write(json.dumps({'id': example_id, 'rep': passage_rep}) + '\n')
    fout.close()
    return passage_ids, passage_reps_list


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def retrieve(args, model, tokenizer, logger, passages_dict, tables_dict, images_dict, prefix=''):
    if prefix == 'test':
        eval_file = args.test_file
    else:
        eval_file = args.dev_file

    # dataset, examples, features = load_and_cache_examples(args, tokenizer, evaluate=True, output_examples=True)
    DatasetClass = RetrieverDataset
    dataset = DatasetClass(eval_file, tokenizer,
                           args.load_small, args.history_num,
                           query_max_seq_length=args.query_max_seq_length,
                           passage_max_seq_length=args.passage_max_seq_length,
                           is_pretraining=args.is_pretraining,
                           given_query=True,
                           given_passage=False,
                           only_positive_passage=args.only_positive_passage,
                           passages_dict=passages_dict,
                           tables_dict=tables_dict,
                           images_dict=images_dict)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    # eval_sampler = SequentialSampler(
    #     dataset) if args.local_rank == -1 else DistributedSampler(dataset)
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(
        dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=args.num_workers)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
        # model.to(f'cuda:{model.device_ids[0]}')
        
    # Eval!
    logger.info("***** Retrieve {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    all_qids = []
    all_query_reps = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        qids = np.asarray(
            batch['qid']).reshape(-1).tolist()
        batch = {k: v.to(args.device)
                 for k, v in batch.items() if k not in ['example_id', 'qid']}
        with torch.no_grad():
            inputs = {}
            inputs['query_input_ids'] = batch['query_input_ids']
            # print(inputs['query_input_ids'], inputs['query_input_ids'].size())
            inputs['query_attention_mask'] = batch['query_attention_mask']
            inputs['query_token_type_ids'] = batch['query_token_type_ids']
            outputs = model(**inputs)
            query_reps = outputs[0]

        all_qids.extend(qids)
        all_query_reps.extend(to_list(query_reps))

    return all_qids, all_query_reps

def evaluate_retriever(args, model, tokenizer, logger, passages_dict, tables_dict, images_dict, idx_id_list, prefix=""):
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)
    predict_dir = os.path.join(args.output_dir, 'predictions')
    if not os.path.exists(predict_dir) and args.local_rank in [-1, 0]:
        os.makedirs(predict_dir)

    passage_ids, passage_reps = gen_passage_rep(args, model, tokenizer, logger, passages_dict, tables_dict, images_dict, idx_id_list)
    passage_reps = np.asarray(passage_reps, dtype='float32')


    passage_ids, passage_reps = [], []
    with open(args.gen_passage_rep_output) as fin:
        for line in tqdm(fin):
            dic = json.loads(line.strip())
            passage_ids.append(dic['id'])
            passage_reps.append(dic['rep'])
    passage_reps = np.asarray(passage_reps, dtype='float32')

    qids, query_reps = retrieve(args, model, tokenizer, logger, passages_dict, tables_dict, images_dict, prefix=prefix)
    query_reps = np.asarray(query_reps, dtype='float32')
        
    index = faiss.IndexFlatIP(args.proj_size)
    index.add(passage_reps)
    D, I = index.search(query_reps, 1000)
    
    # print(qids, query_reps, passage_ids, passage_reps, D, I)

    run = {}
    for qid, retrieved_ids, scores in zip(qids, I, D):
        run[qid] = {passage_ids[retrieved_id]: float(score) for retrieved_id, score in zip(retrieved_ids, scores)}


    with open(args.qrels) as handle:
        qrels = json.load(handle)
    evaluator = pytrec_eval.RelevanceEvaluator(
            qrels, {'ndcg', 'set_recall'})
    metrics = evaluator.evaluate(run)

    mrr_list = [v['ndcg'] for v in metrics.values()]
    recall_list = [v['set_recall'] for v in metrics.values()]
    eval_metrics = {'NDCG': np.average(mrr_list), 'Recall': np.average(recall_list)}

    print(eval_metrics)
    return eval_metrics