import os
import argparse
import logging
import sys
import random
import glob
import timeit
import json
import linecache
import faiss
import numpy as np
import pickle as pkl
from tqdm import tqdm, trange
import pytrec_eval
import scipy as sp
from copy import copy
from PIL import Image
import joblib as jb

os.environ['TOKENIZERS_PARALLELISM'] = "False"

import torch
from torch.utils.data import DataLoader, TensorDataset

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from transformers import WEIGHTS_NAME, AutoTokenizer, AutoConfig, AlbertConfig, AlbertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from utils import (AverageMeter, RawResult, write_final_predictions, write_predictions_v2, 
                   get_retrieval_metrics, gen_reader_features_v2)

from retriever_utils import RetrieverDataset
from modeling import Pipeline, Reader, Retriever, AlbertForRetrieverOnlyPositivePassage
from scorer import quac_eval

logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.INFO)

logger = logging.getLogger(__name__)

def seed_all(seed = 42):
    """
    Fix seed for reproducibility
    """
    # python RNG
    import random
    random.seed(seed)

    # pytorch RNGs
    import torch
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    # numpy RNG
    import numpy as np
    np.random.seed(seed)

def to_list(tensor):
    return tensor.detach().cpu().tolist()

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def gen_query_reps(args, model, batch):
    model.eval()
    batch = {k: v.to(args.device) for k, v in batch.items() 
             if k not in ['example_id', 'qid', 'question_text', 'answer_text', 'answer_start']}

    with torch.no_grad():
        inputs = {}
        inputs['query_input_ids'] = batch['query_input_ids']
        inputs['query_attention_mask'] = batch['query_attention_mask']
        inputs['query_token_type_ids'] = batch['query_token_type_ids']
        outputs = model.retriever(**inputs)
        query_reps = outputs[0]

    return query_reps


def retrieve(args, qids, qid_to_idx, query_reps,
            item_ids, item_id_to_idx, item_reps,
            qrels, qrels_sparse_matrix,
            gpu_index, include_positive_passage = False):

    query_reps = query_reps.detach().cpu().numpy()
    D, I = gpu_index.search(query_reps, args.top_k_for_retriever)

    pidx_for_retriever = np.copy(I)
    qidx = [qid_to_idx[qid] for qid in qids]
    qidx_expanded = np.expand_dims(qidx, axis=1)
    qidx_expanded = np.repeat(qidx_expanded, args.top_k_for_retriever, axis=1)
    labels_for_retriever = qrels_sparse_matrix[qidx_expanded, pidx_for_retriever].toarray()

    if include_positive_passage:
        # ensuring there is atleast 1 positive example
        for i, (qid, labels_per_query) in enumerate(zip(qids, labels_for_retriever)):
                has_positive = np.sum(labels_per_query)
                if not has_positive:
                    positive_pid = list(qrels[qid].keys())[0]
                    positive_pidx = item_id_to_idx[positive_pid]
                    pidx_for_retriever[i][-1] = positive_pidx

        labels_for_retriever = qrels_sparse_matrix[qidx_expanded, pidx_for_retriever].toarray()
        assert np.sum(labels_for_retriever) >= len(labels_for_retriever)

    pids_for_retriever = item_ids[pidx_for_retriever]
    passage_reps_for_retriever = item_reps[pidx_for_retriever]


    scores = D[:, :args.top_k_for_reader]
    retriever_probs = sp.special.softmax(scores, axis=1)
    pidx_for_reader = I[:, :args.top_k_for_reader]
    
    qidx_expanded = np.expand_dims(qidx, axis=1)
    qidx_expanded = np.repeat(qidx_expanded, args.top_k_for_reader, axis=1)
    
    labels_for_reader = qrels_sparse_matrix[qidx_expanded, pidx_for_reader].toarray()

    if include_positive_passage:
        for i, (qid, labels_per_query) in enumerate(zip(qids, labels_for_reader)):
                has_positive = np.sum(labels_per_query)
                if not has_positive:
                    positive_pid = list(qrels[qid].keys())[0]
                    positive_pidx = item_id_to_idx[positive_pid]
                    pidx_for_reader[i][-1] = positive_pidx
        labels_for_reader = qrels_sparse_matrix[qidx_expanded, pidx_for_reader].toarray()
        assert np.sum(labels_for_reader) >= len(labels_for_reader)

    pids_for_reader = item_ids[pidx_for_reader]
    passages_for_reader = get_passages(pidx_for_reader)
    # logger.info(f"Get {pidx_for_reader}")

    return {
        'qidx': qidx,
        'pidx_for_retriever': pidx_for_retriever,
        'pids_for_retriever': pids_for_retriever,
        'passage_reps_for_retriever': passage_reps_for_retriever,
        'labels_for_retriever': labels_for_retriever,
        'retriever_probs': retriever_probs,
        'pidx_for_reader': pidx_for_reader,
        'pids_for_reader': pids_for_reader,
        'passages_for_reader': passages_for_reader, 
        'labels_for_reader': labels_for_reader,
        'no_cut_retriever_probs': D,
    }

def get_passage(i):
    if itemid_modalities[i] == 'text':
        item_context = passages_dict[item_ids[i]]
    elif itemid_modalities[i] == 'table':
        item_context = tables_dict[item_ids[i]]
    elif itemid_modalities[i] == 'image':
        item_context = images_dict[item_ids[i]]
    return item_context
get_passages = np.vectorize(get_passage)


def train(args, train_dataset, model, retriever_tokenizer, reader_tokenizer):
    tb_writer = SummaryWriter(os.path.join(args.output_dir, 'logs'))

    args.train_batch_size = args.per_gpu_train_batch_size
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, num_workers=args.num_workers)

    t_total = len(train_dataloader) * args.num_train_epochs
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    args.warmup_steps = int(t_total * args.warmup_portion)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d",args.per_gpu_train_batch_size)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 1


    for _ in trange(int(args.num_train_epochs), desc='Training'):

        train_loss = AverageMeter()
        qa_train_loss = AverageMeter()
        ret_train_loss = AverageMeter()

        pbar = tqdm(train_dataloader, total=len(train_dataloader), leave=False)

        for step, batch in enumerate(pbar):
            
            # if step > 25:
            #     break

            qids = np.asarray(batch['qid']).reshape(-1).tolist()
            question_texts = np.asarray(batch['question_text']).reshape(-1).tolist()
            answer_texts = np.asarray(batch['answer_text']).reshape(-1).tolist()
            answer_starts = np.asarray(batch['answer_start']).reshape(-1).tolist()

            query_reps = gen_query_reps(args, model, batch)

            retrieval_results = retrieve(args, qids, qid_to_idx, query_reps,
                                         item_ids, item_id_to_idx, item_reps,
                                         qrels, qrels_sparse_matrix,
                                         gpu_index, include_positive_passage=True)

            passage_reps_for_retriever = retrieval_results['passage_reps_for_retriever']
            labels_for_retriever = retrieval_results['labels_for_retriever']

            pids_for_reader = retrieval_results['pids_for_reader']
            passages_for_reader = retrieval_results['passages_for_reader']
            labels_for_reader = retrieval_results['labels_for_reader']

            logger.info(f'QID {qids[0]} Reader IDS {pids_for_reader[0]}')

            model.train()


            # inputs = {'query_input_ids': batch['query_input_ids'].to(args.device),
            #           'query_attention_mask': batch['query_attention_mask'].to(args.device),
            #           'query_token_type_ids': batch['query_token_type_ids'].to(args.device),
            #           'passage_rep': torch.from_numpy(passage_reps_for_retriever).to(args.device),
            #           'retrieval_label': torch.from_numpy(labels_for_retriever).to(args.device)}
            # retriever_outputs = model.retriever(**inputs)
            # retriever_loss = retriever_outputs[0]
            # # retriever_loss = torch.tensor([0.0])

            
            # retriever_loss.backward()



            reader_batch = gen_reader_features_v2(
                                    qids, question_texts, answer_texts, answer_starts,
                                    pids_for_reader, passages_for_reader, labels_for_reader,
                                    reader_tokenizer, args.reader_max_seq_length, 
                                    is_training=True, 
                                    itemid_modalities=itemid_modalities,
                                    item_id_to_idx=item_id_to_idx,
                                    images_titles=images_titles
            )

            internal_batch = len(reader_batch['input_ids'])

            filter_batch = {
                'input_ids': [],
                'input_mask': [],
                'segment_ids': [],
                'start_positions': [],
                'end_positions': [],
                'retrieval_label': [],
                'image_input': [],
            }
            
            for i in range(len(reader_batch['input_ids'])):
                if torch.sum(reader_batch['start_positions'][i]) > 0:
                    for k, v in filter_batch.items():
                        filter_batch[k].append(reader_batch[k][i])
            
            if len(filter_batch['input_ids']) == 0:
                continue

            for k, v in filter_batch.items():
                filter_batch[k] = torch.stack(v)
            
            reader_batch = filter_batch

            mbatch_ds = TensorDataset(reader_batch['input_ids'], 
                                reader_batch['input_mask'],
                                reader_batch['segment_ids'],
                                reader_batch['start_positions'],
                                reader_batch['end_positions'],
                                reader_batch['retrieval_label'],
                                reader_batch['image_input'],
            )

            mbatch_dl = DataLoader(mbatch_ds, batch_size=32, num_workers=4)

            for mini_batch in tqdm(mbatch_dl, desc="Mini-Batch", leave=False, disable=True):
                inputs = {
                    'input_ids': mini_batch[0].to(args.device),
                    'attention_mask': mini_batch[1].to(args.device),
                    'token_type_ids': mini_batch[2].to(args.device),
                    'start_positions': mini_batch[3].to(args.device),
                    'end_positions': mini_batch[4].to(args.device),
                    'retrieval_label': mini_batch[5].to(args.device),
                    'image_input': mini_batch[6].to(args.device),
                }

                reader_outputs = model.reader(**inputs)
                reader_loss, qa_loss, rerank_loss = reader_outputs[0:3]
                
                loss = reader_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                model.zero_grad()

                train_loss.update(loss.item(), len(mini_batch[0]))
                qa_train_loss.update(qa_loss.item(), len(mini_batch[0]))
                ret_train_loss.update(rerank_loss.item(), len(mini_batch[0]))

                pbar.set_postfix(loss=train_loss.avg)

                logger.info(f'Loss {loss.item()} Avg Loss {train_loss.avg}')
                logger.info(f'QA Loss {qa_loss.item()} Avg Loss {qa_train_loss.avg}')
                logger.info(f'Rerank Loss {loss.item()} Avg Loss {ret_train_loss.avg}')



            if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                tb_writer.add_scalar('lr', scheduler.get_last_lr()[0], global_step)
                tb_writer.add_scalar('avg_loss', train_loss.avg, global_step)
                tb_writer.add_scalar('avg_retriever_loss', ret_train_loss.avg, global_step)
                tb_writer.add_scalar('qa_loss', qa_train_loss.avg, global_step)

            if args.save_steps > 0 and global_step % args.save_steps == 0:
                output_dir = os.path.join(args.output_dir, f'checkpoint-{global_step}')
                retriever_model_dir = os.path.join(output_dir, 'retriever')
                reader_model_dir = os.path.join(output_dir, 'reader')

                if not os.path.exists(retriever_model_dir):
                    os.makedirs(retriever_model_dir)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                if not os.path.exists(reader_model_dir):
                    os.makedirs(reader_model_dir)

                
                retriever_model_to_save = model.retriever
                retriever_model_to_save.save_pretrained(retriever_model_dir)

                reader_model_to_save = model.reader
                reader_model_to_save.save_pretrained(reader_model_dir)

                torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                logger.info("Saving model checkpoint to %s", output_dir)

            global_step += 1
            scheduler.step()
    
    tb_writer.close()

    output_dir = os.path.join(args.output_dir, f'checkpoint-{global_step}')
    retriever_model_dir = os.path.join(output_dir, 'retriever')
    reader_model_dir = os.path.join(output_dir, 'reader')
    if not os.path.exists(retriever_model_dir):
        os.makedirs(retriever_model_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(reader_model_dir):
        os.makedirs(reader_model_dir)

    
    retriever_model_to_save = model.retriever
    retriever_model_to_save.save_pretrained(retriever_model_dir)
    reader_model_to_save = model.reader
    reader_model_to_save.save_pretrained(reader_model_dir)

    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
    logger.info("Saving model checkpoint to %s", output_dir)

    return global_step


def evaluate(args, model, retriever_tokenizer, reader_tokenizer, prefix=""):
    if prefix == 'test':
        eval_file = args.test_file
        orig_eval_file = args.test_file
    else:
        eval_file = args.dev_file
        orig_eval_file = args.dev_file

    pytrec_eval_evaluator = evaluator

    dataset = RetrieverDataset(
                eval_file, retriever_tokenizer,
                args.load_small, args.history_num,
                query_max_seq_length=args.retriever_query_max_seq_length,
                is_pretraining=args.is_pretraining,
                prepend_history_questions=args.prepend_history_questions,
                prepend_history_answers=args.prepend_history_answers,
                given_query=True,
                given_passage=False, 
                include_first_for_retriever=args.include_first_for_retriever
    )

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    predict_dir = os.path.join(args.output_dir, 'predictions')

    if not os.path.exists(predict_dir):
        os.makedirs(predict_dir)
    
    args.eval_batch_size = args.per_gpu_eval_batch_size
    eval_dataloader = DataLoader(dataset, batch_size=args.eval_batch_size, num_workers=args.num_workers)

    logger.info(f"***** Running evaluation {prefix} *****")
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    retriever_run_dict, rarank_run_dict = {}, {}
    examples, features = {}, {}
    all_results = []

    model.eval()


    for b_idx, batch in tqdm(enumerate(eval_dataloader), desc="Evaluating", total=len(eval_dataloader)):
        
        # if b_idx > 5:
        #     break

        qids = np.asarray(batch['qid']).reshape(-1).tolist()
        question_texts = np.asarray(batch['question_text']).reshape(-1).tolist()
        answer_texts = np.asarray(batch['answer_text']).reshape(-1).tolist()
        answer_starts = np.asarray(batch['answer_start']).reshape(-1).tolist()

        query_reps = gen_query_reps(args, model, batch)

        retrieval_results = retrieve(args, qids, qid_to_idx, query_reps,
                                     item_ids, item_id_to_idx, item_reps,
                                     qrels, qrels_sparse_matrix,
                                     gpu_index, include_positive_passage=False)
        
        pids_for_retriever = retrieval_results['pids_for_retriever']
        retriever_probs = retrieval_results['retriever_probs']


        # retrieval ground-truth mapping
        for i in range(len(qids)):
            retriever_run_dict[qids[i]] = {}
            for j in range(retrieval_results['no_cut_retriever_probs'].shape[1]):
                retriever_run_dict[qids[i]][pids_for_retriever[i, j]] = int(retrieval_results['no_cut_retriever_probs'][i, j])

        
        pids_for_reader = retrieval_results['pids_for_reader']
        passages_for_reader = retrieval_results['passages_for_reader']
        labels_for_reader = retrieval_results['labels_for_reader']

        # logger.info(f'QID {qids[0]} Reader IDS {pids_for_reader[0]}')

        reader_batch, batch_examples, batch_features = gen_reader_features_v2(qids, question_texts, answer_texts,
                                                                           answer_starts, pids_for_reader,
                                                                           passages_for_reader, labels_for_reader,
                                                                           reader_tokenizer,
                                                                           args.reader_max_seq_length,
                                                                           is_training=False,
                                                                           itemid_modalities=itemid_modalities,
                                                                           item_id_to_idx=item_id_to_idx,
                                                                           images_titles=images_titles)
        

        mbatch_ds = TensorDataset(reader_batch['input_ids'], 
                                reader_batch['input_mask'],
                                reader_batch['segment_ids'],
                                reader_batch['image_input'],

        )
        mbatch_dl = DataLoader(mbatch_ds, batch_size=32, num_workers=4)

        example_ids = reader_batch['example_id']


        examples.update(batch_examples)
        features.update(batch_features)
        
        retriever_probs = retriever_probs.reshape(-1).tolist()

        example_ptr = 0

        for mini_batch in tqdm(mbatch_dl, desc='Mini-Batch', leave=False):
            m_bs = mini_batch[0].shape[0]
            with torch.no_grad():
                inputs = {
                        'input_ids': mini_batch[0].to(args.device),
                        'attention_mask': mini_batch[1].to(args.device),
                        'token_type_ids': mini_batch[2].to(args.device),
                        'image_input': mini_batch[3].to(args.device),
                }
                outputs = model.reader(**inputs)      
            

            for i, ptr in enumerate(range(m_bs)):
                result = RawResult(
                            unique_id = example_ids[example_ptr],
                            start_logits = to_list(outputs[0][i]),
                            end_logits = to_list(outputs[1][i]),
                            retrieval_logits = to_list(outputs[2][i]), 
                            retriever_prob = None
                )
                example_ptr += 1
                all_results.append(result)        

    output_prediction_file = os.path.join(
    predict_dir, "instance_predictions_{}.json".format(prefix))
    output_nbest_file = os.path.join(predict_dir, f"instance_nbest_predictions_{prefix}.json")

    output_final_prediction_file = os.path.join(predict_dir, f"final_predictions_{prefix}.json")

    if args.version_2_with_negative:
        output_null_log_odds_file = os.path.join(predict_dir, "instance_null_odds_{}.json".format(prefix))
    else:
        output_null_log_odds_file = None

    all_predictions = write_predictions_v2(
                    reader_tokenizer, 
                    examples, features, all_results, 
                    args.n_best_size, args.max_answer_length, 
                    args.do_lower_case, 
                    output_prediction_file,
                    output_nbest_file, 
                    output_null_log_odds_file,
                    False,
                    args.version_2_with_negative, 
                    args.null_score_diff_threshold)

    write_final_predictions(
            all_predictions, 
            output_final_prediction_file, 
            use_rerank_prob=args.use_rerank_prob, 
            use_retriever_prob=args.use_retriever_prob
    )
    eval_metrics = quac_eval(args, orig_eval_file, output_final_prediction_file)


    rerank_metrics = get_retrieval_metrics(pytrec_eval_evaluator, all_predictions, eval_retriever_probs=True, retriever_run_dict=retriever_run_dict)
    eval_metrics.update(rerank_metrics)

    metrics_file = os.path.join(predict_dir, f"metrics_{prefix}.json")

    with open(metrics_file, 'w') as fout:
        json.dump(eval_metrics, fout)

    return eval_metrics



parser = argparse.ArgumentParser()

# arguments shared by the retriever and reader

parser.add_argument("--train_file", type=str, required=True, help="open retrieval quac json for training. ")
parser.add_argument("--dev_file", type=str, required=True, help="open retrieval quac json for predictions.")
parser.add_argument("--test_file", type=str, required=True, help="open retrieval quac json for predictions.")

parser.add_argument("--passages_file", help="the file contains passages")
parser.add_argument("--tables_file", help="the file contains passages")
parser.add_argument("--images_file", help="the file contains passages")

parser.add_argument("--qrels", type=str, required=False, help="qrels to evaluate open retrieval")
parser.add_argument("--images_path", type=str, help="the path to images")

parser.add_argument("--gen_passage_rep_output", type=str, help="passage representations")
parser.add_argument("--output_dir", default='./release_test', type=str, required=False, help="The output directory where the model checkpoints and predictions will be written.")
parser.add_argument("--load_small", default=False, type=str2bool, required=False, help="whether to load just a small portion of data during development")
parser.add_argument("--num_workers", default=2, type=int, required=False, help="number of workers for dataloader")

parser.add_argument("--global_mode", default=True, type=str2bool, required=False, help="maxmize the prob of the true answer given all passages")
parser.add_argument("--history_num", default=1, type=int, required=False, help="number of history turns to use")
parser.add_argument("--prepend_history_questions", default=True, type=str2bool, required=False, help="whether to prepend history questions to the current question")
parser.add_argument("--prepend_history_answers", default=False, type=str2bool, required=False, help="whether to prepend history answers to the current question")

parser.add_argument("--do_train", default=True, type=str2bool, help="Whether to run training.")
parser.add_argument("--do_eval", default=True, type=str2bool, help="Whether to run eval on the dev set.")
parser.add_argument("--do_test", default=True, type=str2bool, help="Whether to run eval on the test set.")
parser.add_argument("--best_global_step", default=12000, type=int, required=False, help="used when only do_test")
parser.add_argument("--evaluate_during_training", default=False, type=str2bool, help="Rul evaluation during training at each logging step.")
parser.add_argument("--do_lower_case", default=True, type=str2bool, help="Set this flag if you are using an uncased model.")

parser.add_argument("--per_gpu_train_batch_size", default=1, type=int, help="Batch size per GPU/CPU for training.")
parser.add_argument("--per_gpu_eval_batch_size", default=1, type=int,help="Batch size per GPU/CPU for evaluation.")
parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
parser.add_argument("--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform.")
parser.add_argument("--warmup_portion", default=0.1, type=float, help="Linear warmup over warmup_steps (=t_total * warmup_portion). override warmup_steps ")
parser.add_argument("--verbose_logging", action='store_true',help="If true, all of the warnings related to data processing will be printed. A number of warnings are expected for a normal SQuAD evaluation.")

parser.add_argument('--logging_steps', type=int, default=10, help="Log every X updates steps.")
parser.add_argument('--save_steps', type=int, default=4000, help="Save checkpoint every X updates steps.")
parser.add_argument("--eval_all_checkpoints", default=True, type=str2bool, help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
parser.add_argument("--no_cuda", default=False, type=str2bool, help="Whether not to use CUDA when available")
parser.add_argument('--overwrite_output_dir', default=False, type=str2bool, help="Overwrite the content of the output directory")
parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")




parser.add_argument("--retriever_model_name_or_path", default='albert-base-v2', type=str, required=True, help="retriever model name")
parser.add_argument("--retrieve_checkpoint", type=str, required=True, help="generate query/passage representations with this checkpoint")
parser.add_argument("--retrieve_tokenizer_dir", type=str, required=True, help="dir that contains tokenizer files")

parser.add_argument("--given_query", default=True, type=str2bool, help="Whether query is given.")
parser.add_argument("--given_passage", default=False, type=str2bool, help="Whether passage is given. Passages are not given when jointly train")
parser.add_argument("--is_pretraining", default=False, type=str2bool, help="Whether is pretraining. We fine tune the query encoder in retriever")
parser.add_argument("--include_first_for_retriever", default=True, type=str2bool, help="include the first question in a dialog in addition to history_num for retriever (not reader)")
parser.add_argument("--retriever_query_max_seq_length", default=30, type=int, help="The maximum input sequence length of query.")
parser.add_argument("--retriever_passage_max_seq_length", default=128, type=int, help="The maximum input sequence length of passage (384 + [CLS] + [SEP]).")
parser.add_argument("--proj_size", default=128, type=int, help="The size of the query/passage rep after projection of [CLS] rep.")
parser.add_argument("--top_k_for_retriever", default=2000, type=int, help="retrieve top k passages for a query, these passages will be used to update the query encoder")
parser.add_argument("--use_retriever_prob", default=True, type=str2bool, help="include retriever probs in final answer ranking")

# reader arguments
parser.add_argument("--reader_model_name_or_path", default='bert-base-uncased', type=str, required=False, help="reader model name")
parser.add_argument("--reader_max_seq_length", default=512, type=int, help="The maximum total input sequence length after WordPiece tokenization. Sequences longer than this will be truncated, and sequences shorter than this will be padded.")
parser.add_argument("--doc_stride", default=384, type=int, help="When splitting up a long document into chunks, how much stride to take between chunks.")
parser.add_argument('--null_score_diff_threshold', type=float, default=0.0, help="If null_score - best_non_null is greater than the threshold predict null.")
parser.add_argument("--reader_max_query_length", default=125, type=int, help="The maximum number of tokens for the question. Questions longer than this will be truncated to this length.")
parser.add_argument("--n_best_size", default=20, type=int, help="The total number of n-best predictions to generate in the nbest_predictions.json output file.")
parser.add_argument("--max_answer_length", default=40, type=int, help="The maximum length of an answer that can be generated. This is needed because the start and end predictions are not conditioned on one another.")
parser.add_argument("--qa_loss_factor", default=1.0, type=float, help="total_loss = qa_loss_factor * qa_loss + retrieval_loss_factor * retrieval_loss")
parser.add_argument("--retrieval_loss_factor", default=1.0, type=float, help="total_loss = qa_loss_factor * qa_loss + retrieval_loss_factor * retrieval_loss")
parser.add_argument("--top_k_for_reader", default=10, type=int, help="update the reader with top k passages")
parser.add_argument("--use_rerank_prob", default=True, type=str2bool, help="include rerank probs in final answer ranking")
parser.add_argument('--version_2_with_negative', default=False, type=str2bool, required=False, help='If true, the SQuAD examples contain some that do not have an answer.')
args, unknown = parser.parse_known_args()


if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
    raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

args.retriever_tokenizer_dir = os.path.join(args.output_dir, 'retriever')
args.reader_tokenizer_dir = os.path.join(args.output_dir, 'reader')

seed_all(args.seed)

args.device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

model = Pipeline()

# retriever_config = AutoConfig.from_pretrained(args.retriever_model_name_or_path)
# retriever_config.proj_size = args.proj_size

# retriever_tokenizer = AutoTokenizer.from_pretrained(args.retriever_model_name_or_path)

# model.retriever = Retriever.from_pretrained(args.retrieve_checkpoint, retriever_config)
# # following modules are not required - using DB


retriever_config_class, retriever_model_class, retriever_tokenizer_class = (AlbertConfig, AlbertForRetrieverOnlyPositivePassage, AlbertTokenizer)
retriever_config = retriever_config_class.from_pretrained(args.retrieve_checkpoint)

# load pretrained retriever
retriever_tokenizer = retriever_tokenizer_class.from_pretrained(args.retrieve_tokenizer_dir)
retriever_model = retriever_model_class.from_pretrained(args.retrieve_checkpoint, force_download=True)

model.retriever = retriever_model
model.retriever.passage_encoder = None
model.retriever.passage_proj = None
model.retriever.image_encoder = None
model.retriever.image_proj = None

reader_tokenizer = AutoTokenizer.from_pretrained(args.reader_model_name_or_path, fast=True)

reader_config = AutoConfig.from_pretrained(args.reader_model_name_or_path)
reader_config.num_retrieval_labels = 2
reader_config.num_qa_labels = 2
reader_config.qa_loss_factor = args.qa_loss_factor
reader_config.retrieval_loss_factor = args.retrieval_loss_factor
reader_config.proj_size = args.proj_size

model.reader = Reader(args.reader_model_name_or_path, reader_config)

model.to(args.device)

itemid_modalities = []

passages_dict = {}
with open(args.passages_file, 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = json.loads(line.strip())
        passages_dict[line['id']] = line['text']
        itemid_modalities.append('text')


tables_dict = {}
with open(args.tables_file, 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = json.loads(line.strip())
        table_context = ''
        for row_data in line['table']["table_rows"]:
            for cell in row_data:
                table_context = table_context+" "+cell['text']
        tables_dict[line['id']] = table_context
        itemid_modalities.append('table')

images_dict = {}
with open(args.images_file,'r') as f:
    lines = f.readlines()
    for line in tqdm(lines):
        try:
            line = json.loads(line.strip())
            img_path = os.path.join(args.images_path, line['path'])
            # img_ = Image.open(img_path).convert("RGB")
            images_dict[line['id']] = img_path
            itemid_modalities.append('image')

        except Exception as e:
            print(e, "\t", img_path)
            pass

image_answers_set=set()
with open(args.train_file, "r") as f:
    lines=f.readlines()
    for line in lines:
        image_answers_set.add(json.loads(line.strip())['answer'][0]['answer'])

image_answers_set = sorted([str(x) for x in image_answers_set])
image_answers_str = ' '.join(image_answers_set)

images_titles = {}
with open(args.images_file,'r') as f:
    lines = f.readlines()
    for line in lines:
        line = json.loads(line.strip())
        images_titles[line['id']] = line['title'] + " " + image_answers_str


item_ids, item_reps = [], []
with open(args.gen_passage_rep_output) as fin:
    for line in tqdm(fin):
        dic = json.loads(line.strip())
        item_ids.append(dic['id'])
        item_reps.append(dic['rep'])
item_reps = np.asarray(item_reps, dtype='float32')
item_ids=np.asarray(item_ids)

logger.info('constructing passage faiss_index')
faiss_res = faiss.StandardGpuResources() 
index = faiss.IndexFlatIP(args.proj_size)
index.add(item_reps)
gpu_index=index



logger.info(f'loading qrels from {args.qrels}')
with open(args.qrels) as handle:
    qrels = json.load(handle)

item_id_to_idx = {}
for i, pid in enumerate(item_ids):
    item_id_to_idx[pid] = i

qrels_data, qrels_row_idx, qrels_col_idx = [], [], []
qid_to_idx = {}
for i, (qid, v) in enumerate(qrels.items()):
    qid_to_idx[qid] = i
    for pid in v.keys():
        qrels_data.append(1)
        qrels_row_idx.append(i)
        qrels_col_idx.append(item_id_to_idx[pid])

qrels_data.append(0)
qrels_row_idx.append(5752) # num training samples
qrels_col_idx.append(len(item_id_to_idx)) # total items in DB

qrels_sparse_matrix = sp.sparse.csr_matrix((qrels_data, (qrels_row_idx, qrels_col_idx)))
evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'ndcg', 'set_recall'})




retriever_tokenizer.save_pretrained(args.retriever_tokenizer_dir)
reader_tokenizer.save_pretrained(args.reader_tokenizer_dir)


if args.do_train:
    train_dataset = RetrieverDataset(args.train_file, retriever_tokenizer,
                                 args.load_small, args.history_num,
                                 query_max_seq_length=args.retriever_query_max_seq_length,
                                 is_pretraining=args.is_pretraining,
                                 prepend_history_questions=args.prepend_history_questions,
                                 prepend_history_answers=args.prepend_history_answers,
                                 given_query=True,
                                 given_passage=False, 
                                 include_first_for_retriever=args.include_first_for_retriever)

    global_step = train(args, train_dataset, model, retriever_tokenizer, reader_tokenizer)


results = {}
max_f1 = 0.0
best_metrics = {}

if args.do_eval:
    checkpoints = [args.output_dir]
    if args.eval_all_checkpoints:
        checkpoints = sorted(list(os.path.dirname(os.path.dirname(c)) for c in
                            glob.glob(args.output_dir + '/*/retriever/' + WEIGHTS_NAME, recursive=False)))
    
    logger.info("Evaluate the following checkpoints: %s", checkpoints)


    for checkpoint in checkpoints:
        global_step = checkpoint.split('-')[-1] if len(checkpoint) > 1 else ""
        model = Pipeline()

        model.retriever = retriever_model_class.from_pretrained(os.path.join(checkpoint, 'retriever'), force_download=True)
        model.retriever.passage_encoder = None
        model.retriever.passage_proj = None
        model.retriever.image_encoder = None
        model.retriever.image_proj = None
        model.reader = Reader.from_pretrained(args.reader_model_name_or_path, os.path.join(checkpoint, 'reader'), reader_config)
        model.to(args.device)

        result = evaluate(args, model, retriever_tokenizer, reader_tokenizer, prefix=global_step)

        if result['f1'] > max_f1:
            max_f1 = result['f1']
            best_metrics = copy(result)
            best_metrics['global_step'] = global_step

        
        result = dict((k + ('_{}'.format(global_step) if global_step else ''), v)
                        for k, v in result.items())
        results.update(result)


    best_metrics_file = os.path.join(args.output_dir, 'predictions', 'best_metrics.json')
    with open(best_metrics_file, 'w') as fout:
        json.dump(best_metrics, fout)

    all_results_file = os.path.join(args.output_dir, 'predictions', 'all_results.json')
    with open(all_results_file, 'w') as fout:
        json.dump(results, fout)

    logger.info("Results: {}".format(results))
    logger.info("best metrics: {}".format(best_metrics))


if args.do_test:    
    if args.do_eval:
        best_global_step = best_metrics['global_step'] 
    else:
        best_global_step = args.best_global_step
        
    best_checkpoint = os.path.join(args.output_dir, f'checkpoint-{best_global_step}')
    logger.info("Test the best checkpoint: %s", best_checkpoint)

    model = Pipeline()

    model.retriever = retriever_model_class.from_pretrained(os.path.join(checkpoint, 'retriever'), force_download=True)
    model.retriever.passage_encoder = None
    model.retriever.passage_proj = None
    model.retriever.image_encoder = None
    model.retriever.image_proj = None

    model.reader = Reader.from_pretrained(args.reader_model_name_or_path, os.path.join(checkpoint, 'reader'), reader_config)


    model.to(args.device)

    result = evaluate(args, model, retriever_tokenizer, reader_tokenizer, prefix='test')

    test_metrics_file = os.path.join(args.output_dir, 'predictions', 'test_metrics.json')
    with open(test_metrics_file, 'w') as fout:
        json.dump(result, fout)

    logger.info("Test Result: {}".format(result))