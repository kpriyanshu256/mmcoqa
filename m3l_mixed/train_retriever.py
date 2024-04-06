import argparse
import logging
import os
import sys
import random
import glob
import timeit
import json
import faiss
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler
from PIL import Image

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import tqdm, trange
import pytrec_eval

from transformers import AdamW, get_linear_schedule_with_warmup, AutoConfig, AutoTokenizer
from retriever_utils import RetrieverDataset, GenPassageRepDataset
from modeling import Retriever, Reader
from utils import AverageMeter

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


def train(args, train_dataset, model, tokenizer):
    tb_writer = SummaryWriter(os.path.join(args.output_dir, 'logs'))
    args.train_batch_size = args.per_gpu_train_batch_size

    train_sampler = RandomSampler(train_dataset)

    train_dataloader = DataLoader(
        train_dataset, 
        sampler=train_sampler,
        batch_size=args.train_batch_size, 
        num_workers=args.num_workers
    )
    

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
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", args.train_batch_size)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 1

    for _ in trange(int(args.num_train_epochs), desc='Training'):
        
        loss_metric = AverageMeter()
        pbar = tqdm(train_dataloader, total=len(train_dataloader), leave=False)

        for step, batch in enumerate(pbar):
            model.train()
            # logger.info(f'QID {batch["qid"]}')
            # logger.info(f'Ex {batch["example_id"]}')

            batch = {k: v.to(args.device) for k, v in batch.items() if k not in ['example_id', 'qid']}

            inputs = {}
            if args.given_query:
                inputs['query_input_ids'] = batch['query_input_ids']
                inputs['query_attention_mask'] = batch['query_attention_mask']        
                inputs['query_token_type_ids'] = batch['query_token_type_ids']

            if args.given_passage:
                inputs['passage_input_ids'] = batch['passage_input_ids']
                inputs['passage_attention_mask'] = batch['passage_attention_mask']        
                inputs['passage_token_type_ids'] = batch['passage_token_type_ids']
                inputs['retrieval_label'] = batch['retrieval_label']
                inputs['question_type'] = batch['question_type']
                inputs['image_input'] = batch['image_input'].type(torch.FloatTensor).to(args.device)
            

            # for k, v in inputs.items():
            #     logger.info(f'{k} - {v.shape}')

            outputs = model(**inputs)

            loss = outputs[0]
            # logger.info(f'Step {global_step} Loss {loss.item()}')

            # if global_step > 5:
            #     sys.exit(0)

            loss_metric.update(loss.item(), inputs['query_input_ids'].shape[0])

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            optimizer.step()
            scheduler.step()
            model.zero_grad()
            global_step += 1

            tb_writer.add_scalar('loss', loss_metric.avg, global_step)
            pbar.set_postfix(loss=loss_metric.avg)

            if args.save_steps > 0 and global_step % args.save_steps == 0:
                output_dir = os.path.join(args.output_dir, f'checkpoint-{global_step}')

                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                
                model.save_pretrained(output_dir)
                torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                logger.info(f"Saving model checkpoint to {output_dir}")

    return global_step

def evaluate(args, model, tokenizer, prefix=""):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    predict_dir = os.path.join(args.output_dir, 'predictions')

    if not os.path.exists(predict_dir):
        os.makedirs(predict_dir)

    passage_ids, passage_reps = gen_passage_rep(args, model, tokenizer)
    passage_reps = np.asarray(passage_reps, dtype='float32')


    passage_ids, passage_reps = [], []
    with open(args.gen_passage_rep_output) as fin:
        for line in tqdm(fin):
            dic = json.loads(line.strip())
            passage_ids.append(dic['id'])
            passage_reps.append(dic['rep'])

    passage_reps = np.asarray(passage_reps, dtype='float32')

    qids, query_reps = retrieve(args, model, tokenizer, prefix=prefix)
    query_reps = np.asarray(query_reps, dtype='float32')
        
    index = faiss.IndexFlatIP(args.proj_size)
    index.add(passage_reps)
    D, I = index.search(query_reps, 1000)
    

    run = {}
    for qid, retrieved_ids, scores in zip(qids, I, D):
        run[qid] = {passage_ids[retrieved_id]: float(score) for retrieved_id, score in zip(retrieved_ids, scores)}


    with open(args.qrels) as handle:
        qrels = json.load(handle)

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'ndcg', 'set_recall'})
    metrics = evaluator.evaluate(run)

    mrr_list = [v['ndcg'] for v in metrics.values()]
    recall_list = [v['set_recall'] for v in metrics.values()]
    eval_metrics = {'NDCG': np.average(mrr_list), 'Recall': np.average(recall_list)}

    logger.info(eval_metrics)
    print(eval_metrics)
    return eval_metrics


def retrieve(args, model, tokenizer, prefix=''):
    if prefix == 'test':
        eval_file = args.test_file
    else:
        eval_file = args.dev_file

    dataset = RetrieverDataset(
                eval_file, 
                tokenizer,
                args.load_small, 
                args.history_num,
                query_max_seq_length=args.query_max_seq_length,
                passage_max_seq_length=args.passage_max_seq_length,
                is_pretraining=args.is_pretraining,
                given_query=True,
                given_passage=False,
                only_positive_passage=args.only_positive_passage,
                passages_dict=passages_dict,
                tables_dict=tables_dict,
                images_dict=images_dict
    )

    args.eval_batch_size = args.per_gpu_eval_batch_size
    
    eval_dataloader = DataLoader(
        dataset, 
        batch_size=args.eval_batch_size, 
        num_workers=args.num_workers
    )

        
    logger.info("***** Retrieve {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    all_qids = []
    all_query_reps = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        qids = np.asarray(batch['qid']).reshape(-1).tolist()
        batch = {k: v.to(args.device) for k, v in batch.items() if k not in ['example_id', 'qid']}

        with torch.no_grad():
            inputs = {}
            inputs['query_input_ids'] = batch['query_input_ids']
            inputs['query_attention_mask'] = batch['query_attention_mask']
            inputs['query_token_type_ids'] = batch['query_token_type_ids']
            outputs = model(**inputs)
            query_reps = outputs[0]

        all_qids.extend(qids)
        all_query_reps.extend(to_list(query_reps))

    return all_qids, all_query_reps


def gen_passage_rep(args, model, tokenizer):
    dataset = GenPassageRepDataset(
                args.gen_passage_rep_input,
                args.load_small,
                tokenizer,
                passage_max_seq_length=args.passage_max_seq_length, 
                passages_dict=passages_dict,
                tables_dict=tables_dict,
                images_dict=images_dict,
                idx_id_list=idx_id_list
            )

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size

    eval_dataloader = DataLoader(
            dataset,
            batch_size=args.eval_batch_size, 
            num_workers=args.num_workers)


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
        example_ids = np.asarray(batch['example_id']).reshape(-1).tolist()
        passage_ids.extend(example_ids)

        batch = {k: v.to(args.device) for k, v in batch.items() if k != 'example_id'}

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
        
        for example_id, passage_rep in zip(example_ids, to_list(passage_reps)):
            fout.write(json.dumps({'id': example_id, 'rep': passage_rep}) + '\n')
            
    fout.close()
    return passage_ids, passage_reps_list



parser = argparse.ArgumentParser()

# Required parameters
parser.add_argument("--train_file", type=str, required=True, help="open retrieval quac json for training. ")
parser.add_argument("--dev_file", type=str, required=True, help="open retrieval quac json for predictions.")
parser.add_argument("--test_file", type=str, required=True, help="open retrieval quac json for predictions.")
parser.add_argument("--passages_file", help="the file contains passages")
parser.add_argument("--tables_file", help="the file contains passages")
parser.add_argument("--images_file", help="the file contains passages")
parser.add_argument("--images_path", help="the path to images")


parser.add_argument("--model_name_or_path", default='albert-base-v2', type=str, required=True, help="Path to pre-trained model or shortcut name selected in the list: ")
parser.add_argument("--output_dir", default='./retriever_release_test', type=str, required=False,
                    help="The output directory where the model checkpoints and predictions will be written.")
parser.add_argument("--qrels", type=str, required=True, help="qrels to evaluate open retrieval")

parser.add_argument("--given_query", default=True, type=str2bool, help="Whether query is given.")
parser.add_argument("--given_passage", default=True, type=str2bool, help="Whether passage is given.")
parser.add_argument("--is_pretraining", default=True, type=str2bool, help="Whether is pretraining.")
parser.add_argument("--only_positive_passage", default=True, type=str2bool, help="we only pass the positive passages, the rest of the passges in the batch are considered as negatives")
parser.add_argument("--gen_passage_rep", default=False, type=str2bool, help="generate passage representations for all ")
parser.add_argument("--retrieve_checkpoint", type=str, required=False, help="generate query/passage representations with this checkpoint")
parser.add_argument("--gen_passage_rep_input", type=str, help="generate passage representations for this file that contains passages")
parser.add_argument("--gen_passage_rep_output", type=str, help="passage representations")
parser.add_argument("--retrieve", default=False, type=str2bool, help="generate query reps and retrieve passages")
parser.add_argument("--load_small", default=False, type=str2bool, required=False, help="whether to load just a small portion of data during development")

# Other parameters
parser.add_argument("--query_max_seq_length", default=30, type=int,
                    help="The maximum input sequence length of query (125 + [CLS] + [SEP])."
                         "125 is the max question length in the reader.")
parser.add_argument("--passage_max_seq_length", default=128, type=int,
                    help="The maximum input sequence length of passage (384 + [CLS] + [SEP]).")
parser.add_argument("--proj_size", default=128, type=int,
                    help="The size of the query/passage rep after projection of [CLS] rep.")
parser.add_argument("--do_train", default=True, type=str2bool,
                    help="Whether to run training.")
parser.add_argument("--do_eval", default=True, type=str2bool,
                    help="Whether to run eval on the dev set.")
parser.add_argument("--do_test", default=True, type=str2bool,
                    help="Whether to run eval on the test set.")

parser.add_argument("--per_gpu_train_batch_size", default=20, type=int,
                    help="Batch size per GPU/CPU for training.")
parser.add_argument("--per_gpu_eval_batch_size", default=128, type=int,
                    help="Batch size per GPU/CPU for evaluation.")
parser.add_argument("--learning_rate", default=5e-5, type=float,
                    help="The initial learning rate for Adam.")
parser.add_argument("--weight_decay", default=0.0, type=float,
                    help="Weight decay if we apply some.")
parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                    help="Epsilon for Adam optimizer.")
parser.add_argument("--max_grad_norm", default=1.0, type=float,
                    help="Max gradient norm.")
parser.add_argument("--num_train_epochs", default=22.0, type=float,
                    help="Total number of training epochs to perform.")
parser.add_argument("--warmup_steps", default=0, type=int,
                    help="Linear warmup over warmup_steps.")
parser.add_argument("--warmup_portion", default=0.1, type=float,
                    help="Linear warmup over warmup_steps (=t_total * warmup_portion). override warmup_steps ")
parser.add_argument("--verbose_logging", action='store_true',
                    help="If true, all of the warnings related to data processing will be printed. "
                         "A number of warnings are expected for a normal SQuAD evaluation.")

parser.add_argument('--save_steps', type=int, default=2000,
                    help="Save checkpoint every X updates steps.")
parser.add_argument('--overwrite_output_dir', default=False, type=str2bool,
                    help="Overwrite the content of the output directory")
parser.add_argument('--seed', type=int, default=42,
                    help="random seed for initialization")

parser.add_argument("--history_num", default=1, type=int, required=False,
                    help="number of history turns to use")
parser.add_argument("--num_workers", default=4, type=int, required=False,
                    help="number of workers for dataloader")

args, unknown = parser.parse_known_args()

if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
    raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))


logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d:%H:%M:%S',
            level=logging.INFO)
            
seed_all(args.seed)


args.device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

config = AutoConfig.from_pretrained(args.model_name_or_path)
config.proj_size = args.proj_size

tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

if args.retrieve_checkpoint:
    model = Retriever.from_pretrained(args.retrieve_checkpoint, config)
else:
    model = Retriever(config)



model.to(args.device)

# for i, (k, v) in enumerate(model.named_parameters()):
#     print(k, v.view(-1)[:5])  
# sys.exit(0)


if args.retrieve:
    args.gen_passage_rep = False
    args.do_train = False
    args.do_eval = False
    args.do_test = False
    args.given_query = True
    args.given_passage = False

if args.gen_passage_rep:
    args.do_train = False
    args.do_eval = True
    args.do_test = False


idx_id_list = []
passages_dict = {}
with open(args.passages_file, 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = json.loads(line.strip())
        passages_dict[line['id']] = line['text']
        idx_id_list.append((line['id'], 'text'))


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
        idx_id_list.append((line['id'],'table'))

images_dict = {}
with open(args.images_file,'r') as f:
    lines = f.readlines()
    for line in tqdm(lines):
        try:
            line = json.loads(line.strip())
            img_path = os.path.join(args.images_path, line['path'])
            img_ = Image.open(img_path).convert("RGB")
            images_dict[line['id']] = img_path
            idx_id_list.append((line['id'], 'image'))
        except Exception as e:
            print(e, "\t", img_path)
            pass

if args.do_train:
    train_dataset = RetrieverDataset(
        args.train_file, 
        tokenizer,
        args.load_small, 
        args.history_num,
        query_max_seq_length=args.query_max_seq_length,
        passage_max_seq_length=args.passage_max_seq_length,
        is_pretraining=True,
        given_query=True,
        given_passage=True, 
        only_positive_passage=args.only_positive_passage,
        passages_dict=passages_dict,
        tables_dict=tables_dict,
        images_dict=images_dict
    )
                                
    global_step = train(args, train_dataset, model, tokenizer)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    logger.info("Saving model checkpoint to %s", args.output_dir)
    final_checkpoint_output_dir = os.path.join(args.output_dir, f'checkpoint-{global_step}')
    logger.info("Saving model checkpoint to %s", final_checkpoint_output_dir)

    if not os.path.exists(final_checkpoint_output_dir):
        os.makedirs(final_checkpoint_output_dir)

    model.save_pretrained(final_checkpoint_output_dir)
    tokenizer.save_pretrained(final_checkpoint_output_dir)
    torch.save(args, os.path.join(final_checkpoint_output_dir, 'training_args.bin'))


if args.do_eval:
    model = Retriever.from_pretrained(args.retrieve_checkpoint, config)
    model.to(args.device)
    logger.info("Evaluating")
    result = evaluate(args, model, tokenizer, prefix="eval")
    logger.info(f'Valid Evaluation: {result}')
    # result = evaluate(args, model, tokenizer, prefix="test")
    # logger.info(f'Test Evaluation: {result}')
    
    

# if args.gen_passage_rep:
#     tokenizer = AutoTokenizer.from_pretrained(args.retrieve_checkpoint)
#     logger.info("Gen passage rep with: %s", args.retrieve_checkpoint)

#     model = Retriever.from_pretrained(args.retrieve_checkpoint, config)
#     model.to(args.device)

#     gen_passage_rep(args, model, tokenizer)    
#     logger.info("Gen passage rep complete")