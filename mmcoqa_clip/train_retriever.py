#!/usr/bin/env python
# coding: utf-8
from __future__ import absolute_import, division, print_function

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="7"
import logging
import os
import random
import glob
import json
import pickle
import numpy as np
import torch
import sys
sys.path.append('./mae')
from PIL import Image

from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
import train_options 

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import tqdm, trange

from transformers import WEIGHTS_NAME, BertConfig, BertTokenizer, AlbertConfig, AlbertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from retriever_utils import RetrieverDataset, GenPassageRepDataset
from modeling import BertForRetrieverOnlyPositivePassage,AlbertForRetrieverOnlyPositivePassage
from evaluate_models import evaluate_retriever, gen_passage_rep
import warnings
warnings.filterwarnings("ignore")


logger = logging.getLogger(__name__)

ALL_MODELS = [] # list(BertConfig.pretrained_config_archive_map.keys())

MODEL_CLASSES = {
    'bert': (BertConfig, BertForRetrieverOnlyPositivePassage, BertTokenizer),
    'albert': (AlbertConfig, AlbertForRetrieverOnlyPositivePassage, AlbertTokenizer)
}

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def to_list(tensor):
    return tensor.detach().cpu().tolist()


def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(os.path.join(args.output_dir, 'logs'))

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(
        train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=args.num_workers)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (
            len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(
            train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate, eps=args.adam_epsilon)
    args.warmup_steps = int(t_total * args.warmup_portion)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
        # model.to(f'cuda:{model.device_ids[0]}')

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d",
                args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d",
                args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 1
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs),
                            desc="Epoch", disable=args.local_rank not in [-1, 0])
    # Added here for reproductibility (even between python 2 and 3)
    set_seed(args)
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration",
                              disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
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
            outputs = model(**inputs)
            # model outputs are always tuple in transformers (see doc)
            if args.given_query and args.given_passage:
                loss = outputs[0]

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(
                        amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    # Only evaluate when single GPU otherwise metrics may not average well
                    if args.local_rank == -1 and args.evaluate_during_training:
                        results = evaluate_retriever(args, model, tokenizer, logger, passages_dict, tables_dict, images_dict, idx_id_list, )
                        for key, value in results.items():
                            tb_writer.add_scalar(
                                'eval_{}'.format(key), value, global_step)
                    tb_writer.add_scalar(
                        'lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar(
                        'loss', (tr_loss - logging_loss)/args.logging_steps, global_step)
                    print('loss',loss.item())
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(
                        args.output_dir, 'checkpoint-{}'.format(global_step))
                    
                    os.makedirs(output_dir, exist_ok=True)
                    # Take care of distributed/parallel training
                    model_to_save = model.module if hasattr(
                        model, 'module') else model
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(
                        output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step





# In[7]:


def retrieve(args, model, tokenizer, prefix=''):
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


# In[8]:

if __name__ == "__main__":
    args, unknown = train_options.retriever_args()

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(
            address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
        torch.cuda.set_device(0)
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                        cache_dir=args.cache_dir if args.cache_dir else None)
    config.proj_size = args.proj_size

    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
    model = model_class.from_pretrained(args.model_name_or_path,
                                        from_tf=bool(
                                            '.ckpt' in args.model_name_or_path),
                                        config=config,
                                        cache_dir=args.cache_dir if args.cache_dir else None)
    model = model_class.from_pretrained(args.retrieve_checkpoint)

    if args.local_rank == 0:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Before we do anything with models, we want to ensure that we get fp16 execution of torch.einsum if args.fp16 is set.
    # Otherwise it'll default to "promote" mode, and we'll get fp32 operations. Note that running `--fp16_opt_level="O2"` will
    # remove the need for this code, but it is still valid.
    if args.fp16:
        try:
            import apex
            apex.amp.register_half_function(torch, 'einsum')
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    if args.retrieve:
        args.gen_passage_rep = False
        args.do_train = False
        args.do_eval = False
        args.do_test = False
        args.given_query = True
        args.given_passage = False

    if args.gen_passage_rep:
        args.do_train = False
        args.do_eval = False
        args.do_test = False

    idx_id_list=[]
    #load passages to passages_dict
    passages_dict={}
    with open(args.passages_file,'r') as f:
        lines=f.readlines()
        for line in lines:
            line=json.loads(line.strip())
            passages_dict[line['id']]=line['text']
            idx_id_list.append((line['id'],'text'))

    #load tables to tables_dict
    tables_dict={}
    with open(args.tables_file,'r') as f:
        lines=f.readlines()
        for line in lines:
            line=json.loads(line.strip())
            table_context = ''
            for row_data in line['table']["table_rows"]:
                for cell in row_data:
                    table_context=table_context+" "+cell['text']
            tables_dict[line['id']]=table_context
            idx_id_list.append((line['id'],'table'))
    #load images to images_dict
    # !!!!!!!!!!!!!!!!!!!!!!!还需要精致的修改
    images_dict={}
    # with open(args.images_file,'r') as f:
    #     lines=f.readlines()
    #     for line in lines:
    #         line=json.loads(line.strip())
    #         images_dict[line['id']]=args.images_path+line['path']
    #         idx_id_list.append((line['id'],'image'))


    with open(args.images_file,'r') as f:
        lines=f.readlines()
        for line in lines:
            try:
                line=json.loads(line.strip())
                img_path = os.path.join(args.images_path, line['path'])
                img_ = Image.open(img_path).convert("RGB")
                images_dict[line['id']] = img_path
                idx_id_list.append((line['id'], 'image'))
            except Exception as e:
                print(e, "\t", img_path)
                pass

    # Training
    if args.do_train:
        DatasetClass = RetrieverDataset
        train_dataset = DatasetClass(args.train_file, tokenizer,
                                    args.load_small, args.history_num,
                                    query_max_seq_length=args.query_max_seq_length,
                                    passage_max_seq_length=args.passage_max_seq_length,
                                    is_pretraining=True,
                                    given_query=True,
                                    given_passage=True, 
                                    only_positive_passage=args.only_positive_passage,
                                    passages_dict=passages_dict,
                                    tables_dict=tables_dict,
                                    images_dict=images_dict)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s",
                    global_step, tr_loss)

    # Save the trained model and the tokenizer
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        # Take care of distributed/parallel training
        model_to_save = model.module if hasattr(model, 'module') else model
        final_checkpoint_output_dir = os.path.join(
            args.output_dir, 'checkpoint-{}'.format(global_step))
        if not os.path.exists(final_checkpoint_output_dir):
            os.makedirs(final_checkpoint_output_dir)

        model_to_save.save_pretrained(final_checkpoint_output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(
            final_checkpoint_output_dir, 'training_args.bin'))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(
            final_checkpoint_output_dir, force_download=True)
        tokenizer = tokenizer_class.from_pretrained(
            args.output_dir, do_lower_case=args.do_lower_case)
        model.to(args.device)

    # Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory

    results = {}
    max_recall = 0.0
    best_metrics = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(
            args.output_dir, do_lower_case=args.do_lower_case)
        tb_writer = SummaryWriter(os.path.join(args.output_dir, 'logs'))
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c) for c in sorted(
                glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
    #         logging.getLogger("transformers.modeling_utils").setLevel(
    #             logging.WARN)  # Reduce model loading logs

        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            # Reload the model
            global_step = checkpoint.split(
                '-')[-1] if len(checkpoint) > 1 else ""
            print(global_step, 'global_step')

            model = model_class.from_pretrained(
                checkpoint, force_download=True)
            model.to(args.device)

            # Evaluate
            result = evaluate_retriever(args, model, tokenizer, logger, passages_dict, tables_dict, images_dict, idx_id_list, prefix=global_step)
            if result['Recall'] > max_recall:
                max_recall = result['Recall']
                best_metrics['Recall'] = result['Recall']
                best_metrics['NDCG'] = result['NDCG']
                best_metrics['global_step'] = global_step

            for key, value in result.items():
                tb_writer.add_scalar(
                    'eval_{}'.format(key), value, global_step)

            result = dict((k + ('_{}'.format(global_step) if global_step else ''), v)
                        for k, v in result.items())
            results.update(result)

        best_metrics_file = os.path.join(
            args.output_dir, 'predictions', 'best_metrics.json')
        with open(best_metrics_file, 'w') as fout:
            json.dump(best_metrics, fout)
            
        all_results_file = os.path.join(
            args.output_dir, 'predictions', 'all_results.json')
        with open(all_results_file, 'w') as fout:
            json.dump(results, fout)

        logger.info("Results: {}".format(results))
        logger.info("best metrics: {}".format(best_metrics))


    if args.do_test and args.local_rank in [-1, 0]:
        best_global_step = best_metrics['global_step']
        best_checkpoint = os.path.join(
            args.output_dir, 'checkpoint-{}'.format(best_global_step))
        logger.info("Test the best checkpoint: %s", best_checkpoint)

        model = model_class.from_pretrained(
            best_checkpoint, force_download=True)
        model.to(args.device)

        # Evaluate
        result = evaluate_retriever(args, model, tokenizer, logger, passages_dict, tables_dict, images_dict, idx_id_list,  prefix='test')

        test_metrics_file=os.path.join(
            args.output_dir, 'predictions', 'test_metrics.json')
        with open(test_metrics_file, 'w') as fout:
            json.dump(result, fout)

        logger.info("Test Result: {}".format(result))

    if args.gen_passage_rep and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(
            args.output_dir, do_lower_case=args.do_lower_case)
        logger.info("Gen passage rep with: %s", args.retrieve_checkpoint)

        model = model_class.from_pretrained(
            args.retrieve_checkpoint, force_download=True)
        model.to(args.device)

        # Evaluate
        gen_passage_rep(args, model, tokenizer, logger, passages_dict, tables_dict, images_dict, idx_id_list)
        
        logger.info("Gen passage rep complete")


    # # In[12]:


    # if args.retrieve and args.local_rank in [-1, 0]:
    #     tokenizer = tokenizer_class.from_pretrained(
    #         args.output_dir, do_lower_case=args.do_lower_case)
    #     logger.info("Retrieve with: %s", args.retrieve_checkpoint)
    #     model = model_class.from_pretrained(
    #         args.retrieve_checkpoint, force_download=True)
    #     model.to(args.device)

    #     # Evaluate
    #     qids, query_reps = retrieve(args, model, tokenizer)
    #     query_reps = np.asarray(query_reps, dtype='float32')
        
    #     logger.info("Gen query rep complete")


    # In[13]:


    # passage_ids, passage_reps = [], []
    # with open(args.gen_passage_rep_output) as fin:
    #     for line in tqdm(fin):
    #         dic = json.loads(line.strip())
    #         passage_ids.append(dic['id'])
    #         passage_reps.append(dic['rep'])
    # passage_reps = np.asarray(passage_reps, dtype='float32')

