#!/bin/bash
#SBATCH --job-name=rtp_1
#SBATCH --output=m3l_mixed_no_ret_d2.out
#SBATCH --error=m3l_mixed_no_ret_d2.err
#SBATCH --mem=64g
#SBATCH --gres=gpu:A6000:1
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=8
#SBATCH --mail-user=priyansk@andrew.cmu.edu
source /home/priyansk/.bashrc
conda activate m3l_upd

D=/data/tir/projects/tir7/user_data/priyansk/m3l_base_retriever_dbert_2
D1=/data/tir/projects/tir7/user_data/priyansk/m3l_base_retriever

mkdir $D


# python3 /home/priyansk/m3l_mixed/train_retriever.py \
# --train_file /data/tir/projects/tir7/user_data/priyansk/MMCoQA/MMCoQA/MMCoQA_train.txt \
# --dev_file /data/tir/projects/tir7/user_data/priyansk/MMCoQA/MMCoQA/MMCoQA_dev.txt \
# --test_file /data/tir/projects/tir7/user_data/priyansk/MMCoQA/MMCoQA/MMCoQA_test.txt \
# --passages_file /data/tir/projects/tir7/user_data/priyansk/MMCoQA/MMCoQA/multimodalqa_final_dataset_pipeline_camera_ready_MMQA_texts.jsonl \
# --tables_file /data/tir/projects/tir7/user_data/priyansk/MMCoQA/MMCoQA/multimodalqa_final_dataset_pipeline_camera_ready_MMQA_tables.jsonl \
# --images_file /data/tir/projects/tir7/user_data/priyansk/MMCoQA/MMCoQA/multimodalqa_final_dataset_pipeline_camera_ready_MMQA_images.jsonl \
# --images_path /data/tir/projects/tir7/user_data/priyansk/MMCoQA/MMCoQA/final_dataset_images/ \
# --qrels /data/tir/projects/tir7/user_data/priyansk/MMCoQA/MMCoQA/qrels.txt \
# --retrieve_checkpoint /data/tir/projects/tir7/user_data/priyansk/MMCoQA/retriever_checkpoint/checkpoint-5917 \
# --model_name_or_path albert-base-v2 \
# --output_dir $D \
# --overwrite_output_dir True \
# --do_eval False \
# --num_workers 8 \
# --num_train_epochs 22 \
# --query_max_seq_length 256 \
# --passage_max_seq_length 512 \
# --history_num 5 \
# # --do_train False \

# python3 /home/priyansk/m3l_mixed/train_retriever.py \
# --gen_passage_rep True \
# --retrieve_checkpoint $D/checkpoint-5061 \
# --train_file /data/tir/projects/tir7/user_data/priyansk/MMCoQA/MMCoQA/MMCoQA_train.txt \
# --dev_file /data/tir/projects/tir7/user_data/priyansk/MMCoQA/MMCoQA/MMCoQA_dev.txt \
# --test_file /data/tir/projects/tir7/user_data/priyansk/MMCoQA/MMCoQA/MMCoQA_test.txt \
# --passages_file /data/tir/projects/tir7/user_data/priyansk/MMCoQA/MMCoQA/multimodalqa_final_dataset_pipeline_camera_ready_MMQA_texts.jsonl \
# --tables_file /data/tir/projects/tir7/user_data/priyansk/MMCoQA/MMCoQA/multimodalqa_final_dataset_pipeline_camera_ready_MMQA_tables.jsonl \
# --images_file /data/tir/projects/tir7/user_data/priyansk/MMCoQA/MMCoQA/multimodalqa_final_dataset_pipeline_camera_ready_MMQA_images.jsonl \
# --images_path /data/tir/projects/tir7/user_data/priyansk/MMCoQA/MMCoQA/final_dataset_images \
# --qrels /data/tir/projects/tir7/user_data/priyansk/MMCoQA/MMCoQA/qrels.txt \
# --model_name_or_path albert-base-v2 \
# --output_dir $D \
# --overwrite_output_dir True \
# --per_gpu_eval_batch_size 256 \
# --gen_passage_rep_output $D/dev_blocks.txt \
# --num_workers 8 \
# --query_max_seq_length 256 \
# --passage_max_seq_length 512 \
# --history_num 5 \

# bert

# python3 /home/priyansk/m3l_mixed/train_pipeline.py \
# --train_file /data/tir/projects/tir7/user_data/priyansk/MMCoQA/MMCoQA/MMCoQA_train.txt \
# --dev_file /data/tir/projects/tir7/user_data/priyansk/MMCoQA/MMCoQA/MMCoQA_dev.txt \
# --test_file /data/tir/projects/tir7/user_data/priyansk/MMCoQA/MMCoQA/MMCoQA_test.txt \
# --passages_file /data/tir/projects/tir7/user_data/priyansk/MMCoQA/MMCoQA/multimodalqa_final_dataset_pipeline_camera_ready_MMQA_texts.jsonl \
# --tables_file /data/tir/projects/tir7/user_data/priyansk/MMCoQA/MMCoQA/multimodalqa_final_dataset_pipeline_camera_ready_MMQA_tables.jsonl \
# --images_file /data/tir/projects/tir7/user_data/priyansk/MMCoQA/MMCoQA/multimodalqa_final_dataset_pipeline_camera_ready_MMQA_images.jsonl \
# --images_path /data/tir/projects/tir7/user_data/priyansk/MMCoQA/MMCoQA/final_dataset_images \
# --gen_passage_rep_output $D1/dev_blocks.txt \
# --qrels /data/tir/projects/tir7/user_data/priyansk/MMCoQA/MMCoQA/qrels.txt \
# --retriever_model_name_or_path albert-base-v2 \
# --retrieve_checkpoint $D1/checkpoint-5061 \
# --retrieve_tokenizer_dir $D1 \
# --overwrite_output_dir True \
# --output_dir $D \
# --num_train_epochs 1 \
# --use_retriever_prob False \
# --save_steps 1000 \
# --num_workers 8 \
# --top_k_for_reader 5 \
# --num_train_epochs 1 \
# --learning_rate 5e-5 \
# --history_num 5 \
# --retriever_query_max_seq_length 256 \
# --model_type albert \
# --do_train False \
# --do_eval False \
# --best_global_step 1000 \


# deberta - new

python3 /home/priyansk/m3l_mixed/train_pipeline.py \
--train_file /data/tir/projects/tir7/user_data/priyansk/MMCoQA/MMCoQA/MMCoQA_train.txt \
--dev_file /data/tir/projects/tir7/user_data/priyansk/MMCoQA/MMCoQA/MMCoQA_dev.txt \
--test_file /data/tir/projects/tir7/user_data/priyansk/MMCoQA/MMCoQA/MMCoQA_test.txt \
--passages_file /data/tir/projects/tir7/user_data/priyansk/MMCoQA/MMCoQA/multimodalqa_final_dataset_pipeline_camera_ready_MMQA_texts.jsonl \
--tables_file /data/tir/projects/tir7/user_data/priyansk/MMCoQA/MMCoQA/multimodalqa_final_dataset_pipeline_camera_ready_MMQA_tables.jsonl \
--images_file /data/tir/projects/tir7/user_data/priyansk/MMCoQA/MMCoQA/multimodalqa_final_dataset_pipeline_camera_ready_MMQA_images.jsonl \
--images_path /data/tir/projects/tir7/user_data/priyansk/MMCoQA/MMCoQA/final_dataset_images \
--gen_passage_rep_output $D1/dev_blocks.txt \
--qrels /data/tir/projects/tir7/user_data/priyansk/MMCoQA/MMCoQA/qrels.txt \
--retriever_model_name_or_path albert-base-v2 \
--retrieve_checkpoint $D1/checkpoint-5061 \
--reader_model_name_or_path microsoft/deberta-v3-large \
--retrieve_tokenizer_dir $D1 \
--overwrite_output_dir True \
--output_dir $D \
--num_train_epochs 1 \
--use_retriever_prob False \
--save_steps 1000 \
--num_workers 8 \
--top_k_for_reader 5 \
--num_train_epochs 1 \
--learning_rate 5e-6 \
--model_type albert \
--history_num 5 \
--retriever_query_max_seq_length 256 \
# --do_train False \
# --do_eval False \
# --best_global_step 1000 \



# deberta

# python3 /home/priyansk/m3l_mixed/train_pipeline.py \
# --train_file /data/tir/projects/tir7/user_data/priyansk/MMCoQA/MMCoQA/MMCoQA_train.txt \
# --dev_file /data/tir/projects/tir7/user_data/priyansk/MMCoQA/MMCoQA/MMCoQA_dev.txt \
# --test_file /data/tir/projects/tir7/user_data/priyansk/MMCoQA/MMCoQA/MMCoQA_test.txt \
# --passages_file /data/tir/projects/tir7/user_data/priyansk/MMCoQA/MMCoQA/multimodalqa_final_dataset_pipeline_camera_ready_MMQA_texts.jsonl \
# --tables_file /data/tir/projects/tir7/user_data/priyansk/MMCoQA/MMCoQA/multimodalqa_final_dataset_pipeline_camera_ready_MMQA_tables.jsonl \
# --images_file /data/tir/projects/tir7/user_data/priyansk/MMCoQA/MMCoQA/multimodalqa_final_dataset_pipeline_camera_ready_MMQA_images.jsonl \
# --images_path /data/tir/projects/tir7/user_data/priyansk/MMCoQA/MMCoQA/final_dataset_images \
# --gen_passage_rep_output $D1/dev_blocks.txt \
# --qrels /data/tir/projects/tir7/user_data/priyansk/MMCoQA/MMCoQA/qrels.txt \
# --retriever_model_name_or_path albert-base-v2 \
# --reader_model_name_or_path microsoft/deberta-v3-large \
# --retrieve_checkpoint $D1/checkpoint-5061 \
# --retrieve_tokenizer_dir $D1 \
# --overwrite_output_dir True \
# --output_dir $D \
# --num_train_epochs 1 \
# --use_retriever_prob False \
# --save_steps 1000 \
# --num_workers 8 \
# --top_k_for_reader 5 \
# --num_train_epochs 1 \
# --learning_rate 5e-5 \
# --model_type albert \
# --retriever_query_max_seq_length 256 \
# --history_num 5 \


# --do_train False \
# --do_eval False \
# --best_global_step 1000 \


