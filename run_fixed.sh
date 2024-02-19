#!/bin/bash
#SBATCH --job-name=rtp_1
#SBATCH --output=m3l.out
#SBATCH --error=m3l.err
#SBATCH --mem=64g
#SBATCH --gres=gpu:A6000:1
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=8
#SBATCH --mail-user=priyansk@andrew.cmu.edu
source /home/priyansk/.bashrc
conda activate m3l

D=/data/tir/projects/tir7/user_data/priyansk/m3l_fix3
D1=/data/tir/projects/tir7/user_data/priyansk/m3l_fix

mkdir $D
cd /home/priyansk/m3l


# python3 /home/priyansk/m3l/github_custom/MMCoQA/train_retriever.py \
# --train_file /data/tir/projects/tir7/user_data/priyansk/MMCoQA/MMCoQA/MMCoQA_train.txt \
# --dev_file /data/tir/projects/tir7/user_data/priyansk/MMCoQA/MMCoQA/MMCoQA_dev.txt \
# --test_file /data/tir/projects/tir7/user_data/priyansk/MMCoQA/MMCoQA/MMCoQA_test.txt \
# --passages_file /data/tir/projects/tir7/user_data/priyansk/MMCoQA/MMCoQA/multimodalqa_final_dataset_pipeline_camera_ready_MMQA_texts.jsonl \
# --tables_file /data/tir/projects/tir7/user_data/priyansk/MMCoQA/MMCoQA/multimodalqa_final_dataset_pipeline_camera_ready_MMQA_tables.jsonl \
# --images_file /data/tir/projects/tir7/user_data/priyansk/MMCoQA/MMCoQA/multimodalqa_final_dataset_pipeline_camera_ready_MMQA_images.jsonl \
# --images_path /data/tir/projects/tir7/user_data/priyansk/MMCoQA/MMCoQA/final_dataset_images/ \
# --qrels /data/tir/projects/tir7/user_data/priyansk/MMCoQA/MMCoQA/qrels.txt \
# --retrieve_checkpoint /data/tir/projects/tir7/user_data/priyansk/MMCoQA/retriever_checkpoint/checkpoint-5917 \
# --output_dir $D \
# --overwrite_output_dir True \
# --do_eval False \
# --num_workers 8 \


# python3 /home/priyansk/m3l/github_custom/MMCoQA/train_retriever.py \
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
# --output_dir $D \
# --overwrite_output_dir True \
# --per_gpu_eval_batch_size 256 \
# --gen_passage_rep_output $D/dev_blocks.txt \
# --num_workers 8 \


python3 /home/priyansk/m3l/github-1/MMCoQA/train_updated.py \
--train_file /data/tir/projects/tir7/user_data/priyansk/MMCoQA/MMCoQA/MMCoQA_train.txt \
--dev_file /data/tir/projects/tir7/user_data/priyansk/MMCoQA/MMCoQA/MMCoQA_dev.txt \
--test_file /data/tir/projects/tir7/user_data/priyansk/MMCoQA/MMCoQA/MMCoQA_test.txt \
--passages_file /data/tir/projects/tir7/user_data/priyansk/MMCoQA/MMCoQA/multimodalqa_final_dataset_pipeline_camera_ready_MMQA_texts.jsonl \
--tables_file /data/tir/projects/tir7/user_data/priyansk/MMCoQA/MMCoQA/multimodalqa_final_dataset_pipeline_camera_ready_MMQA_tables.jsonl \
--images_file /data/tir/projects/tir7/user_data/priyansk/MMCoQA/MMCoQA/multimodalqa_final_dataset_pipeline_camera_ready_MMQA_images.jsonl \
--images_path /data/tir/projects/tir7/user_data/priyansk/MMCoQA/MMCoQA/final_dataset_images \
--gen_passage_rep_output $D1/dev_blocks.txt \
--qrels /data/tir/projects/tir7/user_data/priyansk/MMCoQA/MMCoQA/qrels.txt \
--retrieve_checkpoint $D1/checkpoint-5061 \
--overwrite_output_dir True \
--output_dir $D \
--num_train_epochs 1 \
--retrieve_tokenizer_dir $D1 \
--use_retriever_prob False \
--save_steps 300 \
--num_workers 8 \
--do_train False \



# CUDA_VISIBLE_DEVICES=2 python3 /home/priyansk/m3l/github_custom/MMCoQA/train_pipeline.py \
# --do_train False --do_eval False --do_test True --best_global_step 12000 \
# --train_file ./MMCoQA/MMCoQA_train.txt \
# --dev_file ./MMCoQA/MMCoQA_dev.txt \
# --test_file ./MMCoQA/MMCoQA_test.txt \
# --passages_file ./MMCoQA/multimodalqa_final_dataset_pipeline_camera_ready_MMQA_texts.jsonl \
# --tables_file ./MMCoQA/multimodalqa_final_dataset_pipeline_camera_ready_MMQA_tables.jsonl \
# --images_file ./MMCoQA/multimodalqa_final_dataset_pipeline_camera_ready_MMQA_images.jsonl \
# --images_path ./MMCoQA/final_dataset_images \
# --gen_passage_rep_output ./test_copy/dev_blocks.txt \
# --qrels ./MMCoQA/qrels.txt \
# --retrieve_checkpoint ./test_copy/checkpoint-5061 \
# --overwrite_output_dir True \
# --output_dir test_copy \