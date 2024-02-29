#!/bin/bash
#SBATCH --job-name=xfix_squad
#SBATCH --output=logs/output_%A.log
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --gpus=1
#SBATCH --exclude=boston-2-25,boston-2-27,boston-2-29,boston-2-31

# Activate the conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mmcoqa

echo "On host $(hostname)"
nvidia-smi

D0=orig_retriever_checkpoint
D=xfix_squad
mkdir $D

which python

# CUDA_VISIBLE_DEVICES=0 python code/MMCoQA/train_retriever.py \
# --train_file ./MMCoQA/MMCoQA_train.txt \
# --dev_file ./MMCoQA/MMCoQA_dev.txt \
# --test_file ./MMCoQA/MMCoQA_test.txt \
# --passages_file ./MMCoQA/multimodalqa_final_dataset_pipeline_camera_ready_MMQA_texts.jsonl \
# --tables_file ./MMCoQA/multimodalqa_final_dataset_pipeline_camera_ready_MMQA_tables.jsonl \
# --images_file ./MMCoQA/multimodalqa_final_dataset_pipeline_camera_ready_MMQA_images.jsonl \
# --images_path ./MMCoQA/final_dataset_images/ \
# --qrels ./MMCoQA/qrels.txt \
# --retrieve_checkpoint ./retriever_checkpoint/checkpoint-5917 \
# --output_dir $D \
# --overwrite_output_dir True \
# --do_eval False \
# --do_test False \
# --num_workers 8

# CUDA_VISIBLE_DEVICES=0 python code/MMCoQA/train_retriever.py \
# --gen_passage_rep True \
# --retrieve_checkpoint ./$D/checkpoint-5061 \
# --train_file ./MMCoQA/MMCoQA_train.txt \
# --dev_file ./MMCoQA/MMCoQA_dev.txt \
# --test_file ./MMCoQA/MMCoQA_test.txt \
# --passages_file ./MMCoQA/multimodalqa_final_dataset_pipeline_camera_ready_MMQA_texts.jsonl \
# --tables_file ./MMCoQA/multimodalqa_final_dataset_pipeline_camera_ready_MMQA_tables.jsonl \
# --images_file ./MMCoQA/multimodalqa_final_dataset_pipeline_camera_ready_MMQA_images.jsonl \
# --images_path ./MMCoQA/final_dataset_images \
# --qrels ./MMCoQA/qrels.txt \
# --output_dir $D \
# --overwrite_output_dir True \
# --per_gpu_eval_batch_size 256 \
# --gen_passage_rep_output ./$D/dev_blocks.txt \

CUDA_VISIBLE_DEVICES=0 python code/MMCoQA/train_pipeline.py \
--train_file ./MMCoQA/MMCoQA_train.txt \
--dev_file ./MMCoQA/MMCoQA_dev.txt \
--test_file ./MMCoQA/MMCoQA_test.txt \
--passages_file ./MMCoQA/multimodalqa_final_dataset_pipeline_camera_ready_MMQA_texts.jsonl \
--tables_file ./MMCoQA/multimodalqa_final_dataset_pipeline_camera_ready_MMQA_tables.jsonl \
--images_file ./MMCoQA/multimodalqa_final_dataset_pipeline_camera_ready_MMQA_images.jsonl \
--images_path ./MMCoQA/final_dataset_images \
--gen_passage_rep_output ./$D0/dev_blocks.txt \
--qrels ./MMCoQA/qrels.txt \
--retrieve_checkpoint ./$D0/checkpoint-5061 \
--overwrite_output_dir True \
--output_dir $D \
--num_train_epochs 3 \
--retrieve_tokenizer_dir $D0 \
--do_train True

# python3 code/MMCoQA/train_pipeline.py \
# --do_train False --do_eval False --do_test True --best_global_step 12000 \
# --train_file ./MMCoQA/MMCoQA_train.txt \
# --dev_file ./MMCoQA/MMCoQA_dev.txt \
# --test_file ./MMCoQA/MMCoQA_test.txt \
# --passages_file ./MMCoQA/multimodalqa_final_dataset_pipeline_camera_ready_MMQA_texts.jsonl \
# --tables_file ./MMCoQA/multimodalqa_final_dataset_pipeline_camera_ready_MMQA_tables.jsonl \
# --images_file ./MMCoQA/multimodalqa_final_dataset_pipeline_camera_ready_MMQA_images.jsonl \
# --images_path ./MMCoQA/final_dataset_images \
# --gen_passage_rep_output $D1/dev_blocks.txt \
# --qrels ./MMCoQA/qrels.txt \
# --retrieve_checkpoint $D/checkpoint-5061 \
# --overwrite_output_dir True \
# --output_dir $D
