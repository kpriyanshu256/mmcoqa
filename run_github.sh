# !/bin/bash
D=test_rep
mkdir $D

CUDA_VISIBLE_DEVICES=2 python3 github_code/train_retriever.py \
--train_file ./MMCoQA/MMCoQA_train.txt \
--dev_file ./MMCoQA/MMCoQA_dev.txt \
--test_file ./MMCoQA/MMCoQA_test.txt \
--passages_file ./MMCoQA/multimodalqa_final_dataset_pipeline_camera_ready_MMQA_texts.jsonl \
--tables_file ./MMCoQA/multimodalqa_final_dataset_pipeline_camera_ready_MMQA_tables.jsonl \
--images_file ./MMCoQA/multimodalqa_final_dataset_pipeline_camera_ready_MMQA_images.jsonl \
--images_path ./MMCoQA/final_dataset_images/ \
--qrels ./MMCoQA/qrels.txt \
--retrieve_checkpoint ./retriever_checkpoint/checkpoint-5917 \
--output_dir $D \
--overwrite_output_dir True \


# CUDA_VISIBLE_DEVICES=2 python3 github_code/train_retriever.py \
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


# CUDA_VISIBLE_DEVICES=2 python3 github_code/train_pipeline.py \
# --train_file ./MMCoQA/MMCoQA_train.txt \
# --dev_file ./MMCoQA/MMCoQA_dev.txt \
# --test_file ./MMCoQA/MMCoQA_test.txt \
# --passages_file ./MMCoQA/multimodalqa_final_dataset_pipeline_camera_ready_MMQA_texts.jsonl \
# --tables_file ./MMCoQA/multimodalqa_final_dataset_pipeline_camera_ready_MMQA_tables.jsonl \
# --images_file ./MMCoQA/multimodalqa_final_dataset_pipeline_camera_ready_MMQA_images.jsonl \
# --images_path ./MMCoQA/final_dataset_images \
# --gen_passage_rep_output ./$D/dev_blocks.txt \
# --qrels ./MMCoQA/qrels.txt \
# --retrieve_checkpoint ./$D/checkpoint-5061 \
# --overwrite_output_dir True \
# --output_dir $D \
# --num_train_epochs 3 \
# --retrieve_tokenizer_dir $D \
# --do_train True \

# CUDA_VISIBLE_DEVICES=2 python3 github_code/train_pipeline.py \
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
