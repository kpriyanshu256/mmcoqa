# !/bin/bash
CUDA_VISIBLE_DEVICES=2 python3 code/MMCoQA/train_retriever.py \
--train_file ./MMCoQA/MMCoQA_train.txt --dev_file ./MMCoQA/MMCoQA_dev.txt --test_file ./MMCoQA/MMCoQA_test.txt \
--passages_file ./MMCoQA/multimodalqa_final_dataset_pipeline_camera_ready_MMQA_texts.jsonl \
--tables_file ./MMCoQA/multimodalqa_final_dataset_pipeline_camera_ready_MMQA_tables.jsonl \
--images_file ./MMCoQA/multimodalqa_final_dataset_pipeline_camera_ready_MMQA_images.jsonl \
--images_path ./MMCoQA/final_dataset_images/ \
--qrels ./MMCoQA/qrels.txt \
--retrieve_checkpoint ./retriever_checkpoint/checkpoint-5917 \
--model_name_or_path albert/albert-base-v2 \
--num_workers 2 \
--output_dir test0 \
--overwrite_output_dir True \
--num_train_epochs 22

CUDA_VISIBLE_DEVICES=2 python3 code/MMCoQA/train_retriever.py \
--gen_passage_rep True \
--retrieve_checkpoint ./test2/checkpoint-5061 \
--train_file ./MMCoQA/MMCoQA_train.txt \
--dev_file ./MMCoQA/MMCoQA_dev.txt \
--test_file ./MMCoQA/MMCoQA_test.txt \
--passages_file ./MMCoQA/multimodalqa_final_dataset_pipeline_camera_ready_MMQA_texts.jsonl \
--tables_file ./MMCoQA/multimodalqa_final_dataset_pipeline_camera_ready_MMQA_tables.jsonl \
--images_file ./MMCoQA/multimodalqa_final_dataset_pipeline_camera_ready_MMQA_images.jsonl \
--images_path ./MMCoQA/final_dataset_images \
--qrels ./MMCoQA/qrels.txt \
--output_dir test0 \
--overwrite_output_dir True \
--per_gpu_eval_batch_size 256 \
--model_type albert/albert-base-v2 \
--gen_passage_rep_output ./test0/dev_blocks.txt


CUDA_VISIBLE_DEVICES=2 python3 code/MMCoQA/train_pipeline.py \
--train_file ./MMCoQA/MMCoQA_train.txt --dev_file ./MMCoQA/MMCoQA_dev.txt --test_file ./MMCoQA/MMCoQA_test.txt \
--passages_file ./MMCoQA/multimodalqa_final_dataset_pipeline_camera_ready_MMQA_texts.jsonl \
--tables_file ./MMCoQA/multimodalqa_final_dataset_pipeline_camera_ready_MMQA_tables.jsonl \
--images_file ./MMCoQA/multimodalqa_final_dataset_pipeline_camera_ready_MMQA_images.jsonl \
--images_path ./MMCoQA/final_dataset_images \
--gen_passage_rep_output ./test0/dev_blocks.txt \
--qrels ./MMCoQA/qrels.txt \
--retrieve_checkpoint ./test0/checkpoint-5061 \
--overwrite_output_dir True \
--output_dir test0 \
--reader_model_type bert-base-uncased \
--retriever_model_type albert/albert-base-v2 \
--num_train_epochs 3 \
--retrieve_tokenizer_dir test0 \
--do_train True

# CUDA_VISIBLE_DEVICES=2 python3 code_v1/MMCoQA/train_pipeline.py \
# --do_train False --do_eval True --do_test True --best_global_step 4000 \
# --train_file ./MMCoQA/MMCoQA_train.txt --dev_file ./MMCoQA/MMCoQA_dev.txt --test_file ./MMCoQA/MMCoQA_test.txt \
# --passages_file ./MMCoQA/multimodalqa_final_dataset_pipeline_camera_ready_MMQA_texts.jsonl \
# --tables_file ./MMCoQA/multimodalqa_final_dataset_pipeline_camera_ready_MMQA_tables.jsonl \
# --images_file ./MMCoQA/multimodalqa_final_dataset_pipeline_camera_ready_MMQA_images.jsonl \
# --images_path ./MMCoQA/final_dataset_images \
# --gen_passage_rep_output ./test1/dev_blocks.txt \
# --qrels ./MMCoQA/qrels.txt \
# --retrieve_checkpoint ./test1/checkpoint-3169 \
# --overwrite_output_dir True1 \
# --output_dir test1 \