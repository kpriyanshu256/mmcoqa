import json
import logging
import math
import collections
import linecache
import numpy as np
from io import open
from tqdm import tqdm
from torch.utils.data import Dataset


from PIL import Image
import cv2
import torch
import torchvision


logger = logging.getLogger(__name__)

class RetrieverInputExample(object):
    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class RetrieverDataset:
    def __init__(self, 
        filename,
        tokenizer, 
        load_small, 
        history_num, 
        prepend_history_questions=False, 
        prepend_history_answers=False,
        query_max_seq_length=128, 
        passage_max_seq_length=384,
        is_pretraining=False, 
        given_query=False, 
        given_passage=False, 
        only_positive_passage=True, 
        include_first_for_retriever=False,
        passages_dict=None,
        images_dict=None,
        tables_dict=None
    ):
        self._filename = filename
        self._tokenizer = tokenizer
        self._load_small = load_small
        self._history_num = history_num
        self._query_max_seq_length = query_max_seq_length
        self._passage_max_seq_length = passage_max_seq_length
        self._prepend_history_questions = prepend_history_questions
        self._prepend_history_answers = prepend_history_answers
        
        
        self._is_pretraining = is_pretraining 
        self._given_query = given_query
        self._given_passage = given_passage
        
        self._only_positive_passage = only_positive_passage
        
        self._include_first_for_retriever = include_first_for_retriever
        
        self._total_data = 0      
        if self._load_small:
            self._total_data = 100
        else:
            with open(filename, "r") as f:
                self._total_data = len(f.readlines())

        self._passages_dict = passages_dict
        self._images_dict = images_dict
        self._tables_dict = tables_dict

        self._modality_dict = {'text':0,'table':0,'image':1}

        self.img_size = (224, 224)

    def __len__(self):
        return self._total_data

    def _image_transform(self, path):

        trans_f = torchvision.transforms.Compose([
            # torchvision.transforms.Resize((512,512)),
            torchvision.transforms.Resize(self.img_size),

            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        )

        img = cv2.imread(path)
        img = Image.fromarray(img)
        img = trans_f(img)
        print(img.shape)
        img = img.numpy()
        return img
    
    def __getitem__(self, idx):
        line = linecache.getline(self._filename, idx + 1)    
        entry = json.loads(line.strip())
        qas_id = entry["qid"]
        
        return_feature_dict = {}
        
        if self._given_query:
            if self._is_pretraining:
                question_text_for_retriever = entry["gold_question"]
            else:
                question_text_for_retriever = entry["gold_question"]
                question_text = entry["gold_question"]

                orig_question_text = entry["question"]
                history = entry['history']
                question_text_list = []
                if self._history_num > 0:
                    for turn in history[- self._history_num :]:
                        if self._prepend_history_questions:
                            question_text_list.append(turn['question'])
                        if self._prepend_history_answers:
                            question_text_list.append(turn['answer'][0]['answer'])

                question_text_list.append(orig_question_text)
                question_text = f' {self._tokenizer.sep_token} '.join(question_text_list)
                question_text_for_retriever = question_text
    
                # include the first question in addition to history_num for retriever (not reader)
                if self._include_first_for_retriever and len(history) > 0:
                    first_question = history[0]['question']
                    if first_question != question_text_list[0]:                    
                        question_text_for_retriever = first_question + f' {self._tokenizer.sep_token} ' + question_text
                        
                        
            query_example = RetrieverInputExample(guid=qas_id, text_a=question_text_for_retriever)

            query_feature = retriever_convert_example_to_feature(query_example, 
                                                                self._tokenizer, 
                                                                max_length=self._query_max_seq_length)

            query_feature_dict = {'query_input_ids': np.asarray(query_feature['input_ids']), 
                                  'query_token_type_ids': np.asarray(query_feature['token_type_ids']), 
                                  'query_attention_mask': np.asarray(query_feature['attention_mask']), 
                                  'qid': qas_id}

            if not self._is_pretraining:
                query_feature_dict['question_text'] = question_text
                query_feature_dict['answer_text'] = entry['answer'][0]['answer']
                query_feature_dict['answer_start'] = 0
            
            return_feature_dict.update(query_feature_dict)

            if entry['question_type'] == "text":
                modality_label = 0
            elif entry['question_type'] == "table":
                modality_label = 1        
            else:
                modality_label = 2 
            return_feature_dict.update({'modality_label':modality_label})

        if self._given_passage:
            return_feature_dict['question_type'] = self._modality_dict[entry['question_type']]
            
            if self._only_positive_passage:
                if entry['question_type'] == "text":  

                    passage_id = entry['answer'][0]['text_instances'][0]['doc_id']
                    passage = self._passages_dict[passage_id]


                    example_id = '{}_{}'.format(qas_id, passage_id)
                    passage_example = RetrieverInputExample(
                                            guid=example_id,
                                            text_a=passage,
                                            label=1)

                    passage_feature = retriever_convert_example_to_feature(
                                                        passage_example, 
                                                        self._tokenizer, 
                                                        max_length=self._passage_max_seq_length
                    )


                    passage_feature_dict = {
                                    'passage_input_ids': np.asarray(passage_feature['input_ids']), 
                                    'passage_token_type_ids': np.asarray(passage_feature['token_type_ids']), 
                                    'passage_attention_mask': np.asarray(passage_feature['attention_mask']),
                                    'retrieval_label': passage_feature['label'], 
                                    'example_id': example_id,
                                    'image_input':np.zeros([3, *self.img_size])
                    }
                    return_feature_dict.update(passage_feature_dict)

                elif entry['question_type'] == "image":  
                  
                    image_id = entry['answer'][0]['image_instances'][0]['doc_id']
                    image_path = self._images_dict[image_id]
                    img = self._image_transform(image_path)

                    example_id = '{}_{}'.format(qas_id, image_id)


                    passage_feature_dict = {
                                    'passage_input_ids': np.asarray([0]*self._passage_max_seq_length), 
                                    'passage_token_type_ids': np.asarray([0]*self._passage_max_seq_length), 
                                    'passage_attention_mask': np.asarray([0]*self._passage_max_seq_length),
                                    'retrieval_label': 1, 
                                    'example_id': example_id,
                                    'image_input':img
                    }
                    return_feature_dict.update(passage_feature_dict)

                else:
                    table_id=entry['table_id']
                    table = self._tables_dict[table_id]

                    example_id = '{}_{}'.format(qas_id, table_id)
                    table_example = RetrieverInputExample(
                                            guid=example_id,
                                            text_a=table,
                                            label=1)

                    table_feature = retriever_convert_example_to_feature(table_example, 
                                                                        self._tokenizer, 
                                                                        max_length=self._passage_max_seq_length
                    )
                    passage_feature_dict = {
                                    'passage_input_ids': np.asarray(table_feature['input_ids']), 
                                    'passage_token_type_ids': np.asarray(table_feature['token_type_ids']), 
                                    'passage_attention_mask': np.asarray(table_feature['attention_mask']),
                                    'retrieval_label': table_feature['label'], 
                                    'example_id': example_id,
                                    'image_input':np.zeros([3, *self.img_size])
                    }
                    return_feature_dict.update(passage_feature_dict)

        return return_feature_dict


class GenPassageRepDataset():
    def __init__(self, 
                filename, 
                load_small,
                tokenizer, 
                passage_max_seq_length=386,
                passages_dict=None,
                images_dict=None,
                tables_dict=None, 
                idx_id_list=None
        ):
        

        self._tokenizer = tokenizer
        self._load_small = load_small  
        self._passage_max_seq_length = passage_max_seq_length


        self._passages_dict = passages_dict
        self._images_dict = images_dict
        self._tables_dict = tables_dict 
        self._idx_id_list = idx_id_list

        self._total_data = 0      
        if self._load_small:
            self._total_data = 100
        else:
            self._total_data = len(passages_dict) + len(images_dict) + len(tables_dict)
        self._modality_dict = {'text':0,'table':0,'image':1}

        self.img_size = (224, 224)

    def _image_transform(self, path):  
        trans_f = torchvision.transforms.Compose([
            torchvision.transforms.Resize(self.img_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


        try:
            img = Image.open(path).convert("RGB")
        except:
            raise Exception("DataError: " + path)

        try:
            img = trans_f(img)
        except:
            print("Cannot transform, using black image: ", path)
            img = torch.zeros((3, *self.img_size))

        img = img.numpy()
        return img
                
    def __len__(self):
        return self._total_data
                
    def __getitem__(self, idx):
        if self._idx_id_list[idx][1] == 'text':
            passage = self._passages_dict[self._idx_id_list[idx][0]]
            example_id = self._idx_id_list[idx][0]

            passage_example = RetrieverInputExample(guid=example_id, text_a=passage)
            passage_feature = retriever_convert_example_to_feature(
                                                        passage_example, 
                                                        self._tokenizer, 
                                                        max_length=self._passage_max_seq_length)

            batch_feature = {'passage_input_ids': np.asarray(passage_feature['input_ids']), 
                             'passage_token_type_ids': np.asarray(passage_feature['token_type_ids']), 
                             'passage_attention_mask': np.asarray(passage_feature['attention_mask']),
                             'example_id': example_id,
                             'image_input':np.zeros([3, *self.img_size])}

        elif self._idx_id_list[idx][1] == 'table':
            passage = self._tables_dict[self._idx_id_list[idx][0]]
            example_id = self._idx_id_list[idx][0]

            passage_example = RetrieverInputExample(guid=example_id, text_a=passage)
            passage_feature = retriever_convert_example_to_feature(
                                                        passage_example, 
                                                        self._tokenizer, 
                                                        max_length=self._passage_max_seq_length)

            batch_feature = {'passage_input_ids': np.asarray(passage_feature['input_ids']), 
                             'passage_token_type_ids': np.asarray(passage_feature['token_type_ids']), 
                             'passage_attention_mask': np.asarray(passage_feature['attention_mask']),
                             'example_id': example_id,
                             'image_input':np.zeros([3, *self.img_size])}

        elif self._idx_id_list[idx][1] == 'image':
            image_path = self._images_dict[self._idx_id_list[idx][0]]
            example_id = self._idx_id_list[idx][0]

            img = self._image_transform(image_path)

            batch_feature = {'passage_input_ids': np.asarray([0]*self._passage_max_seq_length), 
                             'passage_token_type_ids': np.asarray([0]*self._passage_max_seq_length), 
                             'passage_attention_mask': np.asarray([0]*self._passage_max_seq_length),
                             'example_id': example_id,
                             'image_input':img}

        batch_feature['question_type'] = self._modality_dict[self._idx_id_list[idx][1]]
        return batch_feature


def retriever_convert_example_to_feature(
                            example, tokenizer,
                            max_length=512,
                            pad_on_left=False,
                            pad_token=0,
                            pad_token_segment_id=0,
                            mask_padding_with_zero=True
    ):


    inputs = tokenizer.encode_plus(
        example.text_a,
        example.text_b,
        add_special_tokens=True,
        max_length=max_length,
        truncation=True,
    )
    input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

    attention_mask = inputs['attention_mask']

    # Zero-pad up to the sequence length.
    padding_length = max_length - len(input_ids)
    if pad_on_left:
        input_ids = ([pad_token] * padding_length) + input_ids
        attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
        token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
    else:
        input_ids = input_ids + ([pad_token] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

    
    
    feature = dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                label=example.label
    )

    return feature