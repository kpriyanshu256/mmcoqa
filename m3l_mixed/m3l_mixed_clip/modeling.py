import os
import logging
import collections
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

from transformers import BertModel, BertPreTrainedModel, AlbertModel
from transformers.models.albert import AlbertPreTrainedModel

from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from copy import deepcopy

import torchvision
import clip

logger = logging.getLogger(__name__)

    
    
class Reader(nn.Module):
    def __init__(self, model_name, config):
        super().__init__()
        self.config = config
        self.num_qa_labels = config.num_qa_labels

        self.bert = AutoModel.from_pretrained(model_name) 
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_qa_labels)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

        # self.image_encoder = torchvision.models.resnet50(pretrained=True)
        # self.image_encoder.fc = nn.Linear(self.image_encoder.fc.in_features, config.hidden_size)
        
        self.image_encoder = clip.load("ViT-B/16")[0].float().visual
        self.image_proj = nn.Linear(512, config.hidden_size)


        self.qa_loss_factor = config.qa_loss_factor
        self.retrieval_loss_factor = config.retrieval_loss_factor

        self.query_encoder = AutoModel.from_pretrained(model_name) 
        self.query_proj = nn.Linear(config.hidden_size, config.proj_size)

        self._init_weights(self.qa_outputs)
        self._init_weights(self.classifier)

        # self._init_weights(self.image_encoder.fc)
        self._init_weights(self.query_proj)
        self._init_weights(self.image_proj)




    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def init_weights(self, model_path):
        wts = torch.load(os.path.join(model_path, "pytorch_model.bin"))

        params = [k for k, v in self.named_parameters()]
        weight_params = [k for k, v in wts.items()]

        missing_wts = set(params) - set(weight_params)

        if len(missing_wts) > 0:
            missing_wts = set([x.split('.')[0] for x in missing_wts])
            logger.info(f"Missing weights {missing_wts}")
            self.load_state_dict(wts, strict=False)
        else:
            self.load_state_dict(wts)

        logger.info("Reader Model initialized")
    
    @classmethod
    def from_pretrained(cls, model_name, pretrained_model_name_or_path, config):
        model = cls(model_name, config)
        model.init_weights(pretrained_model_name_or_path)
        return model

    def save_pretrained(self, save_path):
        self.config.save_pretrained(os.path.join(save_path))
        torch.save(self.state_dict(), os.path.join(save_path, "pytorch_model.bin"))

    def forward(self, 
                input_ids=None, 
                attention_mask=None, 
                token_type_ids=None, 
                start_positions=None, 
                end_positions=None, 
                retrieval_label=None,
                image_input=None,
                modality_labels=None,
                item_modality_type=None,
                query_input_ids=None, 
                query_attention_mask=None, 
                query_token_type_ids=None
    ):
        
        # inputs_ids = bs, len
        # image = bs, 3, sz, sz

        batch_size, seq_len = input_ids.size()        
        image_rep = self.image_encoder(image_input.float()) 
        image_rep = self.image_proj(image_rep)
        image_rep = image_rep.view(batch_size, 1, -1)


        outputs = self.bert(
                        input_ids = input_ids,
                        attention_mask = attention_mask,
                        token_type_ids = token_type_ids,
        )

        sequence_output = outputs[0]
        # pooled_output = outputs[1]
        pooled_output = sequence_output[:, 0, :]

        
        sequence_output = sequence_output + image_rep

        qa_logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = qa_logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1) 
        end_logits = end_logits.squeeze(-1)
                
        pooled_output = self.dropout(pooled_output)
        retrieval_logits = self.classifier(pooled_output) 


        outputs = (start_logits, end_logits, retrieval_logits) + outputs[2:]

        if start_positions is not None and end_positions is not None and retrieval_label is not None:
            start_logits = start_logits.view(batch_size, -1)
            end_logits = end_logits.view(batch_size, -1)
            
            retrival_logits = retrieval_logits.squeeze(-1)
            retrieval_logits = retrieval_logits.view(-1)
        
            start_positions = start_positions#.squeeze(-1).max(dim=1).values
            end_positions = end_positions#.squeeze(-1).max(dim=1).values
            
            retrieval_label = retrieval_label.float()#.squeeze(-1).argmax(dim=1)

            start_mask = torch.ones_like(start_positions).float()
            end_mask = torch.ones_like(end_positions).float()

            start_mask[start_positions==0] = 0.01
            end_mask[end_positions==0] = 0.01

            logger.info(f'SP {start_positions}')
            logger.info(f'EP {end_positions}')

            # logger.info(f'SM {start_mask}')
            # logger.info(f'EM {end_mask}')

                
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            qa_loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = qa_loss_fct(start_logits, start_positions) *  start_mask
            end_loss = qa_loss_fct(end_logits, end_positions) * end_mask
            qa_loss = (start_loss + end_loss) / 2
            qa_loss = torch.mean(qa_loss)


            retrieval_loss_fct = nn.BCEWithLogitsLoss()
            retrieval_loss = retrieval_loss_fct(retrieval_logits, retrieval_label)

            total_loss = self.qa_loss_factor * qa_loss \
                        + self.retrieval_loss_factor * retrieval_loss
                               
            outputs = (total_loss, qa_loss, retrieval_loss,) + outputs

        return outputs
    


class Retriever(nn.Module):
    def __init__(self, config, model_path):
        super().__init__()
        self.config = config
        self.query_encoder = AutoModel.from_config(config) 
        self.query_proj = nn.Linear(config.hidden_size, config.proj_size)
        
        self.passage_encoder = AutoModel.from_config(config) 
        self.passage_proj = nn.Linear(config.hidden_size, config.proj_size)
        self.proj_size = config.proj_size
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.image_encoder = torchvision.models.resnet101(pretrained=True)
        self.image_encoder.fc = nn.Linear(self.image_encoder.fc.in_features, config.hidden_size)
        self.image_proj = nn.Linear(config.hidden_size, config.proj_size)

        
        self.init_weights(model_path)
        self._init_weights(self.image_encoder.fc)
        self._init_weights(self.image_proj)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


    def init_weights(self, model_path):
        wts = torch.load(os.path.join(model_path, "pytorch_model.bin"))

        params = [k for k, v in self.named_parameters()]
        weight_params = [k for k, v in wts.items()]

        missing_wts = set(params) - set(weight_params)

        if len(missing_wts) > 0:
            missing_wts = set([x.split('.')[0] for x in missing_wts])
            logger.info(f"Missing weights {missing_wts}")
            self.load_state_dict(wts, strict=False)
        else:
            self.load_state_dict(wts)

        logger.info("Retriever Model initialized")
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, config):
        return cls(config, pretrained_model_name_or_path)

    
    def save_pretrained(self, save_path):
        self.config.save_pretrained(os.path.join(save_path))
        torch.save(self.state_dict(), os.path.join(save_path, "pytorch_model.bin"))



    def forward(self, 
                query_input_ids=None, 
                query_attention_mask=None, 
                query_token_type_ids=None, 
                passage_input_ids=None, 
                passage_attention_mask=None, 
                passage_token_type_ids=None, 
                retrieval_label=None, 
                question_type=None,
                image_input=None,
                query_rep=None, 
                passage_rep=None, 
                modality_labels=None):

        outputs = ()
        
        if query_input_ids is not None:
            query_outputs = self.query_encoder(query_input_ids,
                                attention_mask=query_attention_mask,
                                token_type_ids=query_token_type_ids)
            
            query_pooled_output = query_outputs[1]
            query_pooled_output = self.dropout(query_pooled_output)
            query_rep = self.query_proj(query_pooled_output)    
            outputs = (query_rep, ) + outputs
        
        if passage_input_ids is not None:
            passage_outputs = self.passage_encoder(passage_input_ids,
                                attention_mask=passage_attention_mask,
                                token_type_ids=passage_token_type_ids) 

            passage_pooled_output = passage_outputs[1] 
            passage_pooled_output = self.dropout(passage_pooled_output)
            passage_rep = self.passage_proj(passage_pooled_output) 

            image_outputs = self.image_encoder(image_input)
            image_rep= self.image_proj(image_outputs)

            # obtain the corresponding embedding -- modality_position=question_type:[0,1,0,1]*batchsize+[0,1,2,3]=[0,4,2,6]

            modality_position = question_type * passage_rep.size(0) +\
                         torch.arange(passage_rep.size(0), device=passage_rep.device, dtype=torch.long)


            passage_rep = torch.cat((passage_rep, image_rep), 0)[modality_position]

            outputs = (passage_rep, ) + outputs
                       
        if query_input_ids is not None and passage_input_ids is not None:
            passage_rep_t = passage_rep.transpose(0, 1) # proj_size, batch_size
            retrieval_logits = torch.matmul(query_rep, passage_rep_t) # batch_size, batch_size
            retrieval_label = torch.arange(query_rep.size(0), device=query_rep.device, dtype=retrieval_label.dtype)

            retrieval_loss_fct = CrossEntropyLoss()
            retrieval_loss = retrieval_loss_fct(retrieval_logits, retrieval_label)
            
            outputs = (retrieval_loss, ) + outputs

        if query_input_ids is not None and passage_rep is not None and retrieval_label is not None and len(passage_rep.size()) == 3:
            query_outputs = self.query_encoder(query_input_ids,
                                attention_mask=query_attention_mask,
                                token_type_ids=query_token_type_ids)
            
            query_pooled_output = query_outputs[1]
            query_pooled_output = self.dropout(query_pooled_output)
            query_rep = self.query_proj(query_pooled_output)
            
            batch_size, num_blocks, proj_size = passage_rep.size()
            query_rep = query_rep.unsqueeze(-1) # query_rep (batch_size, proj_size, 1)
            query_rep = query_rep.expand(batch_size, self.proj_size, num_blocks) # batch_size, proj_size, num_blocks)
            query_rep = query_rep.transpose(1, 2) # query_rep (batch_size, num_blocks, proj_size)

            retrieval_logits = query_rep * passage_rep # batch_size, num_blocks, proj_size
            retrieval_logits = torch.sum(retrieval_logits, dim=-1) # batch_size, num_blocks
            retrieval_probs = F.softmax(retrieval_logits, dim=1)
            retrieval_label = retrieval_label.squeeze(-1).argmax(dim=1)
            retrieval_loss_fct = CrossEntropyLoss()
            retrieval_loss = retrieval_loss_fct(retrieval_logits, retrieval_label)
            
            outputs = (retrieval_loss, ) + outputs

        if query_input_ids is not None and modality_labels is not None:
            query_outputs = self.query_encoder(query_input_ids,
                                attention_mask=query_attention_mask,
                                token_type_ids=query_token_type_ids)
            
            query_pooled_output = query_outputs[1]
            query_pooled_output = self.dropout(query_pooled_output)
            query_rep = self.query_proj(query_pooled_output)
            
            outputs = (retrieval_loss, ) + outputs
        
        return outputs    

class Pipeline(nn.Module):
    def __init__(self):
        super(Pipeline, self).__init__()
        
        self.reader = None
        self.retriever = None


class AlbertForRetrieverOnlyPositivePassage(AlbertPreTrainedModel):
    r"""
    
    """
    def __init__(self, config):
        super(AlbertForRetrieverOnlyPositivePassage, self).__init__(config)

        self.query_encoder = AlbertModel(config)
        self.query_proj = nn.Linear(config.hidden_size, config.proj_size)
        
        self.passage_encoder = AlbertModel(config)
        self.passage_proj = nn.Linear(config.hidden_size, config.proj_size)
        self.proj_size = config.proj_size
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.image_encoder=torchvision.models.resnet101(pretrained=True)
        self.image_encoder.fc = nn.Linear(self.image_encoder.fc.in_features, config.hidden_size)
        self.image_proj = nn.Linear(config.hidden_size, config.proj_size)


        
        self.init_weights()





    def forward(self, query_input_ids=None, query_attention_mask=None, query_token_type_ids=None, 
                passage_input_ids=None, passage_attention_mask=None, passage_token_type_ids=None, 
                retrieval_label=None,question_type=None,image_input=None,query_rep=None, passage_rep=None, modality_labels=None):
        outputs = ()
        
        if query_input_ids is not None:
            query_outputs = self.query_encoder(query_input_ids,
                                attention_mask=query_attention_mask,
                                token_type_ids=query_token_type_ids)
            
            query_pooled_output = query_outputs[1]
            query_pooled_output = self.dropout(query_pooled_output)
            query_rep = self.query_proj(query_pooled_output) # batch_size, proj_size    
            # print(query_rep[:, 0])
            outputs = (query_rep, ) + outputs
        
        if passage_input_ids is not None:
            passage_outputs = self.passage_encoder(passage_input_ids,
                                attention_mask=passage_attention_mask,
                                token_type_ids=passage_token_type_ids) 

            passage_pooled_output = passage_outputs[1] 
            passage_pooled_output = self.dropout(passage_pooled_output)
            passage_rep = self.passage_proj(passage_pooled_output) # batch_size, proj_size
            # print(passage_rep[:, 0])

            #####################encode an image
            image_outputs = self.image_encoder(image_input)
            image_rep= self.image_proj(image_outputs) # batch_size, proj_size

            ##############obtain the corresponding embedding     modality_position=question_type:[0,1,0,1]*batchsize+[0,1,2,3]=[0,4,2,6]

            modality_position=question_type*passage_rep.size(0)+torch.arange(passage_rep.size(0), device=passage_rep.device, dtype=torch.long)





            passage_rep=  torch.cat((passage_rep, image_rep), 0)[modality_position]

            outputs = (passage_rep, ) + outputs
                       
        if query_input_ids is not None and passage_input_ids is not None:
            passage_rep_t = passage_rep.transpose(0, 1) # proj_size, batch_size
            retrieval_logits = torch.matmul(query_rep, passage_rep_t) # batch_size, batch_size
            retrieval_label = torch.arange(query_rep.size(0), device=query_rep.device, dtype=retrieval_label.dtype)
            # print('retrieval_label after', retrieval_label.size(), retrieval_label)
            retrieval_loss_fct = CrossEntropyLoss()
            # print('retrieval_logits', retrieval_logits.size(), retrieval_logits)
            # print('retrieval_label', retrieval_label.size(), retrieval_label)
            retrieval_loss = retrieval_loss_fct(retrieval_logits, retrieval_label)
            
            outputs = (retrieval_loss, ) + outputs

        if query_input_ids is not None and passage_rep is not None and retrieval_label is not None and len(passage_rep.size()) == 3:
            # this is during fine tuning
            # passage_rep: batch_size, num_blocks, proj_size      
            query_outputs = self.query_encoder(query_input_ids,
                                attention_mask=query_attention_mask,
                                token_type_ids=query_token_type_ids)
            
            query_pooled_output = query_outputs[1]
            query_pooled_output = self.dropout(query_pooled_output)
            query_rep = self.query_proj(query_pooled_output) # batch_size, proj_size  
            
            batch_size, num_blocks, proj_size = passage_rep.size()
            query_rep = query_rep.unsqueeze(-1) # query_rep (batch_size, proj_size, 1)
            query_rep = query_rep.expand(batch_size, self.proj_size, num_blocks) # batch_size, proj_size, num_blocks)
            query_rep = query_rep.transpose(1, 2) # query_rep (batch_size, num_blocks, proj_size)
            retrieval_logits = query_rep * passage_rep # batch_size, num_blocks, proj_size
            retrieval_logits = torch.sum(retrieval_logits, dim=-1) # batch_size, num_blocks
            retrieval_probs = F.softmax(retrieval_logits, dim=1)
            # print('retrieval_label before', retrieval_label.size(), retrieval_label)
            retrieval_label = retrieval_label.squeeze(-1).argmax(dim=1)
            # print('retrieval_label after', retrieval_label.size(), retrieval_label)
            retrieval_loss_fct = CrossEntropyLoss()
            # print('retrieval_logits', retrieval_logits.size(), retrieval_logits)
            # print('retrieval_label', retrieval_label.size(), retrieval_label)
            retrieval_loss = retrieval_loss_fct(retrieval_logits, retrieval_label)
            
            outputs = (retrieval_loss, ) + outputs

        if query_input_ids is not None and modality_labels is not None:
            # this is during fine tuning
            # passage_rep: batch_size, num_blocks, proj_size      
            query_outputs = self.query_encoder(query_input_ids,
                                attention_mask=query_attention_mask,
                                token_type_ids=query_token_type_ids)
            
            query_pooled_output = query_outputs[1]
            query_pooled_output = self.dropout(query_pooled_output)
            query_rep = self.query_proj(query_pooled_output) # batch_size, proj_size  
            
            
            
            outputs = (retrieval_loss, ) + outputs
        return outputs


from modeling_jina import *
class JinaForRetrieverOnlyPositivePassage(JinaBertPreTrainedModel):
    r"""
    
    """
    def __init__(self, config):
        super().__init__(config, trust_remote_code=True)

        self.query_encoder = JinaBertModel(config)
        self.query_proj = nn.Linear(config.hidden_size, config.proj_size)
        
        self.passage_encoder = JinaBertModel(config)
        self.passage_proj = nn.Linear(config.hidden_size, config.proj_size)
        self.proj_size = config.proj_size
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # self.image_encoder=torchvision.models.resnet101(pretrained=True)
        # self.image_encoder.fc = nn.Linear(self.image_encoder.fc.in_features, config.hidden_size)
        # self.image_proj = nn.Linear(config.hidden_size, config.proj_size)


        self.image_encoder = clip.load("ViT-B/16")[0].float().visual
        self.image_proj = nn.Linear(512, config.proj_size)

        
        self.init_weights()





    def forward(self, query_input_ids=None, query_attention_mask=None, query_token_type_ids=None, 
                passage_input_ids=None, passage_attention_mask=None, passage_token_type_ids=None, 
                retrieval_label=None,question_type=None,image_input=None,query_rep=None, passage_rep=None, modality_labels=None):
        outputs = ()
        
        if query_input_ids is not None:
            query_outputs = self.query_encoder(query_input_ids,
                                attention_mask=query_attention_mask,
                                token_type_ids=query_token_type_ids)
            
            query_pooled_output = query_outputs[1]
            query_pooled_output = self.dropout(query_pooled_output)
            query_rep = self.query_proj(query_pooled_output) # batch_size, proj_size    
            # print(query_rep[:, 0])
            outputs = (query_rep, ) + outputs
        
        if passage_input_ids is not None:
            passage_outputs = self.passage_encoder(passage_input_ids,
                                attention_mask=passage_attention_mask,
                                token_type_ids=passage_token_type_ids) 

            passage_pooled_output = passage_outputs[1] 
            passage_pooled_output = self.dropout(passage_pooled_output)
            passage_rep = self.passage_proj(passage_pooled_output) # batch_size, proj_size
            # print(passage_rep[:, 0])

            #####################encode an image
            image_outputs = self.image_encoder(image_input)
            image_rep= self.image_proj(image_outputs) # batch_size, proj_size

            ##############obtain the corresponding embedding     modality_position=question_type:[0,1,0,1]*batchsize+[0,1,2,3]=[0,4,2,6]

            modality_position=question_type*passage_rep.size(0)+torch.arange(passage_rep.size(0), device=passage_rep.device, dtype=torch.long)





            passage_rep=  torch.cat((passage_rep, image_rep), 0)[modality_position]

            outputs = (passage_rep, ) + outputs
                       
        if query_input_ids is not None and passage_input_ids is not None:
            passage_rep_t = passage_rep.transpose(0, 1) # proj_size, batch_size
            retrieval_logits = torch.matmul(query_rep, passage_rep_t) # batch_size, batch_size
            retrieval_label = torch.arange(query_rep.size(0), device=query_rep.device, dtype=retrieval_label.dtype)
            # print('retrieval_label after', retrieval_label.size(), retrieval_label)
            retrieval_loss_fct = CrossEntropyLoss()
            # print('retrieval_logits', retrieval_logits.size(), retrieval_logits)
            # print('retrieval_label', retrieval_label.size(), retrieval_label)
            retrieval_loss = retrieval_loss_fct(retrieval_logits, retrieval_label)
            
            outputs = (retrieval_loss, ) + outputs

        if query_input_ids is not None and passage_rep is not None and retrieval_label is not None and len(passage_rep.size()) == 3:
            # this is during fine tuning
            # passage_rep: batch_size, num_blocks, proj_size      
            query_outputs = self.query_encoder(query_input_ids,
                                attention_mask=query_attention_mask,
                                token_type_ids=query_token_type_ids)
            
            query_pooled_output = query_outputs[1]
            query_pooled_output = self.dropout(query_pooled_output)
            query_rep = self.query_proj(query_pooled_output) # batch_size, proj_size  
            
            batch_size, num_blocks, proj_size = passage_rep.size()
            query_rep = query_rep.unsqueeze(-1) # query_rep (batch_size, proj_size, 1)
            query_rep = query_rep.expand(batch_size, self.proj_size, num_blocks) # batch_size, proj_size, num_blocks)
            query_rep = query_rep.transpose(1, 2) # query_rep (batch_size, num_blocks, proj_size)
            retrieval_logits = query_rep * passage_rep # batch_size, num_blocks, proj_size
            retrieval_logits = torch.sum(retrieval_logits, dim=-1) # batch_size, num_blocks
            retrieval_probs = F.softmax(retrieval_logits, dim=1)
            # print('retrieval_label before', retrieval_label.size(), retrieval_label)
            retrieval_label = retrieval_label.squeeze(-1).argmax(dim=1)
            # print('retrieval_label after', retrieval_label.size(), retrieval_label)
            retrieval_loss_fct = CrossEntropyLoss()
            # print('retrieval_logits', retrieval_logits.size(), retrieval_logits)
            # print('retrieval_label', retrieval_label.size(), retrieval_label)
            retrieval_loss = retrieval_loss_fct(retrieval_logits, retrieval_label)
            
            outputs = (retrieval_loss, ) + outputs

        if query_input_ids is not None and modality_labels is not None:
            # this is during fine tuning
            # passage_rep: batch_size, num_blocks, proj_size      
            query_outputs = self.query_encoder(query_input_ids,
                                attention_mask=query_attention_mask,
                                token_type_ids=query_token_type_ids)
            
            query_pooled_output = query_outputs[1]
            query_pooled_output = self.dropout(query_pooled_output)
            query_rep = self.query_proj(query_pooled_output) # batch_size, proj_size  
            
            
            
            outputs = (retrieval_loss, ) + outputs
        return outputs
    
