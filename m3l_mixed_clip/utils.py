import sys
import json
import logging
import math
import collections
import linecache
import numpy as np
from io import open
from tqdm import tqdm
import torch
from torch.utils.data import Dataset

from transformers.models.bert.tokenization_bert import BasicTokenizer, whitespace_tokenize

from PIL import Image
import cv2
import torch
import torchvision
import pickle


logger = logging.getLogger(__name__)



class AverageMeter:
    """
    Computes and stores the average and current value
    Source : https://www.kaggle.com/abhishek/bert-base-uncased-using-pytorch/
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False

def image_transform(path, image_size):  
    trans_f = torchvision.transforms.Compose([
        torchvision.transforms.Resize(image_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


    img = Image.open(path).convert("RGB")
    img = trans_f(img).unsqueeze(0)
    img = img.numpy()
    return img


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs



def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heuristic between
    # `pred_text` and `orig_text` to get a character-to-character alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if verbose_logging:
            logger.info(
                "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if verbose_logging:
            logger.info("Length not equal after stripping spaces: '%s' vs '%s'",
                        orig_ns_text, tok_ns_text)
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if verbose_logging:
            logger.info("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if verbose_logging:
            logger.info("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text

RawResult = collections.namedtuple("RawResult",
                ["unique_id", "start_logits", "end_logits", 'retrieval_logits', 'retriever_prob', 'modality_logits'], 
                defaults=(None,)*4+ (1.0,))

class QuacExample(object):
    """
    A single training/test example for the QuAC dataset.
    For examples without an answer, the start and end position are -1.
    """
    def __init__(self,
                 example_id,
                 qas_id,
                 question_text,
                 doc_tokens,
                 context_text=None,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None,
                 is_impossible=None, 
                 followup=None, 
                 yesno=None, 
                 retrieval_label=None, 
                 history=None):
        self.example_id = example_id
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.context_text = context_text
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible
        self.followup = followup
        self.yesno = yesno
        self.retrieval_label = retrieval_label
        self.history = history

def gen_reader_features_v2(qids, question_texts, answer_texts, answer_starts, passage_ids,
                        passages, all_retrieval_labels, reader_tokenizer, max_seq_length, 
                        is_training = False, itemid_modalities = None, item_id_to_idx = None, images_titles = None, 
                        image_size=(512, 512)):

    batch_features = []
    all_examples, all_features = {}, {}
    for (qas_id, question_text, answer_text, answer_start, pids_per_query,
                            paragraph_texts, retrieval_labels) in zip(qids, question_texts, answer_texts, answer_starts, 
                                                   passage_ids, passages, all_retrieval_labels):

        # logger.info("$"*50)

        answer_text = str(answer_text)
        query_context_features = []
        per_query_features = []
        
        # logger.info(f"QID {qas_id} Answer |{answer_text}|")

        for p_idx, (pid, paragraph_text, retrieval_label) in enumerate(zip(pids_per_query, paragraph_texts, retrieval_labels)):
            
            if itemid_modalities[item_id_to_idx[pid]] == 'image':
                image_path = paragraph_text
                paragraph_text = images_titles[pid]
            
            answer_start = paragraph_text.find(str(answer_text))

            # logger.info(f'{pid} | {paragraph_text[:50]}')

            example_id = f'{qas_id}*{pid}'

            # check if answer is not just a substring of some word
            doc_tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True
            for c in paragraph_text:
                if is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)
                    else:
                        doc_tokens[-1] += c
                    prev_is_whitespace = False
                char_to_word_offset.append(len(doc_tokens) - 1)


            is_impossible = False

            if is_training:
                if answer_text in ['CANNOTANSWER', 'NOTRECOVERED'] or retrieval_label == 0 or answer_start==-1:
                    is_impossible = True

                if not is_impossible:
                    orig_answer_text = str(answer_text)
                    answer_offset = answer_start
                    answer_length = len(str(orig_answer_text))
                    start_position = char_to_word_offset[answer_offset]
                    end_position = char_to_word_offset[answer_offset + answer_length - 1]


                    # Only add answers where the text can be exactly recovered from the
                    # document. If this CAN'T happen it's likely due to weird Unicode
                    # stuff so we will just skip the example.
                    #
                    # Note that this means for training mode, every example is NOT
                    # guaranteed to be preserved.

                    actual_text = " ".join(doc_tokens[start_position:(end_position + 1)])
                    # logger.info(f"Space tokens |{actual_text}| Answer |{answer_text}|")
                    cleaned_answer_text = " ".join(whitespace_tokenize(orig_answer_text))

                    if actual_text.find(cleaned_answer_text) == -1 or actual_text != answer_text:
                        logger.warning("Could not find answer: '%s' vs. '%s'", actual_text, cleaned_answer_text)
                        answer_start = -1
                        
                else:
                    answer_start = -1
            

            inputs = reader_tokenizer(
                question_text,
                paragraph_text,
                max_length=max_seq_length,
                truncation="only_second",
                stride=128,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="max_length",
            )

            offset_mapping = inputs.pop("offset_mapping")
            sample_map = inputs.pop("overflow_to_sample_mapping")
            start_positions = []
            end_positions = []

            # logger.info(f"input_ids {inputs['input_ids']}")
            # logger.info(f'offset {offset_mapping}')
            # logger.info(f"sample map {sample_map}")
            # logger.info(f"sequence ids {inputs.sequence_ids(0)}")

            # logger.info(f"Q: {question_text}")
            # logger.info(f"C: {paragraph_text}")
           

            for i, offset in enumerate(offset_mapping):

                sample_idx = sample_map[i]
                answer = answer_text

                # answer not present 
                if answer_start != -1:
                    start_char = answer_start
                    end_char = start_char + len(answer_text)
                    sequence_ids = inputs.sequence_ids(i)

                    # Find the start and end of the context
                    idx = 0
                    while sequence_ids[idx] != 1:
                        idx += 1
                    context_start = idx
                    while sequence_ids[idx] == 1:
                        idx += 1
                    context_end = idx - 1

                    # If the answer is not fully inside the context, label is (0, 0)
                    if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
                        start_positions.append(0)
                        end_positions.append(0)
                    else:
                        # Otherwise it's the start and end token positions
                        idx = context_start
                        while idx <= context_end and offset[idx][0] <= start_char:
                            idx += 1
                        start_positions.append(idx - 1)

                        idx = context_end
                        while idx >= context_start and offset[idx][1] >= end_char:
                            idx -= 1
                        end_positions.append(idx + 1)
                else:
                    start_positions.append(0)
                    end_positions.append(0)

                
                # logger.info(f'start {start_positions[-1]} end {end_positions[-1]}')

                start_ = offset[start_positions[-1]][0]
                end_ = offset[end_positions[-1]][1]
                
                span_text = paragraph_text[start_:end_]


                if is_training and span_text!="" and span_text.strip() != answer_text.strip():
                    print(f"Span mismatch: G |{answer_text}| | P |{span_text}|")
                    assert False

            inputs["start_positions"] = start_positions
            inputs["end_positions"] = end_positions

            num_chunks = len(start_positions)

            # logger.info(f"Q {qas_id} Paragraph {pid} {p_idx} {num_chunks}")

            example = QuacExample(
                example_id=example_id,
                qas_id=qas_id,
                question_text=question_text,
                context_text=paragraph_text,
                doc_tokens=offset_mapping,
                orig_answer_text=answer_text,
                start_position=None,
                end_position=None,
                is_impossible=None,
                retrieval_label=retrieval_label
            )

            # logger.info("Paragraph inputs tokenizer")
            # for k, v in inputs.items():
            #     logger.info(f'{k} - {np.array(v).shape}')
            

            # when evaluating, we save all examples and features
            # so that we can recover answer texts
            if not is_training:
                all_examples[example_id] = example
                all_features[example_id] = inputs


            if is_training:
                if retrieval_label:
                    per_query_feature = {
                                    'input_ids': np.asarray(inputs['input_ids']),
                                    'segment_ids': np.asarray(inputs['token_type_ids']),
                                    'input_mask': np.asarray(inputs['attention_mask']),
                                    'start_positions': np.array(inputs['start_positions']),
                                    'end_positions': np.array(inputs['end_positions']),
                                    'retrieval_label': np.array([retrieval_label] * num_chunks)}

                else:
                    per_query_feature = {
                                    'input_ids': np.asarray(inputs['input_ids']),
                                    'segment_ids': np.asarray(inputs['token_type_ids']),
                                    'input_mask': np.asarray(inputs['attention_mask']),
                                    'start_positions': np.array([0] * num_chunks),
                                    'end_positions': np.array([0] * num_chunks),
                                    'retrieval_label': np.array([retrieval_label] * num_chunks)}
            else:
                per_query_feature = {
                                'input_ids': np.asarray(inputs['input_ids']),
                                'segment_ids': np.asarray(inputs['token_type_ids']),
                                'input_mask': np.asarray(inputs['attention_mask']),
                                'example_id': [example_id] * num_chunks}

            if itemid_modalities[item_id_to_idx[pid]] == 'image':
                per_query_feature['image_input'] = np.concatenate([image_transform(image_path, image_size)]*num_chunks, axis=0)
            else:
                per_query_feature['image_input'] = np.zeros([num_chunks, 3, *image_size])


            if itemid_modalities[item_id_to_idx[pid]] == 'text':
                item_modality_type = 0
            elif itemid_modalities[item_id_to_idx[pid]] == 'table':
                item_modality_type = 1
            else:
                item_modality_type = 2

            per_query_feature['item_modality_type'] = np.array([item_modality_type])

            # logger.info("Per Query Features")
            # for k, v in per_query_feature.items():
            #     logger.info(f'{k} --> {v.shape}')

            # logger.info(f'{pid} --> {per_query_feature["example_id"]}')

            per_query_features.append(per_query_feature)

        collated = {}

        # for x in per_query_features:
        #     logger.info("Features of entire paragraph")
        #     for k, v in x.items():
        #         try:
        #             logger.info(f'\t {k} {v.shape}')
        #         except:
        #             raise Exception(f'{k}')

        # for x in per_query_features:
        #     logger.info(f'Ex {len(x["example_id"])}')

        keys = per_query_features[0].keys()
        for key in keys:
            if key != 'example_id':
                # collated[key] = np.vstack([dic[key] for dic in per_query_features])
                feats = tuple(dic[key] for dic in per_query_features)
                collated[key] = np.concatenate(feats, axis=0)

        if 'example_id' in keys:
            collated['example_id'] = []
            for dic in per_query_features:
                 collated['example_id'] += dic['example_id']

        batch_features.append(collated)
    

    # logger.info("Final features")
    # for x in batch_features:
    #     for k, v in x.items():
    #         logger.info(f'{k} -- {np.array(v).shape}')
    #     logger.info("#"*50)

    batch = {}
    keys = batch_features[0].keys()
    for key in keys:
        if key != 'example_id':
            batch[key] = np.concatenate([dic[key] for dic in batch_features], axis=0)
            # if key != 'segment_ids':
            batch[key] = torch.from_numpy(batch[key])

    if 'example_id' in keys:
        batch['example_id'] = []
        for item in batch_features:
            batch['example_id'].extend(item['example_id'])

    # logger.info("Complete features")
    # for k, v in batch.items():
    #     logger.info(f'{k} -> {v.shape}')


    if is_training:
        return batch
    else:
        return batch, all_examples, all_features 


def write_predictions_v2(reader_tokenizer, all_examples, all_features, all_results, n_best_size,
                      max_answer_length, do_lower_case, output_prediction_file,
                      output_nbest_file, output_null_log_odds_file, verbose_logging,
                      version_2_with_negative, null_score_diff_threshold):
    """Write final predictions to the json file and log-odds of null if needed."""
    logger.info("Writing predictions to: %s" % (output_prediction_file))
    logger.info("Writing nbest to: %s" % (output_nbest_file))

    # example_id_to_features = collections.defaultdict(list)
    # for feature in all_features:
    #     example_id_to_features[feature.example_id].append(feature)

    unique_id_to_result = collections.defaultdict(list)

    for result in all_results:
        unique_id_to_result[result.unique_id].append(result)

    # for k, v in unique_id_to_result.items():
    #     logger.info(f"Result {k} --> {len(v)}")
    
    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit", 'retrieval_logit', 'retriever_prob'])

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    for example_id, example in all_examples.items():

        feature = all_features[example_id]

        # for k, v in feature.items():
        #     logger.info(f'{k} : {np.array(v).shape}')

        # logger.info(f'Write preds {feature}')
        # logger.info(f'Example {example}')


        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive
        min_null_feature_index = 0  # the paragraph slice with min null score
        null_start_logit = 0  # the start logit at the slice with min null score
        null_end_logit = 0  # the end logit at the slice with min null score

        for res_idx, result in enumerate(unique_id_to_result[example_id]):

            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            # if we could have irrelevant answers, get the min score of irrelevant
            if version_2_with_negative:
                feature_null_score = result.start_logits[0] + result.end_logits[0]
                if feature_null_score < score_null:
                    score_null = feature_null_score
                    min_null_feature_index = 0
                    null_start_logit = result.start_logits[0]
                    null_end_logit = result.end_logits[0]
                    null_retrieval_logit = result.retrieval_logits[-1]
                    null_retriever_prob = result.retriever_prob
                    
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    # logger.info(f"Features {feature}")
                    if start_index >= len(example.doc_tokens):
                        continue
                    if end_index >= len(example.doc_tokens):
                        continue
                    # if start_index not in feature.token_to_orig_map:
                    #     continue
                    # if end_index not in feature.token_to_orig_map:
                    #     continue
                    # if not feature.token_is_max_context.get(start_index, False):
                    #     continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=res_idx,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index],
                            retrieval_logit=result.retrieval_logits[-1],
                            retriever_prob=None))
                            
            if version_2_with_negative:
                prelim_predictions.append(
                    _PrelimPrediction(
                        feature_index=0, # min_null_feature_index
                        start_index=0,
                        end_index=0,
                        start_logit=null_start_logit,
                        end_logit=null_end_logit, 
                        retrieval_logit=null_retrieval_logit, 
                        retriever_prob=null_retriever_prob))

        prelim_predictions = sorted(
            prelim_predictions,
            key = lambda x: (x.start_logit + x.end_logit),
            reverse=True
        )

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_logit", "end_logit", 'retrieval_logit', 'retriever_prob'])

        seen_predictions = {}
        nbest = []

        # logger.info(f"{example_id} {'-'*50}")
        # for pred in prelim_predictions:
        #     logger.info(f'Prelim {pred.feature_index} | {pred.start_index} {pred.end_index}')
        # logger.info(f'{"-"*50}')

        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break

            if pred.start_index > 0:  # this is a non-null prediction
                # logger.info(f"Token idx : {pred.start_index} - {pred.end_index}")
                orig_doc_start = example.doc_tokens[pred.feature_index][pred.start_index][0]
                orig_doc_end = example.doc_tokens[pred.feature_index][pred.end_index][1]

                # logger.info(f'Found {orig_doc_start} - {orig_doc_end}')
                predicted_ans = example.context_text[orig_doc_start:orig_doc_end]

                # logger.info(f'Gold {example.orig_answer_text} | Pred {predicted_ans}')

                final_text = get_final_text(predicted_ans, example.orig_answer_text, do_lower_case, verbose_logging)
                if final_text in seen_predictions:
                    continue

                seen_predictions[final_text] = True
            else:
                final_text = "CANNOTANSWER"
                seen_predictions[final_text] = True

            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit, 
                    retrieval_logit=pred.retrieval_logit, 
                    retriever_prob=pred.retriever_prob))

        # if we didn't include the empty option in the n-best, include it
        if version_2_with_negative:
            if "CANNOTANSWER" not in seen_predictions:
                nbest.append(
                    _NbestPrediction(
                        text="CANNOTANSWER",
                        start_logit=null_start_logit,
                        end_logit=null_end_logit, 
                        retrieval_logit=null_retrieval_logit,
                        retriever_prob=null_retriever_prob))
                
            # In very rare edge cases we could only have single null prediction.
            # So we just create a nonce prediction in this case to avoid failure.
            if len(nbest) == 1:
                nbest.insert(0, _NbestPrediction(text="empty", 
                                                    start_logit=0.0, 
                                                    end_logit=0.0, 
                                                    retrieval_logit=0.0, 
                                                    retriever_prob=0.0))

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(_NbestPrediction(text="empty", 
                                            start_logit=0.0, 
                                            end_logit=0.0, 
                                            retrieval_logit=0.0, 
                                            retriever_prob=0.0))

        assert len(nbest) >= 1

        total_scores = []
        best_non_null_entry = None


        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry:
                if entry.text != 'CANNOTANSWER':
                    best_non_null_entry = entry

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            output['retrieval_logit'] = entry.retrieval_logit
            output['retriever_prob'] = entry.retriever_prob
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        if best_non_null_entry is not None:

            if not version_2_with_negative:
                example_prediction = {'text': best_non_null_entry.text, 
                                    'start_logit': best_non_null_entry.start_logit,
                                    'end_logit': best_non_null_entry.end_logit,
                                    'retrieval_logit': best_non_null_entry.retrieval_logit, 
                                    'retriever_prob': best_non_null_entry.retriever_prob,
                                    'example_id': example.example_id}
            else:
                # predict "" iff the null score - the score of best non-null > threshold
                score_diff = (score_null - 
                            best_non_null_entry.start_logit - 
                            best_non_null_entry.end_logit)
                scores_diff_json[example.example_id] = score_diff
                if score_diff > null_score_diff_threshold:
                    example_prediction = {'text': 'CANNOTANSWER',
                                        'start_logit': null_start_logit,
                                        'end_logit': null_end_logit,
                                        'retrieval_logit': null_retrieval_logit, 
                                        'retriever_prob': null_retriever_prob,
                                        'example_id': example.example_id}
                else:
                    example_prediction = {'text': best_non_null_entry.text, 
                                        'start_logit': best_non_null_entry.start_logit,
                                        'end_logit': best_non_null_entry.end_logit,
                                        'retrieval_logit': best_non_null_entry.retrieval_logit, 
                                        'retriever_prob': best_non_null_entry.retriever_prob,
                                        'example_id': example.example_id}
        else:
            example_prediction = {'text': "", 
                                    'start_logit': 0.0,
                                    'end_logit': 0.0,
                                    'retrieval_logit': 0.0, 
                                    'retriever_prob': 0.0,
                                    'example_id': example.example_id}
                    
        if example.qas_id in all_predictions:            
            all_predictions[example.qas_id].append(example_prediction)
        else:
            all_predictions[example.qas_id] = [example_prediction]


    with open(output_prediction_file, "w") as writer:
        writer.write(json.dumps(all_predictions, indent=4) + "\n")

    with open(output_nbest_file, "w") as writer:
        writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

    if version_2_with_negative:
        with open(output_null_log_odds_file, "w") as writer:
            writer.write(json.dumps(scores_diff_json, indent=4) + "\n")

    return all_predictions


def write_final_predictions(all_predictions, final_prediction_file, 
                            use_rerank_prob=True, use_retriever_prob=False):
    """convert instance level predictions to quac predictions"""
    logger.info("Writing final predictions to: %s" % (final_prediction_file))
    
    turn_level_preds = {}
    for qid, preds in all_predictions.items():
        qa_scores = []
        rerank_scores = []
        for pred in preds:
            qa_scores.append(pred['start_logit'] + pred['end_logit'])
            rerank_scores.append(pred['retrieval_logit'])
        qa_probs = np.asarray(_compute_softmax(qa_scores))
        rerank_probs = np.asarray(_compute_softmax(rerank_scores))
        
        total_scores = qa_probs
        if use_rerank_prob:
            total_scores = total_scores * rerank_probs
        if use_retriever_prob:
            total_scores = total_scores * pred['retriever_prob']
      
        best_idx = np.argmax(total_scores)
        turn_level_preds[qid] = preds[best_idx]
        
    dialog_level_preds = {}

    for qid, pred in turn_level_preds.items():
        dialog_id = qid.split('#')[0]
        if dialog_id in dialog_level_preds:
            dialog_level_preds[dialog_id]['best_span_str'].append(pred['text'])
            dialog_level_preds[dialog_id]['qid'].append(qid)
            dialog_level_preds[dialog_id]['yesno'].append('x')
            dialog_level_preds[dialog_id]['followup'].append('y')
        else:
            dialog_level_preds[dialog_id] = {}
            dialog_level_preds[dialog_id]['best_span_str'] = [pred['text']]
            dialog_level_preds[dialog_id]['qid'] = [qid]
            dialog_level_preds[dialog_id]['yesno'] = ['x']
            dialog_level_preds[dialog_id]['followup'] = ['y']
            
    with open(final_prediction_file, 'w') as fout:
        for pred in dialog_level_preds.values():
            fout.write(json.dumps(pred) + '\n')
            
    return dialog_level_preds.values()

def get_retrieval_metrics(evaluator, all_predictions, eval_retriever_probs=False,retriever_run_dict=None): 
    return_dict = {}
    if retriever_run_dict == None:
        retriever_run = {}
        for qid, preds in all_predictions.items():
            retriever_run[qid] = {}
            for pred in preds:
                pid = pred['example_id'].split('*')[1]
                retriever_run[qid][pid] = pred['retriever_prob']

        retriever_metrics = evaluator.evaluate(retriever_run_dict)

        retriever_ndcg_list = [v['ndcg'] for v in retriever_metrics.values()]
        retriever_recall_list = [v['set_recall'] for v in retriever_metrics.values()]    
        return_dict.update({'retriever_ndcg': np.average(retriever_ndcg_list), 
                            'retriever_recall': np.average(retriever_recall_list)})
        print("=======================")
        print('ndcg(@top_k_for_reader)', np.average(retriever_ndcg_list))
        print('set_recall(@top_k_for_reader)', np.average(retriever_recall_list))
        print("=======================")

    else:
        retriever_metrics = evaluator.evaluate(retriever_run_dict)

        retriever_ndcg_list = [v['ndcg'] for v in retriever_metrics.values()]
        retriever_recall_list = [v['set_recall'] for v in retriever_metrics.values()]    
        return_dict.update({'retriever_ndcg': np.average(retriever_ndcg_list), 
                            'retriever_recall': np.average(retriever_recall_list)})
        print("=======================")
        print('ndcg(@top_k_for_retriever)', np.average(retriever_ndcg_list))
        print('set_recall(@top_k_for_retriever)', np.average(retriever_recall_list))
        print("=======================")

    return return_dict