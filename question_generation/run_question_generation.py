#!/usr/bin/env python3
# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Conditional text generation with the auto-regressive models of the library (GPT/GPT-2/Transformer-XL/XLNet)
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import sys
sys.path.insert(0,'../')
import argparse
import logging
import json
import pickle
import random
from tqdm import trange, tqdm
from itertools import repeat, dropwhile, takewhile
from functools import partial
from nltk.translate.meteor_score import single_meteor_score, meteor_score
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from nltk.tokenize import TreebankWordTokenizer

from abc import ABCMeta, abstractmethod
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
import numpy as np
from tensorboardX import SummaryWriter

from transformers import GPT2Config, GPT2Tokenizer, GPT2QGModel
from transformers import get_linear_schedule_with_warmup, AdamW

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

MODEL_CLASSES = {
    'gpt2': (GPT2QGModel, GPT2Tokenizer),
}
nltk_tokenizer = TreebankWordTokenizer()

def _is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False

def tokenize_by_space(context):
    doc_tokens = []
    char_to_word_offset = []
    prev_is_whitespace = True

    for c in context:
        if _is_whitespace(c):
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                doc_tokens.append(c)
            else:
                doc_tokens[-1] += c
            prev_is_whitespace = False
        char_to_word_offset.append(len(doc_tokens) - 1)
    return doc_tokens, char_to_word_offset

def truncate_context(context, ans_pos, max_context_len, a_start, a_end):
    condition = context[:]
    if ans_pos is not None:
        ans_pos = ans_pos[:]
    
    while len(condition)>max_context_len:
        if a_start > len(condition)-a_end-1:
            condition = condition[1:]
            ans_pos = ans_pos[1:] if ans_pos else None
            a_start -= 1
            a_end -= 1
        else:
            condition = condition[:-1]
            ans_pos = ans_pos[:-1] if ans_pos else None
    
    assert ans_pos is None or len(ans_pos) == len(condition), (len(ans_pos), len(context))
    return condition, ans_pos

def get_answer_position(context_id, a_start, a_end, answer_position_encoding):
    if answer_position_encoding is None:
        return None
    ans_pos = []
    if answer_position_encoding == 'distance':
        for i in range(len(context_id)):
            if i < a_start or i > a_end:
                dis = min(abs(i-a_start), abs(i-a_end))+1
            else:
                dis= 1
            ans_pos.append(dis)
    elif answer_position_encoding == 'zero_one':
        ans_pos = [0 if i<a_start or i>a_end else 1
                   for i in range(len(context_id))]
    else:
        raise ValueError('Unknown answer_position_encoding \'%s\''%answer_position_encoding) 

    return ans_pos

def load_squad_data(file_path, tokenizer, max_sequence_length, mode, answer_position_encoding=None):
    examples = []
    with open(file_path) as f:
        data = json.load(f)

    for elem in tqdm(data['data'], desc='process %s'%file_path, dynamic_ncols=True):
        for para in elem['paragraphs']:
            context = para['context']
            # tokenize context by space 
            context_tok, char_to_word_offset = tokenize_by_space(context)
           
            # convert words to subwords
            word_to_tok_index = []
            all_doc_tokens = []
            for (i, token) in enumerate(context_tok):
                word_to_tok_index.append(len(all_doc_tokens))
                sub_tokens = tokenizer.tokenize(token)
                for sub_token in sub_tokens:
                    all_doc_tokens.append(sub_token)

            for qa in para['qas']:
                # question
                question = qa['question'].strip()
                question_tokens = tokenizer.tokenize(question)

                # answer
                assert mode != 'train' or len(qa['answers']) == 1, \
                        'For training data, assuming there is only one answer, but see %d answers.'%len(qa['answers'])
                
                if mode == 'train':
                    answers = qa['answers']
                else:
                    answers = [json.loads(answer) for answer in set([json.dumps(answer)for answer in qa['answers']])]

                for answer in answers:
                    answer_text = answer['text']
                    answer_start = answer['answer_start']
                    answer_end = answer_start+len(answer_text)-1
                    answer_start_position = word_to_tok_index[char_to_word_offset[answer_start]]
                    answer_end_position = word_to_tok_index[char_to_word_offset[answer_end]]

                    answer_tokens = all_doc_tokens[answer_start_position:answer_end_position+1] 

                    ans_pos = get_answer_position(all_doc_tokens,
                                                  answer_start_position,
                                                  answer_end_position,
                                                  answer_position_encoding)
                    assert sum(ans_pos) != 0

                    max_context_len = max_sequence_length - len(question_tokens) -len(answer_tokens) - 3
                    condition, ans_pos = truncate_context(all_doc_tokens, 
                                                          ans_pos, 
                                                          max_context_len, 
                                                          answer_start_position, 
                                                          answer_end_position)

                    condition_ids = tokenizer.convert_tokens_to_ids(condition)
                    question_ids = tokenizer.convert_tokens_to_ids(question_tokens)
                    answer_ids = tokenizer.convert_tokens_to_ids(answer_tokens)
                    
                    # create example
                    example = {'condition':condition_ids,
                               'question':question_ids,
                               'answer':answer_ids,
                               'answer_position':ans_pos,
                               'id':qa['id']} 
                    examples.append(example)
    return examples

def load_qg_data(file_path, tokenizer, max_sequence_length):
    examples = []
    eval_examples = []
    with open(file_path) as f:
        data = json.load(f)

    context_cache = {}
    for elem in tqdm(data, desc='process %s'%file_path,  dynamic_ncols=True):
        context = elem['context']

        if context in context_cache:
            (context_text, context_char_to_token, context_tok_id) = context_cache[context]
        else:
            context_tok = tokenizer.tokenize(context)
            context_tok_id = tokenizer.convert_tokens_to_ids(context_tok)
            context_tok_text = [tokenizer.convert_tokens_to_string([tok]) for tok in context_tok]

            context_text = ''.join(context_tok_text)
            context_char_to_token = get_char_to_token(context_tok_text)
            context_cache[context] = (context_text, context_char_to_token, context_tok_id)

        # answer
        a = elem['answers'][0]
        if a['answer_start'] == -1:
            continue
        a_tok = tokenizer.tokenize(a['text'])
        a_ids = tokenizer.convert_tokens_to_ids(a_tok)
        a_text = tokenizer.convert_tokens_to_string(a_tok)
        
        # find answer to token position
        a_text = a_text.strip()
        for i in range(len(a_text)+1):
            a_match = find_near_matches(a_text, context_text, max_l_dist=i)
            if len(a_match) != 0:
                break
        assert len(a_match) != 0, (a_text, context_text)
        if len(a_match) > 1:
            a_start = a['answer_start']
            dis = [abs(m.start-a_start) for m in a_match]
            idx = np.argmin(dis)
            a_match = a_match[idx]
        else:
            a_match = a_match[0]
        a_start_tok = context_char_to_token[a_match.start]
        a_end_tok = context_char_to_token[a_match.end-1]

        answer_position = get_answer_position(context_tok_id, a_start_tok, a_end_tok)
        if sum(answer_position)==0:
            print ('not found answer position')
            continue

        for question, weight in zip(elem['pred_question'], elem['weight']):
            # question
            q_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(question))
            
            # truncate too long context
            max_context_len = max_sequence_length - len(q_ids) -len(a_ids) - 3
            condition, ans_pos = truncate_context(context_tok_id, answer_position, max_context_len, a_start_tok, a_end_tok)
            
            # create example
            example = {'condition':condition,
                       'question':q_ids,
                       'answer':a_ids,
                       'answer_position':ans_pos,
                       'id':elem['id']} 
            
            examples.append(example)
            eval_examples.append({'context':elem['context'],
                                  'question':question,
                                  'answers':elem['answers']})
    return examples, eval_examples

def load_ag_data(file_path, tokenizer, max_sequence_length):
    examples = []
    eval_examples = []
    with open(file_path) as f:
        data = json.load(f)

    context_cache = {}
    for elem in tqdm(data, desc='process %s'%file_path, dynamic_ncols=True):
        context = elem['context']

        if context in context_cache:
            (context_text, context_char_to_token, context_tok_id) = context_cache[context]
        else:
            context_tok = tokenizer.tokenize(context)
            context_tok_id = tokenizer.convert_tokens_to_ids(context_tok)
            context_tok_text = [tokenizer.convert_tokens_to_string([tok]) for tok in context_tok]

            context_text = ''.join(context_tok_text)
            context_char_to_token = get_char_to_token(context_tok_text)
            context_cache[context] = (context_text, context_char_to_token, context_tok_id)

        # answer
        for a in elem['answers']:
            if a['answer_start'] == -1:
                continue
            a_tok = tokenizer.tokenize(a['text'])
            a_ids = tokenizer.convert_tokens_to_ids(a_tok)
            a_text = tokenizer.convert_tokens_to_string(a_tok)
            
            # find answer to token position
            a_text = a_text.strip()
            #for i in range(len(a_text)+1):
            for i in range(10):
                a_match = find_near_matches(a_text, context_text, max_l_dist=i)
                if len(a_match) != 0:
                    break
            if len(a_match) == 0:
                continue

            if len(a_match) > 1:
                a_start = a['answer_start']
                dis = [abs(m.start-a_start) for m in a_match]
                idx = np.argmin(dis)
                a_match = a_match[idx]
            else:
                a_match = a_match[0]
            a_start_tok = context_char_to_token[a_match.start]
            a_end_tok = context_char_to_token[a_match.end-1]

            answer_position = get_answer_position(context_tok_id, a_start_tok, a_end_tok)
            if sum(answer_position)==0:
                print ('not found answer position')
                continue
            
            # truncate too long context
            max_context_len = max_sequence_length - len(a_ids) - 3
            condition, ans_pos = truncate_context(context_tok_id, answer_position, max_context_len, a_start_tok, a_end_tok)

            # create example
            example = {'condition':condition,
                       'answer':a_ids,
                       'answer_position':ans_pos,
                       'id':elem['id']} 
            
            examples.append(example)
            eval_examples.append({'context':elem['context'],
                                  'answers':[a]})
    return examples, eval_examples

class TextDataset(Dataset):
    def __init__(self, tokenizer, file_path, data_mode,answer_position_encoding,
                 gen=False,
                 max_sequence_length=400,
                 is_qg=False,
                 is_ag=False):

        self.gen=gen
        self.vocab_size = len(tokenizer)
        self.data_mode = data_mode 
        self.max_sequence_length = max_sequence_length

        # special token
        self.bos_id = tokenizer.bos_token_id
        self.eos_id = tokenizer.eos_token_id
        self.sep_id = tokenizer.sep_token_id
        self.mask_id = tokenizer.mask_token_id

        assert os.path.isfile(file_path)
        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(directory, f'cached_{filename}_{answer_position_encoding}')

        if os.path.exists(cached_features_file):
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, 'rb') as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info("Creating features from %s"%file_path )

            if is_qg:
                self.examples = load_qg_data(file_path, tokenizer, max_sequence_length)
            elif is_ag:
                self.examples = load_ag_data(file_path, tokenizer, max_sequence_length)
            else:
                self.examples = load_squad_data(file_path, tokenizer, max_sequence_length,
                                                               mode = 'train' if not evaluate else 'eval',
                                                               answer_position_encoding=answer_position_encoding)

            logger.info("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, 'wb') as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

            '''
            eval_file = os.path.join(directory, f'eval_{filename}')
            with open(eval_file, 'w') as handle:
                json.dump(eval_examples, handle, indent=1, ensure_ascii=False)
            '''


    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        '''
             example = {'condition'         : condition_ids,
                        'question'          : question_ids,
                        'answer'            : answer_ids
                        'answer_position'   : answer_position_encoding} 
           
        '''
        example = self.examples[item]
        condition = example['condition']
        question  = example.get('question', [])
        answer    = example['answer']
        ans_pos   = example['answer_position']

        if self.gen:
            if self.data_mode in ['ca', 'ac']:
                if self.data_mode == 'ac':
                    inputs = answer + [self.sep_id] + condition
                elif self.data_mode == 'ca':
                    inputs = condition + [self.sep_id] + answer
                answer_position = [0]*len(inputs)

            elif self.data_mode == 'cp':
                inputs = condition
                answer_position = ans_pos

            elif self.data_mode in ['cap', 'acp']:
                if self.data_mode == 'acp':
                    inputs = answer + [self.sep_id] + condition
                    answer_position = [1]*(len(answer)+1) + ans_pos
                elif self.data_mode == 'cap':
                    inputs = condition + [self.sep_id] + answer
                    answer_position = ans_pos + [1]*(len(answer)+1)
            else:
                raise Exception
            assert len(inputs) == len(answer_position)

            return torch.tensor(inputs),\
                   torch.tensor(answer_position), \
                   example['id']


        else:
            if self.data_mode in ['ca', 'ac']:
                if self.data_mode == 'ca':
                    inputs = condition + [self.sep_id] + answer +[self.bos_id] + question + [self.eos_id]
                elif self.data_mode == 'ac':
                    inputs = answer + [self.sep_id] + condition + [self.bos_id] + question + [self.eos_id]
                labels = [-1]*(len(answer)+len(condition)+1) + question + [self.eos_id, -1]
                answer_position = [0]*len(inputs)

            elif self.data_mode == 'cp':
                inputs = condition + [self.bos_id] + question + [self.eos_id]
                labels = [-1]*len(condition) + question + [self.eos_id, -1]
                answer_position = ans_pos + [0]*(len(question)+2)
            
            elif self.data_mode in ['cap', 'acp']:
                if self.data_mode == 'cap':
                    inputs = condition + [self.sep_id] + answer + [self.bos_id] + question + [self.eos_id]
                    answer_position =  ans_pos + [1]*(len(answer)+1) + [0]*(len(question)+2)
                elif self.data_mode == 'acp':
                    inputs = answer + [self.sep_id] + condition + [self.bos_id] + question + [self.eos_id]
                    answer_position = [1]*(len(answer)+1) + ans_pos + [0]*(len(question)+2)
                labels = [-1]*(len(answer)+len(condition)+1) + question + [self.eos_id, -1]
            else:
                raise Exception

            assert len(inputs) == len(labels) == len(answer_position)
            
            return torch.tensor(inputs),\
                   torch.tensor(labels), \
                   torch.tensor(answer_position), \
                   example['id']

    def _seq2seq_mask(self, length):
        mask = self.all_one * np.array([[1]*length+[0]*(self.max_sequence_length-length)])
        mask = np.minimum(self.tril + mask, 1)
        return mask

def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    #assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k, dim=-1)[0][:, -1, None]
        logits[indices_to_remove] = filter_value
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


def sample_sequence(model, length, context, num_samples=1, temperature=1, top_k=0, top_p=0.0, device='cpu'):
    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0).repeat(num_samples, 1)
    generated = context
    with torch.no_grad():
        for _ in trange(length):

            inputs = {'input_ids': generated}

            outputs = model(**inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states)
            next_token_logits = outputs[0][0, -1, :] / temperature
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
    return generated

def my_collate(batch, pad_id, is_gen):
    # padding
    new_batch = []
    if not is_gen:
        for i, elem in enumerate(zip(*batch)):
            if i ==0:
                new_batch.append(nn.utils.rnn.pad_sequence(elem, batch_first=True, padding_value=pad_id))
            elif i ==1:
                new_batch.append(nn.utils.rnn.pad_sequence(elem, batch_first=True, padding_value=-1))
            elif i == 2 :
                new_batch.append(nn.utils.rnn.pad_sequence(elem, batch_first=True, padding_value=0))
            elif i == 3:
                new_batch.append(elem)
            else:
                raise Exception
    else:
        for i, elem in enumerate(zip(*batch)):
            if i ==0:
                new_batch.append(nn.utils.rnn.pad_sequence(elem, batch_first=True, padding_value=pad_id))
            elif i == 1 :
                new_batch.append(nn.utils.rnn.pad_sequence(elem, batch_first=True, padding_value=0))
            elif i == 2:
                new_batch.append(elem)
            else:
                raise Exception

    return new_batch

def train(args, train_dataset, dev_dataset, model, tokenizer):
    """ Train the model """
    summary_writer = SummaryWriter(os.path.join(args.save_dir, 'summary'))

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
                                  collate_fn=partial(my_collate, pad_id=tokenizer.pad_token_id, is_gen=False))
    if args.epoch != 0:
        t_total = int(args.epoch*len(train_dataset)//(args.train_batch_size*args.gradient_accumulation_steps))
    else:
        t_total = args.train_steps


    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.fp16_opt_level)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num training steps = %d", t_total)
    logger.info("  batch size = %d", args.train_batch_size)
    logger.info("  gradient accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  true batch size = %d", args.train_batch_size*args.gradient_accumulation_steps)

    def repeater(data_loader):
        for loader in repeat(data_loader):
            for data in loader:
                yield data

    pad_id = tokenizer.pad_token_id
    bos_id = tokenizer.bos_token_id
    unk_id = tokenizer.unk_token_id

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)

    loss_fct = nn.CrossEntropyLoss(ignore_index=-1, reduction='sum')

    # parameters controling when to earlystop
    max_fail_time = 5
    fail =  0
    max_decay = 2
    decay = 0

    min_loss = np.inf
    train_iterator = trange(t_total, desc="Train")
    for i in train_iterator:
        loss_sum = 0
        batches = [next(repeater(train_dataloader)) for _ in range(args.gradient_accumulation_steps)]
        total_num = sum([torch.sum(batch[1]!=-1) for batch in batches])
        for batch in batches:
            batch = [t.to(args.device) for t in batch[:-1]] + batch[-1:]
            
            input_ids = batch[0]
            labels = batch[1]
            answer_position = batch[2]
            attention_mask = input_ids != pad_id

            inputs = {'input_ids':input_ids,
                      'answer_position':answer_position,
                      'attention_mask':attention_mask}
            outputs = model(**inputs)

            #compute loss
            logits = outputs[0]

            loss = loss_fct(logits.view(-1, logits.size(-1)),
                            labels.view(-1))
            loss = loss/total_num
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            loss_sum += loss.item()

        if args.fp16:
            torch.nn.utils.clip_grad_norm_(
                amp.master_params(optimizer), args.max_grad_norm)
        else:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.max_grad_norm)

        optimizer.step()
        scheduler.step()  # Update learning rate schedule
        model.zero_grad()
        train_iterator.set_postfix(loss=float(loss_sum))

        # write summary
        if (i+1) % 100 == 0:
            summary_writer.add_scalar('lr', scheduler.get_lr()[0], i)
            summary_writer.add_scalar('loss', loss_sum, i)

        # eval in training
        if i == t_total-1 or (i+1)%args.eval_step == 0:
            # Log metrics
            loss = None
            if args.evaluate_during_training:
                loss = evaluate(args, dev_dataset, model, tokenizer)
                summary_writer.add_scalar('eval_loss', loss, i)
                logger.info('eval loss : %f'%loss)

            if loss is None or loss < min_loss:
                min_loss = loss
                output_dir = os.path.join(args.save_dir, 'checkpoint')
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                model.save_pretrained(output_dir)
                logger.info("Saving model checkpoint to %s", output_dir)
                fail = 0
            else :
                fail += 1
            
            if fail == max_fail_time and decay < max_decay:
                scheduler.decay_lr(0.5)
                fail=0
                decay += 1
            
            if fail >= max_fail_time:
                break
            model.train()
    
    if args.eval_step == -1: 
        output_dir = os.path.join(args.save_dir, 'checkpoint')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        model.save_pretrained(output_dir)
        logger.info("Saving model checkpoint to %s", output_dir)

def evaluate(args, dataset, model, tokenizer):
    dataloader = DataLoader(dataset, batch_size=args.eval_batch_size,
                            collate_fn=partial(my_collate, pad_id=tokenizer.pad_token_id, is_gen=False))
    model.eval()
    
    loss_fct = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
    results = {}
    for batch in tqdm(dataloader, desc='eval',  dynamic_ncols=True):
        batch = [t.to(args.device) for t in batch[:-1]] + batch[-1:]
        
        input_ids = batch[0]
        labels = batch[1]
        answer_position = batch[2]
        ids = batch[3]
        attention_mask = input_ids != tokenizer.pad_token_id

        with torch.no_grad():
            inputs = {'input_ids':input_ids,
                      'answer_position':answer_position,
                      'attention_mask':attention_mask}
            outputs = model(**inputs)

            #compute loss
            logits = outputs[0]
            log_probs = loss_fct(logits.view(-1, logits.size(-1)),
                                 labels.view(-1))
            log_probs = torch.sum(log_probs.view(-1, logits.size(1)), dim=-1).cpu().numpy()
            num_word = torch.sum(labels!=-1, dim=-1).cpu().numpy()

            for log_prob, n, example_id in zip(log_probs, num_word, ids):
                if example_id not in results or results[example_id][2]>log_prob/n:
                    results[example_id] =(log_prob, n, log_prob/n)
    
    token_num = 0
    loss = 0
    for _, v in results.items():
        token_num += v[1]
        loss += v[0]

    return loss/token_num

class Searcher(metaclass=ABCMeta):
    def __init__(self, model, tokenizer, vocab_mask=None):
        self._model = model
        self._tokenizer = tokenizer
        self._bos_id = tokenizer.bos_token_id
        self._eos_id = tokenizer.eos_token_id
        self._pad_id = tokenizer.pad_token_id
        self._vocab_mask = vocab_mask
    
    def decode(self, input_ids, answer_position, attention_mask, max_len, device='cpu', **wargs):
        '''
            Shape of each input :
                inputs_ids : [batch, len]
                answer_position : [batch, len]
                init_attention_mask : [batch, 1, len]
        '''
        bos_id = self._bos_id
        pad_id = self._pad_id
        with torch.no_grad():
            # first we get model cached of input sequences
            inputs = {'input_ids':input_ids,
                      'answer_position':answer_position,
                      'attention_mask':attention_mask}

            outputs = self._model(**inputs)
            cached_state = outputs[1]

            # next run a for loop to generate sequences
            # prepare buffer
            batch_size = input_ids.size(0)
            self._initialize_buffer(batch_size, device, **wargs)

            # initial input
            mask = input_ids != pad_id
            input_ids = torch.ones((batch_size, 1),dtype=input_ids.dtype, device=device)*bos_id # [batch, 1]
            answer_position = torch.zeros((batch_size, 1), dtype=answer_position.dtype, device=device)
            attention_mask = F.pad(mask.unsqueeze(1).type(attention_mask.dtype), (0,1), value=1) # [batch, 1, len+1]
            position_ids = torch.sum(mask, dim=-1, keepdim=True)+1 # [batch, 1]
            
            next_inputs = {'input_ids':input_ids,
                           'answer_position':answer_position,
                           'attention_mask':attention_mask,
                           'past':cached_state,
                           'position_ids':position_ids}

            for step in range(max_len):
                next_inputs = self._step(next_inputs, step, **wargs)
                if self.is_finished:
                    break
        return self._output_tokens, self._output_log_probs

    @abstractmethod
    def _step(self):
        pass

    @abstractmethod
    def _initialize_buffer(self):
        pass

    @property
    def is_finished(self):
        return all(self._is_finished)


class GreedySearcher(Searcher):
    def _initialize_buffer(self, batch_size, device, **wargs):
        self._is_finished = torch.zeros(batch_size, dtype=torch.uint8, device=device)

    def _step(self, inputs, step, **wargs):
        # get outputs
        outputs = self._model(**inputs)

        # new cached
        cached_state = outputs[1]

        # mask some words in vocab (such as <PAD>, <BOS>, <MASK>,...)
        next_token_logits = outputs[0][:, -1, :] # [batch, voceb_size]
        if self._vocab_mask is not None:
            vocab_mask = self._vocab_mask
            # can not output <EOS> in first time step
            if step == 0:
                vocab_mask = vocab_mask.clone()
                vocab_mask[0, self._eos_id] = torch.min(vocab_mask)
            next_token_logits += vocab_mask

        # get next token and token log prob via argmax function
        log_probs, next_token = self._get_next_token(next_token_logits, **wargs)

        # replace next token with pad_id if has already finished
        next_token = torch.where(self._is_finished, torch.ones_like(next_token)*self._pad_id, next_token)
        log_probs = torch.where(self._is_finished, torch.zeros_like(log_probs), log_probs)
        
        # update is_finished
        self._is_finished = self._is_finished | (next_token==self._eos_id).type(self._is_finished.dtype)

        # save result to buffer
        next_token = next_token.unsqueeze(-1)
        if step == 0:
            self._output_tokens = next_token # [batch, 1]
            self._output_log_probs = log_probs # [batch]
        else:
            self._output_tokens = torch.cat((self._output_tokens,next_token), dim=-1)
            self._output_log_probs += log_probs

        # update inputs
        next_inputs = {'input_ids':next_token,
                       'answer_position':inputs['answer_position'],
                       'attention_mask':F.pad(inputs['attention_mask'], (0,1), value=1),
                       'past':cached_state,
                       'position_ids':inputs['position_ids']+1}

        return next_inputs

    def _get_next_token(self, next_token_logits, **wargs):
        log_probs = F.log_softmax(next_token_logits, dim=-1)
        log_probs, next_token = torch.max(log_probs, dim=-1)
        return log_probs, next_token 

class SampleSearcher(GreedySearcher):
    def _get_next_token(self, next_token_logits, temperature, top_k, top_p, **wargs):
        next_token_logits = next_token_logits / temperature
        filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
        next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)

        next_token = next_token.squeeze(-1)
        log_probs = F.log_softmax(next_token_logits, dim=-1)
        log_probs = torch.stack([log_prob[token] for log_prob, token in zip(log_probs, next_token)], dim=0)
        
        return log_probs, next_token 

class BeamSearcher(Searcher):
    def _initialize_buffer(self, batch_size, device, beam_width, **wargs):
        self._is_finished = torch.zeros(batch_size*beam_width, dtype=torch.uint8, device=device)
        self._beam_scores = torch.zeros((batch_size, beam_width), dtype=torch.float, device=device) # [batch, beam_width]
        self._beam_log_probs = torch.zeros((batch_size, beam_width), dtype=torch.float, device=device) # [batch, beam_width]

    def _step(self, inputs, step, beam_width, temperature, top_k, rank_penalty, **wargs):
        # get outputs
        outputs = self._model(**inputs)
        
        # new cached
        del inputs['past']
        cached_state = outputs[1]

        # mask some words in vocab (such as <PAD>, <BOS>, <MASK>,...)
        next_token_logits = outputs[0][:, -1, :] # [batch, voceb_size]
        if self._vocab_mask is not None:
            vocab_mask = self._vocab_mask
            # can not output <EOS> in first time step
            if step == 0:
                vocab_mask = vocab_mask.clone()
                vocab_mask[0, self._eos_id] = torch.min(vocab_mask)
            next_token_logits += vocab_mask

        # keep top k for each beam if top_k is not 0
        if top_k != 0 and step != 0:
            next_token_logits = top_k_top_p_filtering(next_token_logits, top_k)
        
        logits = F.log_softmax(next_token_logits/temperature, dim=-1)
        vocab_size = logits.size(-1)

        if step == 0:
            # logits : [batch, vocab_size]
            scores = logits
        else:
            # logit : [batch*beam_width, vocab_size]
            finished_logit = -torch.ones(vocab_size, dtype=logits.dtype, device=logits.device)*1e4
            finished_logit[self._pad_id] = 0
            finished_logit = finished_logit.unsqueeze(0).repeat(logits.size(0),1)  # [-1, vocab_size]

            finished_masks = self._is_finished.unsqueeze(1).type(logits.dtype) # [-1, 1]
            
            logits = logits*(1-finished_masks) + finished_masks*finished_logit
            '''
            for idx, f in enumerate(self._is_finished):
                if f == 1:
                    logits[idx] = finished_logit
            '''
            
            logits = logits.view(-1, beam_width, vocab_size) # [batch_size, beam_width, vocab_size]

            # rank penalty
            if rank_penalty == 0:
                penalty = 0
            else:
                penalty = torch.arange(0,vocab_size, device=logits.device, dtype=logits.dtype)*rank_penalty
                penalty = penalty.unsqueeze(0).unsqueeze(0).repeat(logits.size(0), beam_width,1)
                rank = torch.argsort(logits, dim=-1, descending=True) # [batch, beam, vocab]
                penalty = penalty.scatter(-1, rank, penalty)

            scores = (logits - penalty + self._beam_scores.unsqueeze(-1)).view(-1, beam_width*vocab_size) # [batch_size, beam_width*vocab_size]
        top_k_score = scores.topk(beam_width, dim=-1) # [batch_size, beam_width]
        beam_index = top_k_score.indices//vocab_size # [batch_size, beam_size]
        
        # update score and log prob
        beam_logits = torch.stack([blp[i] for blp, i in zip(self._beam_log_probs, beam_index)]) # [batch, beam_width]
        new_logits = torch.stack([s[i]for s, i in zip(logits.view(logits.size(0), -1), top_k_score.indices)], dim=0)
        self._beam_log_probs = beam_logits + new_logits # [batch, beam]
        self._beam_scores = top_k_score.values

        '''
        beam_scores = torch.stack([blp[i] for blp, i in zip(self._beam_scores, beam_index)]) # [batch, beam_width]
        new_logits = torch.stack([s[i]for s, i in zip(logits.view(logits.size(0), -1), top_k_score.indices)], dim=0)
        self._beam_scores = beam_scores + new_logits # [batch, beam]
        '''
        # reorder cache
        batch_start = torch.arange(0, beam_index.size(0), device=beam_index.device, dtype=beam_index.dtype).view(-1, 1) # [batch_size, 1]
        if step == 0:
            cached_index = (batch_start+beam_index).view(-1) # [batch_size*beam_width]
        else:
            cached_index = (batch_start*beam_width+beam_index).view(-1) # [batch_size*beam_width]
        
        cached_state = tuple([state[:,cached_index] for state in cached_state])

        # update output token 
        next_token = (top_k_score.indices%vocab_size).unsqueeze(-1) # [batch, beam_width, 1]
        if step == 0:
            self._output_tokens = next_token # [batch, beam_size, 1]
        else:
            self._output_tokens = torch.stack([self._output_tokens[i, ind]
                                        for i, ind in enumerate(beam_index)], dim=0) # [batch, beam_width, n]
            self._output_tokens = torch.cat((self._output_tokens, next_token), dim=-1) # [batch, beam_width, n+1]

        # update is_finished
        self._is_finished = self._is_finished[cached_index]|(next_token.view(-1)==self._eos_id).type(self._is_finished.dtype)

        # update inputs
        if step == 0:
            answer_position = inputs['answer_position']
            inputs['answer_position'] = torch.zeros((inputs['input_ids'].size(0)*beam_width, 1), 
                                                    dtype=answer_position.dtype, 
                                                    device=answer_position.device) # [batch, 1]

        next_inputs = {'input_ids':next_token.view(-1, 1),
                       'answer_position':inputs['answer_position'],
                       'attention_mask':F.pad(inputs['attention_mask'][cached_index], (0,1), value=1),
                       'past':cached_state,
                       'position_ids':inputs['position_ids'][cached_index]+1}

        return next_inputs

    @property
    def _output_log_probs(self):
        return self._beam_log_probs

class DiverseBeamSearcher(Searcher):
    def _initialize_buffer(self, batch_size, device, beam_width, **wargs):
        self._is_finished = torch.zeros(batch_size*beam_width, dtype=torch.uint8, device=device)
        self._beam_scores = torch.zeros((batch_size, beam_width), dtype=torch.float, device=device) # [batch, beam_width]
        self._beam_log_probs = torch.zeros((batch_size, beam_width), dtype=torch.float, device=device) # [batch, beam_width]

    def _step(self, inputs, step, beam_width, group_num, diversity_strength, temperature, top_k, **wargs):
        assert beam_width%group_num ==0
        group_beam_width = beam_width // group_num

        # get outputs
        outputs = self._model(**inputs)

        # new cached
        cached_state = outputs[1]

        # mask some words in vocab (such as <PAD>, <BOS>, <MASK>,...)
        next_token_logits = outputs[0][:, -1, :] # [batch, voceb_size]
        if self._vocab_mask is not None:
            vocab_mask = self._vocab_mask
            # can not output <EOS> in first time step
            if step == 0:
                vocab_mask = vocab_mask.clone()
                vocab_mask[0,self._eos_id] = torch.min(vocab_mask)
            next_token_logits += vocab_mask

        logits = F.log_softmax(next_token_logits/temperature, dim=-1) # [batch*beam_width, vocab_size]
        vocab_size = logits.size(-1)

        if step == 0:
            scores = [logits.unsqueeze(1).clone()]*group_num # list of [batch, 1, vocab_size]
            logits = [logits.unsqueeze(1)]*group_num # list of [batch, 1, vocab_size]
        else:
            finished_logit = -torch.ones(vocab_size, dtype=logits.dtype, device=logits.device)*np.inf
            finished_logit[self._pad_id] = 0
            for idx, f in enumerate(self._is_finished):
                if f == 1:
                    logits[idx] = finished_logit

            logits = logits.view(-1, beam_width, vocab_size) # [batch, beam_width, vocab_size] 
            scores = (logits + self._beam_scores.unsqueeze(-1)) # [batch, beam_width, vocab_size]

            # split logit and scores by groups
            logits = torch.chunk(logits, group_num, dim=1)
            scores = torch.chunk(scores, group_num, dim=1)

        beam_index = []
        new_scores = []
        new_log_probs = []
        next_token = []
        
        diverse_buf = self._beam_scores.new_zeros((self._beam_scores.size(0), vocab_size))
        for g in range(group_num):
            group_logits = logits[g] # [batch, group_beam_width, vocab_size]
            group_scores = scores[g] # [batch, group_beam_width, vocab_size]
            if g > 0:
                # add diversity penalty
                group_scores -= diverse_buf.unsqueeze(1)*diversity_strength

            batch_size = group_scores.size(0)
            
            # choose top group_beam_width words
            top_k = torch.topk(group_scores.view(batch_size,-1), group_beam_width, dim=-1) # [batch, group_beam_width]
            
            if step == 0:
                # for step=0, beam index should be all zero
                beam_index.append(top_k.indices//vocab_size) # list of [batch, group_beam_width]
            else:
                beam_index.append(top_k.indices//vocab_size+g*group_beam_width) # list of [batch, group_beam_width]
            tokens = (top_k.indices%vocab_size) # [batch, group_beam_width]
            next_token.append(tokens.unsqueeze(-1)) # list of [batch, group_beam_width, 1]
            
            # update diversity buf
            inc = tokens.new_ones(tokens.size(), dtype=diverse_buf.dtype)
            inc[tokens==self._pad_id]=0
            diverse_buf.scatter_add_(1, tokens, inc)

            # get beam_score
            new_scores.append(top_k.values) # list of [batch, group_beam_width]

            # get beam_log_prob
            group_beam_log_probs = torch.stack([blp[i] for blp, i in zip(self._beam_log_probs, beam_index[-1])]) # [batch, group_beam_width]
            
            new_logits = torch.stack([s[i]for s, i in zip(group_logits.view(batch_size,-1), top_k.indices)], dim=0)
            new_log_probs.append(group_beam_log_probs + new_logits) # list of [batch, group_beam_width]

        beam_index = torch.cat(beam_index, dim=1)  # [batch, beam_width]
        next_token = torch.cat(next_token, dim=1) # [batch, beam_width, 1]
        new_scores = torch.cat(new_scores, dim=1) # [batch, beam_width]
        new_log_probs = torch.cat(new_log_probs, dim=1)

        # update score
        self._beam_scores = new_scores # [batch, beam_width]
        self._beam_log_probs = new_log_probs
        
        # reorder cache
        batch_start = torch.arange(0, batch_size, device=beam_index.device).view(-1, 1) # [batch, 1]
        if step == 0:
            cached_index = (batch_start+beam_index).view(-1) # [batch_size*beam_width]
        else:
            cached_index = (batch_start*beam_width+beam_index).view(-1) # [batch_size*beam_width]
        cached_state = tuple([state[:,cached_index] for state in cached_state])

        # update output token 
        if step == 0:
            self._output_tokens = next_token # [batch, beam_size, 1]
        else:
            self._output_tokens = torch.stack([self._output_tokens[i, ind]
                                        for i, ind in enumerate(beam_index)], dim=0) # [batch, beam_width, n]
            self._output_tokens = torch.cat((self._output_tokens, next_token), dim=-1) # [batch, beam_width, n+1]

        # update is_finished
        self._is_finished = self._is_finished[cached_index]|(next_token.view(-1)==self._eos_id).type(self._is_finished.dtype)

        # update inputs
        if step == 0:
            answer_position = inputs['answer_position']
            inputs['answer_position'] = torch.zeros((inputs['input_ids'].size(0)*beam_width, 1), 
                                                    dtype=answer_position.dtype, 
                                                    device=answer_position.device) # [batch, 1]

        next_inputs = {'input_ids':next_token.view(-1, 1),
                       'answer_position':inputs['answer_position'],
                       'attention_mask':F.pad(inputs['attention_mask'][cached_index], (0,1), value=1),
                       'past':cached_state,
                       'position_ids':inputs['position_ids'][cached_index]+1}

        return next_inputs

    @property
    def _output_log_probs(self):
        return self._beam_log_probs

def search(model, input_ids, answer_position, attention_mask, max_len, tokenizer, mode, cached_state,
           vocab_mask=None, num_samples=1, temperature=1, top_k=0, top_p=0.0, device='cpu'):
    # special token
    bos_id = tokenizer.bos_token_id
    end_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id

    batch_size = input_ids.size(0)
    
    # buffer
    is_finished = torch.zeros(batch_size, dtype=torch.uint8, device=device)
    position_ids = torch.sum(input_ids!=pad_id, dim=-1, keepdim=True) # [batch, 1]
    basic_mask = (input_ids != pad_id).unsqueeze(1).type(attention_mask.dtype) # [batch, 1, length]

    with torch.no_grad():
        # start token
        input_ids = torch.ones((batch_size, 1),dtype=input_ids.dtype, device=device)*bos_id
        answer_position = torch.zeros((batch_size, 1), dtype=answer_position.dtype, device=device)
        next_mask = torch.ones((batch_size, 1, 1), dtype=attention_mask.dtype, device=device)

        length = 0
        output_tokens = []
        output_log_probs = []
        while not is_finished.all() and length<max_len:
            # update attention mask
            basic_mask = torch.cat((basic_mask, next_mask), dim=-1)
            attention_mask = basic_mask

            # get outputs
            position_ids += 1
            inputs = {'input_ids':input_ids,
                      'answer_position':answer_position,
                      'attention_mask':attention_mask,
                      'past':cached_state,
                      'position_ids':position_ids}
            outputs = model(**inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states)

            # update cached
            cached_state = outputs[1]

            # compute next token
            next_token_logits = outputs[0][:, -1, :] # [batch, voceb_size]
            if vocab_mask is not None:
                next_token_logits += vocab_mask
            if mode == 'greedy':
                log_probs = F.log_softmax(next_token_logits, dim=-1)
                log_probs, next_token = torch.max(log_probs, dim=-1)
            elif mode == 'sample':
                raise NotImplementedError()
                next_token_logits = next_token_logits / temperature
                filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
                next_token = next_token.squeeze(1)

            # update inputs
            next_token = torch.where(is_finished, torch.ones_like(next_token)*pad_id, next_token)
            log_probs = torch.where(is_finished, torch.zeros_like(log_probs), log_probs)
            input_ids = next_token.unsqueeze(1)
            output_tokens.append(input_ids)
            output_log_probs.append(log_probs.unsqueeze(1))

            # update is_finished
            is_finished = is_finished | (next_token==end_id).type(is_finished.dtype)
            length += 1

        output_tokens = torch.cat(output_tokens, dim=1)
        output_log_probs = torch.sum(torch.cat(output_log_probs, dim=1), dim=1)

    return output_tokens, output_log_probs

def beam_search(model, input_ids, answer_position, attention_mask, max_len, tokenizer, cached_state,
                beam_width, vocab_mask=None, temperature=1, top_k=None, device='cpu'):
    # special token
    bos_id = tokenizer.bos_token_id
    end_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    
    batch_size = input_ids.size(0)
    
    # buffer
    is_finished = torch.zeros((batch_size*beam_width), dtype=torch.uint8, device=device) # [batch*beam_width]
    position_ids = torch.sum(input_ids!=pad_id, dim=-1, keepdim=True) # [batch, 1]
    basic_mask = (input_ids != pad_id).unsqueeze(1).type(attention_mask.dtype) # [batch, 1, length]
    with torch.no_grad():
        # start token
        input_ids = torch.ones((batch_size, 1),dtype=input_ids.dtype, device=device)*bos_id # [batch, 1]
        
        # first attentioin mask
        next_mask = torch.ones((batch_size, 1, 1), dtype=attention_mask.dtype, device=device)
        attention_mask = torch.cat((basic_mask, next_mask), dim=-1)

        length = 0 
        beam_scores = torch.zeros((batch_size, beam_width), dtype=torch.float, device=device) # [batch, beam_width]
        while not is_finished.all() and length<max_len:
            if length == 0:
                answer_position = torch.zeros((batch_size, 1), dtype=answer_position.dtype, device=device) # [batch, 1]
            else:
                answer_position = torch.zeros((batch_size*beam_width, 1), dtype=answer_position.dtype, device=device) # [batch, 1]

            # get outputs
            position_ids += 1
            inputs = {'input_ids':input_ids,
                      'answer_position':answer_position,
                      'attention_mask':attention_mask,
                      'past':cached_state,
                      'position_ids':position_ids}

            outputs = model(**inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states)

            # update cached
            cached_state = outputs[1]
            
            # compute next token
            next_token_logits = outputs[0][:, -1, :] # [batch_size*beam_width, vocab_size]
            if vocab_mask is not None:
                next_token_logits += vocab_mask
            
            # keep only top k for each beam
            if top_k != 0 and length != 0:
                next_token_logits = top_k_top_p_filtering(next_token_logits, top_k)
            
            logits = F.log_softmax(next_token_logits/temperature, dim=-1)
            vocab_size = logits.size(-1)
            if length == 0:
                # logits : [batch, vocab_size]
                scores = logits
                top_k_score = torch.topk(scores, beam_width, dim=-1) # [batch_size, beam_width]
            else:
                # logit : [batch*beam_width, vocab_size]
                finished_logit = -torch.ones(vocab_size, dtype=logits.dtype, device=device)*np.inf
                finished_logit[pad_id] = 0
                for idx, f in enumerate(is_finished):
                    if f == 1:
                        logits[idx] = finished_logit
                logits = logits.view(batch_size, beam_width, -1) # [batch_size, beam_width, vocab_size]
                scores = (logits+beam_scores.unsqueeze(-1)).view(batch_size, -1) # [batch_size, beam_width*vocab_size]
                top_k_score = torch.topk(scores, beam_width, dim=-1) # [batch_size, beam_width]

            beam_index = top_k_score.indices//vocab_size # [batch_size, beam_size]
            
            # update score
            beam_scores = top_k_score.values
            assert beam_scores.size(0)==batch_size and beam_scores.size(1)==beam_width
            
            # reorder cache
            batch_start = torch.arange(0,batch_size, device=device).view(batch_size, 1) # [batch_size, 1]
            if length == 0:
                cached_index = (batch_start+beam_index).view(-1) # [batch_size*beam_width]
            else:
                cached_index = (batch_start*beam_width+beam_index).view(-1) # [batch_size*beam_width]
            cached_state = (state[:,cached_index] for state in cached_state)
            '''
            new_state = ()
            for state in cached_state:
                # state : [:, batch*beam_width, :, :, :]
                new_state = new_state + (state[:,cached_index], )
            cached_state = new_state
            '''
            # reorder attention mask
            new_mask = torch.ones((batch_size*beam_width, 1, 1), dtype=attention_mask.dtype, device=device)
            pre_attention_mask = attention_mask[cached_index]
            attention_mask = torch.cat((pre_attention_mask, new_mask), dim=-1)
           
            # reorder position_ids
            position_ids = position_ids[cached_index]
            
            # update output token 
            is_finished = is_finished[cached_index]
            new_index = (top_k_score.indices%vocab_size) # [batch, beam_width]
            if length == 0:
                output_tokens = new_index.unsqueeze(-1) # [batch, beam_size, 1]
            else:
                new_index = torch.where(is_finished.view(batch_size, beam_width), torch.ones_like(new_index)*pad_id, new_index)
                old_index = torch.stack([output_tokens[i, ind]
                                            for i, ind in enumerate(beam_index)], dim=0) # [batch, beam_width, n]
                output_tokens = torch.cat((old_index, new_index.unsqueeze(-1)), dim=-1) # [batch, beam_width, n+1]

            # update is_finished
            is_finished = is_finished | (new_index.view(-1)==end_id).type(is_finished.dtype)

            # update inputs
            input_ids = new_index.view(batch_size*beam_width, 1) # [batch * beam_width, 1]

            # update length
            length += 1
    return output_tokens, beam_scores

def diverse_beam_decode(model, input_ids, answer_position, attention_mask, max_len, tokenizer, cached_state,
                beam_width, vocab_mask=None, temperature=1, top_k=None, device='cpu'):
    # special token
    bos_id = tokenizer.bos_token_id
    end_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    
    batch_size = input_ids.size(0)
    is_finished = torch.zeros((batch_size*beam_width), dtype=torch.uint8, device=device) # [batch*beam_width]
    position_ids = torch.sum(input_ids!=pad_id, dim=-1, keepdim=True) # [batch, 1]
    basic_mask = (input_ids != pad_id).unsqueeze(1).type(attention_mask.dtype) # [batch, 1, length]
    with torch.no_grad():
        # start token
        input_ids = torch.ones((batch_size, 1),dtype=input_ids.dtype, device=device)*bos_id # [batch, 1]
        
        # first attentioin mask
        next_mask = torch.ones((batch_size, 1, 1), dtype=attention_mask.dtype, device=device)
        attention_mask = torch.cat((basic_mask, next_mask), dim=-1)

        length = 0 
        beam_scores = torch.zeros((batch_size, beam_width), dtype=torch.float, device=device) # [batch, beam_width]
        while not is_finished.all() and length<max_len:
            if length == 0:
                answer_position = torch.zeros((batch_size, 1), dtype=answer_position.dtype, device=device) # [batch, 1]
            else:
                answer_position = torch.zeros((batch_size*beam_width, 1), dtype=answer_position.dtype, device=device) # [batch, 1]

            # get outputs
            position_ids += 1
            inputs = {'input_ids':input_ids,
                      'answer_position':answer_position,
                      'attention_mask':attention_mask,
                      'past':cached_state,
                      'position_ids':position_ids}

            outputs = model(**inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states)

            # update cached
            cached_state = outputs[1]
            
            # compute next token
            next_token_logits = outputs[0][:, -1, :] # [batch_size*beam_width, vocab_size]
            if vocab_mask is not None:
                next_token_logits += vocab_mask
            
            # keep only top k for each beam
            if top_k is not None and length != 0:
                next_token_logits = top_k_top_p_filtering(next_token_logits, top_k)
            
            logits = F.log_softmax(next_token_logits/temperature, dim=-1)
            vocab_size = logits.size(-1)
            if length == 0:
                # logits : [batch, vocab_size]
                scores = logits
                top_k_score = torch.topk(scores, beam_width, dim=-1) # [batch_size, beam_width]
            else:
                # logit : [batch*beam_width, vocab_size]
                finished_logit = -torch.ones(vocab_size, dtype=logits.dtype, device=device)*np.inf
                finished_logit[pad_id] = 0
                for idx, f in enumerate(is_finished):
                    if f == 1:
                        logits[idx] = finished_logit
                logits = logits.view(batch_size, beam_width, -1) # [batch_size, beam_width, vocab_size]
                scores = (logits+beam_scores.unsqueeze(-1)).view(batch_size, -1) # [batch_size, beam_width*vocab_size]
                top_k_score = torch.topk(scores, beam_width, dim=-1) # [batch_size, beam_width]

            beam_index = top_k_score.indices//vocab_size # [batch_size, beam_size]
            
            # update score
            beam_scores = top_k_score.values
            assert beam_scores.size(0)==batch_size and beam_scores.size(1)==beam_width
            
            # update cache
            batch_start = torch.arange(0,batch_size, device=device).view(batch_size, 1) # [batch_size, 1]
            if length == 0:
                cached_index = (batch_start+beam_index).view(-1) # [batch_size*beam_width]
            else:
                cached_index = (batch_start*beam_width+beam_index).view(-1) # [batch_size*beam_width]
            new_state = ()
            for state in cached_state:
                # state : [:, batch*beam_width, :, :, :]
                new_state = new_state + (state[:,cached_index], )
            cached_state = new_state
            
            # update attention mask
            new_mask = torch.ones((batch_size*beam_width, 1, 1), dtype=attention_mask.dtype, device=device)
            pre_attention_mask = attention_mask[cached_index]
            attention_mask = torch.cat((pre_attention_mask, new_mask), dim=-1)
           
            # update position_ids
            position_ids = position_ids[cached_index]
            
            # update output token 
            is_finished = is_finished[cached_index]
            new_index = (top_k_score.indices%vocab_size) # [batch, beam_width]
            if length == 0:
                output_tokens = new_index.unsqueeze(-1) # [batch, beam_size, 1]
            else:
                new_index = torch.where(is_finished.view(batch_size, beam_width), torch.ones_like(new_index)*pad_id, new_index)
                old_index = torch.stack([output_tokens[i, ind]
                                            for i, ind in enumerate(beam_index)], dim=0) # [batch, beam_width, n]
                output_tokens = torch.cat((old_index, new_index.unsqueeze(-1)), dim=-1) # [batch, beam_width, n+1]

            # update is_finished
            is_finished = is_finished | (new_index.view(-1)==end_id).type(is_finished.dtype)

            # update inputs
            input_ids = new_index.view(batch_size*beam_width, 1) # [batch * beam_width, 1]

            # update length
            length += 1
    return output_tokens, beam_scores

def generate(model, inputs, ans_pos, attention_mask, max_len, tokenizer, mode, cached_state,
             num_samples=1, temperature=1, top_k=None, top_p=None, beam_width=None, device='cpu'):
    # special token
    bos_id = tokenizer.bos_token_id
    end_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    mask_id = tokenizer.mask_token_id
    sep_id = tokenizer.sep_token_id
    cls_id = tokenizer.cls_token_id

    # mask impossible grnerated token
    mask_word_id = [bos_id, pad_id, mask_id, sep_id, cls_id]
    vocab_mask = torch.zeros([1,len(tokenizer)], device=device, dtype=torch.float)
    for idx in mask_word_id :
        vocab_mask[0,idx] = -1e4

    if mode == 'greedy':
        return greedy_search()
    elif mode == 'sample':
        return sample_search()
    elif mode == 'beam':
        return beam_search(model, inputs, ans_pos, attention_mask, max_len, tokenizer, cached_state,
                           vocab_mask=vocab_mask,
                           beam_width=beam_width, temperature=temperature, 
                           top_k=top_k, device=device)
    elif mode == 'diverse-beam':
        return diverse_beam_search()

    else:
        return decode(model, inputs, ans_pos, attention_mask, max_len, tokenizer, mode, cached_state,
                      vocab_mask=vocab_mask, num_samples=num_samples, temperature=temperature, top_k=top_k, top_p=top_p, 
                      device=device)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action",choices=['train', 'eval', 'gen'])

    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(MODEL_CLASSES.keys())
        )
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--max_len", type=int, default=100)
    parser.add_argument("--atten_drop_rate", type=float, default=0.1)
    parser.add_argument("--data_mode", choices=['ca','ac','cp','cap','acp'], default='acp')
    parser.add_argument("--answer_position_encoding", choices=['distance', 'zero_one'], default='zero_one')
    parser.add_argument("--is_qg", action="store_true")

    # training options
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--train_steps", type=int, default=0)
    parser.add_argument("--epoch", type=float, default=1)
    parser.add_argument("--warmup_steps", default=1000, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--eval_step", default=10000, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=4, type=int)
    parser.add_argument("--evaluate_during_training", action='store_true')

    #inference options
    parser.add_argument("--inference", default='greedy', choices=['greedy','sample', 'beam', 'diverse_beam'])
    parser.add_argument("--beam_width", default=10, type=int)
    parser.add_argument("--rank_penalty", default=0, type=float)

    parser.add_argument("--group_beam_width", default=1, type=int)
    parser.add_argument("--diversity_strength", default=0.8, type=float)

    parser.add_argument("--sample_num", default=5, type=int)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.0)
    parser.add_argument("--ag", action='store_true')

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")

    # file path
    parser.add_argument("--save_dir", default=None)
    parser.add_argument("--train_file_path", default='data/squad/train.json')
    parser.add_argument("--dev_file_path", default='data/squad/dev.json')
    parser.add_argument("--output_file", default=None)
    parser.add_argument("--eval_file", default='data/squad/eval_dev.json')

    args = parser.parse_args()

    if args.save_dir is not None:
        os.makedirs(args.save_dir, exist_ok=True)

    # check devices 
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    logger.warning("Process device: %s, n_gpu: %s, 16-bits training: %s",
                   args.device, args.n_gpu, args.fp16)

    # get tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    special_tokens_dict = {'pad_token': '<PAD>',
                           'bos_token': '<BOS>',
                           'eos_token': '<EOS>',
                           'cls_token': '<CLS>',
                           'sep_token': '<SEP>',
                           'mask_token': '<MASK>'}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    logger.info('We have added %d tokens'%num_added_toks)

    # load model from pretrained gpt2 or finetuned gpt2
    config = GPT2Config.from_pretrained(args.model_name_or_path)
    if args.action == 'train':
        config.attn_pdrop = args.atten_drop_rate
        config.data_mode = args.data_mode
        config.answer_position_encoding = args.answer_position_encoding

        if 'p' in config.data_mode:
            assert config.answer_position_encoding in ['distance', 'zero_one'], \
            'If using answer position embedding, make sure args.answer_position_encoding should be \'distance\' or \'zero_one\''

    logger.info('Data mode : %s'%(config.data_mode))
    logger.info('Answer Position Encoding : %s'%(config.answer_position_encoding))
    
    model = GPT2QGModel.from_pretrained(args.model_name_or_path, config=config)
    model.resize_token_embeddings(len(tokenizer))  # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e. the length of the tokenizer.
    model.to(args.device)

    if args.fp16:
        try:
            import apex
            apex.amp.register_half_function(torch, 'einsum')
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    # action
    if args.action == 'train':
        train_dataset = TextDataset(tokenizer, args.train_file_path, config.data_mode,
                                    is_qg=args.is_qg,
                                    gen=False,
                                    answer_position_encoding=config.answer_position_encoding)
        dev_dataset = TextDataset(tokenizer, args.dev_file_path, config.data_mode, 
                                  gen=False, 
                                  answer_position_encoding=config.answer_position_encoding)
        train(args, train_dataset, dev_dataset, model, tokenizer)
    
    elif args.action == 'eval':
        dataset = TextDataset(tokenizer, args.dev_file_path, config.data_mode,
                              gen=False,
                              answer_position_encoding=config.answer_position_encoding)
        result = evaluate(args, dataset, model, tokenizer)
        print ('%s,%f'%(args.model_name_or_path.split('/')[-2],result))

    elif args.action == 'gen':
        from collections import defaultdict
        assert args.output_file is not None
        bos_id = tokenizer.bos_token_id
        eos_id = tokenizer.eos_token_id
        pad_id = tokenizer.pad_token_id

        if args.ag:
            logger.info('Input file without questions!')
        dataset = TextDataset(tokenizer, args.dev_file_path, config.data_mode,
                              gen=True, is_ag=args.ag,
                              answer_position_encoding=config.answer_position_encoding)

        dataloader = DataLoader(dataset, batch_size=args.eval_batch_size,
                            collate_fn=partial(my_collate, pad_id=pad_id, is_gen=True))

        model.eval()
        
        pred_all_questions = defaultdict(list)
        pred_all_questions_token_num = defaultdict(list)
        pred_all_log_probs = defaultdict(list)
        
        # mask impossible grnerated token
        bos_id = tokenizer.bos_token_id
        pad_id = tokenizer.pad_token_id
        mask_id = tokenizer.mask_token_id
        sep_id = tokenizer.sep_token_id
        cls_id = tokenizer.cls_token_id

        mask_word_id = [bos_id, pad_id, mask_id, sep_id, cls_id]
        vocab_mask = torch.zeros([1,len(tokenizer)], device=args.device, dtype=torch.float)
        vocab_mask[0,mask_word_id] = -1e4
   
        #vocab_mask = None
        if args.inference == 'greedy':
            searcher = GreedySearcher(model, tokenizer, vocab_mask=vocab_mask)
        elif args.inference == 'sample':
            searcher = SampleSearcher(model, tokenizer, vocab_mask=vocab_mask)
        elif args.inference == 'beam':
            searcher = BeamSearcher(model, tokenizer, vocab_mask=vocab_mask)
        elif args.inference == 'diverse_beam':
            searcher = DiverseBeamSearcher(model, tokenizer, vocab_mask=vocab_mask)

        for batch in tqdm(dataloader, desc='generate', dynamic_ncols=True):
            batch = [t.to(args.device) for t in batch[:-1]] + batch[-1:]
            
            input_ids = batch[0]
            answer_position = batch[1]
            example_ids = batch[2]
            attention_mask = input_ids != pad_id


            if args.inference == 'greedy':
                pred_questions, log_probs = searcher.decode(input_ids, answer_position, attention_mask, args.max_len,
                                                             device=args.device)
                pred_questions = [[q] for q in pred_questions.cpu().numpy()]
                log_probs = [[p] for p in log_probs.cpu().numpy().tolist()]

            elif args.inference == 'sample':
                pred_questions = []
                log_probs = []
                for _ in range(args.sample_num):
                    pred_question, log_prob = searcher.decode(input_ids, answer_position, attention_mask, args.max_len,
                                                              device=args.device, 
                                                              top_k=args.top_k, top_p=args.top_p, temperature=args.temperature)

                    pred_question = pred_question.cpu().numpy().tolist()
                    log_prob = log_prob.cpu().numpy().tolist()
                    pred_questions.append(pred_question)
                    log_probs.append(log_prob)
                pred_questions = list(zip(*pred_questions))
                log_probs = list(zip(*log_probs))
            elif args.inference in ['beam','diverse_beam']:

                if args.inference == 'beam':
                    pred_questions, log_probs = searcher.decode(input_ids, answer_position, attention_mask, args.max_len,
                                                                device=args.device, 
                                                                beam_width=args.beam_width,
                                                                top_k=args.top_k,
                                                                temperature=args.temperature,
                                                                rank_penalty=args.rank_penalty)
                elif args.inference == 'diverse_beam':
                    group_num = args.beam_width//args.group_beam_width
                    pred_questions, log_probs = searcher.decode(input_ids, answer_position, attention_mask, args.max_len,
                                                                device=args.device, 
                                                                beam_width=args.beam_width,
                                                                group_num=group_num,
                                                                diversity_strength=args.diversity_strength,
                                                                top_k=args.top_k,
                                                                temperature=args.temperature)


                pred_questions = pred_questions.cpu().numpy().tolist()
                log_probs = log_probs.cpu().numpy().tolist()

            assert len(example_ids) == len(pred_questions)
            for example_id, pred, log_prob in zip(example_ids, pred_questions, log_probs):
                for q, p in zip(pred, log_prob):
                    q = list(takewhile(lambda x:x!=eos_id, q))
                    pred_all_questions_token_num[example_id].append(len(q)+1)
                    q = tokenizer.decode(q, clean_up_tokenization_spaces=True, skip_special_tokens=True).strip()
                    pred_all_questions[example_id].append(q)
                    pred_all_log_probs[example_id].append(p)

            assert len(pred_all_questions)  == len(pred_all_log_probs) == len(pred_all_questions_token_num)


        # merge same questions and sorted by generated probabilities
        results = {}
        for key in pred_all_questions:
            questions = pred_all_questions[key]
            token_nums = pred_all_questions_token_num[key]
            log_probs = pred_all_log_probs[key]

            table = {}
            for q, n, p in zip(questions, token_nums, log_probs):
                # keep large probability
                if q not in table or p > table[q][0]:
                    table[q] = (p, n)
            results[key] = [elem for elem in sorted(((q,)+v for q,v in table.items()), key=lambda x:x[1], reverse=True)]

        with open(args.dev_file_path) as f:
            data = json.load(f)


        generated_data = []
        data_num = 0
        count = 0
        for elem in data['data']:
            for para in elem['paragraphs']:
                context = para['context']
                for qa in para['qas']:
                    example_id = qa['id']
                    if example_id in results:
                        pred_questions, log_probs, token_nums = list(zip(*results[example_id]))
                        generated_data.append({'context':context,
                                               'question':qa['question'],
                                               'answers':qa['answers'],
                                               'pred_question':pred_questions,
                                               'pred_question_token_num':token_nums,
                                               'pred_question_log_prob':log_probs,
                                               'id':example_id})
                        count += 1
                    data_num += 1
        print ('%d/%d'%(count, data_num))

        '''
        for elem, pred, pred_token_num, log_prob in zip(eval_file, pred_all_questions, pred_all_questions_token_num,
                                        pred_all_log_probs):
            elem['pred_question']=pred
            elem['pred_question_token_num'] = pred_token_num
            elem['pred_question_log_prob']=log_prob
        '''
        output_dir = os.path.dirname(args.output_file)
        if output_dir != '':
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output_file,'w') as f:
            json.dump(generated_data, f, indent=1, ensure_ascii=False)

if __name__ == '__main__':
    main()
