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
import argparse
import logging
import json
import pickle
import random
import collections
from tqdm import trange, tqdm
from itertools import repeat, dropwhile, takewhile
from functools import partial
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
import numpy as np
from tensorboardX import SummaryWriter

from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from transformers import get_linear_schedule_with_warmup, AdamW

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

#nltk_tokenizer = TreebankWordTokenizer()

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

class Example:
    def __init__(self, context, question, answer, answer_start_position, answer_end_position,
                 example_id):
        self.context = context
        self.question = question
        self.answer = answer

        self.answer_start_position = answer_start_position
        self.answer_end_position = answer_end_position

        self.example_id = example_id

class Feature:
    def __init__(self, inputs, labels, example_id):
        self.inputs = inputs
        self.labels = labels
        self.example_id = example_id

def load_squad_example(file_path, tokenizer, mode):
    with open(file_path) as f:
        data = json.load(f)

    examples = []
    context_token_cache = {}
    for elem in tqdm(data['data'], desc='process'):
        for para in elem['paragraphs']:
            context = para['context']
            if context not in context_token_cache:
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
                context_token_cache[context]={'context_tok':context_tok,
                                              'doc_tok':all_doc_tokens,
                                              'char_to_word_offset':char_to_word_offset,
                                              'word_to_tok_index':word_to_tok_index}
            else:
                char_to_word_offset = context_token_cache[context]['char_to_word_offset']
                word_to_tok_index = context_token_cache[context]['word_to_tok_index']

            for qa in para['qas']:
                # question
                question = qa['question'].strip()
                example_id = qa['id']

                # answer
                assert mode != 'train' or len(qa['answers']) == 1, 'For training data, assuming there is only one answer.'
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

                    example = Example(context, 
                                      question, 
                                      answer, 
                                      answer_start_position, 
                                      answer_end_position,
                                      example_id)
                    examples.append(example)
    return examples, context_token_cache

def load_qg_example(file_path, tokenizer):
    with open(file_path) as f:
        data = json.load(f)

    examples = []
    context_token_cache = {}
    for elem in tqdm(data, desc='process example', dynamic_ncols=True):
        context = elem['context']
        if context not in context_token_cache:
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
            context_token_cache[context]={'context_tok':context_tok,
                                          'doc_tok':all_doc_tokens,
                                          'char_to_word_offset':char_to_word_offset,
                                          'word_to_tok_index':word_to_tok_index}
        else:
            char_to_word_offset = context_token_cache[context]['char_to_word_offset']
            word_to_tok_index = context_token_cache[context]['word_to_tok_index']

        answers = [json.loads(answer) for answer in set([json.dumps(answer)for answer in elem['answers']])]
        example_id = elem['id']
        for answer in answers:
            answer_text = answer['text']
            answer_start = answer['answer_start']
            answer_end = answer_start+len(answer_text)-1
            answer_start_position = word_to_tok_index[char_to_word_offset[answer_start]]
            answer_end_position = word_to_tok_index[char_to_word_offset[answer_end]]
             
            # question
            question_num = len(elem['pred_question'])
            for question_id, question in enumerate(elem['pred_question']):
                example = Example(context, 
                                   question, 
                                   answer, 
                                   answer_start_position, 
                                   answer_end_position,
                                   '%s_%d'%(example_id, question_id))
                examples.append(example)

    return examples, context_token_cache

def convert_to_features(examples, tokenizer, context_token_cache, max_seq_len, data_mode):
    features = []

    for example in tqdm(examples, desc='convert to feature'):
        # process question
        question = example.question
        question_tok = tokenizer.tokenize(question)

        # convert to feature
        question_tok_id = tokenizer.convert_tokens_to_ids(question_tok)
        assert len(question_tok_id) < 400
        if data_mode == 'q':
            inputs = [tokenizer.bos_token_id] + question_tok_id 
            labels = question_tok_id + [tokenizer.eos_token_id]

        elif data_mode == 'cq':
            # process context
            context = example.context
            context_tok = context_token_cache[context]['doc_tok'][:]
            max_context_len = max_seq_len - len(question_tok) - 1 if max_seq_len else None 

            # trunacte long context
            if max_context_len:
                answer_start_position = example.answer_start_position
                answer_end_position = example.answer_end_position
                while len(context_tok) > max_context_len:
                    if answer_start_position > abs(len(context_tok)-answer_end_position):
                        context_tok = context_tok[1:]
                        answer_start_position -= 1
                        answer_end_position -= 1
                    else:
                        context_tok = context_tok[:-1]
            context_tok_id = tokenizer.convert_tokens_to_ids(context_tok)

            inputs = context_tok_id + [tokenizer.bos_token_id] + question_tok_id
            labels = [-1]*len(context_tok_id) + question_tok_id + [tokenizer.eos_token_id]

        assert len(inputs) == len(labels)
        features.append(Feature(inputs=inputs,
                                labels=labels,
                                example_id = example.example_id))
    return features

class TextDataset(Dataset):
    def __init__(self, features):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, item):
        return torch.tensor(self.features[item].inputs),\
               torch.tensor(self.features[item].labels), \
               self.features[item].example_id

def load_and_cached_data(config, tokenizer, file_path, mode, max_seq_len=400, is_qg=False,
                         output_example=False,
                         save_cache=False):
    assert os.path.isfile(file_path), file_path
    directory, filename = os.path.split(file_path)

    # cache file path
    cached_features_file = os.path.join(directory, f'cached_qlm-{config.data_mode}_{filename}')


    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        with open(cached_features_file, 'rb') as handle:
            cache_data = pickle.load(handle)
        examples = cache_data['examples']
        features = cache_data['features']
    else:
        logger.info("Creating features from file %s", file_path)

        if is_qg:
            examples, context_token_cache = load_qg_example(file_path, tokenizer) 
        else:
            examples, context_token_cache = load_squad_example(file_path, tokenizer, mode) 

        logger.info('Example num : %d'%len(examples))
        features = convert_to_features(examples, tokenizer, context_token_cache, max_seq_len, config.data_mode)
        
        if save_cache:
            logger.info("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, 'wb') as handle:
                pickle.dump({'features':features,'examples':examples},
                            handle, protocol=pickle.HIGHEST_PROTOCOL)

    dataset = TextDataset(features)
    if output_example:
        return dataset, examples
    return dataset

def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def my_collate(batch, pad_id):
    # padding
    new_batch = []
    for i, elem in enumerate(zip(*batch)):
        if i == 0:
            new_batch.append(nn.utils.rnn.pad_sequence(elem, batch_first=True, padding_value=pad_id))
        elif i == 1:
            new_batch.append(nn.utils.rnn.pad_sequence(elem, batch_first=True, padding_value=-1))
        elif i == 2 :
            new_batch.append(elem)
        else:
            raise ValueError

    return new_batch

def train(args, train_dataset, dev_dataset, model, tokenizer):
    """ Train the model """
    summary_writer = SummaryWriter(os.path.join(args.save_dir, 'summary'))

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
                                  collate_fn=partial(my_collate, pad_id=tokenizer.pad_token_id))

    t_total = int(args.epoch*len(train_dataset)//(args.train_batch_size*args.gradient_accumulation_steps))

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
    logger.info("  true batch size = %d", args.gradient_accumulation_steps*args.train_batch_size)

    def repeater(data_loader):
        for loader in repeat(data_loader):
            for data in loader:
                yield data

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(t_total, desc="Train")
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    loss_fct = nn.CrossEntropyLoss(ignore_index=-1, reduction='sum')
    model.train()
    max_score = 0
    min_loss = np.inf

    pad_id = tokenizer.pad_token_id
    bos_id = tokenizer.bos_token_id
    unk_id = tokenizer.unk_token_id
    tolerance = args.early_stop_tolerance
    for i in train_iterator:
        loss_sum = 0
        batches = [next(repeater(train_dataloader)) for _ in range(args.gradient_accumulation_steps)]
        total_num = sum([torch.sum(batch[1]!=-1) for batch in batches])
        for batch in batches:
            ## from train dataset
            batch = [t.to(args.device) for t in batch[:-1]] + batch[-1:]

            inputs = batch[0]
            labels = batch[1]

            outputs = model(inputs)

            #compute loss
            logits = outputs[0]

            # Flatten the tokens
            loss = loss_fct(logits.view(-1, logits.size(-1)),
                            labels.view(-1))
            loss /= total_num 

            loss_sum += loss
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

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
        if (i+1)%args.eval_step == 0 or (i+1) == t_total:
            # Log metrics
            results = evaluate(args, dev_dataset, model, tokenizer)
            for key, value in results.items():
                summary_writer.add_scalar('eval_{}'.format(key), value, i)

            if results['loss'] < min_loss:
                min_loss = results['loss']
                output_dir = os.path.join(args.save_dir, 'checkpoint')
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                model.save_pretrained(output_dir)
                logger.info("Saving model checkpoint to %s", output_dir)
                tolerance = args.early_stop_tolerance
            else:
                tolerance -= 1
            
            model.train()
        if tolerance == 0:
            break

def evaluate(args, dataset, model, tokenizer):
    dataloader = DataLoader(dataset, batch_size=args.eval_batch_size,
                            collate_fn=partial(my_collate, pad_id=tokenizer.pad_token_id))
    model.eval()
    loss_fct = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
    
    results = {}
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='eval', ncols=50):
            batch = [t.to(args.device) for t in batch[:-1]] + batch[-1:]
            inputs = batch[0]
            labels = batch[1]
            example_id = batch[2]

            outputs = model(inputs)

            #compute loss
            logits = outputs[0] # [batch, len, vocab_size]

            # Flatten the tokens, note that output of loss_fct is positive
            log_prob = loss_fct(logits.view(-1, logits.size(-1)),labels.view(-1))
            log_prob = torch.sum(log_prob.view(-1, logits.size(1)), dim=-1).cpu().numpy() # [batch]
            num_word = torch.sum(labels != -1, dim=-1).cpu().numpy()

            assert len(log_prob.shape) == len(num_word.shape)

            # in dev set, one question has multiple answers, we keep the lowest ppl one.
            for log_p, n, e_id in zip(log_prob, num_word, example_id):
                if e_id not in results or results[e_id][2]>log_p/n:
                    results[e_id] = (log_p, n, log_p/n)
    log_p = 0
    num_word = 0

    for _, v in results.items():
        log_p += v[0]
        num_word += v[1]

    log_ppl = log_p/num_word
    ppl = np.exp(log_ppl)
    logger.info('%s,%f'%(args.dev_file_path.split('/')[-2], ppl))
    return {'loss':ppl}

def score(args, dataset, model, tokenizer):
    dataloader = DataLoader(dataset, batch_size=args.eval_batch_size,
                            collate_fn=partial(my_collate, pad_id=tokenizer.pad_token_id))
    def log_softmax(x):
        x = np.exp(x-np.max(x))
        return np.log(x/np.sum(x, axis=1)+1e-12)
        
    loss_fct = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
    results = {}
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='score', ncols=50):
            batch = [t.to(args.device) for t in batch[:-1]] + batch[-1:]
            
            inputs=batch[0]
            labels = batch[1]
            example_ids = batch[2]
            

            outputs = model(inputs)

            # get logits
            logits = outputs[0]

            # flatten the tokens
            log_probs = -loss_fct(logits.view(-1, logits.size(-1)),
                                  labels.view(-1))
            log_probs = torch.sum(log_probs.view(logits.size(0), -1), dim=1).cpu().numpy().tolist()
            token_nums = torch.sum(labels!=-1, dim=1).cpu().numpy().tolist()

            for log_prob, token_num, example_id in zip(log_probs, token_nums, example_ids):
                if example_id not in results or results[example_id][2]<log_prob/token_num:
                    results[example_id]=(log_prob, token_num, log_prob/token_num)

    return results

def write_output(results, input_file, output_file):
    with open(input_file, 'r') as f:
        data = json.load(f) 
    for elem in data:
        example_id = elem['id']
        elem['lm_log_prob'] = [0]*len(elem['pred_question'])
        elem['lm_token_num'] = [0]*len(elem['pred_question'])
        for q_id, pred_question in enumerate(elem['pred_question']):
            question_id = '%s_%d'%(example_id, q_id)
            elem['lm_log_prob'][q_id] = results[question_id][0]
            elem['lm_token_num'][q_id] = results[question_id][1]
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=1, ensure_ascii=False)

    #final_outputs = {}
    #for example in examples:
    #    example_id = example.example_id
    #    unique_id = example.unique_id
    #    result = results[unique_id]
    #    if example.example_id in final_outputs:
    #        elem = final_outputs[example.example_id]
    #    else:
    #        elem = {'context':example.context,
    #                'answers':example.answer,
    #                'pred_question':['']*example.question_num,
    #                'lm_log_prob':[-1e4]*example.question_num,
    #                'lm_token_num':[0]*example.question_num}
    #        final_outputs[example.example_id] = elem
    #    elem['pred_question'][example.question_id] = example.question
    #    elem['lm_log_prob'][example.question_id] = result.log_prob
    #    elem['lm_token_num'][example.question_id] = result.token_num

    #final_outputs = [final_outputs[i] for i in range(len(final_outputs))]
    #with open(output_file, 'w') as f:
    #    json.dump(final_outputs, f, indent=1, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action",choices=['train', 'eval', 'score'])

    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name")
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--max_len", type=int, default=100)
    parser.add_argument("--data_mode", choices=['q', 'cq'], default='q', 
                        help='Control input features: q means using questions only, cq means using both contexts and questions.')

    # training options
    parser.add_argument("--learning_rate", type=float, default=5e-5)
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
    parser.add_argument("--is_qg", action='store_true')
    parser.add_argument("--early_stop_tolerance", default=5, type=int)
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")

    # file path
    parser.add_argument("--save_dir", default=None)
    parser.add_argument("--train_file_path", default='data/squad/train.json')
    parser.add_argument("--dev_file_path", default='data/squad/dev.json')
    parser.add_argument("--eval_file", default='data/squad/eval_dev.json')
    parser.add_argument("--score_file_path", default=None)
    parser.add_argument("--output_file", default=None)

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
                           'cls_token': '<CLS>'}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    logger.info('We have added %d tokens'%num_added_toks)
    
    # load model from pretrained gpt2 or finetuned gpt2
    config = GPT2Config.from_pretrained(args.model_name_or_path)
    if args.action == 'train':
        config.data_mode = args.data_mode

    model = GPT2LMHeadModel.from_pretrained(args.model_name_or_path, config=config)
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
        train_dataset = load_and_cached_data(config, tokenizer, args.train_file_path, mode='train', max_seq_len=300,save_cache=True)
        dev_dataset = load_and_cached_data(config, tokenizer, args.dev_file_path, mode='dev', save_cache=True)
        train(args, train_dataset, dev_dataset, model, tokenizer)
    elif args.action == 'eval':
        dataset = load_and_cached_data(config, tokenizer, args.dev_file_path, mode='dev', 
                                       is_qg=args.is_qg, save_cache=False)
        result = evaluate(args, dataset, model, tokenizer)
        print (args.model_name_or_path, result)
    elif args.action == 'score':
        assert args.output_file is not None
        dataset, examples = load_and_cached_data(config, tokenizer, args.score_file_path, mode='score', is_qg=True,
                                                 output_example=True, save_cache=False)
        results = score(args, dataset, model, tokenizer)
        write_output(results, args.score_file_path, args.output_file)
if __name__ == '__main__':
    main()
