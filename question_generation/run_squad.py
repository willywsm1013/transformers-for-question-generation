# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
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
""" Finetuning the library models for question-answering on SQuAD (DistilBERT, Bert, XLM, XLNet)."""

from __future__ import absolute_import, division, print_function
import sys
sys.path.insert(0, '../')
import argparse
import logging
import os
import re
import random
import json
import glob
import timeit
import numpy as np
import torch
import math
import string

import torch.nn as nn
from torch.utils.data import (
    DataLoader, RandomSampler, SequentialSampler, TensorDataset, ConcatDataset)
from torch.utils.data.distributed import DistributedSampler

from tqdm import tqdm, trange
from collections import defaultdict, namedtuple

from transformers import (
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    squad_convert_examples_to_features,
)
from transformers.data.metrics.squad_metrics import (
    compute_predictions_log_probs,
    compute_predictions_logits,
    squad_evaluate,
    compute_scores,
)
from transformers.data.processors.squad import SquadResult, SquadV1Processor, SquadV2Processor, SquadQGProcessor

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_QUESTION_ANSWERING_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def to_list(tensor):
    return tensor.detach().cpu().tolist()


def train(args, train_dataset, model, tokenizer, eval_dataset_and_example=None):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(os.path.join(args.output_dir, 'summary'))

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    if args.squad_file is not None and args.sampler == 'balance':
        logger.info('Use BalanceRandomSampler')
        assert args.balance_ratio is not None
        train_sampler = BalanceRandomSampler(train_dataset, tuple(args.balance_ratio))
    else:
        assert args.sampler == 'random'
        logger.info('Use RandomSampler')
        train_sampler = RandomSampler(
            train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (
            len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(
            train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate, eps=args.adam_epsilon)

    if args.warmup_steps != 0:
        assert args.warmup_ratio == 0
        warmup_steps = args.warmup_steps
    elif args.warmup_ratio != 0:
        warmup_steps = int(t_total*args.warmup_ratio)
    else:
        warmup_steps = 0
    logger.info('Warm-up steps : %d'%warmup_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    '''
    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, 'optimizer.pt')) and os.path.isfile(os.path.join(args.model_name_or_path, 'scheduler.pt')):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(
            os.path.join(args.model_name_or_path, 'optimizer.pt')))
        scheduler.load_state_dict(torch.load(
            os.path.join(args.model_name_or_path, 'scheduler.pt')))
    '''
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d",
                args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d",
                args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    '''
    if os.path.exists(args.model_name_or_path):
        # set global_step to gobal_step of last saved checkpoint from model path
        global_step = int(args.model_name_or_path.split('-')[-1].split('/')[0])
        epochs_trained = global_step // (len(train_dataloader) //
                                         args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (
            len(train_dataloader) // args.gradient_accumulation_steps)

        logger.info(
            "  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch",
                    steps_trained_in_current_epoch)
    '''
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(epochs_trained, int(
        args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    
    # Added here for reproductibility (even between python 2 and 3)
    #set_seed(args)

    max_f1 = 0
    fail=0
    logger.info('GT + GEN example num : %d'%len(train_dataset))
    if args.sampler == 'balance':
        logger.info('batch num in one epoch after balancing (ratio : %s): %d'%(args.balance_ratio, len(train_dataloader)))
    elif args.sampler =='random':
        logger.info('batch num in one epoch without balancing: %d'%len(train_dataloader))
    else:
        raise Exception
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration",
                              disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)

            inputs = {
                'input_ids':       batch[0],
                'attention_mask':  batch[1],
                'token_type_ids': None if args.model_type in ['xlm', 'roberta', 'distilbert'] else batch[2],
                'start_positions': batch[3],
                'end_positions':   batch[4]
            }

            if args.model_type in ['xlnet', 'xlm']:
                inputs.update({'cls_index': batch[5],
                               'p_mask':       batch[6]})
                if args.version_2_with_negative:
                    inputs.update({'is_impossible': batch[7]})
            outputs = model(**inputs)
            # model outputs are always tuple in transformers (see doc)
            loss = outputs[0]

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(
                        amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                # Log metrics
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    tb_writer.add_scalar(
                        'lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar(
                        'loss', (tr_loss - logging_loss)/args.logging_steps, global_step)
                    logging_loss = tr_loss

                # Save model checkpoint
                if (args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0) or global_step >=t_total:
                    # Only evaluate when single GPU otherwise metrics may not average well
                    results = {}
                    if args.local_rank == -1 and args.evaluate_during_training:
                        results = evaluate(args, model, tokenizer, eval_dataset_and_example=eval_dataset_and_example)
                        for key, value in results.items():
                            tb_writer.add_scalar(
                                'eval_{}'.format(key), value, global_step)
                            logger.info('eval_{} : {}'.format(key, value))


                    if not args.evaluate_during_training or results['f1'] > max_f1:
                        max_f1 = results['f1'] if 'f1' in results else 0
                        output_dir = os.path.join(args.output_dir, 'checkpoint')
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        # Take care of distributed/parallel training
                        model_to_save = model.module if hasattr(
                            model, 'module') else model
                        model_to_save.save_pretrained(output_dir)
                        tokenizer.save_pretrained(output_dir)

                        torch.save(args, os.path.join(
                            output_dir, 'training_args.bin'))
                        logger.info("Saving model checkpoint to %s", output_dir)

                        torch.save(optimizer.state_dict(), os.path.join(
                            output_dir, 'optimizer.pt'))
                        torch.save(scheduler.state_dict(), os.path.join(
                            output_dir, 'scheduler.pt'))
                        logger.info(
                            "Saving optimizer and scheduler states to %s", output_dir)
                        fail=0
                    else:
                        fail=0
                        #fail += 1

            if (args.max_steps > 0 and global_step > args.max_steps) or fail>=10:
                epoch_iterator.close()
                break
        if (args.max_steps > 0 and global_step > args.max_steps) or fail>=10:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, eval_dataset_and_example=None, prefix="", cache_data=True):
    if eval_dataset_and_example is None:
        dataset, examples, features = load_and_cache_examples(
            args, tokenizer, args.predict_file, is_qg=args.evaluate_is_qg,
            evaluate=True, output_examples=True,  cache_data=cache_data, sort=True)
    else:
        dataset, examples, features = eval_dataset_and_example

    #if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
    #    os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(
        dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    all_results = []
    start_time = timeit.default_timer()

    import torch.nn.functional as F
    model.eval()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(args.device) for t in batch)

        try:
            max_token_num = batch[0][0].cpu().numpy().tolist().index(tokenizer.pad_token_id)
        except:
            max_token_num = len(batch[0][0].cpu().numpy().tolist())

        with torch.no_grad():
            inputs = {
                'input_ids':      batch[0][:,:max_token_num],
                'attention_mask': batch[1][:,:max_token_num],
                'token_type_ids': batch[2][:,:max_token_num],
            }
           
            if args.model_type in ["xlm", "roberta", "distilbert", "camembert"]:
                del inputs["token_type_ids"]

            feature_indices = batch[3]

            # XLNet and XLM use more arguments for their predictions
            if args.model_type in ['xlnet', 'xlm']:
                inputs.update({'cls_index': batch[4], 'p_mask': batch[5]})
                # for lang_id-sensitive xlm models
                if hasattr(model, "config") and hasattr(model.config, "lang2id"):
                    inputs.update(
                        {"langs": (torch.ones(batch[0].shape, dtype=torch.int64) * args.lang_id).to(args.device)}
                    )

            outputs = model(**inputs)
        for i, feature_index in enumerate(feature_indices):
            eval_feature = features[feature_index.item()]
            unique_id = int(eval_feature.unique_id)

            output = [to_list(output[i]) for output in outputs]

            # Some models (XLNet, XLM) use 5 arguments for their predictions, while the other "simpler"
            # models only use two.
            if len(output) >= 5:
                start_logits = output[0]
                start_top_index = output[1]
                end_logits = output[2]
                end_top_index = output[3]
                cls_logits = output[4]

                result = SquadResult(
                    unique_id,
                    start_logits,
                    end_logits,
                    start_top_index=start_top_index,
                    end_top_index=end_top_index,
                    cls_logits=cls_logits
                )

            else:
                start_logits, end_logits = output
                result = SquadResult(unique_id, start_logits, end_logits)

            all_results.append(result)

    evalTime = timeit.default_timer() - start_time
    logger.info("  Evaluation done in total %f secs (%f sec per example)",
                evalTime, evalTime / len(dataset))

    # Compute predictions
    '''
    output_prediction_file = os.path.join(
        args.output_dir, "predictions_{}.json".format(prefix))
    output_nbest_file = os.path.join(
        args.output_dir, "nbest_predictions_{}.json".format(prefix))
    '''
    output_prediction_file=None
    output_nbest_file=None
    if args.version_2_with_negative:
        output_null_log_odds_file = os.path.join(
            args.output_dir, "null_odds_{}.json".format(prefix))
    else:
        output_null_log_odds_file = None

    # XLNet and XLM use a more complex post-processing procedure
    if args.model_type in ['xlnet', 'xlm']:
        start_n_top = model.config.start_n_top if hasattr(
            model, "config") else model.module.config.start_n_top
        end_n_top = model.config.end_n_top if hasattr(
            model, "config") else model.module.config.end_n_top

        predictions = compute_predictions_log_probs(examples, features, all_results, args.n_best_size,
                                                    args.max_answer_length, output_prediction_file,
                                                    output_nbest_file, output_null_log_odds_file,
                                                    start_n_top, end_n_top,
                                                    args.version_2_with_negative, tokenizer, args.verbose_logging)
    else:
        predictions = compute_predictions_logits(
                        examples, 
                        features, 
                        all_results, 
                        args.n_best_size,
                        args.max_answer_length, 
                        args.do_lower_case, 
                        output_prediction_file,
                        output_nbest_file, 
                        output_null_log_odds_file, 
                        args.verbose_logging,
                        args.version_2_with_negative, 
                        args.null_score_diff_threshold,
                        tokenizer,)

    # Compute the F1 and exact scores.
    results = squad_evaluate(examples, predictions)
    return results

def score(args, model, tokenizer, prefix="", cache_data=True):
    dataset, examples, features = load_and_cache_examples(
        args, tokenizer, args.score_file, is_qg=args.score_is_qg, 
        evaluate=False, output_examples=True, cache_data=cache_data, sort=True)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(
        dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    all_results = []
    all_losses = []
    start_time = timeit.default_timer()

    import torch.nn.functional as F
    model.eval()
    for batch_idx, batch in enumerate(tqdm(eval_dataloader, desc="Scoring")):
        batch = tuple(t.to(args.device) for t in batch)
            
        try:
            max_token_num = batch[0][0].cpu().numpy().tolist().index(tokenizer.pad_token_id)
        except:
            max_token_num = len(batch[0][0].cpu().numpy().tolist())

        input_ids = batch[0][:,:max_token_num]
        attention_mask = batch[1][:,:max_token_num]

        with torch.no_grad():
            inputs = {
                'input_ids':      batch[0][:,:max_token_num],
                'attention_mask': batch[1][:,:max_token_num],
                'token_type_ids': batch[2][:,:max_token_num],
            }
           
            if args.model_type in ["xlm", "roberta", "distilbert", "camembert"]:
                del inputs["token_type_ids"]

            feature_indices = batch[-1]

            # XLNet and XLM use more arguments for their predictions
            if args.model_type in ['xlnet', 'xlm']:
                inputs.update({'cls_index': batch[4], 'p_mask': batch[5]})

            outputs = model(**inputs)

        input_ids = batch[0]
        start_positions = batch[3]
        end_positions = batch[4]
        elems = zip(feature_indices, input_ids, start_positions, end_positions)

        for i, (example_index, input_id, start_pos, end_pos) in enumerate(elems):
            eval_feature = features[example_index.item()]
            unique_id = int(eval_feature.unique_id)

            output = [to_list(output[i]) for output in outputs]

            # Some models (XLNet, XLM) use 5 arguments for their predictions, while the other "simpler"
            # models only use two.
            if len(output) >= 5:
                start_logits = output[0]
                start_top_index = output[1]
                end_logits = output[2]
                end_top_index = output[3]
                cls_logits = output[4]

                result = SquadResult(
                    unique_id, start_logits, end_logits,
                    start_top_index=start_top_index,
                    end_top_index=end_top_index,
                    cls_logits=cls_logits
                )

            else:
                start_logits, end_logits = output

                result = SquadResult(
                    unique_id, start_logits, end_logits
                )
            
            result.start_position = start_pos
            result.end_position = end_pos
            all_results.append(result)

    evalTime = timeit.default_timer() - start_time
    logger.info("  Evaluation done in total %f secs (%f sec per example)",
                evalTime, evalTime / len(dataset))

    # Compute predictions
    predictions = compute_scores(examples, features, all_results, 
                                 args.n_best_size, args.max_answer_length, args.do_lower_case)

    # qas_id to probs
    unique_id_to_result = {result.unique_id:result for result in all_results}

    def normalize_answer(s):
        """Lower text and remove punctuation, articles and extra whitespace."""
        def remove_articles(text):
            regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
            return re.sub(regex, ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()
        return white_space_fix(remove_articles(remove_punc(lower(s))))

    ret = defaultdict(lambda:{'em':[], 'f1':[], 'qa_prob':[], 'answer_rank':[], 'pred_answer':[], 'merge_qa_prob':[]})  
    seen = {}
    for example in examples:
        qas_id = '_'.join(example.qas_id.split('_')[:-1])
        if qas_id in seen:
            continue
        seen[qas_id] = 1

        if qas_id not in predictions:
            logger.warning('Example %s lost!'%qas_id)
            continue
        answer = example.answers
        pred = predictions[qas_id]
        # merge answers which are the same after normalizing
        table = {}
        for nbest in pred['nbest']:
            nbest['qa_prob'] = np.exp(nbest['start_logit']+nbest['end_logit'])
            key = normalize_answer(nbest['text'])
            if key in table:
                table[key]['qa_prob'] += nbest['qa_prob']
            else:
                table[key] = nbest
        
        # sort answers by qa prob
        pred['nbest'] = sorted([v for k,v in table.items()], key= lambda x:x['qa_prob'], reverse=True)
        pred['merge_qa_prob'] = 0
        for i, nbest in enumerate(pred['nbest']):
            if any([normalize_answer(a['text']) == normalize_answer(nbest['text']) for a in answer]):
                pred['merge_qa_prob'] = nbest['qa_prob']
                pred['answer_rank'] = i
                break

        example_id = qas_id.split('_')[0]
        ret[example_id]['em'].append(pred['em'])
        ret[example_id]['f1'].append(pred['f1'])
        ret[example_id]['qa_prob'].append(math.exp(pred['answer_log_prob']))
        ret[example_id]['answer_rank'].append(pred.get('answer_rank',len(pred['nbest'])))

    #print (len(examples))
    #print (len(features))
    #print (len(predictions))
    #print (len(ret))
    return ret

class BalanceRandomSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, ratio):
        super(BalanceRandomSampler, self).__init__(dataset)
        assert isinstance(dataset, ConcatDataset)
        assert len(dataset.cumulative_sizes) == 2, 'Only accept 2 datasets.'
        assert len(ratio) == 2
        self.dataset = dataset
        self.ratio = ratio

        sizes = dataset.cumulative_sizes
        self.data_size = [ s if i ==0 else s-sizes[i-1]for i, s in enumerate(sizes)]
        
        n1, n2 = self.data_size[0], self.data_size[1]
        assert n1 < n2
        r1, r2 = ratio[0], ratio[1]
        
        r = r1+r2
        #self.len = min(n1//r1*r, n2//r2*r)
        self.len = n1//r1*r
        self.n2_list_remain = []
    
    def __iter__(self):
        r1, r2 = self.ratio[0], self.ratio[1]
        r = r1 + r2
        
        n1, n2 = self.data_size[0], self.data_size[1]

        n1_list = torch.randperm(n1).tolist()
        
        n2_list = self.n2_list_remain + (torch.randperm(n2)+n1).tolist()

        idxs = [ n1_list[(i//r*r1 + i%r)%n1] if i%r<r1 else n2_list[(i//r*r2 + i%r - r1)%n2]
                        for i in range(self.len)]

        last_n2_list_idx = (self.len-1)//r*r2 + min((self.len-1)%r - r1, 0)
        if last_n2_list_idx > 0 and last_n2_list_idx < n2:
            self.n2_list_remain = n2_list[last_n2_list_idx:]
        else:
            self.n2_list_remain = []
        return iter(idxs)
    
    def __len__(self):
        return self.len

def load_qg_examples(args, tokenizer, file_path, evaluate=False, cache_data=True, output_examples=False, sort=False):

    # Load data features from cache or dataset file
    data_dir, filename = os.path.split(file_path)
    input_dir = args.data_dir if args.data_dir else data_dir

    # cache file name
    cached_features_file = os.path.join(input_dir, 'cached_{}_{}_{}'.format(
        re.sub('\.json','',filename),
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length)))
    if args.sort_mode != None:
        cached_features_file += '_sort-mode-%s'%args.sort_mode
    
    if args.topk != -1:
        cached_features_file += '_topk-%d'%args.topk
    
    if args.use_cluster is not None:
        cached_features_file += '_cluster-%s'%args.use_cluster
       
    if args.a_prob_threshold is not None:
        #if args.sort_mode not in ['a', 'qa'] :
        #    raise Exception('If set a_prob_threshold, sort_mode should be (a, qa) but found (%s).'%args.sort_mode)
        cached_features_file += '_a-threshold-%.1f'%args.a_prob_threshold

    if args.rank_threshold is not None:
        cached_features_file += '_rank-threshold-%d'%args.rank_threshold

    # Init features and dataset from cache if it exists
    if os.path.exists(cached_features_file) and not args.overwrite_cache and not output_examples:
        logger.info("Loading features from cached file %s",
                    cached_features_file)
        features_and_dataset = torch.load(cached_features_file)
        features, dataset = features_and_dataset["features"], features_and_dataset["dataset"]
    else:
        logger.info("Creating features from dataset file at %s", input_dir)

        processor = SquadQGProcessor(args)
            
        if evaluate:
            examples = processor.get_dev_examples(file_path)
        else:
            examples = processor.get_train_examples(file_path)
 
        features, dataset = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=not evaluate,
            return_dataset='pt',
            threads=args.threads,
            sort=sort,
        )
       
        if args.local_rank in [-1, 0] and cache_data:
            logger.info("Saving features into cached file %s",
                        cached_features_file)
            torch.save({"features": features, "dataset": dataset},
                       cached_features_file)
    
    if output_examples:
        return dataset, examples, features
    return dataset

def load_squad_examples(args, tokenizer, file_path, evaluate=False, cache_data=True, output_examples=False, sort=False):
    # Load data features from cache or dataset file
    data_dir, filename = os.path.split(file_path)
    input_dir = args.data_dir if args.data_dir else data_dir

    # cache file name
    cached_features_file = os.path.join(input_dir, 'cached_{}_{}_{}'.format(
        re.sub('\.json','',filename),
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length))
    )

    # Init features and dataset from cache if it exists
    if os.path.exists(cached_features_file) and not args.overwrite_cache and not output_examples:
        logger.info("Loading features from cached file %s",
                    cached_features_file)
        features_and_dataset = torch.load(cached_features_file)
        features, dataset = features_and_dataset["features"], features_and_dataset["dataset"]
    else:
        logger.info("Creating features from dataset file at %s", input_dir)
        processor = SquadV2Processor() if args.version_2_with_negative else SquadV1Processor()
        
        if evaluate:
            examples = processor.get_dev_examples(args.data_dir, filename=file_path)
        else:
            examples = processor.get_train_examples(args.data_dir, filename=file_path)
        features, dataset = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=not evaluate,
            return_dataset='pt',
            threads=args.threads,
            sort=sort,
        )
        if args.local_rank in [-1, 0] and cache_data:
            logger.info("Saving features into cached file %s",
                        cached_features_file)
            torch.save({"features": features, "dataset": dataset},
                       cached_features_file)
    
    if output_examples:
        return dataset, examples, features
    return dataset

def load_and_cache_examples(args, tokenizer, file_path, is_qg=False,
                            evaluate=False, cache_data=True, output_examples=False, 
                            squad_file_path=None, sort=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()
    
    if is_qg:
        load_function = load_qg_examples
    else:
        load_function = load_squad_examples
    
    ret = load_function(args, tokenizer, file_path, evaluate, cache_data, output_examples, sort)

    if output_examples:
        dataset, examples, features = ret
    else:
        dataset = ret

    logger.info('Data num : %d'%len(dataset))
    if squad_file_path is not None:
        logger.info('Train with squad file!')
        assert not output_examples 
        squad_dataset = load_squad_examples(args, tokenizer, squad_file_path,
                                            evaluate=False, cache_data=True, output_examples=False)
        dataset = ConcatDataset([squad_dataset, dataset]) 
    
    if output_examples:
        return dataset, examples, features
    return dataset

def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_TYPES))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pretrained model or model identifier from huggingface.co/models")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")

    # Other parameters
    parser.add_argument("--data_dir", default=None, type=str,
                        help="The input data dir. Should contain the .json files for the task." +
                             "If no data dir or train/predict files are specified, will run with tensorflow_datasets.")
    parser.add_argument("--train_file", default=None, type=str,
                        help="The input training file. If a data dir is specified, will look for the file there" +
                             "If no data dir or train/predict files are specified, will run with tensorflow_datasets.")
    parser.add_argument("--predict_file", default=None, type=str,
                        help="The input evaluation file. If a data dir is specified, will look for the file there" +
                             "If no data dir or train/predict files are specified, will run with tensorflow_datasets.")
    parser.add_argument("--score_file", default=None, type=str,
                        help="The input scoring file. If a data dir is specified, will look for the file there" +
                             "If no data dir or train/predict files are specified, will run with tensorflow_datasets.")

    parser.add_argument("--squad_file", default=None, type=str)
    parser.add_argument("--sampler", default='random', type=str, choices=['balance','random'])
    parser.add_argument("--balance_ratio", default=None, nargs=2, type=int)

    parser.add_argument("--output_file", default=None, type=str)

    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")

    parser.add_argument('--version_2_with_negative', action='store_true',
                        help='If true, the SQuAD examples contain some that do not have an answer.')
    parser.add_argument('--null_score_diff_threshold', type=float, default=0.0,
                        help="If null_score - best_non_null is greater than the threshold predict null.")
    parser.add_argument('--train_is_qg', action='store_true',
                        help='If true, input training file is generted by QG model.')
    parser.add_argument('--evaluate_is_qg', action='store_true',
                        help='If true, input evaluate file is generted by QG model.')
    parser.add_argument('--score_is_qg', action='store_true',
                        help='If true, input score file is generted by QG model.')

    parser.add_argument("--max_seq_length", default=384, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--doc_stride", default=128, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument("--max_query_length", default=64, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_score", action='store_true',
                        help="Whether to run score on the dev set.")
    parser.add_argument("--do_calibrate", action='store_true',
                        help="Whether to run calibration on the dev set.")

    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--warmup_ratio", default=0, type=float,
                        help="Linear warmup over warmup_ratio.")

    parser.add_argument("--n_best_size", default=20, type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions.json output file.")
    parser.add_argument("--max_answer_length", default=30, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start "
                             "and end predictions are not conditioned on one another.")
    parser.add_argument("--verbose_logging", action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")

    parser.add_argument('--threads', type=int, default=8, help='multiple threads for converting example to features')
    
    parser.add_argument('--topk', type=int, default=-1)
    parser.add_argument('--sort_mode', type=str, default=None, choices=['qg','qg_ln','a','qa'])
    parser.add_argument('--a_prob_threshold', type=float, default=None)
    parser.add_argument('--rank_threshold', type=int, default=None)
    parser.add_argument('--use_cluster', type=str, default=None, choices=['use','infersent'])
    parser.add_argument('--qa_alpha', type=float, default=1.0)
    parser.add_argument('--temperature', type=float, default=1.0)

    parser.add_argument('--hidden_dropout_prob', type=float, default=0.1)
    parser.add_argument('--final_dropout_prob', type=float, default=0.)
    parser.add_argument('--max_position_embeddings', type=int, default=512)

    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(
            address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    #set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    args.model_type = args.model_type.lower()

    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    config.hidden_dropout_prob = args.hidden_dropout_prob
    config.final_dropout_prob = args.final_dropout_prob

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model = AutoModelForQuestionAnswering.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )


    '''
    if args.do_train and config.max_position_embeddings < args.max_position_embeddings:
        logger.warning('Positional embedding length not match! Set positional embeddings length to %d'%args.max_position_embeddings)
        old_embed = model.bert.embeddings.position_embeddings.weight.data
        new_pos_embedding=nn.Embedding(args.max_position_embeddings, config.hidden_size)
        new_pos_embedding.weight.data[:config.max_position_embeddings,:]=old_embed
        config.max_position_embeddings = args.max_position_embeddings
        model.bert.embeddings.position_embeddings=new_pos_embedding
    '''
    if args.local_rank == 0:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Before we do anything with models, we want to ensure that we get fp16 execution of torch.einsum if args.fp16 is set.
    # Otherwise it'll default to "promote" mode, and we'll get fp32 operations. Note that running `--fp16_opt_level="O2"` will
    # remove the need for this code, but it is still valid.
    if args.fp16:
        try:
            import apex
            apex.amp.register_half_function(torch, 'einsum')
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(
            args, tokenizer, args.train_file, is_qg=args.train_is_qg,
            evaluate=False, output_examples=False,
            squad_file_path=args.squad_file)
        eval_dataset_and_example=None
        if args.evaluate_during_training:
            eval_dataset_and_example = load_and_cache_examples(
                args, tokenizer, args.predict_file, is_qg=args.evaluate_is_qg,
                evaluate=True, output_examples=True,  cache_data=False,
                sort=True)

        global_step, tr_loss = train(args, train_dataset, model, tokenizer, eval_dataset_and_example)
        logger.info(" global_step = %s, average loss = %s",
                    global_step, tr_loss)

    '''
    # Save the trained model and the tokenizer
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        # Take care of distributed/parallel training
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))
        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(
            args.output_dir, force_download=True)
        tokenizer = tokenizer_class.from_pretrained(
            args.output_dir, do_lower_case=args.do_lower_case)
        model.to(args.device)
    '''
    # Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory
    if args.do_eval and args.local_rank in [-1, 0]:
        results = {}
        if args.do_train:
            logger.info("Loading checkpoints saved during training for evaluation")
            checkpoints = [args.output_dir]
            if args.eval_all_checkpoints:
                checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
                logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce model loading logs
        else:
            logger.info("Loading checkpoint %s for evaluation", args.model_name_or_path)
            checkpoints = [args.model_name_or_path]

        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        
        # Evaluate
        result = evaluate(args, model, tokenizer, prefix='', cache_data=False)

        print ('%s, %f, %f'%(args.model_name_or_path.split('/')[-2], result['exact'], result['f1']))

    if args.do_score and args.local_rank in [-1, 0]:
        assert args.output_file is not None
        logger.info("Loading checkpoint %s for scoring", args.model_name_or_path)
        checkpoints = [args.model_name_or_path]

        logger.info("Score the following checkpoints: %s", checkpoints)

        model.to(args.device)

        # Evaluate
        predictions = score(args, model, tokenizer, prefix='', cache_data=False)

        with open(args.score_file, 'r') as f:
            data = json.load(f)

        for elem in data:
            idx = elem['id']
            if idx in predictions:
                for key, value in predictions[idx].items() :
                    elem[key]=value
            else:
                print ('%s not found'%idx)
        output_dir = os.path.dirname(args.output_file)
        if output_dir != '':
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output_file, 'w') as f:
            json.dump(data, f, indent=1, ensure_ascii=False)

if __name__ == "__main__":
    main()
