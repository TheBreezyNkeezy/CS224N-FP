import argparse
import json
import os
import torch
import random
import csv
import util

import numpy as np
from copy import deepcopy
import pprint as pp
from time import time

from collections import OrderedDict
from collections import Counter, OrderedDict, defaultdict as ddict

from transformers import DistilBertTokenizerFast
from transformers import DistilBertForQuestionAnswering
from transformers import AdamW

from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler

from args import get_train_test_args
from util import *
from train import prepare_train_data

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer


# DATA PROCESSING
def pqa_format_squad(path):
    print(f'BEGINNING DATA FORMATTING FOR {path}...')
    t0 = time()
    with open(path, 'rb') as f:
        squad_dict = json.load(f)
    pqa_list = [] # (p, q, id, (a_start, a_text))
    for group in squad_dict['data']:
        for passage in group['paragraphs']:
            context = passage['context']
            for qa in passage['qas']:
                id_num = qa['id']
                question = qa['question']
                if len(qa['answers']) == 0:
                    answer = None
                    pqa = (context, question, id_num, answer)
                    pqa_list.append(pqa)
                else:
                    for answer in qa['answers']:
                        start = answer["answer_start"]
                        text = answer["text"]
                        pqa = (context, question, id_num, (start, text))
                        pqa_list.append(pqa)
    final_pqa = list(set([i for i in pqa_list]))
    plist = list(set([p for p,q,i,a in final_pqa]))
    print(f'DATA FORMATTING FOR {path} FINISHED IN {(time()-t0):0.3f}s.\n')
    return plist, final_pqa

def topics_pqa(plist, topic_num):
    # pp.pprint(len(plist))
    print('BEGINNING VECTORIZATION...')
    t0 = time()
    vectorizer = CountVectorizer(max_df = 0.85, min_df = 2, max_features = 500, stop_words = 'english')
    vectorized_plist = vectorizer.fit_transform(plist)
    print('VECTORIZATION FINISHED in %0.3fs.\n' % (time() - t0))

    print('BEGINNING LDA FITTING...')
    t0 = time()
    lda = LatentDirichletAllocation(n_components = topic_num, max_iter = 5, random_state = 0)
    ptmatrix = lda.fit_transform(vectorized_plist)
    print('LDA FITTING FINISHED in %0.3fs.\n' % (time() - t0))

    return ptmatrix   

def topicalize_pqa(pqa_list, plist, ptmatrix):
    print('BEGINNING TOPICALIZAITON OF DATA...')
    t0 = time()
    num_paragraphs, num_topics = ptmatrix.shape
    topic_counts = dict([(i,{'paragraphs': 0, 'data_points': 0}) for i in range(num_topics)])
    topicalized = dict([(i,[]) for i in range(num_topics)])
    for n, curr in enumerate(plist):
        topic_most_pr = ptmatrix[n].argmax()
        matching = [
            (p,q,i,a)
            for p,q,i,a in pqa_list
            if p == curr
        ]
        topicalized[topic_most_pr].extend(matching)
        topic_counts[topic_most_pr]['paragraphs'] += 1
        topic_counts[topic_most_pr]['data_points'] += len(matching)
        # print(f"DOCUMENT: {n} --- TOPIC: {topic_most_pr}")
    print('TOPICALIZATION FINISHED in %0.3fs.\n' % (time() - t0))
    return topicalized, topic_counts

def support_query_split(topicalized):
    print('BEGINNING SUPPORT/QUERY SPLIT OF TOPICALIZED DATA...')
    t0 = time()
    split_topicalized = {}
    split_topic_counts = {}
    for topic_num in topicalized:
        topic_data = topicalized[topic_num][:]
        random.shuffle(topic_data)
        support = topic_data[:int((len(topic_data)+1)*.70)]
        query = topic_data[int((len(topic_data)+1)*.70):]
        split_topicalized[topic_num] = {'support': support, 'query': query}
        split_topic_counts[topic_num] = {'support': len(support), 'query': len(query)}
    print('SUPPORT/QUERY SPLIT FINISHED in %0.3fs.' % (time() - t0))
    return split_topicalized, split_topic_counts

def meta_train_format_dataset(data_path, eval = False):
    save_path = 'meta_train_data.json' if not eval else 'meta_eval_data.json'
    file = Path(save_path)
    split_topicalized = {}
    if file.exists():
        with open(file, 'r') as f:
            split_topicalized = json.load(f)
    else:
        plist, formatted = pqa_format_squad(data_path)
        ptmatrix = topics_pqa(plist, 100)
        topicalized, topic_counts = topicalize_pqa(formatted, plist, ptmatrix)
        split_topicalized, split_topic_counts = support_query_split(topicalized)
        with open(file, 'w') as f:
            json.dump(split_topicalized, f)
    return split_topicalized

def convert_task_data(data_list):
    # [(p,q,i,(a_s, a_text))]
    data_dict = {'question': [], 'context': [], 'id': [], 'answer': []}
    for p,q,i,a in data_list:
        data_dict['context'].append(p)
        data_dict['question'].append(q)
        data_dict['id'].append(i)
        answer_dict = {'answer_start': a[0], 'text': a[1]}
        data_dict['answer'].append(answer_dict)

    id_map = ddict(list)
    for idx, qid in enumerate(data_dict['id']):
        id_map[qid].append(idx)

    data_dict_collapsed = {'question': [], 'context': [], 'id': [], 'answer': []}
    for qid in id_map:
        ex_ids = id_map[qid]
        data_dict_collapsed['question'].append(data_dict['question'][ex_ids[0]])
        data_dict_collapsed['context'].append(data_dict['context'][ex_ids[0]])
        data_dict_collapsed['id'].append(qid)
        all_answers = [data_dict['answer'][idx] for idx in ex_ids]
        data_dict_collapsed['answer'].append({
            'answer_start': [answer['answer_start'] for answer in all_answers],
            'text': [answer['text'] for answer in all_answers]
        })
    return data_dict_collapsed

def sample_tasks(data_map, batch_size):
    sampled = {}
    selected_indices = np.random.choice(len(data_map.keys()), batch_size)
    for idx in selected_indices:
        sampled[idx] = data_map[idx]
    return sampled

def get_loader_batches(task_data, tokenizer):
    final_task_format = convert_task_data(task_data)
    encoded = prepare_train_data(final_task_format, tokenizer)
    meta_train_data = QADataset(encoded, train = True)
    meta_train_loader = DataLoader(
        meta_train_data,
        batch_size=16,
        sampler=RandomSampler(meta_train_data)
    )
    return meta_train_loader

def get_task_distribution(data_map, batch_size):
    tasks = list(data_map.keys())[:]
    random.shuffle(tasks)
    samp_batch_idxs = np.array_split(np.array(tasks), batch_size)
    sample_batches = []
    for indices in samp_batch_idxs:
        samp_batch = {}
        for idx in indices:
            samp_batch[idx] = data_map[idx]
        sample_batches.append(samp_batch)
    return sample_batches

def read_squad(path):
    with open(path, 'rb') as f:
        squad_dict = json.load(f)
    data_dict = {'question': [], 'context': [], 'id': [], 'answer': []}
    for group in squad_dict['data']:
        for passage in group['paragraphs']:
            context = passage['context']
            for qa in passage['qas']:
                question = qa['question']
                if len(qa['answers']) == 0:
                    data_dict['question'].append(question)
                    data_dict['context'].append(context)
                    data_dict['id'].append(qa['id'])
                else:
                    for answer in  qa['answers']:
                        data_dict['question'].append(question)
                        data_dict['context'].append(context)
                        data_dict['id'].append(qa['id'])
                        data_dict['answer'].append(answer)
    id_map = ddict(list)
    for idx, qid in enumerate(data_dict['id']):
        id_map[qid].append(idx)

    data_dict_collapsed = {'question': [], 'context': [], 'id': []}
    if data_dict['answer']:
        data_dict_collapsed['answer'] = []
    for qid in id_map:
        ex_ids = id_map[qid]
        data_dict_collapsed['question'].append(data_dict['question'][ex_ids[0]])
        data_dict_collapsed['context'].append(data_dict['context'][ex_ids[0]])
        data_dict_collapsed['id'].append(qid)
        if data_dict['answer']:
            all_answers = [data_dict['answer'][idx] for idx in ex_ids]
            data_dict_collapsed['answer'].append({'answer_start': [answer['answer_start'] for answer in all_answers],
                                                  'text': [answer['text'] for answer in all_answers]})
    return data_dict_collapsed


# TRAINING AND TESTING
class MetaTrain:
    def __init__(self, args):
        self.num_epochs = 30 # Decide
        self.device = args.device
        self.path = os.path.join(args.save_meta_dir, 'checkpoint')
        self.alpha = args.alpha
        self.beta = args.beta
        self.num_iteration = args.num_iteration
        self.batch = args.batch
        self.epoch_num = args.epoch_num
        self.save_dir = args.save_dir
        
    def save(self, model):
        model.save_pretrained(self.path)
    
    def meta_test(self, model_phi, query_batches, data_dict):
        device = self.device
        model_phi.to(device)
        model_phi.eval()

        pred_dict = {}
        all_start_logits = []
        all_end_logits = []

        with torch.no_grad(), tqdm(total=len(query_batches.dataset)) as progress_bar:
            for batch in query_batches:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                start_positions = batch['start_positions'].to(device)
                end_positions = batch['end_positions'].to(device)
                outputs = model_phi(input_ids, 
                                    attention_mask = attention_mask, 
                                    start_positions = start_positions, 
                                    end_positions = end_positions)
           
                start_logits, end_logits = outputs.start_logits, outputs.end_logits
                all_start_logits.append(start_logits)
                all_end_logits.append(end_logits)
                progress_bar.update(len(input_ids))

            # Get F1 and EM scores
            start_logits = torch.cat(all_start_logits).cpu().numpy()
            end_logits = torch.cat(all_end_logits).cpu().numpy()

            preds = util.postprocess_qa_predictions(data_dict, 
                                                    query_batches.dataset.encodings,
                                                    (start_logits, end_logits))

            results = util.eval_dicts(data_dict, preds)
            results_list = [('F1', results['F1']),
                            ('EM', results['EM'])]

            score = OrderedDict(results_list) 

        return preds, score


    def meta_train(self, model_theta, task_batch, val_dict, tokenizer, eval = False):
        device = self.device
        all_gradient = []
        num_task = len(task_batch)
        best_scores = {'F1': -1.0, 'EM': -1.0}

        optim_1 = AdamW(model_theta.parameters(), lr = self.beta)
        model_theta.train()
        
        for t, T in enumerate(task_batch):
            support_batches = get_loader_batches(task_batch[T]['support'], tokenizer) # get Dataloader for this
            query_batches = get_loader_batches(task_batch[T]['query'], tokenizer) # get Dataloader for this

            model_phi = deepcopy(model_theta) 
            model_phi.to(device)
            optim_2 = AdamW(model_phi.parameters(), lr = self.beta)
            model_phi.train()

            print("Learning Task: ", t)

            for k in range(self.num_iteration):
                all_loss = []
                with torch.enable_grad(), tqdm(total=len(support_batches.dataset)) as progress_bar:
                    for batch in support_batches: 
                        input_ids = batch['input_ids'].to(device) # number representation of 
                        attention_mask = batch['attention_mask'].to(device)
                        start_positions = batch['start_positions'].to(device)
                        end_positions = batch['end_positions'].to(device)
                        
                        outputs = model_phi(input_ids, attention_mask = attention_mask, 
                                            start_positions = start_positions, 
                                            end_positions = end_positions)

                        loss = outputs[0]
                        loss.sum().backward()
                        optim_2.step()
                        optim_2.zero_grad()

                        all_loss.append(loss)
                        progress_bar.update(len(input_ids))
                        
                print("Average loss after step: ", k, torch.mean(torch.stack(all_loss)))    
               
            # model_phi.to(torch.device('cpu'))

            if not eval:
                for i,(param_theta, param_phi) in enumerate(zip(list(model_theta.parameters()),list(model_phi.parameters()))):
                    gradient = param_theta - param_phi
                    
                    if t == 0:
                        all_gradient.append(gradient)
                    else:
                        all_gradient[i] += gradient

            # Meta testing
            preds, curr_score = self.meta_test(model_phi, query_batches, val_dict)

            if curr_score['F1'] >= best_scores['F1']:
                best_scores = curr_score
                self.save(model_theta)

            # model_phi.to(torch.device('cpu'))
            if not eval:
                for i in range(len(all_gradient)): all_gradient[i] /= float(num_task)
                    
                # Update theta model parameter
                for i, params in enumerate(model_theta.parameters()): params.grad = all_gradient[i]
                optim_1.step()
                optim_1.zero_grad()

        return best_scores if not eval else preds, best_scores
        
    def train(self, model_theta, train_tasks, val_dict, tokenizer):        
        for epoch_num in range(self.epoch_num): 
            task_distribution = get_task_distribution(train_tasks, self.batch)
            
            for i , sampled_train_tasks in enumerate(task_distribution):
                print("\n----------------- Meta-Training in Session -----------------\n")
                best_scores = self.meta_train(model_theta, sampled_train_tasks, val_dict, tokenizer, eval = False)

                print(f'\n----------------- Training step {i} with current best_scores {best_scores} -----------------\n')
    
    def evaluate(self, model_theta, test_tasks, val_dict, tokenizer):
        # for epoch_num in range(self.epoch_num): 
        task_distribution = get_task_distribution(test_tasks, self.batch)

        eval_preds = []
        eval_scores = []
        for i , sampled_eval_tasks in enumerate(task_distribution):
            print("\n----------------- Evaluation in Session -----------------\n")
            eval_preds_curr, eval_scores_curr = self.meta_train(model_theta, sampled_eval_tasks, val_dict, tokenizer, eval = True)
            eval_preds.extend(eval_preds_curr)
            eval_scores.extend(eval_scores_curr)
            print(f'\n----------------- Evaluation step {i} with current eval_scores {eval_scores_curr} -----------------\n')
    
        return eval_preds, eval_scores

def main():
    # define parser and arguments
    args = get_train_test_args()
    util.set_seed(args.seed)
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    
    if args.do_train: # python MetaTrain.py  --do-train
        model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")
        model.to(args.device)
        train_tasks = meta_train_format_dataset(args.meta_train_dir)
        val_dict = read_squad(args.meta_train_dir)
        best_scores = MetaTrain(args).train(model, train_tasks, val_dict, tokenizer)    
    if args.do_eval: ## python MetaTrain.py --do-eval --sub-file mtl_submission.csv --save-dir save/meta_train_models
        
        split_name = 'test' if 'test' in args.eval_dir else 'validation'
        checkpoint_path = os.path.join(args.save_meta_dir)
        model = DistilBertForQuestionAnswering.from_pretrained(checkpoint_path)
        model.to(args.device)
        
        eval_tasks = meta_train_format_dataset(args.meta_val_dir, eval = True)
        
        val_dict = read_squad(args.meta_val_dir)
        eval_preds, eval_scores = MetaTrain(args).evaluate(model, eval_tasks, val_dict, tokenizer)
        
        # Write submission file
        sub_path = os.path.join(args.save_dir, split_name + '_' + args.sub_file)
        with open(sub_path, 'w', newline='', encoding='utf-8') as csv_fh:
            csv_writer = csv.writer(csv_fh, delimiter=',')
            csv_writer.writerow(['Id', 'Predicted'])
            for eval_pred in eval_preds:
                for uuid in sorted(eval_pred):
                    csv_writer.writerow([uuid, eval_pred[uuid]])
            

if __name__ == '__main__':
    main()