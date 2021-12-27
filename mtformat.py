import json
import random
import argparse
import numpy as np
import pprint as pp
from util import *
from time import time
from pathlib import Path
from train import prepare_train_data
from collections import Counter, OrderedDict, defaultdict as ddict
from transformers import DistilBertTokenizerFast

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler

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
    # pp.pprint(vectorized_plist.toarray().shape)
    # np.savetxt('vectorized_plist.csv', vectorized_plist.toarray(), delimiter=',')
    print('VECTORIZATION FINISHED in %0.3fs.\n' % (time() - t0))

    print('BEGINNING LDA FITTING...')
    t0 = time()
    lda = LatentDirichletAllocation(n_components = topic_num, max_iter = 5, random_state = 0)
    ptmatrix = lda.fit_transform(vectorized_plist)
    print('LDA FITTING FINISHED in %0.3fs.\n' % (time() - t0))
    # pp.pprint(lda_plist.shape)
    # pp.pprint(lda.components_.shape)
    # np.savetxt('lda_plist.csv', lda_plist, delimiter=',')
    # np.savetxt('lda_components.csv', lda.components_, delimiter=',')
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

def meta_train_format_dataset(data_path, save_path='meta_train_data.json'):
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str)
    args = parser.parse_args()
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    processed_data = meta_train_format_dataset(args.data_path)
    # task_distribution = get_task_distribution(processed_data, 16)
    # for task_set in task_distribution:
    #     print(task_set.keys())
    # first_sample_key = list(task_distribution[0].keys())[0]
    # meta_train_loader = get_loader_batches(task_distribution[0][first_sample_key]['support'], tokenizer)
    # for batch in meta_train_loader:
    #     print(batch['input_ids'])

    # train_tasks = meta_train_format_dataset(train_path)
    # test_tasks = meta_train_format_dataset(val_path)

    # MetaTrain(model, train_tasks, test_tasks)

    # for epoch in num_epochs:
    #     sampled_tasks = sample_tasks(processed_data, batch_size)
    #     for T in sampled_tasks:
    #         meta_train_loader = get_loader_batches(T['support'], tokenizer)
    #         trainer.meta_train(meta_train_loader)