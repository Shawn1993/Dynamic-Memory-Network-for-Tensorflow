import numpy as np
import os
import pickle

class Batch(object):
    """data structure for batch"""
    def __init__(self, contexts, fact_masks, questions, answers):
        self.contexts = contexts
        self.questions = questions
        self.answers = answers
        self.fact_masks = fact_masks

class DataSet(object):
    ''' data structure for data
    '''
    def __init__(self, batch_size, contexts, fact_masks, questions, answers):

        self.contexts = contexts # shape of (N, S)
        self.fact_masks = fact_masks # shape of (N,S)
        self.questions = questions # shape of (N, Q)
        self.answers = answers # shape of (N, )
        self.size = len(answers)
        self.indices = np.arange(self.size) # shape of (N, )

        self.batch_size = batch_size
        assert self.batch_size <= self.size
        self.batch_index = 0

    def shuffle(self):
        ''' Shuffle
        While you use mini-batch, the last batch may not be a full batch.
        we would drop the last batch. To avoid the last batch is always not used,
        you have to shuffle the dataset explicitly
        '''
        np.random.shuffle(self.indices)

    def split(self, ratio, pre_batch_size=None, pos_batch_size=None):
        ''' Split the dataset to tow dataset by ratio 
        
        Args:
            ratio: the ratio for split, for example:
                    dataset --> shape of (100,)
                    previous, posterior = dataset.split(0.1)
                    previous --> shape of (90,)
                    posterior --> shape of (10,)
            pre_batch_size: Optional, the batch size for previous set
            pos_batch_size: Optional, the batch size for posterior set


        Return:
            previous: the previos subset
            posterior: the posterior subset
        '''
        mid = int(ratio * self.size)
        pos_batch_size = pos_batch_size if pos_batch_size else self.batch_size
        pre_batch_size = pre_batch_size if pre_batch_size else self.batch_size
        if pos_batch_size > mid:
            raise ValueError('batch_size must be smaller than dataset size. Please set a smaller pos_batch_size or a larger ratio.')
        if pre_batch_size > self.size - mid:
            raise ValueError('batch_size must be smaller than dataset size. Please set a smaller pre_batch_size or a smaller ratio.')

        previous = DataSet(pre_batch_size, 
                          self.contexts[:-mid], 
                          self.fact_masks[:-mid], 
                          self.questions[:-mid], 
                          self.answers[:-mid])
        posterior = DataSet(pos_batch_size, 
                            self.contexts[-mid:], 
                            self.fact_masks[-mid:], 
                            self.questions[-mid:], 
                            self.answers[-mid:])
        return previous, posterior

    def __iter__(self):
        return self

    def __next__(self):
        if self.batch_index + self.batch_size <= self.size:
            start = self.batch_index
            self.batch_index += self.batch_size
            end = self.batch_index

            batch_indices = self.indices[start:end]
            return Batch(self.contexts[batch_indices],
                         self.fact_masks[batch_indices],
                         self.questions[batch_indices], 
                         self.answers[batch_indices])
        else:
            self.batch_index = 0
            raise StopIteration



import pandas as pd

class WordTable(object):
    """data structure for word embeddings 
    """

    def __init__(self, dim, pretrain=None):
        '''
        Parameters
        ----------
        dim : int
            If pretrain is given, dim doesn't work. Otherwise, it must be given.
        pretrain : dict, optional
            The pretrain word embeddings. If given, the dim will be infered.
            If not given, dim must be given explicitly.
        '''
        if pretrain:
            assert type(pretrain) == dict
            self.pre_vocab = pd.Series(pretrain)
            self.embed_size = len(self.pre_vocab[0])
        else :
            assert dim is not None
            self.pre_vocab = None
            self.embed_size = dim
       
        self.vocab = pd.Series()
        
    @property
    def vocab_size(self):
        return self.vocab.size

    def add(self, words):
        for w in words:
            if (self.pre_vocab is not None) and (w in self.pre_vocab.index):
                self.vocab[w] = self.pre_vocab[w]
            else :
                self.vocab[w] = list(np.random.uniform(size=self.embed_size))

    def word2vec(self, words):
        '''word to vector
        Args:
            words: str or list of str.
        Return:
            a list of word vector with the same shape of word.
        '''
        return self.vocab[words] if(type(words)!=list) else list(self.vocab[words].values)

    def word2index(self, words):
        '''word to index'''
        return self.vocab.index.get_loc(words) if(type(words)!=list) else list(self.vocab.index.get_indexer(words))

""" a neat code from https://github.com/therne/dmn-tensorflow """
def load_glove(dim):
    """ Loads GloVe data.
    :param dim: word vector size (50, 100, 200)
    :return: GloVe word table
    """
    print("Loading Glove data...")
    word2vec = {}

    path = "data/glove/glove.6B/glove.6B." + str(dim) + 'd'
    if os.path.exists(path + '.cache'):
        with open(path + '.cache', 'rb') as cache_file:
            word2vec = pickle.load(cache_file)

    else:
        # Load n create cache
        with open(path + '.txt') as f:
            for line in f:
                l = line.split()
                word2vec[l[0]] = [float(x) for x in l[1:]]

        with open(path + '.cache', 'wb') as cache_file:
            pickle.dump(word2vec, cache_file)

    print("Loaded Glove data")
    return word2vec

from copy import deepcopy

""" a neat code from https://github.com/therne/dmn-tensorflow """ 
def load_babi(data_dir, task_id, type='train'):
    """ Load bAbi Dataset.
    :param data_dir
    :param task_id: bAbI Task ID
    :param type: "train" or "test"
    :return: dict
    """

    # Parsing the file name
    print("Loading data from bAbI {} task {}...".format(type, task_id))
    files = os.listdir(data_dir)
    files = [os.path.join(data_dir, f) for f in files]
    s = 'qa{}_'.format(task_id)
    file_name = [f for f in files if s in f and type in f][0]

    # Parsing the specific file
    tasks = []
    for i, line in enumerate(open(file_name)):
        id = int(line[0:line.find(' ')])
        if id == 1:
            curr_task = {"C": [], "Q": "", "A": ""}

        line = line.strip()
        line = line.replace('.', ' . ')
        line = line[line.find(' ') + 1:]
        if line.find('?') == -1:
            curr_task["C"].append(line)
        else:
            idx = line.find('?')
            tmp = line[idx + 1:].split('\t')
            curr_task["Q"] = line[:idx]
            curr_task["A"] = tmp[1].strip()
            tasks.append(deepcopy(curr_task))
    print("Loaded {} data from bAbI {} task {}".format(len(tasks), type, task_id))

    return tasks

""" a neat code from https://github.com/therne/dmn-tensorflow 
I modify something
"""
from tqdm import tqdm
def process_babi(tasks, word_table):
    
    """ Tokenizes sentences.
    :param raw: dict returned from load_babi
    :param word_table: WordTable
    :return:
    """

    # Define the fuction for tokenization
    def token(sent):return [w for w in sent.lower().split() if len(w)>0 ]

    from functools import reduce
    def flatten(lists):return reduce(lambda x,y:x+y, lists)

    index = word_table.word2index

    questions = []
    inputs = []
    answers = []
    fact_masks = []
    max_sentence_len = 0
    max_facts_len = 0
    max_question_len = 0

    for task in tqdm(tasks):
        
        # find something from task['C']
        input_indices = []
        masks = []
        for sent in task["C"]:
            sent_words = token(sent)
            word_table.add(sent_words)
            input_indices.append(index(sent_words))
            m = np.zeros(len(sent_words))
            m[-1] = 1
            masks.append(list(m))
        
        inputs.append(flatten(input_indices))
        fact_masks.append(flatten(masks))
        
        # find something from task['Q']
        q_words = token(task['Q'])
        word_table.add(q_words)
        q_indices = index(q_words)
        questions.append(q_indices)
        
        # find something from task['A']
        # NOTE: here we assume the answer is one word!
        answers.append(index(task["A"]))  

    return inputs, fact_masks, questions, answers

def get_babi(data_dir, task_id, task_type, batch_size, word_table):
    data = process_babi(load_babi(data_dir,task_id,task_type), word_table)
    return DataSet(batch_size, *data)

def get_wordtable(dim, name=None):
    if not name:
        return WordTable(dim)
    if name.lower() == 'glove':
        return WordTable(dim, pretrain=load_glove(dim))

def get_max_len(*datasets):
    max_sentence_len = max_question_len = max_facts_len = 0
    for data in datasets:
        for c, q, f in zip(data.contexts, data.questions, data.fact_masks):
            max_sentence_len = max(max_sentence_len, len(c))
            max_question_len = max(max_question_len, len(q))
            max_facts_len = max(max_facts_len, np.sum(f))
    
    return int(max_sentence_len), int(max_question_len), int(max_facts_len)

def padding_datasets(config, *datasets):

    def padding(array, max_size):
        return list(np.append(array, np.zeros(max_size-len(array))))

    for _, data in enumerate(datasets):
        for n in range(data.size):
            data.contexts[n] = padding(data.contexts[n], config.max_sentence_len)
            data.fact_masks[n] = padding(data.fact_masks[n], config.max_sentence_len)
            data.questions[n] = padding(data.questions[n], config.max_question_len)

        data.contexts = np.array(data.contexts, dtype=int)
        data.fact_masks = np.array(data.fact_masks, dtype=int)
        data.questions = np.array(data.questions, dtype=int)
        data.answers = np.array(data.answers, dtype=int)

    return datasets
































