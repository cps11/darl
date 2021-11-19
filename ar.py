import os
import pickle
import random
from collections import defaultdict


def _count_stop_words(corpus, min_count):
    word_count = defaultdict(int)
    for category in corpus:
        for words in category:
            for w in words:
                word_count[w] += 1
            
    stop_words = set()
    for w, count in word_count.items():
        if count <= min_count:
            stop_words.add(w)
    
    return stop_words

class Ar(object):
    def __init__(self, target, sample_size=2000):
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.dataset_dir = os.path.join(self.current_dir, 'data', 'ar')
        self.vocab_path = os.path.join(self.dataset_dir, 'vocab_{}_{}.pkl'.format(target, sample_size))
        self.corpus_path = os.path.join(self.dataset_dir, 'corpus_{}_{}.pkl'.format(target, sample_size))
        self.idx_path = os.path.join(self.dataset_dir, 'idx_{}_{}.pkl'.format(target, sample_size))
        self.files = ['toys', 'pet', 'beauty', 'food', 'baby']
        self.target = target 
        self.sample_size = sample_size

        self.corpus, self.word_to_id, self.id_to_word = self.__word_to_value()
    
    def __readfile(self, path):
        corpus = []
        with open(path, 'rb') as fd:
            for line in fd.readlines():
                words = ['<go>'] + [w for w in line.decode(errors='ignore').strip().split(' ')] + ['<cls>']
                l = len(words)
                if l < 15 + 2 or l > 30 + 2:
                    continue

                corpus.append(words)

        random.shuffle(corpus)
        return corpus[:int(self.sample_size)]
    
    def __load_corpus(self):
        corpus = {}
        for name in self.files:
            file_path = os.path.join(self.dataset_dir, '{}.txt'.format(name))
            corpus[name] = self.__readfile(file_path)
        
        file_path = os.path.join(self.dataset_dir, '{}.txt'.format(self.target))
        corpus[self.target] = self.__readfile(file_path)
        
        return corpus

    def __load_vocab(self):
        if os.path.exists(self.vocab_path):
            with open(self.vocab_path, 'rb') as f:
                word_to_id, id_to_word = pickle.load(f)
            return word_to_id, id_to_word
        
        self.corpus = self.__load_corpus()
        stop_word = _count_stop_words(self.corpus.values(), 1)
        
        word_to_id = {}
        id_to_word = {}
        for category in self.corpus.values():
            for words in category:
                for w in words:
                    if w in stop_word:
                        w = '<unk>'
                    
                    if w in word_to_id:
                        continue
                        
                    tmp_id = len(word_to_id)
                    word_to_id[w] = tmp_id
                    id_to_word[tmp_id] = w
        
        print('stop words {}'.format(len(stop_word)))
        with open(self.vocab_path, 'wb') as f:
            pickle.dump((word_to_id, id_to_word), f)

        return word_to_id, id_to_word
    
    def __word_to_value(self):
        word_to_id, id_2_word = self.__load_vocab()
        
        if os.path.exists(self.corpus_path):
            with open(self.corpus_path, 'rb') as f:
                corpus = pickle.load(f)
            return corpus, word_to_id, id_2_word
        
        cls_id = word_to_id['<cls>']
        unk_id = word_to_id['<unk>']
        
        corpus = {}
        for name, category in self.corpus.items():
            docs = []
            for doc in category:
                words = [word_to_id.get(w, unk_id) for w in doc]
                words += [cls_id] * (32 - len(words))
                docs.append(words)
            corpus[name] = docs

        with open(self.corpus_path, 'wb') as f:
            pickle.dump(corpus, f)

        return corpus, word_to_id, id_2_word

    def load_vocab(self):
        return self.word_to_id, self.id_to_word
    
    def load_sources(self, sources):
        return [self.corpus[n] for n in sources]
    
    def __load_idx(self, target, shot):
        key = '{}_{}'.format(target, shot)
        
        idx = {}
        if os.path.exists(self.idx_path):
            with open(self.idx_path, 'rb') as f:
                idx = pickle.load(f)
                if key in idx:
                    return idx[key]

        idx[key] = random.sample(range(self.sample_size), k=shot)
        with open(self.idx_path, 'wb') as f:
            pickle.dump(idx, f)
        
        return idx[key]


    def load_target(self, target, shot):
        return [self.corpus[target][i] for i in self.__load_idx(target, shot)]
    
    def load_other_target(self, target, shot):
        idx = self.__load_idx(target, shot)
        return [self.corpus[target][i] for i in range(self.sample_size) if i not in idx]

        
if __name__ == '__main__':
    data = Ar('office')
    word_to_id, _ = data.load_vocab()
    print('words {}'.format(len(word_to_id)))
