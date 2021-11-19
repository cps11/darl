import torch
from torch.optim import Adam
import torch.nn.functional as F
from tqdm import tqdm
import math
import random
from tqdm import tqdm
from classifiers import Inductionlm
import os
import pickle

def build_rec_loss(prob, target, reward):
    prob = F.log_softmax(prob, dim=-1)
    one_hot = torch.zeros_like(prob)
    one_hot.scatter_(1, target.data.view((-1,1)), 1)
    one_hot = one_hot.type(torch.bool).to(prob.device)

    reward = torch.exp(reward)

    loss = torch.masked_select(prob, one_hot)
    loss = loss * reward
    loss =  -torch.mean(loss)
    return loss

class Darl(object):
    def __init__(self, model, cfg, i2w, w2i) -> None:
        self.model = model
        self.cfg = cfg
        self.i2w = i2w
        self.w2i = w2i
    
        root = os.path.split(os.path.realpath(__file__))[0]
        # weights and vocabulary are pre-trained by https://github.com/zhongyuchen/few-shot-text-classification
        weights = pickle.load(open(os.path.join(root, 'model', 'weights'), 'rb'))
        vocabulary = pickle.load(open(os.path.join(root, 'model', 'vocabulary'), 'rb'))

        self.classifier = Inductionlm(2, cfg.shot, weights, vocabulary)
        self.classifier.to(self.cfg.gpu_device)
        self.optim4cls = Adam(self.classifier.parameters(), lr=self.cfg.pre_cls_lr)
       
        self.S = None
        self.S_L = None

    def pretrain_discriminator(self, corpus):
    
        domains = []
        for docs in corpus:
            domains.append(self.id2id(docs))
        corpus = [torch.tensor(docs, device=self.cfg.gpu_device) for docs in domains]

        task_num = len(corpus)
        task_ids = [i for i in range(task_num)]

        pbar = tqdm(range(self.cfg.pre_cls_epochs))
        for i in pbar:        
            pbar.set_description('discriminator pre-training {}/{}'.format(i, self.cfg.pre_cls_epochs))
            
            # choice two tasks at random
            data, label = [], []
            for l, t in enumerate(random.sample(task_ids, 2)):
                idx = torch.randperm(corpus[t].shape[0])
                docs = corpus[t][idx]

                data.append(docs[:self.cfg.S+self.cfg.Q])
                label.append(torch.tensor([l for _ in range(self.cfg.S+self.cfg.Q)], dtype=torch.long, device=self.cfg.gpu_device))

            idx_q = torch.randperm(self.cfg.Q*2)

            S = torch.cat([item[:self.cfg.S] for item in data], dim=0)
            Q = torch.cat([item[self.cfg.S:] for item in data], dim=0)[idx_q]

            S_l = torch.cat([item[:self.cfg.S] for item in label], dim=0)
            Q_l = torch.cat([item[self.cfg.S:] for item in label], dim=0)[idx_q]

            data = torch.cat([S, Q], dim=0)
            label = torch.cat([S_l, Q_l], dim=0)
        
            self.optim4cls.zero_grad()
            loss, acc = self.classifier(data, label)
            loss.backward()
            self.optim4cls.step()

            pbar.set_postfix(acc=acc.item(), loss=loss.item())

    
    def id2id(self, corpus):
        docs = []
        for doc in corpus.tolist():
            words = [self.classifier.vocabulary[self.i2w[w]] for w in doc if w not in (self.w2i['<cls>'], self.w2i['<go>'])]
            docs.append(words + [self.classifier.vocabulary.padding_idx for _ in range(30-len(words))])
        
        return docs

    def dataloader(self, real, fake):
        real = self.id2id(real) * 2
        fake = self.id2id(fake)

        self.idxq = torch.randperm(self.cfg.shot*2)

        self.S = torch.tensor(real[:self.cfg.shot]+fake[:self.cfg.shot], device=self.cfg.gpu_device)
        Q = torch.tensor(real[self.cfg.shot:]+fake[self.cfg.shot:], device=self.cfg.gpu_device)[self.idxq]
        
        self.S_L = torch.tensor([0 for _ in range(self.cfg.shot)] + [1 for _ in range(self.cfg.shot)], dtype=torch.long, device=self.cfg.gpu_device)
        Q_L = torch.tensor([0 for _ in range(self.cfg.shot)] + [1 for _ in range(self.cfg.shot)], dtype=torch.long, device=self.cfg.gpu_device)[self.idxq]
        
        data = torch.cat([self.S, Q], dim=0) 
        label = torch.cat([self.S_L, Q_L], dim=0)

        return data, label

    def train_discriminator(self, corpus):
        data, label = self.dataloader(corpus, self.model.sample(self.classifier.induction.S*2, 30))

        self.optim4cls.zero_grad()
        loss, acc = self.classifier(data, label)
        loss.backward()
        self.optim4cls.step()

        return loss.item(), acc.item()


    def pretrain_generator(self, corpus):
        optim = Adam(lr=self.cfg.lr, params=self.model.parameters())

        train_corpus = torch.cat([docs for docs in corpus])
        data_size = train_corpus.shape[0]
        max_iters = math.ceil(data_size/self.cfg.batch_size)
        
        for epoch in range(self.cfg.epochs):
            idx = torch.randperm(data_size)
            xs = train_corpus[idx]

            pbar = tqdm(range(max_iters))
            total_loss = 0
            for i in pbar:
                pbar.set_description('generator pretraining{}/{}'.format(epoch, self.cfg.epochs))

                offset = i * self.cfg.batch_size
                
                xs_batch = xs[offset:offset+self.cfg.batch_size]
                ts_batch = xs_batch[:,1:]
                
                logit = self.model(xs_batch[:,:-1])
                N, T, D = logit.shape
                loss = F.cross_entropy(logit.reshape(N*T, D), ts_batch.reshape(N*T))

                optim.zero_grad()
                loss.backward()
                optim.step()

                total_loss += loss.item()
                pbar.set_postfix(loss=loss.item())
            
            pbar.set_postfix(loss=total_loss/max_iters)

    def train_generator(self, corpus, R):
        r = torch.rand(1, dtype=torch.float)
        # RL training of generator
        if r < R:
            samples = self.model.sample(32, 30)
            zero = torch.zeros([32, 1], dtype=torch.long, device=self.model.device)
            xs = torch.cat([zero, samples], 1)
            ts = xs[:,1:]

            rewards = self.get_rewards(samples, self.cfg.sample_num, self.cfg.target)
                    
            logit = self.model(xs[:,:-1])
            N, T, D = logit.shape
            loss = build_rec_loss(logit.reshape(N*T, D), ts.reshape(N*T), rewards.reshape(N*T))
        # MLE training of generator
        else:
            xs = corpus[:]
            ts = xs[:,1:]

            logit = self.model(xs[:,:-1])
            N, T, D = logit.shape
            loss = F.cross_entropy(logit.reshape(N*T, D), ts.reshape(N*T))
        
        return loss

    def train(self, corpus: torch.Tensor):
        self.model.train()
        optim = Adam(lr=self.cfg.lr, params=self.model.parameters())

        R = int(self.cfg.epochs * self.cfg.R)
        R = [1] * R + [0] * (self.cfg.epochs - R)
        random.shuffle(R)

        pbar = tqdm(range(self.cfg.epochs))
        for i in pbar:
            pbar.set_description('Processing {}/{}'.format(i, self.cfg.epochs))
            # traning of generator
            optim.zero_grad()
            gloss = self.train_generator(corpus, R[i]) 
            gloss.backward()
            optim.step()
            
            # traning of discriminator
            dloss, acc = self.train_discriminator(corpus)

            pbar.set_postfix(acc=acc, gloss=gloss.item(), dloss=dloss)
    
    def get_rewards(self, samples, numbers):
        batch_size, max_length = samples.shape

        rewards = []
        with torch.no_grad():
            for i in range(numbers):
                for length in range(1, max_length):
                    data = samples[:, 0:length]
                    new_samples = self.model.sample(batch_size, max_length-length, data)
                    new_samples = torch.cat([data, new_samples], dim=1)
                    
                    Q = torch.tensor(self.id2id(new_samples), dtype=torch.long, device=self.cfg.gpu_device)
                    data = torch.cat([self.S, Q], dim=0)
                    pred = self.classifier.fit(data)
                    pred = pred[:,1]

                    if i == 0:
                        rewards.append(pred)
                    else:
                        rewards[length-1] += pred

                Q = torch.tensor(self.id2id(samples), dtype=torch.long, device=self.cfg.gpu_device)
                data = torch.cat([self.S, Q], dim=0)
                pred = self.classifier.fit(data)
                pred = pred[:, 1]

                if i == 0:
                    rewards.append(pred)
                else:
                    rewards[max_length-1] += pred
                
        rewards = torch.stack(rewards) / (1.0 * numbers)
        return rewards.T
