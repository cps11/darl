
import torch
from torch.optim import Adam
import torch.nn.functional as F
from tqdm import tqdm
import copy
import math


class Maml(object):
    def __init__(self, model, cfg) -> None:
        self.model = model
        self.cfg = cfg
    
    def meta_train(self, corpus):
        self.model.train()
        
        # the number of source domains
        task_num = len(corpus)
        # the number of samples in each source domain
        data_size = corpus[0].shape[0]
        
        max_iters = math.ceil(data_size/self.cfg.batch_size)
        
        for epoch in range(self.cfg.epochs):

            meta_optim = Adam(lr=self.cfg.meta_lr, params=self.model.parameters())
            optim = Adam(lr=self.cfg.inner_lr, params=self.model.parameters())

            idx = torch.randperm(data_size)
            xs = [docs[idx] for docs in corpus]

            init_state = copy.deepcopy(self.model.state_dict())

            pbar = tqdm(range(max_iters))
            for i in pbar:
                offset = i * self.cfg.batch_size
                
                total_loss = 0
                task_loss = 0
                meta_loss = 0
                pbar.set_description('meta training {}/{}'.format(epoch, self.cfg.epochs))
                for k in range(task_num):
                    
                    self.model.load_state_dict(init_state)
                    optim.zero_grad()

                    xs_batch = xs[k][offset:offset+self.cfg.batch_size]
                    ts_batch = xs_batch[:,1:]
                    
                    # update parameters for each task
                    logit = self.model(xs_batch[:,:-1])
                    N, T, D = logit.shape
                    loss = F.cross_entropy(logit.reshape(N*T, D), ts_batch.reshape(N*T))
                    loss.backward()
                    optim.step()

                    task_loss += loss.item()
                
                    # loss for the meta-update   
                    logit = self.model(xs_batch[:,:-1])
                    N, T, D = logit.shape
                    loss = F.cross_entropy(logit.reshape(N*T, D), ts_batch.reshape(N*T))
                    total_loss = total_loss + loss

                    meta_loss += loss.item()
        
                pbar.set_postfix(task_loss=task_loss/task_num, meta_loss=meta_loss/task_num)
                
                self.model.load_state_dict(init_state)

                # update parameters for meta learner
                meta_optim.zero_grad()
                loss = total_loss / task_num
                loss.backward()
                meta_optim.step()

                init_state = copy.deepcopy(self.model.state_dict())
    
    def meta_test(self, corpus):
        self.model.train()

        # the number of samples in target domain
        data_size = corpus.shape[0]
        max_iters = math.ceil(data_size/self.cfg.batch_size)

        optim = Adam(lr=self.cfg.test_lr, params=self.model.parameters())

        pbar = tqdm(range(self.cfg.epochs))
        for i in pbar:
            pbar.set_description('meta testing {}/{}'.format(i, self.cfg.epochs))
        
            idx = torch.randperm(data_size)
            xs = corpus[idx]
            
            ave_loss = 0
            for i in range(max_iters):
                offset = i * self.cfg.batch_size
                
                optim.zero_grad()
                    
                xs_batch = xs[offset:offset+self.cfg.batch_size]
                ts_batch = xs_batch[:,1:]

                logit = self.model(xs_batch[:,:-1])                
                N, T, D = logit.shape
                loss = torch.nn.functional.cross_entropy(logit.reshape(N*T, D), ts_batch.reshape(N*T))
                loss.backward()
                optim.step()

                ave_loss += loss.item()
            
            pbar.set_postfix(loss=ave_loss/max_iters)
