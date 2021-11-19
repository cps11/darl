import torch
from torch.optim import Adam, SGD
from tqdm import tqdm
import copy
import random
from tqdm import tqdm
from classifiers import CNNlm
from darl import Darl


class DarlCNN(Darl):
    def __init__(self, model, cfg, i2w, w2i) -> None:
        self.model = model
        self.cfg = cfg
        self.i2w = i2w
        self.w2i = w2i
    
        self.classifier = CNNlm(2, 32, 30, 64, len(self.w2i), [2, 3, 4, 5], [200], self.w2i['<cls>'])
        self.classifier.to(self.cfg.gpu_device)
        self.optim4cls = Adam(self.classifier.parameters(), lr=self.cfg.cls_lr)

    def dataloader(self, real, fake):
        data = torch.cat([fake, real[:, 1:-1]], dim=0) 
        labels = torch.tensor([1 for _ in range(fake.shape[0])] + [0 for _ in range(real.shape[0])], dtype=torch.long, device=self.cfg.gpu_device)

        return data, labels

    def pretrain_discriminator(self, postive, verbose=True):
        
        total_acc = 0
        total_loss = 0

        pbar = tqdm(range(self.cfg.d))
        for i in pbar:
            if verbose:
                pbar.set_description('discriminator pretraining {}/{}'.format(i, self.cfg.d))
            
            size = postive.shape[0]
            data, labels = self.dataloader(postive, self.model.sample(size, 30))

            d_loss = 0
            d_acc = 0
            for _ in range(self.cfg.k):
                idx = torch.randperm(size*2)
                data = data[idx]
                labels = labels[idx]

                count = 0
                b_loss = 0
                b_acc = 0
                for j in range(0, size*2, self.cfg.batch_size):
                    loss, acc  = self.classifier(data[j:j+self.cfg.batch_size], labels[j:j+self.cfg.batch_size])
                    self.optim4cls.zero_grad()
                    loss.backward()
                    self.optim4cls.step()
                    
                    count += 1
                    b_loss += loss.item()
                    b_acc += acc.item()

                b_loss /= count
                b_acc /= count
                
                d_loss += b_loss
                d_acc += b_acc
                if verbose:
                    pbar.set_postfix(loss=b_loss, acc=b_acc)
            
            d_loss /= self.cfg.k
            d_acc /= self.cfg.k
                
            total_loss += d_loss
            total_acc += d_loss
            if verbose:
                pbar.set_postfix(loss=d_loss, acc=d_loss)
        
        return total_loss, total_acc

    
    def train_discriminator(self, corpus):
        return self.pretrain_discriminator(corpus, False)


    def train(self, c4g, c4d):
        """
        c4g: corpus for generator
        c4d: corpus for discriminator
        """
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
            gloss = self.train_generator(c4g, R[i]) 
            gloss.backward()
            optim.step()
            
            # traning of discriminator
            dloss, acc = self.train_discriminator(c4d)

            pbar.set_postfix(acc=acc, gloss=gloss.item(), dloss=dloss)
    

class DarlMamlCNN(DarlCNN):
    def __init__(self, model, cfg, i2w, w2i) -> None:
        super().__init__(model, cfg, i2w, w2i)
        # delete the optimizer
        delattr(self, 'optim4cls')
        
        self.meta_optim4cls = Adam(self.classifier.parameters(), lr=self.cfg.cls_meta_lr)
        self.task_optim4cls = SGD(self.classifier.parameters(), lr=self.cfg.cls_inner_lr)

        self.cls_meta_loss = None
        self.state_dict = None
    
    def pretrain_discriminator(self, corpus):
        task_nums = len(corpus)
        task_ids = []
        for i in range(task_nums-1):
            for j in range(i, task_nums):
                task_ids.append((i, j))

        epochs = max(int(self.cfg.pre_cls_epochs / task_nums), 1)
        pbar = tqdm(range(epochs))
        for i in pbar:
            pbar.set_description('discriminator pre-training {}/{}'.format(i, epochs))
            state_dict = copy.deepcopy(self.classifier.state_dict())

            mloss, tloss = 0, 0
            macc, tacc = 0, 0
            count = 0 
            for ids in task_ids:
                count += 1
                self.classifier.load_state_dict(state_dict)

                data = []
                label = []
                
                for l, t in enumerate(ids):
                    idx = torch.randperm(corpus[t].shape[0])
                    docs = corpus[t][idx]

                    data.append(docs[:self.cfg.S+self.cfg.Q])
                    label.append(torch.tensor([l for _ in range(self.cfg.S+self.cfg.Q)], dtype=torch.long, device=self.cfg.gpu_device))

                idx_s = torch.randperm(self.cfg.S*2)
                idx_q = torch.randperm(self.cfg.Q*2)

                S = torch.cat([item[:self.cfg.S] for item in data], dim=0)[idx_s]
                Q = torch.cat([item[self.cfg.S:] for item in data], dim=0)[idx_q]

                S_l = torch.cat([item[:self.cfg.S] for item in label], dim=0)[idx_s]
                Q_l = torch.cat([item[self.cfg.S:] for item in label], dim=0)[idx_q]
            
                self.task_optim4cls.zero_grad()
                loss, acc = self.classifier(S, S_l)
                loss.backward()
                self.task_optim4cls.step()
                tloss += loss.item()
                tacc += acc.item()
                
                loss, acc = self.classifier(Q, Q_l)
                mloss += loss
                macc += acc.item()

            self.classifier.load_state_dict(state_dict)
            self.meta_optim4cls.zero_grad()
            loss = mloss / count
            loss.backward()
            self.meta_optim4cls.step()

            pbar.set_postfix(task_acc=tacc/count, task_loss=tloss/count, meta_acc=macc/count, meta_loss=loss.item())
        
        self.state_dict = copy.deepcopy(self.classifier.state_dict())
    

    def train_discriminator(self, corpus):
        accs, dloss = 0, 0

        if self.cls_meta_loss is not None:
            self.classifier.load_state_dict(self.state_dict)
            self.meta_optim4cls.zero_grad()
            self.cls_meta_loss.backward()
            self.meta_optim4cls.step()

            self.state_dict = copy.deepcopy(self.classifier.state_dict())

        for _ in range(self.cfg.d):
            S, S_l = self.dataloader(corpus, self.model.sample(self.cfg.shot, 30))
            Q, Q_l = self.dataloader(corpus, self.model.sample(self.cfg.shot, 30))
            mloss = 0
            tloss = 0
            macc = 0
            for _ in range(self.cfg.k):
                self.classifier.load_state_dict(self.state_dict)
                idx = torch.randperm(self.cfg.shot*2)

                self.task_optim4cls.zero_grad()
                loss, _ = self.classifier(S[idx], S_l[idx])
                loss.backward()
                self.task_optim4cls.step()
                tloss += loss

                loss, acc = self.classifier(Q[idx], Q_l[idx])
                macc += acc
                mloss += loss
            
            self.loss = mloss
            accs += macc/self.cfg.k
            dloss += mloss.item()/self.cfg.k
        
        return dloss/self.cfg.d, accs/self.cfg.d


