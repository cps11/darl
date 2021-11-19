import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CNNClassifier(nn.Module):
    def __init__(self, k_label, embed_dim, max_seq_len, num_rep, vocab_size, filter_sizes, num_filters, padding_idx, dropout=0.25, weights=None):
        super(CNNClassifier, self).__init__()
        self.k_label = k_label
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.feature_dim = sum(num_filters)
        self.emb_dim_single = int(embed_dim / num_rep)

        self.embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        if weights is not None:
            self.embeddings.weight.data.copy_(torch.from_numpy(weights))

        self.convs = nn.ModuleList([
            nn.Conv2d(1, n, (f, embed_dim)) for (n, f) in zip(num_filters, filter_sizes)
        ]) 
        self.highway = nn.Linear(self.feature_dim, self.feature_dim)
        self.feature2out = nn.Linear(self.feature_dim, 100)
        self.out2logits = nn.Linear(100, k_label) 
        self.dropout = nn.Dropout(dropout)

        self.init_params()

    def forward(self, inp):
        """
        Get logits of discriminator
        :param inp: batch_size * seq_len * vocab_size
        :return logits: [batch_size * num_rep] (1-D tensor)
        """
        emb = self.embeddings(inp).unsqueeze(1)  # batch_size * 1 * max_seq_len * embed_dim

        convs = [F.relu(conv(emb)).squeeze(3) for conv in self.convs]  # [batch_size * num_filter * length]
        pools = [F.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in convs]  # [batch_size * num_filter]
       
        pred = torch.cat(pools, 1)  # batch_size * feature_dim
        highway = self.highway(pred)
        pred = torch.sigmoid(highway) * F.relu(highway) + (1. - torch.sigmoid(highway)) * pred  # highway, same dim

        pred = self.feature2out(self.dropout(pred))
        logits = self.out2logits(self.dropout(pred)).squeeze(1)  # vanilla, batch_size * k_label
       
        return logits
    
    def init_params(self):
        for param in self.parameters():
            if param.requires_grad and len(param.shape) > 0:
                stddev = 1 / math.sqrt(param.shape[0])
                torch.nn.init.normal_(param, std=stddev)


class CNNlm(CNNClassifier):
    def __init__(self, k_label, embed_dim, max_seq_len, num_rep, vocab_size, filter_sizes, num_filters, padding_idx, dropout=0.25, weights=None):
         super(CNNlm, self).__init__(k_label, embed_dim, max_seq_len, num_rep, vocab_size, filter_sizes, num_filters, padding_idx, dropout, weights)
         self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, inp, target):
        logits = super().forward(inp)
        loss = self.criterion(logits, target)
        logits = F.softmax(logits, dim=-1)
        logits = logits.argmax(axis=-1).flatten()
        acc = torch.sum(logits == target).float() / target.shape[0]
        return loss, acc
    
    def fit(self, inp):
        with torch.no_grad():
            logits = super().forward(inp)
            return nn.functional.log_softmax(logits, dim=-1)
