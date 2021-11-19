from model import Generator
from ar import Ar
import torch
import datetime
import os
import config as opt
from maml import Maml
from darl import Darl
from darlCNN import (
    DarlCNN,
    DarlMamlCNN
)

def bulid_model_path(opt):
    now = datetime.datetime.now().strftime('%m%d')
    root = os.path.split(os.path.realpath(__file__))[0]

    filename = '{}-{}-{}-{}-{}'.format(opt.mode, '_'.join(opt.sources), opt.epochs, opt.target, opt.shot)
    if opt.mode == 'MAML':
        filename = '{}-{}-{}'.format(filename, opt.meta_lr, opt.inner_lr)
    elif opt.mode == 'DARL':
        filename = '{}-{}-{}'.format(filename, opt.pre_train_lr, opt.lr)
    
    filename = '{}-{}.pth'.format(filename, now)
    
    path = os.path.join(root, 'model')
    if not os.path.exists(path):
        os.makedirs(path)

    return os.path.join(path, filename)


if __name__ == '__main__':
    # load dataset and vocabulary
    dataset = Ar(opt.target)
    word_to_id, id_to_word = dataset.load_vocab()

    opt.gpu_device = torch.device("cuda:{}".format(opt.gpu_device) if torch.cuda.is_available() else "cpu")
    n_vocab = len(word_to_id)
    
    # load k-shot target domain
    corpus = dataset.load_target(opt.target, opt.shot)
    target_corpus = torch.tensor(corpus, device=opt.gpu_device)

    # load source domains
    corpus = dataset.load_sources(opt.sources)
    source_corpus = [torch.tensor(docs, device=opt.gpu_device) for docs in corpus]
    
    # initialize the generator
    model = Generator(n_vocab, opt.n_embed, opt.n_hidden, opt.gpu_device)
    model.to(opt.gpu_device)

    if opt.mode == 'MAML':
        tool = Maml(model, opt)
        tool.meta_train(source_corpus)
        tool.meta_test(target_corpus)
        checkpoint = model
    elif opt.mode == 'DARL':
        tool = Darl(model, opt, id_to_word, word_to_id)
        # pretrain generator on source domains
        tool.pretrain_generator(source_corpus)
        # pretrain discriminator on source domains
        tool.pretrain_discriminator(source_corpus)
        # pretrain discriminator on target domains
        tool.train_discriminator(target_corpus)
        # train generator and discriminator 
        tool.train(target_corpus)
        checkpoint = {
            'genertor': model,
            'discriminator': {
                'model': tool.classifier,
                'S': tool.S,
                'label': tool.S_L
            }
        }
    elif opt.mode in {'DARL_CNN_CLASSIC', 'DARL_CNN_TRANSITION'}:
        tool = DarlCNN(model, opt, id_to_word, word_to_id)
        if opt.mode == 'DARL_CNN_CLASSIC':
            pre_dis_corpus = torch.tensor(dataset.load_sources([opt.target])[0], device=opt.gpu_device)
        else:
            pre_dis_corpus = target_corpus
        # pretrain generator on source domains
        tool.pretrain_generator(source_corpus)
        # pretrain discriminator on source domains
        tool.pretrain_discriminator(pre_dis_corpus)
        # train generator and discriminator 
        tool.train(target_corpus, pre_dis_corpus)
        checkpoint = {
            'genertor': model,
            'discriminator': tool.classifier
        }
    elif opt.mode == 'DARL_CNN_FEW_SHOT':
        tool = DarlMamlCNN(model, opt, id_to_word, word_to_id)
        # pretrain generator on source domains
        tool.pretrain_generator(source_corpus)
        # pretrain discriminator on source domains
        tool.pretrain_discriminator(source_corpus)
        # pretrain discriminator on target domains
        tool.train_discriminator(target_corpus)
        # train generator and discriminator 
        tool.train(target_corpus, target_corpus)
        checkpoint = {
            'genertor': model,
            'discriminator': tool.classifier
        }
    else:
        raise ValueError('undefined training mode: {}'.format(opt.mode))

    # save model
    path = bulid_model_path(opt)
    torch.save(checkpoint, path)
