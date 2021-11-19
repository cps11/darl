# the configuration of trainng
gpu_device = 0
epochs = 1
batch_size = 128

# the mode of training methods, inlcuding DARL, MAML, DARL_CNN_CLASSIC, DARL_CNN_TRANSITION, DARL_CNN_FEW_SHOT
mode = 'DARL_CNN_FEW_SHOT'

# the configuration of Generator
n_hidden = 256
n_embed = 300

# source domains
sources = ['beauty', 'pet', 'toys', 'baby', 'food']

# target domain
target = 'phone'

# the configuration of few shot learning
# size of target samples
shot = 5
# query dateset size of
Q = 32
# suport dateset size 
S = shot

# the configuration of MAML
# the meta lr 
meta_lr = 0.003
# the inner lr
inner_lr = 0.003
# the test lr
test_lr = 0.003

# the configuration of DARL
# the pre-traing lr of generator
pre_train_lr = 0.0001
# the traing lr of generator
lr = 0.001
# pre-traing lr of discriminator
pre_cls_lr = 1e-4
# epochs of discriminator pre-training
pre_cls_epochs = 1
# number of Monte Carlo search
sample_num = 16
# the ratio of RL training and MLE training
R = 0.5

# the configuration of DARL with CNN discriminator
# d-steps of discriminator training
d = 1
# epochs in each discriminator training
k = 1
# the meta lr of discriminator trained by classic or transition
cls_lr = 1e-3
# the meta lr of discriminator trained by few-shot learning
cls_meta_lr= 0.001
# the inner lr of discriminator trained by few-shot learning
cls_inner_lr = 0.01