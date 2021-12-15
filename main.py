# -*- coding: utf-8 -*-

from Model import PoissonModel, GCN, GATModel
import numpy as np
import torch
from utils import  load_data_1, DelDiagEdgeIndex, kl_categorical, contrastive_loss
import torch.nn.functional as F
import random
import itertools
import scipy.io as scio
import sys




weight_decay = 1e-2 
epochs = 500  
learning_rate = 0.1 
hidden_num = 128 
dropout = 0.8 


eta = 0.9 
alpha_contra = 1.3 
T2=10 

seed = 12345
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)

adj, features, labels_1, idx_train, idx_val, idx_test = load_data_1('cora', 1, 20)
num_classes = int(labels_1.max())+1
edge_index = adj.coalesce().indices()
edge_index = DelDiagEdgeIndex(edge_index) 






Poimodel = PoissonModel(fea=features.cuda(), 
                        nhidden=hidden_num, 
                        edge_indices_no_diag=edge_index.cuda(),
                        idx_train=idx_train.cuda(), 
                        labels=labels_1.cuda(),
                        alpha=0.1,
                        adj=[False, 1],
                        dropout = dropout,
                        T=T2)
GCNmodel = GCN(nfeat=features.shape[1],
               nhid=hidden_num ,
               nclass=num_classes,
               dropout=dropout)
Poimodel.cuda()
GCNmodel.cuda()

idx_val_tes = torch.cat([idx_val, idx_test])

optimizer = torch.optim.Adam(itertools.chain(Poimodel.parameters(), GCNmodel.parameters()), lr=learning_rate, weight_decay=weight_decay)
max_val = 0
max_test = 0
for epoch in range(1, epochs+1):
    Poimodel.train()
    GCNmodel.train()
    optimizer.zero_grad()
    output_Poi = Poimodel(features.cuda())
    output_GCN = GCNmodel(features.cuda(), Poimodel.A_ds_no_diag)
    
    loss = eta*F.nll_loss(F.log_softmax(output_Poi[idx_train,:], dim=1), labels_1[idx_train].cuda().long())
    loss += F.nll_loss(F.log_softmax(output_GCN[idx_train,:], dim=1), labels_1[idx_train].cuda().long())
    loss += kl_categorical(output_Poi[idx_val_tes], output_GCN[idx_val_tes])
    loss += alpha_contra*contrastive_loss(output_Poi[idx_val_tes], output_GCN[idx_val_tes])
    
    loss.backward()
    optimizer.step()
    
    Poimodel.eval()
    GCNmodel.eval()
    output_Poi = Poimodel(features.cuda())
    output_GCN = GCNmodel(features.cuda(), Poimodel.A_ds_no_diag)
    preds_2 = torch.argmax(output_Poi, dim=1)
    acc_2 = torch.sum(preds_2[idx_test] == labels_1.cuda()[idx_test]).float() / labels_1[idx_test].shape[0]
    acc_val = torch.sum(preds_2[idx_val] == labels_1.cuda()[idx_val]).float() / labels_1[idx_val].shape[0]
    if acc_val>max_val:
        max_val = acc_val
        max_test = acc_2.detach().clone().cpu().numpy()

scio.savemat('max_test.mat',{'max_test':np.array(max_test)})
print(max_test)







        
        
 