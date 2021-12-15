import torch
import numpy as np
import scipy.sparse as sparse
import graphlearning as gl
import math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
from scipy import sparse
from torch_geometric.nn import GATConv
from torch.nn import Parameter, Linear, LeakyReLU, Dropout
from inits_gat import glorot, zeros
from utils import kl_categorical

        
        
        

class PoissonModel(torch.nn.Module):
    def __init__(self, fea, nhidden, edge_indices_no_diag, idx_train, labels, alpha, adj, dropout, T):
        super(PoissonModel, self).__init__()    
        
        self.edge_indices_no_diag = edge_indices_no_diag
        self.in_features = fea.shape[1]
        self.out_features = nhidden
        self.num_classes = max(labels)+1
        self.W = Linear(self.in_features, self.out_features, bias=False)
        self.a = Parameter(torch.Tensor(2*self.out_features, 1))
        self.W1 = Linear(self.in_features, self.num_classes, bias=False)
        self.I = idx_train
        self.g = labels[idx_train]
        self.num1 = fea.shape[0]
        self.features = fea
        self.leakyrelu = LeakyReLU(alpha)
        self.isadj = adj[0]
        self.adj = adj[1]
        self.dropout = dropout
        self.T1 = T
        self.tmp = []
        
        
        self.reset_parameters()
        
    def reset_parameters(self):
        glorot(self.W.weight)
        glorot(self.a)
        glorot(self.W1.weight)

        
    def forward(self, h):        
        Wh = self.W(h)
        
        self.A_ds_no_diag = self.CalAttenA(Wh)
        if self.isadj:
            print('popisson')
        
        return self.PoiConv()
        
        
        
    def PoiConv(self):
        Kg = torch.zeros([self.num_classes, self.num1]).cuda()
        Kg[self.g, self.I] = 1
        c = torch.sum(Kg, axis=1)/len(self.I)
        b = torch.transpose(Kg, 0, 1)
        b[self.I,:] = b[self.I,:]-c
        self.tmp = b
        
        D = self.A_ds_no_diag + 1e-10*torch.eye(self.num1).cuda()
        D = torch.sum(D, 1)**-1
        D = torch.diag(D)
        
        P = torch.mm(D, self.A_ds_no_diag.t())
        Db = torch.mm(D, b)
    
        
        ut = torch.zeros([self.num1, self.num_classes]).cuda()
        T = 0
        while T < self.T1:
            ut = torch.mm(P,ut)+Db
            T = T+1
            if not self.isadj:
                if T == self.T1-3:    
                    ut = ut + self.W1(self.features)
                    ut = F.dropout(ut, self.dropout, training=self.training)
  
                
        
        return ut
        
        
    def CalAttenA(self, Wh):
        indices = self.edge_indices_no_diag.clone()
        fea1 = Wh[indices[0,:],:]
        fea2 = Wh[indices[1,:],:]
        fea12 = torch.cat((fea1, fea2), 1)
        atten_coef = torch.exp(self.leakyrelu(torch.mm(fea12, self.a))).flatten()
        A_atten = torch.zeros([self.num1, self.num1]).cuda()
        A_atten[indices[0,:],indices[1,:]] = atten_coef
        s1 = A_atten.sum(1)
        pos1 = torch.where(s1==0)[0]
        A_atten[pos1, pos1] = 1
        A_atten = A_atten.t()/A_atten.sum(1)
        return A_atten.t()
    
    def DiagMatMulA(self, diag_ind, diag_values, indices, atten_coef, size_12): 
        row_idx_edge = indices[0, :].clone()
        vec1 = torch.zeros([row_idx_edge.shape[0]]).cuda()
        for row_idx in range(self.num1):
            pos0 = torch.where(row_idx_edge==row_idx)[0]
            pos1 = torch.where(diag_ind==row_idx)[0]
            vec1[pos0] = diag_values[pos1]
        return torch.sparse.FloatTensor(indices, atten_coef*vec1, size_12)
        
    
    
    def GetSpIdentity(self, size_n):
        mat1 = torch.eye(size_n)
        indices = torch.nonzero(mat1).t()
        values = mat1[indices[0], indices[1]]
        return torch.sparse.FloatTensor(indices, values, mat1.size())
        
    def torch_sparse(self, A):

        A = A.tocoo()
        values = A.data
        indices = np.vstack((A.row, A.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = A.shape
        return torch.sparse.FloatTensor(i, v, torch.Size(shape))
        
        
        
        
        
class GCN(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x
    
class GATModel(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GATModel, self).__init__()

        self.gc1 = GATConv(in_channels = nfeat, out_channels = nhid, dropout = dropout)
        self.gc2 = GATConv(in_channels = nhid, out_channels = nclass, dropout = dropout)
         
        self.dropout = dropout

    def forward(self, x, edge_index): 
        x = F.relu(self.gc1(x, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, edge_index)
        return F.log_softmax(x, dim=1)
    
class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)



    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
