import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import torch.nn as nn
import sys
import pickle as pkl
import networkx as nx
import random

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

def to_sparse(x):
    """ converts dense tensor x to sparse format """
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)

    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return torch.sparse.FloatTensor(indices, values, x.size())

def load_data(path="./data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)


    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def IterGCNLoss(weights, preds, K, K_min):
    values, indices = preds.topk(K+1, dim=1)
    secondary_values = values[:, 1:1+K]
    secondary_values = torch.mean(secondary_values, dim=1)
    min_values, _ = preds.topk(K_min, dim=1, largest=False)
    min_values = torch.mean(min_values, dim=1)
    max_value = values[:, 0]
    return torch.mean((max_value - secondary_values)/(max_value - min_values) * weights)
        
    
def DelTensorElem(arr, index):
    arr1 = arr[0:index]
    arr2 = arr[index+1:]
    return torch.cat((arr1,arr2),dim=0) 

def RemovFromVec(vec1_, vec2_): 
    vec1 = vec1_.detach().clone()
    vec2 = vec2_.copy()
    vec2_num = vec2.shape[0]
    for vec2_idx in range(vec2_num):
        elem = vec2[vec2_idx]
        elem_idx = torch.where(vec1 == elem)[0][0]
        vec1 = DelTensorElem(vec1, elem_idx)
    return vec1    

def SelectPseudoForPoi(GCN_output, ori_idx, ori_labels, selected_num):

    selected_num = selected_num
    GCN_output = GCN_output.detach().clone()
    [confidence_values, col_idx_max] = GCN_output.max(axis=1)
    values_rank, idx_rank = confidence_values.sort(dim = 0, descending = True) 
    idx_rank = RemovFromVec(idx_rank, ori_idx.numpy())
    selected_idx = idx_rank[0:selected_num]   
    confident_u_res = torch.zeros([GCN_output.shape[0], GCN_output.shape[1]]).cuda()
    confident_u_res[ori_idx, ori_labels] = 1
    confident_u_res[selected_idx, :] = GCN_output[selected_idx, :]
    return confident_u_res
            
def KLDivLoss_1(y, x):
    x = F.log_softmax(x)
    y = F.softmax(y, dim=1)
    criterion = nn.KLDivLoss()
    return criterion(x, y)

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def load_data_1(dataset_str, tr_num, val_num):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("./data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("./data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features = normalize(features)
    features = torch.FloatTensor(np.array(features.todense()))
    
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]
    
    labels_vec = labels.argmax(1)
    labels_num = labels_vec.max()+1
    idx_train = []
    idx_val = []
    idx_test = torch.tensor(range(labels.shape[0]))
    for label_idx in range(labels_num):
        pos0 = np.argwhere(labels_vec == label_idx).flatten()
        tr_val_pos = random.sample(pos0.tolist(), tr_num+val_num)
        
        idx_train.append(tr_val_pos[0:tr_num])
        idx_val.append(tr_val_pos[-val_num:])
        
    idx_train = np.array(idx_train).flatten()
    idx_val = np.array(idx_val).flatten()
    idx_test = RemovFromVec(idx_test, idx_train)
    idx_test = RemovFromVec(idx_test, idx_val)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    
    
        

    

    return adj, features, torch.LongTensor(labels_vec), idx_train, idx_val, idx_test

def load_data_std(dataset_str, tr_num, val_num):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("./data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("./data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features = normalize(features)
    features = torch.FloatTensor(np.array(features.todense()))
    
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_test_std = torch.LongTensor(idx_test)
    idx_train = range(len(y))
    idx_train_std = torch.LongTensor(idx_train)
    idx_val = range(len(y), len(y)+500)
    idx_val_std = torch.LongTensor(idx_val)
    

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]
    
    labels_vec = labels.argmax(1)
    labels_num = labels_vec.max()+1
    idx_train = []
    idx_val = []
    idx_test = torch.tensor(range(labels.shape[0]))
    for label_idx in range(labels_num):
        pos0 = np.argwhere(labels_vec == label_idx).flatten()
        tr_val_pos = random.sample(pos0.tolist(), tr_num+val_num)
        
        idx_train.append(tr_val_pos[0:tr_num])
        idx_val.append(tr_val_pos[-val_num:])
        
    idx_train = np.array(idx_train).flatten()
    idx_val = np.array(idx_val).flatten()
    idx_test = RemovFromVec(idx_test, idx_train)
    idx_test = RemovFromVec(idx_test, idx_val)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    
    
        
    
    
    

    return adj, features, torch.LongTensor(labels_vec), idx_train_std, idx_val_std, idx_test_std      


def load_new_dataset_random_1(data_name, tr_num, val_num):
    [adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask] = np.load(data_name, allow_pickle=True)
    pos_adj = np.argwhere(adj==1)
    adj[pos_adj[:,1],pos_adj[:,0]]=1
    
    
    adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    
    
    
    features = normalize(features)
    features = torch.FloatTensor(np.array(features.todense()))
    labels_1 = torch.LongTensor(np.argmax(y_train+ y_val+ y_test, 1))

    labels_vec = np.argmax(y_train+ y_val+ y_test, 1)
    labels_num = labels_1.max()+1
    idx_train = []
    idx_val = []
    idx_test = torch.tensor(range(labels_1.shape[0]))
    
    independent_idx = []
    for i in range(adj.shape[0]):
        if adj[i,i]==1:
            independent_idx.append(i)
    for label_idx in range(labels_num):
        pos0 = np.argwhere(labels_vec == label_idx).flatten()
        while(1):
            tr_val_pos = random.sample(pos0.tolist(), tr_num+val_num)
            list1 = tr_val_pos[0:tr_num]
            for j in list1:
                if j in independent_idx:
                    continue
            break
            
        
        
        
            
        idx_train.append(tr_val_pos[0:tr_num])
        idx_val.append(tr_val_pos[-val_num:])
        
    idx_train = np.array(idx_train).flatten()
    idx_val = np.array(idx_val).flatten()
    idx_test = RemovFromVec(idx_test, idx_train)
    idx_test = RemovFromVec(idx_test, idx_val)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    
    
    
    

    return adj, features, torch.LongTensor(labels_vec), idx_train, idx_val, idx_test



def DelDiagEdgeIndex(edge_index): 
    vec1 = edge_index[0,:] - edge_index[1,:]
    pos1 = torch.where(vec1!=0)
    return edge_index[:, pos1[0]]

def kl_categorical(p_logit, q_logit):
    p = F.softmax(p_logit, dim=-1)
    _kl = torch.sum(p * (F.log_softmax(p_logit, dim=-1)
                                  - F.log_softmax(q_logit, dim=-1)), 1)
    return torch.mean(_kl)

def sim(z1: torch.Tensor, z2: torch.Tensor):
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    return torch.mm(z1, z2.t())

def semi_loss(z1: torch.Tensor, z2: torch.Tensor, tau):
    f = lambda x: torch.exp(x / tau)
    refl_sim = f(sim(z1, z1))
    between_sim = f(sim(z1, z2))
    return -torch.log(
        between_sim.diag()
        / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))    

def contrastive_loss(z1: torch.Tensor, z2: torch.Tensor,
         mean: bool = True):
    h1 = z1
    h2 = z2
    l1 = semi_loss(h1, h2, 0.5)
    l2 = semi_loss(h2, h1, 0.5)

    ret = (l1 + l2) * 0.5
    ret = ret.mean() if mean else ret.sum()

    return ret    