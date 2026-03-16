import numpy as np
from torch_sparse import SparseTensor
from torch_sparse import fill_diag, mul
from torch_sparse import sum as sparsesum
import torch
from scipy import sparse as sp

def rwr(A, restart_prob):
    """
    Random Walk with Restart (RWR) on similarity network.
    :param A: n x n, similarity matrix
    :param restart_prob: probability of restart
    :return: n x n, steady-state probability
    """
    A = A.toarray()
    n = A.shape[0]
    A = (A + A.T) / 2
    A = A - np.diag(np.diag(A))
    A = A + np.diag(sum(A) == 0)
    P = A / sum(A)
    Q = np.linalg.inv(np.eye(n) - (1 - restart_prob) * P) @ (restart_prob * np.eye(n))
    return Q

def convert_sp_mat_to_sp_tensor(X):
    return SparseTensor.from_scipy(X)

def _convert_sp_mat_to_sp_tensor(X):
    coo = X.tocoo()
    i = torch.LongTensor([coo.row, coo.col])
    v = torch.from_numpy(coo.data).float()
    return torch.sparse.FloatTensor(i, v, coo.shape)

def norm_adj(adj_t):
    deg = sparsesum(adj_t, dim=1)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
    adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))
    adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
    return adj_t

def InfoNCE(view1, view2, temperature: float, b_cos: bool = True):
    """
    Args:
        view1: (torch.Tensor - N x D)
        view2: (torch.Tensor - N x D)
        temperature: float
        b_cos (bool)

    Return: Average InfoNCE Loss
    """
    if b_cos:
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)

    pos_score = (view1 @ view2.T) / temperature
    score = torch.diag(F.log_softmax(pos_score, dim=1))
    return -score.mean()

def bpr_loss(user_emb, pos_item_emb, neg_item_emb, n_negs=1):
    # user_emb: (batch_size, embedding_dim)
    # pos_item_emb: (batch_size, embedding_dim)
    # neg_item_emb: (batch_size * n_negs, embedding_dim)
    
    # reshape (batch_size, n_negs, embedding_dim)
    neg_item_emb = neg_item_emb.view(-1, n_negs, neg_item_emb.size(-1))
    
    # (batch_size,)
    pos_score = torch.mul(user_emb, pos_item_emb).sum(dim=1)
    
    # user_emb.unsqueeze(1): (batch_size, 1, embedding_dim)
    # neg_item_emb: (batch_size, n_negs, embedding_dim)
    neg_scores = torch.mul(user_emb.unsqueeze(1), neg_item_emb).sum(dim=2)
    loss = -torch.log(10e-6 + torch.sigmoid(pos_score.unsqueeze(1) - neg_scores))
    
    return loss.mean()


def l2_reg_loss(reg, *args):
    emb_loss = 0
    for emb in args:
        emb_loss += torch.norm(emb, p=2)/emb.shape[0]
    return emb_loss * reg

def build_augmented_adjacency(R_augmented):
    """从破坏后的R构建增广邻接矩阵并归一化"""
    n_proteins, n_terms = R_augmented.shape
    adj_mat = sp.dok_matrix((n_proteins + n_terms, n_proteins + n_terms), dtype=np.float32)
    adj_mat = adj_mat.tolil()
    R_augmented = R_augmented.tolil()
    # 构建增广邻接矩阵
    adj_mat[:n_proteins, n_proteins:] = R_augmented
    adj_mat[n_proteins:, :n_proteins] = R_augmented.T
    
    # 添加自环
    adj_mat = adj_mat + sp.eye(adj_mat.shape[0])
    adj_mat = adj_mat.todok()
    # 对称归一化
    rowsum = np.array(adj_mat.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    
    norm_adj = d_mat_inv_sqrt.dot(adj_mat).dot(d_mat_inv_sqrt)
    return norm_adj.tocsr()

def load_hpo_index(hpo_index_file):
    hpo_to_idx = {}
    idx_to_hpo = {}
    with open(hpo_index_file) as f:
        for line in f:
            hpo_id, idx = line.strip().split('\t')
            idx = int(idx)
            hpo_to_idx[hpo_id] = idx
            idx_to_hpo[idx] = hpo_id
    return hpo_to_idx, idx_to_hpo