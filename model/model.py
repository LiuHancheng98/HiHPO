import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import norm
import numpy as np
import random
from scipy import sparse as sp
from utils.utils import convert_sp_mat_to_sp_tensor, _convert_sp_mat_to_sp_tensor, norm_adj, build_augmented_adjacency, load_hpo_index


class HiHPO(nn.Module):
    def __init__(self, pro_num, term_num, esm_dim, pub_dim, ontology, device, gcn_layers=1):
        super(HiHPO, self).__init__()
        self.hpo = ontology
        self.device = device
        self.pro_num = pro_num
        self.term_num = term_num
        self.gcn_layers = gcn_layers

        self.dnn_exp = nn.Linear(53, 256)
        self.dnn_esm = nn.Linear(esm_dim, 256)
        self.dnn_ppi = nn.Linear(4919, 256)
        self.dense_pub0 = nn.Linear(pub_dim, 256)
        self.dense_pub1 = nn.Linear(pub_dim, 256)
        self.dense_pub2 = nn.Linear(pub_dim, 256)

        self.bn_exp = norm.BatchNorm(256)
        self.bn_esm = norm.BatchNorm(256)
        self.bn_ppi = norm.BatchNorm(256)
        self.bm_pub0 = norm.BatchNorm(256)
        self.bm_pub1 = norm.BatchNorm(256)
        self.bm_pub2 = norm.BatchNorm(256)

        
        self.drop_rate = 0.1 
        self.temperature = 0.1 
        self.proj_pro = nn.Sequential(  
            nn.Linear(256, 256))
        self.proj_term = nn.Sequential(
            nn.Linear(256, 256))
        self.current_epoch = 0
        self._build_depth_index()
    
    def _build_depth_index(self):
        hpo_to_idx, _ = load_hpo_index('./data/hpo_list.txt')
        max_idx = max(hpo_to_idx.values()) if hpo_to_idx else 0
        self.depth_array = np.ones(max_idx + 1, dtype=np.float32)
        for hpo_id, term in self.hpo.items():
            if hpo_id in hpo_to_idx:
                self.depth_array[hpo_to_idx[hpo_id]] = term.depth

    def corrupt_relation_matrix(self, R):
        R = R.tocoo()
        n_proteins, n_terms = R.shape
        rows, cols, data = R.row, R.col, R.data
        hpo_depths = self.depth_array[cols]
        weights = 1.0 / hpo_depths
        num_edges = len(data)
        num_drop = int(num_edges * self.drop_rate)
        drop_idx = np.random.choice(num_edges, num_drop, replace=False, p=weights/weights.sum())
        # drop_idx = np.random.choice(num_edges, num_drop, replace=False)
        keep_mask = np.ones(num_edges, dtype=bool)
        keep_mask[drop_idx] = False
        new_R = sp.coo_matrix((data[keep_mask], (rows[keep_mask], cols[keep_mask])), 
                            shape=(n_proteins, n_terms))
        return new_R.todok()
    
    def update_augmented_graphs(self, A_original, epoch):
        if self.current_epoch != (epoch + 1):
            self.current_epoch = (epoch + 1)
            R1 = self.corrupt_relation_matrix(A_original)
            self.A_cop = _convert_sp_mat_to_sp_tensor(build_augmented_adjacency(R1))
    
    def forward(self, epoch, pro_idx, hpo_idx, X_exp, X_esm, X_ppi, X_term, A_ppi, A_rel, A_original):

        # original view
        pro_exp, term_exp, pro_esm, term_esm, pro_ppi, term_ppi, pro_exp_cl, pro_esm_cl, pro_ppi_cl, term_exp_cl, term_esm_cl, term_ppi_cl = self._forward_view(
            X_exp, X_esm, X_ppi, X_term, A_ppi, A_rel)

        # augmented view
        self.update_augmented_graphs(A_original, epoch)
        pro_exp1, term_exp1, pro_esm1, term_esm1, pro_ppi1, term_ppi1, pro_exp_cl1, pro_esm_cl1, pro_ppi_cl1, term_exp_cl1, term_esm_cl1, term_ppi_cl1 = self._forward_view(
            X_exp, X_esm, X_ppi, X_term, A_ppi, self.A_cop.to(self.device))
        

        pro_idx_set = torch.unique(torch.Tensor(pro_idx).type(torch.long)).cuda()
        hpo_idx_set = torch.unique(torch.Tensor(hpo_idx).type(torch.long)).cuda()

        intra_cl_loss_pro = (self.InfoNCE_pro(pro_exp[pro_idx_set], pro_exp1[pro_idx_set]) + \
                self.InfoNCE_pro(pro_esm[pro_idx_set], pro_esm1[pro_idx_set]) + \
                self.InfoNCE_pro(pro_ppi[pro_idx_set], pro_ppi1[pro_idx_set]))/3
        intra_cl_loss_term = (self.InfoNCE_term(term_exp[hpo_idx_set], term_exp1[hpo_idx_set]) + \
                self.InfoNCE_term(term_esm[hpo_idx_set], term_esm1[hpo_idx_set]) + \
                self.InfoNCE_term(term_ppi[hpo_idx_set], term_ppi1[hpo_idx_set]))/3
        
        return pro_exp, term_exp, pro_esm, term_esm, pro_ppi, term_ppi, pro_exp_cl, pro_esm_cl, pro_ppi_cl, (intra_cl_loss_pro + intra_cl_loss_term)/2
    
    def _forward_view(self, X_exp, X_esm, X_ppi, X_term, A_ppi, A_rel):
        # protein embeddings
        pro_exp = F.leaky_relu(self.dnn_exp(X_exp))
        pro_exp = self.bn_exp(pro_exp)
        pro_esm = F.leaky_relu(self.dnn_esm(X_esm))
        pro_esm = self.bn_esm(pro_esm)
        pro_ppi = F.leaky_relu(self.dnn_ppi(X_ppi))
        pro_ppi = self.bn_ppi(pro_ppi)

        # HPO embeddings
        X_pub0 = F.leaky_relu(self.dense_pub0(X_term))
        X_pub0 = self.bm_pub0(X_pub0)
        X_pub1 = F.leaky_relu(self.dense_pub1(X_term))
        X_pub1 = self.bm_pub1(X_pub1)
        X_pub2 = F.leaky_relu(self.dense_pub2(X_term))
        X_pub2 = self.bm_pub2(X_pub2)

        pro_exp_final, term_exp_final = self._propagate(pro_exp, X_pub0, A_rel)
        pro_esm_final, term_esm_final = self._propagate(pro_esm, X_pub1, A_rel)
        pro_ppi_final1, term_ppi_final = self._propagate(pro_ppi, X_pub2, A_rel)
        pro_ppi_final = A_ppi.matmul(pro_ppi_final1)
        
        return pro_exp_final, term_exp_final, pro_esm_final, term_esm_final, pro_ppi_final, term_ppi_final, pro_exp, pro_esm, pro_ppi, X_pub0, X_pub1, X_pub2
    
    def _propagate(self, pro_emb, term_emb, A_rel):
        ego_embeddings = torch.cat([pro_emb, term_emb], 0)
        all_embeddings = []
        for k in range(self.gcn_layers):
            ego_embeddings = torch.sparse.mm(A_rel, ego_embeddings)
            all_embeddings.append(ego_embeddings)
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        pro_final = all_embeddings[:self.pro_num]
        term_final = all_embeddings[self.pro_num:]
        return pro_final, term_final
    
    def InfoNCE_pro(self, view1, view2, b_cos: bool = True):
        """
        Args:
            view1: (torch.Tensor - N x D)
            view2: (torch.Tensor - N x D)
            b_cos (bool)

        Return: Average InfoNCE Loss
        """
        view1 = self.proj_pro(view1)
        view2 = self.proj_pro(view2)
        if b_cos:
            view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)

        pos_score = (view1 @ view2.T) / self.temperature
        score = torch.diag(F.log_softmax(pos_score, dim=1))
        return -score.mean()
    
    def InfoNCE_term(self, view1, view2, b_cos: bool = True):
        """
        Args:
            view1: (torch.Tensor - N x D)
            view2: (torch.Tensor - N x D)
            b_cos (bool)

        Return: Average InfoNCE Loss
        """
        view1 = self.proj_term(view1)
        view2 = self.proj_term(view2)
        if b_cos:
            view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)

        pos_score = (view1 @ view2.T) / self.temperature
        score = torch.diag(F.log_softmax(pos_score, dim=1))
        return -score.mean()