import numpy as np
import random as rd
import scipy.sparse as sp
import os
from random import shuffle, sample
from logzero import logger
import torch
import pickle
import networkx as nx
from utils.utils import rwr
import pandas as pd

root_terms = {'pa': 'HP:0000118'}
class Data(object):
    def __init__(self, path, batch_size):
        self.path = path
        self.batch_size = batch_size

        train_file = f'./data/{path}/train.txt'
        test_file = f'./data/{path}/test.txt'

        # get number of proteins and terms
        self.n_proteins, self.n_terms = 0, 0
        self.n_train, self.n_test = 0, 0
        neg_pools = {}

        self.exist_proteins = []

        # get number of proteins and terms from training set
        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    terms = [int(l[1])]
                    uid = int(l[0])
                    self.exist_proteins.append(uid)
                    self.n_terms = max(self.n_terms, max(terms))
                    self.n_proteins = max(self.n_proteins, uid)
                    self.n_train += len(terms)
        # get number of proteins and terms from test set
        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    terms = [int(l[1])]
                    # print(terms)
                    self.n_terms = max(self.n_terms, max(terms))
                    self.n_test += len(terms)
        self.n_terms += 1
        self.n_proteins += 1
        logger.info('n_proteins=%d, n_terms=%d' % (self.n_proteins, self.n_terms))
        logger.info('n_interactions=%d' % (self.n_train + self.n_test))
        logger.info('n_train=%d, n_test=%d, sparsity=%.5f' % (self.n_train, self.n_test, (self.n_train + self.n_test)/(self.n_proteins * self.n_terms)))

        self.R = sp.dok_matrix((self.n_proteins, self.n_terms), dtype=np.float32)

        self.train_terms, self.test_set = {}, {}
        with open(train_file) as f_train:
            with open(test_file) as f_test:
                for l in f_train.readlines():
                    if len(l) == 0:
                        break
                    l = l.strip('\n')
                    terms = [int(i) for i in l.split(' ')]
                    uid, train_uid_term = terms[0], terms[1]
                    if uid not in self.train_terms:
                        self.train_terms[uid] = []
                    self.train_terms[uid].append(train_uid_term)

                for uid in self.train_terms:
                    for i in self.train_terms[uid]:
                        self.R[uid, i] = 1.0

                for l in f_test.readlines():
                    if len(l) == 0: break
                    l = l.strip('\n')
                    try:
                        terms = [int(i) for i in l.split(' ')]
                    except Exception:
                        continue

                    uid, test_uid_term = terms[0], terms[1]
                    if uid not in self.test_set:
                        self.test_set[uid] = []
                    self.test_set[uid].append(test_uid_term)

        self.training_data = []
        for user in self.train_terms:
            for item in self.train_terms[user]:
                self.training_data.append((user, item))
        self.all_neg_pools = {}
        for protein in self.train_terms:
            self.all_neg_pools[protein] = list(set(range(self.n_terms)) - set(self.train_terms[protein]))
    
    def next_batch_pairwise(self, n_negs=1):
        shuffle(self.training_data)
        ptr = 0
        data_size = len(self.training_data)
        while ptr < data_size:
            if ptr + self.batch_size < data_size:
                batch_end = ptr + self.batch_size
            else:
                batch_end = data_size
            proteins = [self.training_data[idx][0] for idx in range(ptr, batch_end)]
            terms = [self.training_data[idx][1] for idx in range(ptr, batch_end)]
            ptr = batch_end
            u_idx, i_idx, j_idx = [], [], []
            term_list = list(range(self.n_terms))
            for i, protein in enumerate(proteins):
                i_idx.append(terms[i])
                u_idx.append(protein)
                neg_pool = self.all_neg_pools[protein]
                neg_samples = sample(neg_pool, n_negs)
                j_idx.extend(neg_samples)
            yield u_idx, i_idx, j_idx

    def create_adj_mat(self):
        # create adjacency matrix for proteins and terms, (n_proteins + n_terms) * (n_proteins + n_terms)
        adj_mat = sp.dok_matrix((self.n_proteins + self.n_terms, self.n_proteins + self.n_terms), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.R.tolil()

        adj_mat[:self.n_proteins, self.n_proteins:] = R
        adj_mat[self.n_proteins:, :self.n_proteins] = R.T
        adj_mat = adj_mat.todok()

        def mean_adj_single(adj):
            # D^-1 * A
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            # norm_adj = adj.dot(d_mat_inv)
            logger.info('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()

        def normalized_adj_single(adj):
            # D^-1/2 * A * D^-1/2
            rowsum = np.array(adj.sum(1))

            d_inv_sqrt = np.power(rowsum, -0.5).flatten()
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
            d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
            bi_lap = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
            return bi_lap.tocoo()
        norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0])).tocsr()
        return norm_adj_mat, R
    
    def load_protein_embedding(self):
        protein2idx = {}
        with open('./data/pro_list.txt', 'r') as f:
            for line in f:
                protein, idx = line.strip().split('\t')
                protein2idx[protein] = int(idx)
        #load ppi mat
        ppi_mat_path = f'./data/{self.path}/ppi_mat.npz'
        if os.path.exists(ppi_mat_path):
            ppi_mat = sp.load_npz(ppi_mat_path)
        else:
            ppi_mat = sp.dok_matrix((self.n_proteins, self.n_proteins), dtype=np.float32) # 用字典的方式获取ppi矩阵
            with open('./data/network.v12.0.txt', 'r') as f:
                for line in f:
                    protein1, protein2, score = line.strip().split(' ')
                    pid1 = protein2idx.get(protein1, -1)
                    pid2 = protein2idx.get(protein2, -1)
                    if pid1!= -1 and pid2!= -1:
                        ppi_mat[pid1, pid2] = float(score)
                        ppi_mat[pid2, pid1] = float(score)
            #self-loop
            for i in range(self.n_proteins):
                ppi_mat[i,i] = 1.0
            sp.save_npz(ppi_mat_path, ppi_mat.tocsr())

        # protein esm embedding
        esm_embedding_dic = pickle.load(open('./data/protein_embeddings_esm2_3B_36.pkl', 'rb'))
        esm_embedding = [esm_embedding_dic[uid] for uid, pid in sorted(protein2idx.items(), key=lambda x:x[1])]
        protein_esm_embedding = torch.from_numpy(np.stack(esm_embedding, axis=0)).float()
        protein_ppi_embedding = torch.from_numpy(rwr(ppi_mat, 0.9)).float()

        # Read the preprocessed gene expression data
        df = pd.read_csv('./data/preprocessed_expression.tsv', sep='\t')
        # Identify identifier columns to exclude from the feature matrix
        identifier_cols = ['Index', 'Protein ID', 'Gene ID', 'Gene Name']
        # Select numerical feature columns (expression values or PCA components, plus engineered features)
        feature_cols = [col for col in df.columns if col not in identifier_cols]
        # Extract the feature matrix as a numpy array
        feature_matrix = df[feature_cols].to_numpy()
        # Convert the feature matrix to a PyTorch tensor
        protein_exp_embedding = torch.tensor(feature_matrix, dtype=torch.float32)

        return protein_exp_embedding, protein_ppi_embedding, protein_esm_embedding, ppi_mat
    
    def load_hpo_embedding(self):
        hpo_similarity = sp.load_npz(f'./data/{self.path}/hpo_similarity.npz')
        hpo_embedding = torch.from_numpy(rwr(hpo_similarity, 0.9)).float()
        return hpo_embedding