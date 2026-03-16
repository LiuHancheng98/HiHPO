import numpy as np
import os
import logging
import argparse
import torch
from torch import optim
from logzero import logger
from model.model import HiHPO
from scipy import sparse as sp
from utils.load_data import Data
from utils.evaluation import find_k_largest, ranking_evaluation
from utils.ontology import HumanPhenotypeOntology
from utils.utils import convert_sp_mat_to_sp_tensor, _convert_sp_mat_to_sp_tensor, norm_adj, bpr_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='temporal', help='temporal or random')
    parser.add_argument('--batch_size', type=int, default=2048, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--n_negs', type=int, default=1, help='number of negative samples')
    parser.add_argument('--device_id', type=str, default='0', help='gpu id for training')
    parser.add_argument('--alpha', type=float, default=0.001, help='balance parameter for contrastive loss')
    parser.add_argument('--topN', type=int, default=50, help='top N for evaluation')
    args = parser.parse_args()


    # load data
    data = Data(args.dataset, args.batch_size)
    norm_adj_mat, A_original = data.create_adj_mat()
    protein_exp_embedding, protein_ppi_embedding, protein_esm_embedding, ppi_mat = data.load_protein_embedding()
    hpo_embedding = data.load_hpo_embedding()

    device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")

    train_data = {'X_exp': protein_exp_embedding.to(device),
            'X_esm': protein_esm_embedding.to(device),
            'X_ppi': protein_ppi_embedding.to(device),
            'X_term': hpo_embedding.to(device),
            'A_ppi': norm_adj(convert_sp_mat_to_sp_tensor(ppi_mat)).to(device),
            'A_rel': _convert_sp_mat_to_sp_tensor(norm_adj_mat).to(device),
            'A_original': A_original}

    ontology = HumanPhenotypeOntology('./data/hp.obo', version="202401")
    model = HiHPO(data.n_proteins, data.n_terms, protein_esm_embedding.shape[1], hpo_embedding.shape[1], ontology, device)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epoch_loss = []
    bestPerformance_list = [[]]
    for epoch in range(args.n_epochs):
        optimizer.zero_grad()
        loss = 0
        model.train()
        for n, batch in enumerate(data.next_batch_pairwise(args.n_negs)):
            pro_idx, pos_idx, neg_idx = batch

            pro_exp, term_exp, pro_esm, term_esm, pro_ppi, term_ppi, pro_exp_cl, pro_esm_cl, pro_ppi_cl, intra_cl_loss = model(epoch=epoch, pro_idx=pro_idx, hpo_idx=pos_idx, **train_data)
            pro_exp_batch, term_exp_pos_batch, term_exp_neg_batch = pro_exp[pro_idx], term_exp[pos_idx], term_exp[neg_idx]
            pro_esm_batch, term_esm_pos_batch, term_esm_neg_batch = pro_esm[pro_idx], term_esm[pos_idx], term_esm[neg_idx]
            pro_ppi_batch, term_ppi_pos_batch, term_ppi_neg_batch = pro_ppi[pro_idx], term_ppi[pos_idx], term_ppi[neg_idx]


            batch_loss_exp = bpr_loss(pro_exp_batch, term_exp_pos_batch, term_exp_neg_batch)
            batch_loss_esm = bpr_loss(pro_esm_batch, term_esm_pos_batch, term_esm_neg_batch)
            batch_loss_ppi = bpr_loss(pro_ppi_batch, term_ppi_pos_batch, term_ppi_neg_batch)


            batch_loss = batch_loss_exp + batch_loss_esm + batch_loss_ppi + intra_cl_loss * args.alpha
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            loss += batch_loss.item()

        epoch_loss.append(loss)
        logger.info(f"Epoch {epoch+1} loss: {loss:.4f}")

        topN=[args.topN]
        if epoch % 1 == 0:
            for i in range(0, len(topN)):
                topNi = topN[i]
                bestPerformancei = bestPerformance_list[i]


                all_prot_list = [uid for uid in data.test_set]
                model.eval()
                with torch.no_grad():

                    pro_exp, term_exp, pro_esm, term_esm, pro_ppi, term_ppi, _, _, _, _ = model(epoch=epoch, pro_idx=[0], hpo_idx=[0], **train_data)
                    
                    test_pro_exp, test_pro_esm, test_pro_ppi = pro_exp[all_prot_list], pro_esm[all_prot_list], pro_ppi[all_prot_list]
                    pred_mat1 = torch.matmul(test_pro_exp, term_exp.t()).detach().cpu().numpy()
                    pred_mat2 = torch.matmul(test_pro_esm, term_esm.t()).detach().cpu().numpy()
                    pred_mat3 = torch.matmul(test_pro_ppi, term_ppi.t()).detach().cpu().numpy()
                    pred_mat = pred_mat1 + pred_mat2 + pred_mat3

                rec_list = {}
                origin = {}
                for i, uid in enumerate(all_prot_list):
                    if uid in data.test_set and uid in data.train_terms:
                        test_uid_terms = data.test_set[uid]
                        train_uid_terms = data.train_terms[uid]
                        for terms in train_uid_terms:
                            pred_mat[i][terms] = -10e8
                        origin[uid] = {}
                        for terms in test_uid_terms:
                            origin[uid][terms] = 1
                        candidates = pred_mat[i, :]
                        ids, scores = find_k_largest(np.max(topNi), candidates)
                        rec_list[uid] = list(zip(ids, scores))

                measure = ranking_evaluation(origin, rec_list, [topNi])  # [args.topN])

                if len(bestPerformancei) > 0:
                    count = 0
                    performance = {}
                    for m in measure[1:]:
                        k, v = m.strip().split(':')
                        performance[k] = float(v)
                    for k in bestPerformancei[1]:
                        if k not in ['NDCG']:
                            continue
                        if bestPerformancei[1][k] > performance[k]:
                            count += 1
                        else:
                            count -= 1
                    if count < 0:
                        bestPerformancei[1] = performance
                        bestPerformancei[0] = epoch + 1
                else:
                    bestPerformancei.append(epoch + 1)
                    performance = {}
                    for m in measure[1:]:
                        k, v = m.strip().split(':')
                        performance[k] = float(v)
                    bestPerformancei.append(performance)

                logger.info('-' * 120)
                logger.info(f'Real-Time Ranking Performance Top-{topNi} Term Quality')
                measure = [m.strip() for m in measure[1:]]
                logger.info('*Current Performance*')
                measurement = ' | '.join(measure)
                logger.info(f'Epoch: {str(epoch + 1)}, {measurement}')

                bp = ''
                bp += 'Hit Ratio' + ':' + str(bestPerformancei[1]['Hit Ratio']) + '  |  '
                bp += 'Precision' + ':' + str(bestPerformancei[1]['Precision']) + '  |  '
                bp += 'Recall' + ':' + str(bestPerformancei[1]['Recall']) + '  |  '
                bp += 'NDCG' + ':' + str(bestPerformancei[1]['NDCG']) + '  |  '
                logger.info('*Best Performance* ')
                logger.info(f'Epoch: {str(bestPerformancei[0])},  |  {bp}')
                logger.info('-' * 120)

if __name__ == '__main__':
    main()




