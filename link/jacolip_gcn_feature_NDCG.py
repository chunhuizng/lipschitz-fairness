from __future__ import division
from __future__ import print_function

import argparse
import time

import numpy as np
import scipy.sparse as sp
import torch
from torch import optim
from torch import autograd
from model import GCNModelVAE, GCN, GCN_inform
from optimizer import loss_function, loss_function_gcn
from utils import load_data, mask_test_edges, preprocess_graph, get_roc_score, load_data2, load_data_news, load_data3, \
    load_AN, get_roc_score_GCN
from numpy import *

import os


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gcn', help="models used")
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--hidden1', type=int, default=32, help='Number of units in hidden layer 1.')
parser.add_argument('--hidden2', type=int, default=16, help='Number of units in hidden layer 2.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default='facebook', help='dataset', choices=["BlogCatalog", "facebook", "Flickr"])
parser.add_argument('--u_lip', type=float, default=0.1, help='upper bound of lipschitz constant.')
args = parser.parse_args()

model_name = args.model
dataset = args.dataset
u_lip = args.u_lip
def save_log(*args, **kwargs):
    path = 'log/' + str(model_name) + '/'  + 'feat/'+ str(dataset) + '/'
    if not os.path.exists(path):
        os.makedirs(path)
        print('create path: ', path)
    print(*args, **kwargs)
    with open(path + str(u_lip)+'.txt', 'a') as f:
        print(*args, file=f, **kwargs)

def simi(output):  
    a = output.norm(dim=1)[:, None]
    the_ones = torch.ones_like(a)
    a = torch.where(a == 0, the_ones, a)
    a_norm = output / a
    b_norm = output / a
    res = 5 * (torch.mm(a_norm, b_norm.transpose(0, 1)) + 1)

    return res


def jaccard_simi(adj):
    adj = torch.FloatTensor(adj.A).cuda()
    simi = torch.zeros_like(adj)
    for i in range(adj.shape[0]):
        one = torch.sum(adj[i, :].repeat(adj.shape[0], 1).mul(adj), axis=1)
        two = torch.sum(adj[i, :]) * torch.sum(adj, axis=1)
        simi[i, :] = (one / two).T

    return simi.cuda()



def idcg_computation(x_sorted_scores, top_k):
    c = 2 * torch.ones_like(x_sorted_scores)[:top_k]
    numerator = c.pow(x_sorted_scores[:top_k]) - 1
    denominator = torch.log2(2 + torch.arange(x_sorted_scores[:top_k].shape[0], dtype=torch.float)).cuda()
    final = numerator / denominator

    return torch.sum(final)


def dcg_computation(score_rank, top_k):
    c = 2 * torch.ones_like(score_rank)[:top_k]
    numerator = c.pow(score_rank[:top_k]) - 1
    denominator = torch.log2(2 + torch.arange(score_rank[:top_k].shape[0], dtype=torch.float))
    final = numerator / denominator

    return torch.sum(final)


def ndcg_exchange_abs(x_corresponding, j, k, idcg, top_k):
    new_score_rank = x_corresponding
    dcg1 = dcg_computation(new_score_rank, top_k)
    the_index = np.arange(new_score_rank.shape[0])
    temp = the_index[j]
    the_index[j] = the_index[k]
    the_index[k] = temp
    new_score_rank = new_score_rank[the_index]
    dcg2 = dcg_computation(new_score_rank, top_k)

    return torch.abs((dcg1 - dcg2) / idcg)

def avg_ndcg(x_corresponding, x_similarity, x_sorted_scores, y_ranks, top_k):
    c = 2 * torch.ones_like(x_sorted_scores[:, :top_k])
    numerator = c.pow(x_sorted_scores[:, :top_k]) - 1
    denominator = torch.log2(2 + torch.arange(x_sorted_scores[:, :top_k].shape[1], dtype=torch.float)).repeat(x_sorted_scores.shape[0], 1).cuda()
    idcg = torch.sum((numerator / denominator), 1)
    new_score_rank = torch.zeros(y_ranks.shape[0], y_ranks[:, :top_k].shape[1])
    numerator = c.pow(x_corresponding.cuda()[:, :top_k]) - 1
    denominator = torch.log2(2 + torch.arange(new_score_rank[:, :top_k].shape[1], dtype=torch.float)).repeat(x_sorted_scores.shape[0], 1).cuda()
    ndcg_list = torch.sum((numerator / denominator), 1) / idcg
    avg_ndcg = torch.mean(ndcg_list)
    save_log("Now Average NDCG@k = ", avg_ndcg.item())

    return avg_ndcg.item()

def ndcg_computer(embedding):
    y_similarity1 = simi(embedding)
    x_similarity = simi(features)

    lambdas1, x_sorted_scores, y_sorted_idxs, x_corresponding = lambdas_computation(x_similarity, y_similarity1, top_k)
    assert lambdas1.shape == y_similarity1.shape
    save_log("Ranking optimizing... ")
    avg_ndcg(x_corresponding, x_similarity, x_sorted_scores, y_sorted_idxs, top_k)


def lambdas_computation(x_similarity, y_similarity, top_k):
    max_num = 2000000
    x_similarity[range(x_similarity.shape[0]), range(x_similarity.shape[0])] = max_num * torch.ones_like(x_similarity[0, :])
    y_similarity[range(y_similarity.shape[0]), range(y_similarity.shape[0])] = max_num * torch.ones_like(y_similarity[0, :])

    
    (x_sorted_scores, x_sorted_idxs) = x_similarity.sort(dim=1, descending=True)
    (y_sorted_scores, y_sorted_idxs) = y_similarity.sort(dim=1, descending=True)
    y_ranks = torch.zeros(y_similarity.shape[0], y_similarity.shape[0])
    the_row = torch.arange(y_similarity.shape[0]).view(y_similarity.shape[0], 1).repeat(1, y_similarity.shape[0])
    y_ranks[the_row, y_sorted_idxs] = 1 + torch.arange(y_similarity.shape[1]).repeat(y_similarity.shape[0], 1).float()

    
    sigma_tuned = sigma_1
    length_of_k = k_para * top_k
    y_sorted_scores = y_sorted_scores[:, 1 :(length_of_k + 1)]
    y_sorted_idxs = y_sorted_idxs[:, 1 :(length_of_k + 1)]
    x_sorted_scores = x_sorted_scores[:, 1 :(length_of_k + 1)]
    pairs_delta = torch.zeros(y_sorted_scores.shape[1], y_sorted_scores.shape[1], y_sorted_scores.shape[0])

    for i in range(y_sorted_scores.shape[0]):
        pairs_delta[:, :, i] = y_sorted_scores[i, :].view(y_sorted_scores.shape[1], 1) - y_sorted_scores[i, :].float()

    fraction_1 = - sigma_tuned / (1 + (pairs_delta * sigma_tuned).exp())
    x_delta = torch.zeros(y_sorted_scores.shape[1], y_sorted_scores.shape[1], y_sorted_scores.shape[0])
    x_corresponding = torch.zeros(x_similarity.shape[0], length_of_k)

    for i in range(x_corresponding.shape[0]):
        x_corresponding[i, :] = x_similarity[i, y_sorted_idxs[i, :]]

    for i in range(x_corresponding.shape[0]):
        
        x_delta[:, :, i] = x_corresponding[i, :].view(x_corresponding.shape[1], 1) - x_corresponding[i, :].float()

    S_x = torch.sign(x_delta)
    zero = torch.zeros_like(S_x)
    S_x = torch.where(S_x < 0, zero, S_x)

    
    ndcg_delta = torch.zeros(x_corresponding.shape[1], x_corresponding.shape[1], x_corresponding.shape[0])
    for i in range(y_similarity.shape[0]):
        if i >= 0.6 * y_similarity.shape[0]:
            break
        idcg = idcg_computation(x_sorted_scores[i, :], top_k)
        for j in range(x_corresponding.shape[1]):
            for k in range(x_corresponding.shape[1]):
                if S_x[j, k, i] == 0:
                    continue
                if j < k:
                    the_delta = ndcg_exchange_abs(x_corresponding[i, :], j, k, idcg, top_k)
                    
                    ndcg_delta[j, k, i] = the_delta
                    ndcg_delta[k, j, i] = the_delta

    without_zero = S_x * fraction_1 * ndcg_delta
    lambdas = torch.zeros(x_corresponding.shape[0], x_corresponding.shape[1])

    for i in range(lambdas.shape[0]):
        for j in range(lambdas.shape[1]):
            lambdas[i, j] = torch.sum(without_zero[j, :, i]) - torch.sum(without_zero[:, j, i])   

    mid = torch.zeros_like(x_similarity)
    the_x = torch.arange(x_similarity.shape[0]).repeat(length_of_k, 1).transpose(0, 1).reshape(length_of_k * x_similarity.shape[0], 1).squeeze()
    the_y = y_sorted_idxs.reshape(length_of_k * x_similarity.shape[0], 1).squeeze()
    the_data = lambdas.reshape(length_of_k * x_similarity.shape[0], 1).squeeze()
    mid.index_put_((the_x, the_y.long()), the_data.cuda())

    return mid, x_sorted_scores, y_sorted_idxs, x_corresponding




def train_fair(top_k, epoch, model, optimizer, features, adj_norm, embedding):
    model.train()
    y_similarity1 = simi(embedding)
    x_similarity = simi(features)

    lambdas1, x_sorted_scores, y_sorted_idxs, x_corresponding = lambdas_computation(x_similarity, y_similarity1, top_k)
    assert lambdas1.shape == y_similarity1.shape
    save_log('Epoch ', epoch, ' : ')
    save_log("Ranking optimizing... ")

    should_return = avg_ndcg(x_corresponding, x_similarity, x_sorted_scores, y_sorted_idxs, top_k)

    y_similarity1.backward(lambdas_para * lambdas1)
    optimizer.step()

    return should_return


lambdas_para = 1
k_para = 1
sigma_1 = -1
top_k = 10
all_ndcg_list_train = []
auc = []
ap = []

auc_before = []
auc_after = []
ap_before = []
ap_after = []
fair_before = []
fair_after = []

if dataset == "facebook":
    adj, features = load_AN()
    sigma_1 = 30e-3
    args.epochs = 50
    pre_train = 200
else:
    adj, features = load_data2(dataset)
    if dataset == "Flickr":
        sigma_1 = 100e-5
        args.epochs = 100
        pre_train = 200
    else:
        sigma_1 = 8e-4
        args.epochs = 60
        pre_train = 200

save_log("Using {} dataset".format(dataset))

n_nodes, feat_dim = features.shape


adj_orig = adj
adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
adj_orig.eliminate_zeros()


adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(
    adj)
adj = adj_train



adj_norm = preprocess_graph(adj)
adj_label = adj_train + sp.eye(adj_train.shape[0])
adj_label = torch.FloatTensor(adj_label.toarray())

pos_weight = torch.Tensor([float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()])
norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
hidden_emb = None

model = GCN_inform(feat_dim, args.hidden1, args.dropout)
optimizer = optim.Adam(model.parameters(), lr=args.lr)

model.cuda()
features = features.cuda()
adj_norm = adj_norm.cuda()
adj_label = adj_label.cuda()
pos_weight = pos_weight.cuda()



def train(epoch, flag=0):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    recovered, embedding = model(features, adj_norm)

    loss = loss_function_gcn(preds=recovered, labels=adj_label,
                             pos_weight=pos_weight)

    cur_loss = loss.item()

    roc_curr, ap_curr = get_roc_score_GCN(recovered.cpu().detach().numpy(), adj_orig, val_edges, val_edges_false)

    save_log("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(cur_loss),
          "val_ap=", "{:.5f}".format(ap_curr),
          "time=", "{:.5f}".format(time.time() - t)
          )

    roc_score, ap_score = get_roc_score_GCN(recovered.cpu().detach().numpy(), adj_orig, test_edges, test_edges_false)
    save_log('Test AUC score: ' + str(roc_score))
    save_log('Test AP score: ' + str(ap_score))

    if flag == 1:
        loss.backward()
        optimizer.step()
    else:
        loss.backward(retain_graph=True)
        auc.append(roc_score)
        ap.append(ap_score)

    if (epoch + 1) % 5 == 0:
        ndcg_computer(embedding=embedding)

    return embedding


def train_lipgrad(epoch, flag=0, lip_grad=False, u=0.0):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    recovered, embedding = model(features, adj_norm)

    loss = loss_function_gcn(preds=recovered, labels=adj_label,
                             pos_weight=pos_weight)

    cur_loss = loss.item()

    roc_curr, ap_curr = get_roc_score_GCN(recovered.cpu().detach().numpy(), adj_orig, val_edges, val_edges_false)

    save_log("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(cur_loss),
          "val_ap=", "{:.5f}".format(ap_curr),
          "time=", "{:.5f}".format(time.time() - t)
          )

    roc_score, ap_score = get_roc_score_GCN(recovered.cpu().detach().numpy(), adj_orig, test_edges, test_edges_false)
    save_log('Test AUC score: ' + str(roc_score))
    save_log('Test AP score: ' + str(ap_score))

    if lip_grad == True:
        lip_mat = []
        input = features.detach().clone()
        input.requires_grad_(True)
        _, out = model(input, adj_norm)
        for i in range(out.shape[1]):
            v = torch.zeros_like(out)
            v[:, i] = 1
            gradients = autograd.grad(outputs=out, inputs=input, grad_outputs=v,
                                      create_graph=True, retain_graph=True, only_inputs=True)[0]
            grad_norm = torch.norm(gradients, dim=1).unsqueeze(dim=1)
            lip_mat.append(grad_norm)

        input.requires_grad_(False)
        lip_concat = torch.cat(lip_mat, dim=1)
        
        lip_concat = torch.mm(lip_concat, lip_concat.t()) 
        lip_con_norm = torch.norm(lip_concat, dim=1)
        lip_loss = torch.max(lip_con_norm)
        loss = loss + u * lip_loss

    if flag == 1:
        loss.backward()
        optimizer.step()
    else:
        loss.backward(retain_graph=True)
        auc.append(roc_score)
        ap.append(ap_score)

    if (epoch + 1) % 10 == 0:
        ndcg_computer(embedding=embedding)

    return embedding


save_log('1. model, 2. dataset, and 3. u_lip are: ' + str(args.model) + ' ' + str(args.dataset) + ' ' + str(args.u_lip))
for epoch in range(pre_train):
    
    _ = train_lipgrad(epoch, flag=1, lip_grad=True, u = args.u_lip)

for epoch in range(args.epochs):
    embedding = train(epoch)
    all_ndcg_list_train.append(train_fair(top_k, epoch, model, optimizer, features, adj_norm, embedding))

auc_before.append(auc[0])
auc_after.append(auc[-1])

ap_before.append(ap[0])
ap_after.append(ap[-1])

fair_before.append(all_ndcg_list_train[0])
fair_after.append(all_ndcg_list_train[-1])

save_log('auc_before' + str(auc_before))
save_log('auc_after' + str(auc_after))
save_log('ap_before' + str(ap_before))
save_log('ap_after' + str(ap_after))
save_log('fair_before' + str(fair_before))
save_log('fair_after' + str(fair_after))

torch.cuda.empty_cache()
