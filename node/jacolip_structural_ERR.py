import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import load_citation, sgc_precompute, set_seed
from models import get_model
from metrics import accuracy
import pickle as pkl
from time import perf_counter
from data_loader.structural_utils import load_data2, load_npz
from numpy import *
import scipy.sparse as sp
from torch import autograd
import os
from scipy.sparse import csr_matrix
import wandb

def jaccard_simi(dataset):

    print('start')

    if dataset == 'ACM':
        simi = torch.FloatTensor(sp.load_npz('10acm.npz').A)
    elif dataset == 'coauthor-cs':
        simi = torch.FloatTensor(sp.load_npz('10cs.npz').A)
    else:
        simi = torch.FloatTensor(sp.load_npz('10phy.npz').A)


    print('finished')
    return simi.cuda()





def simi(output):  
    a = output.norm(dim=1)[:, None]
    the_ones = torch.ones_like(a)
    a = torch.where(a==0, the_ones, a)
    a_norm = output / a
    b_norm = output / a
    res = 5 * (torch.mm(a_norm, b_norm.transpose(0, 1)) + 1)

    return res



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


def err_computation(score_rank, top_k):
    the_maxs = torch.max(score_rank).repeat(1, score_rank.shape[0])
    c = 2 * torch.ones_like(score_rank)
    score_rank = (( c.pow(score_rank) - 1) / c.pow(the_maxs))[0]
    the_ones = torch.ones_like(score_rank)
    new_score_rank = torch.cat((the_ones, 1 - score_rank))

    for i in range(score_rank.shape[0] - 1):
        score_rank = torch.mul(score_rank, new_score_rank[-score_rank.shape[0] - 1 - i : -1 - i])
    the_range = torch.arange(0., score_rank.shape[0]) + 1

    final = (1 / the_range[0:]) * score_rank[0:]

    return torch.sum(final)



def err_exchange_abs(x_corresponding, j, k, top_k):
    new_score_rank = x_corresponding
    err1 = err_computation(new_score_rank, top_k)

    the_index = np.arange(new_score_rank.shape[0])
    temp = the_index[j]
    the_index[j] = the_index[k]
    the_index[k] = temp
    new_score_rank = new_score_rank[the_index]

    err2 = err_computation(new_score_rank, top_k)

    return torch.abs(err1 - err2)


def avg_err(x_corresponding, x_similarity, x_sorted_scores, y_ranks, top_k):

    the_maxs, _ = torch.max(x_corresponding, 1)
    the_maxs = the_maxs.reshape(the_maxs.shape[0], 1).repeat(1, x_corresponding.shape[1])
    c = 2 * torch.ones_like(x_corresponding)
    x_corresponding = ( c.pow(x_corresponding) - 1) / c.pow(the_maxs)
    the_ones = torch.ones_like(x_corresponding)
    new_x_corresponding = torch.cat((the_ones, 1 - x_corresponding), 1)
    for i in range(x_corresponding.shape[1] - 1):
        x_corresponding = torch.mul(x_corresponding, new_x_corresponding[:, -x_corresponding.shape[1] - 1 - i : -1 - i])

    the_range = torch.arange(0., x_corresponding.shape[1]).repeat(x_corresponding.shape[0], 1) + 1
    score_rank = (1 / the_range[:, 0:]) * x_corresponding[:, 0:]
    final = torch.mean(torch.sum(score_rank, axis=1))
    print("Now Average_ERR@k = ", final.item())
    
    wandb.log({"Average_ERR@k": final.item()})

    return final.item()



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
    print("Now Average NDCG@k = ", avg_ndcg.item())
    
    wandb.log({"Average_NDCG@k": avg_ndcg.item()})
    return avg_ndcg.item()


def lambdas_computation(x_similarity, y_similarity, top_k, sigma_1, k_para = 1):
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
        
        
        for j in range(x_corresponding.shape[1]):
            for k in range(x_corresponding.shape[1]):
                if S_x[j, k, i] == 0:
                    continue
                if j < k:
                    the_delta = err_exchange_abs(x_corresponding[i, :], j, k, top_k)
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


def lambdas_computation_only_review(x_similarity, y_similarity, top_k, k_para = 1):
    max_num = 2000000
    x_similarity[range(x_similarity.shape[0]), range(x_similarity.shape[0])] = max_num * torch.ones_like(x_similarity[0, :])
    y_similarity[range(y_similarity.shape[0]), range(y_similarity.shape[0])] = max_num * torch.ones_like(y_similarity[0, :])
    (x_sorted_scores, x_sorted_idxs) = x_similarity.sort(dim=1, descending=True)
    (y_sorted_scores, y_sorted_idxs) = y_similarity.sort(dim=1, descending=True)
    y_ranks = torch.zeros(y_similarity.shape[0], y_similarity.shape[0])
    the_row = torch.arange(y_similarity.shape[0]).view(y_similarity.shape[0], 1).repeat(1, y_similarity.shape[0])
    y_ranks[the_row, y_sorted_idxs] = 1 + torch.arange(y_similarity.shape[1]).repeat(y_similarity.shape[0], 1).float()
    length_of_k = k_para * top_k - 1
    y_sorted_idxs = y_sorted_idxs[:, 1 :(length_of_k + 1)]
    x_sorted_scores = x_sorted_scores[:, 1 :(length_of_k + 1)]
    x_corresponding = torch.zeros(x_similarity.shape[0], length_of_k)

    for i in range(x_corresponding.shape[0]):
        x_corresponding[i, :] = x_similarity[i, y_sorted_idxs[i, :]]

    return x_sorted_scores, y_sorted_idxs, x_corresponding

def train_fair(epoch, model_name, adj, output, model, features, idx_train, idx_test, top_k, all_ndcg_list_test, optimizer, sigma_1, lambdas_para=1, jaccard_adj_orig = None):
    model.train()
    y_similarity1 = simi(output[idx_train])
    x_similarity = jaccard_adj_orig[idx_train, :][:, idx_train]

    lambdas1, x_sorted_scores, y_sorted_idxs, x_corresponding = lambdas_computation(x_similarity, y_similarity1, top_k, sigma_1=sigma_1)
    assert lambdas1.shape == y_similarity1.shape

    y_similarity = simi(output[idx_test])
    x_similarity = jaccard_adj_orig[idx_test, :][:, idx_test]

    print("Ranking optimizing... ")
    x_sorted_scores, y_sorted_idxs, x_corresponding = lambdas_computation_only_review(x_similarity, y_similarity, top_k)
    all_ndcg_list_test.append(avg_err(x_corresponding, x_similarity, x_sorted_scores, y_sorted_idxs, top_k))

    y_similarity1.backward(lambdas_para * lambdas1)
    
    
    optimizer.step()
    return all_ndcg_list_test



def err_computer(output, idx_test, jaccard_adj_orig, top_k):

    y_similarity = simi(output[idx_test])
    x_similarity = jaccard_adj_orig[idx_test, :][:, idx_test]

    print("Ranking optimizing... ")
    x_sorted_scores, y_sorted_idxs, x_corresponding = lambdas_computation_only_review(x_similarity, y_similarity, top_k)
    avg_err(x_corresponding, x_similarity, x_sorted_scores, y_sorted_idxs, top_k)

def train_lipgrad(epoch, model_name, adj, model=None, flag=1, lip_grad = False, u=0.0, features=None, idx_train=None, idx_val=None, idx_test=None, labels=None, optimizer=None, jaccard_adj_orig=None):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    if model_name == 'SGC':
        output1 = model(features)
    else:
        output1 = model(features, adj)

    loss_train = F.cross_entropy(output1[idx_train], labels[idx_train])
    acc_train = accuracy(output1[idx_train], labels[idx_train])

    if lip_grad == True:
        lip_mat = []
        input = features.detach().clone()
        input.requires_grad_(True)
        if model_name == 'SGC':
            out = model(input)
        else:
            out = model(input, adj)
        for i in range(out.shape[1]):
            v = torch.zeros_like(out)
            v[:, i] = 1
            gradients = autograd.grad(outputs=out, inputs=input, grad_outputs=v,
                                      create_graph=True, retain_graph=True, only_inputs=True)[0]
            gradients = gradients[idx_train]  
            grad_norm = torch.norm(gradients, dim=1).unsqueeze(dim=1)
            lip_mat.append(grad_norm)

        input.requires_grad_(False)
        lip_concat = torch.cat(lip_mat, dim=1)
        lip_con_norm = torch.norm(lip_concat, dim=1)
        lip_loss = torch.max(lip_con_norm)
        loss_train = loss_train + u * lip_loss
    else:
        loss_train = loss_train

    if flag == 0:
        loss_train.backward(retain_graph=True)
        
    else:
        loss_train.backward()
        
        optimizer.step()

    model.eval()
    if model_name == 'SGC':
        output = model(features)
    else:
        output = model(features, adj)


    loss_val = F.cross_entropy(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))
    
    wandb.log({"loss_train": loss_train.item(), "acc_train": acc_train.item(), "loss_val": loss_val.item(), "acc_val": acc_val.item()})

    if (epoch+1) % 10 == 0:
        err_computer(output1, idx_test=idx_test, jaccard_adj_orig=jaccard_adj_orig, top_k=10)

    return output1


def train(epoch, model_name, adj, flag=0, model=None, features=None, labels=None, idx_train=None, idx_val=None, optimizer=None, idx_test=None, jaccard_adj_orig=None):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    if model_name == 'SGC':
        output1 = model(features)
    else:
        output1 = model(features, adj)

    loss_train = F.cross_entropy(output1[idx_train], labels[idx_train])
    acc_train = accuracy(output1[idx_train], labels[idx_train])

    if flag == 0:
        loss_train.backward(retain_graph=True)
    else:
        loss_train.backward()
        optimizer.step()

    model.eval()
    if model_name == 'SGC':
        output = model(features)
    else:
        output = model(features, adj)

    loss_val = F.cross_entropy(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))
    
    wandb.log({"loss_train": loss_train.item(), "acc_train": acc_train.item(), "loss_val": loss_val.item(), "acc_val": acc_val.item()})

    if (epoch+1) % 10 == 0:
        err_computer(output1, idx_test=idx_test, jaccard_adj_orig=jaccard_adj_orig, top_k=10) 

    return output1


def test(model, features, adj, labels, idx_test, model_name):
    model.eval()
    if model_name == 'SGC':
        output = model(features)
    else:
        output = model(features, adj)

    loss_test = F.cross_entropy(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    
    wandb.log({'test_loss': loss_test.item(), 'test_acc': acc_test.item()})
    return acc_test.item()


def get_citation_args():
    args = argparse.Namespace()
    args.seed = 42
    args.epochs = 500
    args.lr = 0.01
    args.weight_decay = 5e-6
    args.hidden = 16
    args.dropout = 0.3
    args.dataset = wandb.config.dataset
    args.feature = 'mul'
    args.normalization = 'AugNormAdj'
    args.degree = 2
    args.per = -1
    args.experiment = 'base-experiment'
    args.tuned = True
    args.u_lip = wandb.config.u_lip
    args.model_name = wandb.config.model_name
    
    args.save_dir = 'debug'

    args.cuda = True

    return args

def train_model():
    wandb.init(project="fairnode-struct-err")
    args = get_citation_args()
    dataset = args.dataset  
    model_name = args.model_name  
    args.save_dir = 'log/err/' + 'struc/' + str(model_name) + '/' + str(dataset) + '/'
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        print('create path: ', args.save_dir)
    else:
        print('exist path: ', args.save_dir)
    all_ndcg_list_test = []
    lambdas_para = 1
    k_para = 1
    sigma_1 = -1
    top_k = 10
    pre_train = 200

    if model_name == "GCN":
        if dataset == "ACM":
            adj, features, labels, idx_train, idx_val, idx_test, _, adj_ori = load_data2("ACM")
            sigma_1 = 1e-3
            args.epochs = 200
        else:
            adj, features, labels, idx_train, idx_val, idx_test, _, adj_ori = load_npz(dataset)
            if dataset == "coauthor-cs":
                sigma_1 = 3e-3
                args.epochs = 300
            else:
                sigma_1 = 12e-3
                args.epochs = 600
    else:
        if dataset == "ACM":
            adj, features, labels, idx_train, idx_val, idx_test, _, adj_ori = load_data2("ACM")
            sigma_1 = 5e-4  
            args.epochs = 50
            pre_train = 200
        else:
            adj, features, labels, idx_train, idx_val, idx_test, _, adj_ori = load_npz(dataset)
            if dataset == "coauthor-cs":
                sigma_1 = 5e-3  
                args.epochs = 40
                pre_train = 500
            else:
                sigma_1 = 3e-2  
                args.epochs = 30
                pre_train = 500


    print("Using {} dataset".format(dataset))

    jaccard_adj_orig = jaccard_simi(dataset)
    model = get_model(model_name, features.size(1), labels.max().item() + 1, args.hidden, args.dropout,
                      args.cuda)

    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)

    if args.model_name == "SGC":
        features, precompute_time = sgc_precompute(features, adj, args.degree)

    if args.cuda:
        model.cuda()
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()
  
    for epoch in range(pre_train):
        output = train_lipgrad(epoch, model_name, adj, flag=1, model=model, lip_grad=True, u=args.u_lip, jaccard_adj_orig=jaccard_adj_orig,
                            features=features, optimizer=optimizer, labels=labels, idx_train=idx_train, idx_val=idx_val, idx_test=idx_test)
        
        
        

    for epoch in range(args.epochs):
        output = train(epoch, model_name, adj, flag=0, model=model, features=features, labels=labels, idx_train=idx_train, jaccard_adj_orig=jaccard_adj_orig,
                       idx_val=idx_val, optimizer=optimizer, idx_test=idx_test)
        
        all_ndcg_list_test = train_fair(epoch, model_name, adj, output, model=model, features=features, idx_train=idx_train,
                    idx_test=idx_test, top_k=top_k, all_ndcg_list_test=all_ndcg_list_test,
                    optimizer=optimizer, lambdas_para=lambdas_para, sigma_1=sigma_1, jaccard_adj_orig=jaccard_adj_orig)
        
        
        if (epoch + 1) % 10 == 0 or (epoch + 1) % 5 == 0 and epoch >= (args.epochs // 2):
            test(model, features, adj, labels, idx_test, model_name)
    
    test(model, features, adj, labels, idx_test, model_name)
    wandb.finish()

if __name__ == '__main__':
    
    from pprint import pprint

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    
    sweep_config = {
        'method': 'grid',
        'parameters': {
            'dataset': {
                'values': ["ACM", "coauthor-cs", "coauthor-phy"]
            },
            'model_name': {
                'values': ["SGC", "GCN"]
            },
            'u_lip': {
                'values': [0.01, 0.001, 0.0001, 0.00001, 0.000001]
            },
        }
    }

    sweep_id = wandb.sweep(sweep_config, project="fairnode-struct-err")
    
    wandb.agent(sweep_id, function=train_model)