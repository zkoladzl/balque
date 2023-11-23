import torch.utils.data as data
import torch
import numpy as np
import torch.nn.functional as F
import copy
from sklearn.metrics import pairwise_distances
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import euclidean_distances
import math
class CustomSquentialSampler(data.sampler.Sampler):
    '''
    按顺序返回indice中的元素, indice中的元素不一定是连续的, 而data.SquentialSampler 要求 indice 中的元素是连续的
    '''
    def __init__(self, indice):
        self.indice = indice
    def __iter__(self):
        return (self.indice[i] for i in range(len(self.indice)))
    def __len__(self):
        return len(self.indice)
def get_embedding(model, data_loader, embedding_dim = 512, return_last_embedding = False, last_embedding_dim = -1):
    penultimate_embedding = torch.zeros([len(data_loader.dataset), embedding_dim])
    if return_last_embedding:
        last_embedding = torch.zeros([len(data_loader.dataset), last_embedding_dim])
    with torch.no_grad():
        for inputs in data_loader:
            datas, labels, indice = inputs
            datas = datas.cuda()
            results = model(datas, return_penultimate_features = True)
            if return_last_embedding:last_embedding[indice] = results['outputs'].cpu()
            penultimate_embedding[indice, :] = results['penultimate_features']
    if return_last_embedding:return penultimate_embedding, last_embedding
    return penultimate_embedding

def get_grad_embedding(model, data_loader, in_dsitribution_classes, embedding_dim = 512, average_score = None):
    '''
        @param num_classes 包括 in dsitribution 和 out distribution 的类别数
    '''
    model.eval()
    embedding = np.zeros([len(data_loader.dataset), embedding_dim * in_dsitribution_classes])
    with torch.no_grad():
        for datas, labels, indice in data_loader:
            datas = datas.cuda()
            results = model(datas, return_penultimate_features = True)
            penultimate_features = results['penultimate_features'].numpy()
            batchProbs = F.softmax(results['outputs'], dim=1).cpu().numpy()
            if average_score is not None:batchProbs = average_score[indice]
            maxInds = np.argmax(batchProbs,1)
            for j in range(len(indice)):
                for c in range(in_dsitribution_classes):
                    if c == maxInds[j]:
                        embedding[indice[j]][embedding_dim * c : embedding_dim * (c+1)] = penultimate_features[j] * (1 - batchProbs[j][c])
                        # embedding[indice[j]][embedding_dim * c : embedding_dim * (c+1)] = penultimate_features[j] * (batchProbs[j][c])
                    else:
                        embedding[indice[j]][embedding_dim * c : embedding_dim * (c+1)] = penultimate_features[j] * (-1 * batchProbs[j][c])
        return embedding
# kmeans ++ initialization
def init_centers(X, K):
    ind = np.argmax([np.linalg.norm(s, 2) for s in X])
    mu = [X[ind]]
    indsAll = [ind]
    centInds = [0.] * len(X)
    cent = 0
    print('#Samps\tTotal Distance')
    while len(mu) < K:
        if len(mu) == 1:
            D2 = pairwise_distances(X, mu).ravel().astype(float)
        else:
            newD = pairwise_distances(X, [mu[-1]]).ravel().astype(float)
            for i in range(len(X)):
                if D2[i] >  newD[i]:
                    centInds[i] = cent
                    D2[i] = newD[i]
        if (len(mu) + 1)%100 == 0:print(str(len(mu)) + '\t' + str(sum(D2)), flush=True)
        D2 = D2.ravel().astype(float)
        Ddist = (D2 ** 2)/ sum(D2 ** 2)
        customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
        ind = customDist.rvs(size=1)[0]
        while ind in indsAll: ind = customDist.rvs(size=1)[0]
        mu.append(X[ind])
        indsAll.append(ind)
        cent += 1
    return indsAll

def predict_prob(model, pool_loader, num_classes):
    probs = torch.zeros([len(pool_loader.dataset), num_classes])
    with torch.no_grad():
        for datas, labels, idxs in pool_loader:
            datas, labels = datas.cuda(), labels.cuda()
            results = model(datas)
            prob = F.softmax(results['outputs'], dim=1)
            probs[idxs] = prob.cpu()
    
    return probs

def store_queried_information(file_prefix, infomation, unqueried_indice):
    '''
        information的下标和 unqueried indices 对应
    '''
    file_name = file_prefix + '_infomation.npy'
    np.save(file_name, infomation)
    file_name = file_prefix + '_unqueried_indice.npy'
    np.save(file_name, unqueried_indice)
def store_queried_indice(file_path, indice):
    np.save(file_path, indice)

def furthest_first(pool_embeddings, train_embeddings, n):
    m = np.shape(pool_embeddings)[0]
    if np.shape(train_embeddings)[0] == 0:
        min_dist = np.tile(float("inf"), m)
    else:
        dist_ctr = pairwise_distances(pool_embeddings, train_embeddings)
        min_dist = np.amin(dist_ctr, axis=1)
    idxs = []
    for i in range(n):
        idx = min_dist.argmax()
        idxs.append(idx)
        dist_new_ctr = pairwise_distances(pool_embeddings, pool_embeddings[[idx], :])
        for j in range(m):
            min_dist[j] = min(min_dist[j], dist_new_ctr[j, 0])
    return idxs
