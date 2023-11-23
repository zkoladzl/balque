'''2018-ICLR-Active learning for convolutional neural networks: A core-set approach'''
import numpy as np
from datetime import datetime
from sklearn.metrics import pairwise_distances
import torch.utils.data as data
import torch
import strategy.utils as utils

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

def core_set_sampling(active_learner, model, queried_indice, unqueried_indice, batch_size):
    active_learner.train_classifier(model, active_learner.epoches)
    model.load_state_dict(torch.load(active_learner.current_max_train_acc_model_path))
    
    queried_and_unqueried_indices = queried_indice + unqueried_indice.tolist()
    all_sampler = utils.CustomSquentialSampler(queried_and_unqueried_indices)
    all_loader = data.DataLoader(active_learner.pool_dataset, active_learner.test_batch_size, sampler= all_sampler,num_workers=4)
    embedding = utils.get_embedding(model, all_loader)
    embedding = embedding.numpy()
    chosen = furthest_first(embedding[unqueried_indice, :], embedding[queried_indice, :], batch_size)
    return unqueried_indice[chosen]
