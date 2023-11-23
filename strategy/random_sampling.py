import numpy as np
import strategy.utils as utils
import torch.utils.data as data
import torch
def random_sampling(active_learner, model, queried_indice, unqueried_indice, batch_size):
    '''
        return unqueried_indice 的下标，下标对应的值对应样本的下标
    '''
    active_learner.train_classifier(model, active_learner.epoches)
    perm = np.random.permutation(unqueried_indice.shape[0])
    return unqueried_indice[perm[0:min(batch_size, unqueried_indice.shape[0])]]

