'''2020-ICLR-deep_batch_active_learning_by_DIVERSE, UNCERTAIN GRADIENT LOWER BOUNDS'''
import torch.utils.data as data
import torch
import numpy as np
import strategy.utils as utils
from sklearn.metrics import euclidean_distances
def badge_sampling(active_learner, model, queried_indice, unqueried_indice, batch_size):
    active_learner.train_classifier(model, active_learner.epoches)
    model.load_state_dict(torch.load(active_learner.current_max_train_acc_model_path))
    
    unlabeled_sampler = utils.CustomSquentialSampler(unqueried_indice.tolist())
    unlabeled_loader = data.DataLoader(active_learner.pool_dataset, batch_size = active_learner.test_batch_size, sampler = unlabeled_sampler,num_workers=4)
    
    gradEmbedding = utils.get_grad_embedding(model, unlabeled_loader, in_dsitribution_classes = len(active_learner.args.in_distribution_labels))
    chosen = utils.init_centers(gradEmbedding[unqueried_indice], batch_size)
    return unqueried_indice[chosen]