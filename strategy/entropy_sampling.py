import strategy.utils as utils
import torch
import numpy as np
import torch.utils.data as data
import copy
def entropy_sampling(active_learner, model, queried_indice, unqueried_indice, batch_size):
    active_learner.train_classifier(model, active_learner.epoches)
    model.load_state_dict(torch.load(active_learner.current_max_train_acc_model_path))
    pool_sampler = utils.CustomSquentialSampler(unqueried_indice.tolist())
    pool_loader = data.DataLoader(active_learner.pool_dataset, batch_size = active_learner.test_batch_size, sampler = pool_sampler, num_workers= 4)
    model.eval()
    probs = utils.predict_prob(model, pool_loader, active_learner.args.in_distribution_num_classes)
    log_probs = torch.log(probs)
    score = (probs*log_probs).sum(1).numpy()
    unlabeled_score = score[unqueried_indice]
    sorted_idx = np.argsort(unlabeled_score)
    selected_indice = unqueried_indice[sorted_idx[0:batch_size]]
    return selected_indice
