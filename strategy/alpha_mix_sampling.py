# 2022, cvpr, Active Learning by Feature Mixing
import numpy as np
import torch.utils.data as data
import torch
import numpy as np
import strategy.utils as utils
import torch.nn.functional as F
import copy
import math
from sklearn.cluster import KMeans

def calculate_optimum_alpha(eps, labeled_embedding, unlabeled_embedding, ulb_grads):
    z = (labeled_embedding - unlabeled_embedding) #* ulb_grads
    alpha = (eps * z.norm(dim=1) / ulb_grads.norm(dim=1)).unsqueeze(dim=1).repeat(1, z.size(1)) * ulb_grads / (z + 1e-8)
    return alpha

def find_candidate_set(linear_model, labeled_embedding, unlabeled_embedding, pseudo_labels, unlabeled_probs, alpha_cap, Y,grads, indistribution_num_classes):
    unlabeled_size = unlabeled_embedding.shape[0]
    embedding_size = labeled_embedding.shape[1]

    min_alphas = torch.ones((unlabeled_size, embedding_size), dtype=torch.float)
    pred_change = torch.zeros(unlabeled_size, dtype=torch.bool)

    alpha_cap /= math.sqrt(embedding_size)
        
    for i in range(indistribution_num_classes):
        emb = labeled_embedding[Y == i]
        if emb.size(0) == 0:
            emb = labeled_embedding
        anchor_i = emb.mean(dim=0).view(1, -1).repeat(unlabeled_size, 1)

        embed_i, ulb_embed = anchor_i.cuda(), unlabeled_embedding.cuda()
        alpha = calculate_optimum_alpha(alpha_cap, embed_i, ulb_embed, grads)

        embedding_mix = (1 - alpha) * ulb_embed + alpha * embed_i
        out= linear_model(embedding_mix)
        out = out.detach().cpu()
        alpha = alpha.cpu()

        pc = out.argmax(dim=1) != pseudo_labels

        torch.cuda.empty_cache()

        alpha[~pc] = 1.
        pred_change[pc] = True
        is_min = min_alphas.norm(dim=1) > alpha.norm(dim=1)
        min_alphas[is_min] = alpha[is_min]
        
    return pred_change, min_alphas

def sample(batch_size, feats):
    feats = feats.numpy()
    cluster_learner = KMeans(n_clusters=batch_size)
    cluster_learner.fit(feats)

    cluster_idxs = cluster_learner.predict(feats)
    centers = cluster_learner.cluster_centers_[cluster_idxs]
    dis = (feats - centers) ** 2
    dis = dis.sum(axis=1)
    return np.array(
        [np.arange(feats.shape[0])[cluster_idxs == i][dis[cluster_idxs == i].argmin()] for i in range(batch_size) if
            (cluster_idxs == i).sum() > 0])

def alpha_mix_sampling(active_learner, model, queried_indice, unqueried_indice, batch_size):
    active_learner.train_classifier(model, active_learner.epoches)

    unlabeled_sampler = utils.CustomSquentialSampler(unqueried_indice.tolist())
    unlabeled_loader = data.DataLoader(active_learner.pool_dataset, batch_size = active_learner.test_batch_size, sampler = unlabeled_sampler,num_workers=4)
    unlabeled_probs_distribution = utils.predict_prob(model, unlabeled_loader, active_learner.args.in_distribution_num_classes)[unqueried_indice.tolist()]
    unlabeled_probs, pseudo_labels= torch.max(unlabeled_probs_distribution, dim = 1)
    if active_learner.args.data_name == 'SVHN':true_labels = np.array(active_learner.args.train_dataset.dataset.labels)
    else:true_labels = np.array(active_learner.args.train_dataset.dataset.train_labels)
    all_sampler = utils.CustomSquentialSampler(queried_indice + unqueried_indice.tolist())
    all_loader = data.DataLoader(active_learner.pool_dataset, batch_size = active_learner.test_batch_size, sampler = all_sampler,num_workers=4)
    all_embedding = utils.get_embedding(model, all_loader)

    unlabeled_embedding = all_embedding[unqueried_indice.tolist()]
    labeled_embedding = all_embedding[queried_indice]

    unlabeled_size = unlabeled_embedding.shape[0]
    embedding_size = labeled_embedding.shape[1]

    min_alphas = torch.ones((unlabeled_size, embedding_size), dtype=torch.float)
    candidate = torch.zeros(unlabeled_size, dtype=torch.bool)

    var_emb = unlabeled_embedding.clone().detach().requires_grad_(True).cuda()
    linear_model = copy.deepcopy(model.fc)
    out = linear_model(var_emb)
    loss = F.cross_entropy(out, pseudo_labels.cuda()).mean()
    grads = torch.autograd.grad(loss, var_emb)[0].data
    del loss, var_emb, out

    alpha_cap, alpha_cap_initial = 0., 0.03125
    while alpha_cap < 1.0:
        alpha_cap += alpha_cap_initial

        tmp_pred_change, tmp_min_alphas = find_candidate_set(linear_model, labeled_embedding, unlabeled_embedding, 
            pseudo_labels, unlabeled_probs, alpha_cap, true_labels[queried_indice], grads,
            active_learner.args.in_distribution_num_classes
        )

        is_changed = min_alphas.norm(dim=1) >= tmp_min_alphas.norm(dim=1)

        min_alphas[is_changed] = tmp_min_alphas[is_changed]
        candidate += tmp_pred_change

        print('With alpha_cap set to %f, number of inconsistencies: %d' % (alpha_cap, int(tmp_pred_change.sum().item())))

        if candidate.sum() > batch_size:
            break

    if candidate.sum() > 0:
        print('Number of inconsistencies: %d' % (int(candidate.sum().item())))

        print('alpha_mean_mean: %f' % min_alphas[candidate].mean(dim=1).mean().item())
        print('alpha_std_mean: %f' % min_alphas[candidate].mean(dim=1).std().item())
        print('alpha_mean_std %f' % min_alphas[candidate].std(dim=1).mean().item())

        c_alpha = F.normalize(unlabeled_embedding[candidate].view(candidate.sum(), -1), p=2, dim=1).detach()

        selected_idxs = sample(min(batch_size, candidate.sum().item()), feats=c_alpha)
        selected_idxs = unqueried_indice[candidate][selected_idxs]
    else:
        selected_idxs = np.array([], dtype=np.int)

    if selected_idxs.shape[0] < batch_size:
        remained = batch_size - len(selected_idxs)
        indicator = np.ones(len(queried_indice) + unqueried_indice.shape[0])
        indicator[queried_indice + selected_idxs.tolist()] = 0
        selected_idxs = np.concatenate([selected_idxs, np.random.permutation(np.where(indicator == 1)[0])[0:remained]])
        print('picked %d samples from RandomSampling.' % (remained))
    return selected_idxs