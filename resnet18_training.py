import config
import torch
import model.active_learner as active_learner
from strategy import forgetting_events_sampling, badge, core_set, random_sampling, entropy_sampling, alpha_mix_sampling, utils
import numpy as np
import copy
import os
torch.cuda.set_device(0)
base_path = "/mnt/sda/hyc/open_source/"
args = config.get_config(base_path)

# set sampling methods
query_functions = [
    forgetting_events_sampling.forgetting_events_sampling_V4,
    badge.badge_sampling,
    core_set.core_set_sampling,
    random_sampling.random_sampling,
    entropy_sampling.entropy_sampling,
    alpha_mix_sampling.alpha_mix_sampling
]
# set str name of sampling methods
query_names = [
    'forgetting_events_sampling_V4',
    'badge_sampling',
    'core_set',
    'random_sampling',
    'entropy_sampling',
    'alpha_mix_sampling'
]

path = copy.deepcopy(args.results_path)
for i in range(len(query_functions)):
    dir = path + query_names[i] + "/"
    if os.path.exists(dir) is False:
        os.makedirs(dir)
    args.results_path = dir + args.data_name + '_' + query_names[i] + "_"
    args.sampling_name = query_names[i]
    learner = active_learner.ActiveLearner(args, query_functions[i])
    learner.adversarial_perturbations = None;learner.active_train()
    os.remove(learner.current_max_train_acc_model_path)
    print(query_names[i] + ":\n train acc:{}, \n test_acc:{}".format(
        learner.train_acc, learner.test_acc)
    )