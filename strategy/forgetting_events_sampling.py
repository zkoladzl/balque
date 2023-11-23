import copy
import torch.utils.data as data
import torch
import numpy as np
import strategy.utils as utils
import numpy as np
def forgetting_events_sampling_common(active_learner, model, queried_indice, unqueried_indice, batch_size, interval):
    train_loader = active_learner.train_loader
    unlabeled_sampler = utils.CustomSquentialSampler(unqueried_indice.tolist())
    unlabeled_loader = data.DataLoader(active_learner.pool_dataset, batch_size = active_learner.test_batch_size, sampler = unlabeled_sampler,num_workers=4)
    scores = np.zeros((len(queried_indice) + unqueried_indice.shape[0], active_learner.args.in_distribution_num_classes))
    model = model.cuda();model.train()
    if active_learner.optimizer_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr = active_learner.lr, weight_decay = active_learner.weight_decay)
    training_size = len(train_loader.sampler)
    print('training size:{}'.format(training_size))
    max_train_acc= 0
    for epoche in range(active_learner.epoches):
        for iter, inputs in enumerate(train_loader):
            datas, labels, indice = inputs
            datas, labels = datas.cuda(), labels.cuda()
            results = model(datas)
            optimizer.zero_grad()
            loss = active_learner.loss_function(results['outputs'], labels)
            loss.backward()
            optimizer.step()
        
        if (epoche + 1)%10 == 0:
            model.eval()
            train_acc, test_acc = active_learner.acc_evaluation(model, train = True), active_learner.acc_evaluation(model)
            if(train_acc >= max_train_acc):
                max_train_acc = train_acc
                active_learner.current_max_train_acc = train_acc
                active_learner.current_optimal_test_acc = test_acc
                torch.save(model.state_dict(), active_learner.current_max_train_acc_model_path)
            print("epoche:{}, train acc:{}, test acc:{}".format(epoche + 1, train_acc, test_acc))
            model.train()
        if active_learner.args.data_name == 'SVHN' and (epoche + 1)%5 == 0:
            model.eval()
            with torch.no_grad():
                for iter, inputs in enumerate(unlabeled_loader):
                    datas, labels, indice = inputs
                    datas, labels = datas.cuda(), labels.cuda()
                    results = model(datas)
                    pseudo_labels = torch.argmax(results['outputs'], dim = 1).cpu().numpy()
                    scores[indice, pseudo_labels] +=1
            model.train()
        elif active_learner.args.data_name.startswith('CIFAR') and (epoche + 1)%interval == 0:
            model.eval()
            with torch.no_grad():
                for iter, inputs in enumerate(unlabeled_loader):
                    datas, labels, indice = inputs
                    datas, labels = datas.cuda(), labels.cuda()
                    results = model(datas)
                    pseudo_labels = torch.argmax(results['outputs'], dim = 1).cpu().numpy()
                    scores[indice, pseudo_labels] +=1
            model.train()
        if active_learner.args.early_stop and max_train_acc >= 0.995:
            print('trigger eraly stop')
            break
    scores = scores[unqueried_indice,:]
    
    np.save(active_learner.results_path_base + 'infomation_{}.npy'.format(active_learner.current_cycle), scores)
    np.save(active_learner.results_path_base + 'infomation_{}_unqueried_indice.npy'.format(active_learner.current_cycle), unqueried_indice)
    
    pseudo_label_categories = np.sum(scores[0,])
    proba = scores/pseudo_label_categories
    proba += 1e-9
    scores = np.sum(-np.log10(proba)*proba, axis=1)
    print('infomation entropy bound:min:{}, max:{}'.format(np.min(scores), np.max(scores)))
    sorted_indice = np.argsort(scores)
    chosen = sorted_indice[max(0, scores.shape[0] - batch_size):scores.shape[0]]
    return unqueried_indice[chosen]
def forgetting_events_sampling_V1(active_learner, model, queried_indice, unqueried_indice, batch_size):
    return forgetting_events_sampling_common(active_learner, model, queried_indice, unqueried_indice, batch_size, 1)

def forgetting_events_sampling_V2(active_learner, model, queried_indice, unqueried_indice, batch_size):
    return forgetting_events_sampling_common(active_learner, model, queried_indice, unqueried_indice, batch_size, 5)

def forgetting_events_sampling_V3(active_learner, model, queried_indice, unqueried_indice, batch_size):
    return forgetting_events_sampling_common(active_learner, model, queried_indice, unqueried_indice, batch_size, 10)

def forgetting_events_sampling_V4(active_learner, model, queried_indice, unqueried_indice, batch_size):
    return forgetting_events_sampling_common(active_learner, model, queried_indice, unqueried_indice, batch_size, 20)

def forgetting_events_sampling_V5(active_learner, model, queried_indice, unqueried_indice, batch_size):
    return forgetting_events_sampling_common(active_learner, model, queried_indice, unqueried_indice, batch_size, 50)
