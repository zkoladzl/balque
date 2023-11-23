import copy
import torch.utils.data as data
import torch
import torch.nn.functional as F
import numpy as np
import strategy.utils as utils
class ActiveLearner():
    def __init__(self, args, query_function):
        '''
            @param pool_indice np.array 下标对应的值为0表示该样本属于训练集, 为1表示属于 pool set, pool_indice的大小为train set的样本数
        '''
        self.query_function = query_function
        self.model = args.model
        self.epoches = args.epoches
        self.train_indice = copy.deepcopy(args.initial_indice)
        self.in_distribution_labels = args.in_distribution_labels
        self.out_distribution_labels = args.out_distribution_labels
        #unlabeled in and out distribution indice
        self.pool_indice = copy.deepcopy(args.pool_indice) #被查询标签的样本用0标记，为被查询的样本用1标记
        self.pool_dataset = args.train_dataset
        self.test_dataset = args.test_dataset
        self.training_batch_size = args.training_batch_size
        self.test_batch_size = args.test_batch_size
        self.test_in_distribution_indice = args.test_in_distribution_indice
        self.validate_indice = copy.deepcopy(args.validate_indice)
        #path/data_name_queri_name_*
        self.results_path_base = args.results_path
        self.model_params_path = []
        self.train_acc, self.test_acc= [], []
        self.query_cnts = args.query_cnts
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.optimizer_type = args.optimizer_type
        self.loss_function = args.loss_function.cuda()

        test_sampler = data.sampler.SubsetRandomSampler(args.test_in_distribution_indice)
        self.test_loader = data.DataLoader(self.test_dataset, args.test_batch_size, test_sampler, num_workers=4)
        validate_sampler = data.sampler.SubsetRandomSampler(args.validate_indice)
        self.validate_loader = data.DataLoader(self.pool_dataset, args.test_batch_size, validate_sampler, num_workers=4)
        self.query_batch_size = args.query_batch_size
        self.selected_indistribution_indice, self.selected_out_distribution_indice = [], []
        self.current_cycle = 0 #当前是第几次查询
        self.args = args
        self.train_sampler, self.train_loader = None, None
        self.training_model = None
        self.current_max_train_acc_model_path = self.results_path_base + 'current_max_train_acc_model.pt'
        self.current_max_train_acc, self.current_optimal_test_acc = None, None

        self.adversarial_perturbations = None
    def acc_evaluation(self, model, train = False):
        datas_size = 0
        if train:
            train_sampler = data.SubsetRandomSampler(self.train_indice)
            data_loader = data.DataLoader(self.pool_dataset, batch_size= 128, sampler = train_sampler, num_workers=4)
            datas_size = len(self.train_indice)
        else:
            datas_size = len(self.test_in_distribution_indice)
            test_sampler = data.sampler.SubsetRandomSampler(self.test_in_distribution_indice)
            data_loader = data.DataLoader(self.test_dataset, self.test_batch_size, sampler = test_sampler, num_workers=4)
        cnt = 0
        for iter, inputs in enumerate(data_loader):
            datas, labels, indice = inputs
            datas, labels = datas.cuda(), labels.cuda()
            results = model(datas)
            predictions = torch.argmax(F.softmax(results['outputs'], dim = 1), dim = 1)
            cnt += int((predictions == labels).sum())
        return cnt/datas_size
    def train_classifier(self, model, epoches):
        model = model.cuda()
        model.train()
        training_size = len(self.train_loader.sampler)
        optimizer = None
        if self.optimizer_type == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr = self.lr, weight_decay=self.weight_decay)
        max_train_acc = 0.
        for epoche in range(epoches):
            loss_sum = torch.tensor(0.)
            for iter, inputs in enumerate(self.train_loader):
                datas, labels, indice = inputs
                datas, labels = datas.cuda(), labels.cuda()
                results = model(datas)
                optimizer.zero_grad()
                loss = self.loss_function(results['outputs'], labels)
                loss_sum += loss.data.cpu()
                loss.backward()
                optimizer.step()
            if (epoche + 1)%20 == 0:
                model.eval()
                test_acc = self.acc_evaluation(model)
                train_acc = self.acc_evaluation(model, train = True)
                if train_acc >= max_train_acc:
                    max_train_acc = train_acc
                    self.current_max_train_acc, self.current_optimal_test_acc = train_acc, test_acc
                    torch.save(model.state_dict(), self.current_max_train_acc_model_path)
                print('epoch:{}, loss:{}, train acc:{}, test acc:{}'.format(epoche + 1, loss_sum/training_size,train_acc, self.current_optimal_test_acc))
                model.train()
        print('max train acc:{}, test acc:{}'.format(max_train_acc, self.current_optimal_test_acc))
        return model
    def store_queried_indice(self, selected_in_distribution_indice, selected_out_distribution_indice):
        path = self.results_path_base + 'query_indice(in_distribution)_cycle_' + str(self.current_cycle) + '.npy'
        np.save(path, selected_in_distribution_indice)
        path = self.results_path_base + 'query_indice(out_distribution)_cycle_' + str(self.current_cycle) + '.npy'
        np.save(path, selected_out_distribution_indice)
        
    def active_train(self):
        # 存储模型初始化参数
        path = self.results_path_base + 'query_cnt(-1).pt'
        torch.save(self.model.state_dict(), path)
        for cnt in range(self.query_cnts):
            # load model and data loader
            self.train_sampler = data.sampler.SubsetRandomSampler(self.train_indice)
            self.train_loader = data.DataLoader(self.pool_dataset, self.training_batch_size, sampler = self.train_sampler, num_workers=4)
            self.current_cycle = cnt
            model = copy.deepcopy(self.model)
            
            #1. 查询样本并分离出 in distribution 的样本下标和 out distribution 的下标
            unqueried_indice = np.where(self.pool_indice == 1)[0]
            # training model and query samples
            selected_indice = self.query_function(self, model, copy.deepcopy(self.train_indice), unqueried_indice, self.query_batch_size)
            
            self.test_acc.append(self.current_optimal_test_acc);self.train_acc.append(self.current_max_train_acc)
            optimal_model_state_dict = torch.load(self.current_max_train_acc_model_path)
            path = self.results_path_base + 'query_cnt({}).pt'.format(cnt);torch.save(optimal_model_state_dict, path)

            if self.args.data_name != 'SVHN':
                labels = np.array(self.pool_dataset.dataset.targets)
            else:
                labels = np.array(self.pool_dataset.dataset.labels)
            indicator = np.ones_like(labels) #1表示被选中的 in distribution 下标，0表示被选中的 out distribution 下标, -1表示没被选中的下标
            indicator = -indicator
            indicator[selected_indice] = 0
            selected_in_distribution_indice, selected_out_distribution_indice = [], []
            for label in self.in_distribution_labels:
                indicator[ selected_indice[np.where(labels[selected_indice] == label)[0]] ] = 1
            selected_in_distribution_indice = np.where(indicator == 1)[0].tolist()
            selected_out_distribution_indice = np.where(indicator == 0)[0].tolist()
            
            #2. 更新训练集的下标和 pool set 的下标
            self.train_indice += selected_in_distribution_indice
            self.pool_indice[selected_indice] = 0
            self.selected_indistribution_indice += selected_in_distribution_indice
            self.selected_out_distribution_indice += selected_out_distribution_indice
            
            # 存储当前iteration 查询样本的下标
            self.store_queried_indice(np.array(selected_in_distribution_indice), np.array(selected_out_distribution_indice))
            
            print('numbers of selected instance:{}, numbers of in-distribution instance:{}, numbers of out-distribution instance:{}'.format(
                selected_indice.shape[0], len(selected_in_distribution_indice), len(selected_out_distribution_indice)
            ))


        self.train_sampler = data.sampler.SubsetRandomSampler(self.train_indice)
        self.train_loader = data.DataLoader(self.pool_dataset, self.training_batch_size, sampler = self.train_sampler, num_workers=4)
        model = copy.deepcopy(self.model)
        self.train_classifier(model, self.epoches)
        self.test_acc.append(self.current_optimal_test_acc);self.train_acc.append(self.current_max_train_acc)
        path = self.results_path_base + 'query_cnt({}).pt'.format(self.current_cycle + 1)
        optimal_model_state_dict = torch.load(self.current_max_train_acc_model_path)
        torch.save(optimal_model_state_dict, path)