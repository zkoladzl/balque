import argparse
import torchvision.transforms as transforms
import datasets.datasets as datasets
import torch
print(torch.__version__)
import numpy as np
import model.resnet as resnet
def get_config(base_path):
    '''
        @param base_path the path of dataset and results
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--sampling_name', type = str, default = None)
    parser.add_argument('--model', type = object, default = None)
    parser.add_argument('--train_dataset', type = object, default = None)
    parser.add_argument('--test_dataset', type = object, default = None)
    parser.add_argument('--test_in_distribution_indice', type = list, default = [])
    parser.add_argument('--validate_nums', type = int, default = 0)
    parser.add_argument('--validate_indice', type = list, default = [])

    parser.add_argument('--in_distribution_labels', type = list, default = [])
    parser.add_argument('--out_distribution_labels', type = list, default = [])
    parser.add_argument('--out_distribution_ratio', type = float, default = 0.0)

    parser.add_argument('--loss_function', type = object, default = torch.nn.CrossEntropyLoss())
    
    
    parser.add_argument('--initial_indice', type = list, default = [])
    parser.add_argument('--pool_indice', type = list, default = [])
    
    #updating ... 
    parser.add_argument('--results_path', type = str, default = base_path + "results/")
    parser.add_argument('--data_name', type = str, default = 'CIFAR10')
    parser.add_argument('--data_path', type = str, default = base_path + 'datas/cifar10')
    parser.add_argument('--in_distribution_num_classes', type = int, default = 10)
    parser.add_argument('--total_num_classes', type = int, default = 10)
    parser.add_argument('--initial_num_per_class', type = int, default = 100)# svhn:300;cifar:100
    parser.add_argument('--pool_size', help = "the size of pool", type = int, default = 49000)#training set size: 73257
    parser.add_argument('--epoches', type = int, default = 200)
    parser.add_argument('--lr', type = float, default = 0.00005)
    parser.add_argument('--min_lr', type = float, default = 0.001)
    parser.add_argument('--weight_decay', type = float, default = 0.00005)
    parser.add_argument('--optimizer_type', type = str, default = 'adam')
    parser.add_argument('--query_cnts', type = int, default = 2)
    parser.add_argument('--query_batch_size', type = int, default = 2000)
    parser.add_argument('--training_batch_size', type = int, default = 32)
    parser.add_argument('--test_batch_size', type = int, default = 128)
    parser.add_argument('--early_stop', type = bool, default = True)

    args = parser.parse_args()
    

    args.in_distribution_labels = [i for i in range(args.in_distribution_num_classes)]
    args.out_distribution_labels = [i for i in range(args.in_distribution_num_classes, args.total_num_classes, 1)]
    args.model = resnet.resnet18(num_classes = args.in_distribution_num_classes)
    if args.data_name == 'CIFAR10':
        train_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        args.train_dataset = datasets.CIFAR10_train(args.data_path, train_transforms)
        args.test_dataset = datasets.CIFAR10_test(args.data_path, test_transforms)
    elif args.data_name == 'CIFAR100':
        train_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        args.train_dataset = datasets.CIFAR100_train(args.data_path, train_transforms)
        args.test_dataset = datasets.CIFAR100_test(args.data_path, test_transforms)
    elif args.data_name == 'SVHN':
        common_transforms = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
        ])
        args.train_dataset = datasets.SVHN_train(args.data_path, common_transforms)
        args.test_dataset = datasets.SVHN_test(args.data_path, common_transforms)
    elif args.data_name == 'tiny-imagenet-200':
        train_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(32),
            transforms.ToTensor(),
            transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)),
        ])
        test_transforms = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821))
        ])
        args.train_dataset = datasets.TinyImagenet200_train(args.data_path, train_transforms)
        args.test_dataset = datasets.TinyImagenet200_test(args.data_path, test_transforms)
    
    if args.data_name == 'tiny-imagenet-200':
        # store_tiny_imagenet_200_dataset(args.results_path + args.data_name + '_', args.train_dataset.dataset)
        train_labels = np.array(args.train_dataset.dataset.targets)
    elif args.data_name != 'SVHN':
        # store_cifar_dataset(args.results_path + args.data_name + '_', args.train_dataset.dataset)
        train_labels = np.array(args.train_dataset.dataset.targets)
    else:
        # store_svhn_dataset(args.results_path + args.data_name + '_', args.train_dataset.dataset)
        train_labels = np.array(args.train_dataset.dataset.labels)

    indice_grouped_by_label = []
    for label in range(len(args.in_distribution_labels) + len(args.out_distribution_labels)):
        indice_grouped_by_label.append(np.random.permutation(np.where(train_labels == label)[0]).tolist() )
    
    in_distribution_cnts = 0
    for label in range(len(args.in_distribution_labels)):
        in_distribution_cnts += len(indice_grouped_by_label[label])
    out_distribution_cnts = 0
    for label in range(len(args.in_distribution_labels), len(args.in_distribution_labels) + len(args.out_distribution_labels), 1):
        out_distribution_cnts += len(indice_grouped_by_label[label])

    # initial indices
    indice = []
    for label in range(len(args.in_distribution_labels)):
        indice += indice_grouped_by_label[label][0:min(args.initial_num_per_class, len(indice_grouped_by_label[label]))]
        indice_grouped_by_label[label] = indice_grouped_by_label[label][min(args.initial_num_per_class, len(indice_grouped_by_label[label])):len(indice_grouped_by_label[label])]
        in_distribution_cnts -= min(args.initial_num_per_class, len(indice_grouped_by_label[label]))
    args.initial_indice = indice
    args.initial_indice = np.random.permutation(args.initial_indice).tolist()
    
    out_distribution_unlabeled_size = int(args.pool_size*args.out_distribution_ratio)
    in_distribution_unlabeled_size = args.pool_size - out_distribution_unlabeled_size

    # unlabeled in distribution indices
    indice = [];cnt = 0
    if in_distribution_cnts > 0:
        for label in args.in_distribution_labels:
            if label == len(args.in_distribution_labels) - 1:nums = in_distribution_unlabeled_size - cnt
            else:nums = int(in_distribution_unlabeled_size*len(indice_grouped_by_label[label])/in_distribution_cnts)
            indice += indice_grouped_by_label[label][0:nums]
            indice_grouped_by_label[label] = indice_grouped_by_label[label][nums:len(indice_grouped_by_label[label])]
            cnt += nums
    args.pool_indice += indice
    # unlabeled out distribution indices
    indice = [];cnt = 0
    for label in range(len(args.in_distribution_labels), len(args.in_distribution_labels) + len(args.out_distribution_labels), 1):
        if label == args.total_num_classes - 1:nums = out_distribution_unlabeled_size - cnt
        else:nums = int(out_distribution_unlabeled_size*len(indice_grouped_by_label[label])/out_distribution_cnts)
        indice += indice_grouped_by_label[label][0:nums]
        indice_grouped_by_label[label] = indice_grouped_by_label[label][nums:len(indice_grouped_by_label[label])]
        cnt += nums
    args.pool_indice += indice
    indicator = np.zeros(len(args.train_dataset))

    if args.validate_nums > 0:
        tmp_pool_indice = np.random.permutation(args.pool_indice).tolist()
        args.validate_indice = tmp_pool_indice[0:args.validate_nums]
        args.pool_indice = tmp_pool_indice[args.validate_nums:len(tmp_pool_indice)]
    indicator[args.pool_indice] = 1
    args.pool_indice = indicator
    print('training size:{}, validation size:{}, pool size:{}, test size:{}'.format(
            len(args.initial_indice),
            len(args.validate_indice),
            np.where(args.pool_indice == 1)[0].shape[0],
            len(args.test_dataset.dataset)
        )
    )
    # store_indice(args.results_path  + args.data_name + '_', np.array(args.initial_indice), np.where(args.pool_indice == 1)[0], np.array(args.validate_indice))
    
    #test in distribution indice
    if args.data_name != 'SVHN':
        test_labels = np.array(args.test_dataset.dataset.targets)
    else:
        test_labels = np.array(args.test_dataset.dataset.labels)
    indice = []
    for label in args.in_distribution_labels:
        indice += np.where(test_labels == label)[0].tolist()
    args.test_in_distribution_indice = np.random.permutation(np.array(indice)).tolist()

    return args
def store_cifar_dataset(path, train_dataset):
    labels = np.array(train_dataset.targets)
    datas = train_dataset.data
    np.save(path + 'train_dataset.npy', datas)
    np.save(path + 'train_labels.npy', labels)
def store_svhn_dataset(path, train_dataset):
    labels = np.array(train_dataset.labels)
    datas = train_dataset.data
    np.save(path + 'train_dataset.npy', datas)
    np.save(path + 'train_labels.npy', labels)
def store_tiny_imagenet_200_dataset(path, train_dataset):
    labels = np.array(train_dataset.targets)
    datas = train_dataset.imgs
    np.save(path + 'train_dataset.npy', datas)
    np.save(path + 'train_labels.npy', labels)
def store_indice(path, initial_indice, pool_indice, val_indice):
    np.save(path + 'initial_indice.npy', initial_indice)
    np.save(path + 'pool_indice.npy', pool_indice)
    np.save(path + 'validate_indice.npy', pool_indice)
