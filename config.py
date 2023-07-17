import argparse

import models
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from data_loader import (get_imbalanced, get_oversampled, get_smote,
                         make_longtailed_imb)
from fuzzywuzzy import process
from torchvision import datasets, transforms
from utils import InputNormalize, sum_t

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True
if torch.cuda.is_available():
    N_GPUS = torch.cuda.device_count()
else:
    N_GPUS = 0

def get_class_sample_counts(dataset):
    class_sample_counts = [0] * len(dataset.classes)
    
    for _, label in dataset:
        class_sample_counts[label] += 1
    
    return class_sample_counts

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--model', default='resnet32', type=str,
                        help='model type (default: ResNet18)')
    parser.add_argument('--batch-size', default=128, type=int, help='batch size')
    parser.add_argument('--epoch', default=200, type=int,
                        help='total epochs to run')
    parser.add_argument('--seed', default=None, type=int, help='random seed')
    parser.add_argument('--dataset', required=True,
                        choices=['cifar10', 'cifar100', 'svhn', 'celebA', 'sun397'], help='Dataset')
    parser.add_argument('--decay', default=2e-4, type=float, help='weight decay')
    parser.add_argument('--no-augment', dest='augment', action='store_false',
                        help='use standard augmentation (default: True)')

    parser.add_argument('--name', default='0', type=str, help='name of run')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    parser.add_argument('--net_g', default=None, type=str,
                        help='checkpoint path of network for generation')
    parser.add_argument('--net_g2', default=None, type=str,
                        help='checkpoint path of network for generation')
    parser.add_argument('--net_t', default=None, type=str,
                        help='checkpoint path of network for train')
    parser.add_argument('--net_both', default=None, type=str,
                        help='checkpoint path of both networks')

    parser.add_argument('--beta', default=0.999, type=float, help='Hyper-parameter for rejection/sampling')
    parser.add_argument('--lam', default=0.5, type=float, help='Hyper-parameter for regularization of translation')
    parser.add_argument('--warm', default=160, type=int, help='Deferred strategy for re-balancing')
    parser.add_argument('--gamma', default=0.99, type=float, help='Threshold of the generation')

    parser.add_argument('--eff_beta', default=1.0, type=float, help='Hyper-parameter for effective number')
    parser.add_argument('--focal_gamma', default=1.0, type=float, help='Hyper-parameter for Focal Loss')

    parser.add_argument('--gen', '-gen', action='store_true', help='')
    parser.add_argument('--step_size', default=0.1, type=float, help='')
    parser.add_argument('--attack_iter', default=10, type=int, help='')

    parser.add_argument('--imb_type', default='longtail', type=str,
                        choices=['none', 'longtail', 'step'],
                        help='Type of artificial imbalance')
    parser.add_argument('--loss_type', default='CE', type=str,
                        choices=['CE', 'Focal', 'LDAM'],
                        help='Type of loss for imbalance')
    parser.add_argument('--ratio', default=100, type=int, help='max/min')
    parser.add_argument('--imb_start', default=5, type=int, help='start idx of step imbalance')

    parser.add_argument('--smote', '-s', action='store_true', help='oversampling')
    parser.add_argument('--cost', '-c', action='store_true', help='oversampling')
    parser.add_argument('--effect_over', action='store_true', help='Use effective number in oversampling')
    parser.add_argument('--no_over', dest='over', action='store_false', help='Do not use over-sampling')

    return parser.parse_args()


ARGS = parse_args()
if ARGS.seed is not None:
    SEED = ARGS.seed
else:
    SEED = np.random.randint(10000)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DATASET, _ = process.extractOne(ARGS.dataset.lower(), datasets.__all__)
BATCH_SIZE = ARGS.batch_size
MODEL = ARGS.model

LR = ARGS.lr
EPOCH = ARGS.epoch
START_EPOCH = 0

LOGFILE_BASE = f"S{SEED}_{ARGS.name}_" \
    f"L{ARGS.lam}_W{ARGS.warm}_" \
    f"E{ARGS.step_size}_I{ARGS.attack_iter}_" \
    f"{DATASET}_R{ARGS.ratio}_{MODEL}_G{ARGS.gamma}_B{ARGS.beta}"

# Data
print('==> Preparing data: %s' % DATASET)

if DATASET == 'cifar100':
    N_CLASSES = 100
    N_SAMPLES = 500
    mean = torch.tensor([0.5071, 0.4867, 0.4408])
    std = torch.tensor([0.2675, 0.2565, 0.2761])
elif DATASET == 'cifar10':
    N_CLASSES = 10
    N_SAMPLES = 5000
    mean = torch.tensor([0.4914, 0.4822, 0.4465])
    std = torch.tensor([0.2023, 0.1994, 0.2010]) # [0.247,  0.2435, 0.2616]
elif DATASET == 'svhn':
    N_CLASSES = 10
    N_SAMPLES = [4948, 13861, 10585, 8497, 7458, 6882, 5727, 5595, 5045, 4659]
    mean = torch.tensor([0.4377, 0.4438, 0.4728])
    std = torch.tensor([0.198, 0.201, 0.197])
elif DATASET == 'sun397':
    N_CLASSES = 397
    N_SAMPLES = [484, 115, 1091, 324, 316, 219, 750, 173, 527, 164, 169, 354, 218, 146, 164, 283, 139, 310, 159, 114, 127, 332, 340, 242, 463, 209, 221, 482, 125, 177, 202, 112, 141, 296, 738, 279, 140, 287, 152, 295, 111, 951, 117, 166, 156, 111, 1194, 549, 2084, 120, 100, 100, 144, 123, 111, 210, 483, 263, 211, 131, 410, 124, 124, 847, 325, 187, 112, 182, 227, 139, 318, 104, 476, 174, 193, 490, 259, 255, 222, 125, 392, 590, 1100, 135, 238, 635, 430, 569, 314, 101, 174, 102, 287, 159, 236, 921, 225, 130, 234, 199, 300, 336, 458, 695, 161, 186, 105, 833, 527, 100, 445, 339, 134, 332, 229, 376, 147, 100, 565, 547, 172, 186, 142, 154, 242, 166, 313, 195, 164, 118, 117, 197, 243, 1180, 137, 130, 550, 165, 281, 125, 126, 101, 139, 127, 121, 125, 394, 334, 101, 259, 169, 343, 248, 294, 283, 111, 167, 154, 103, 265, 162, 502, 152, 251, 212, 265, 221, 171, 100, 328, 803, 101, 161, 140, 814, 357, 136, 372, 205, 131, 250, 144, 101, 221, 833, 144, 178, 120, 203, 116, 241, 120, 492, 955, 103, 113, 207, 133, 253, 219, 399, 166, 134, 120, 330, 111, 193, 185, 174, 179, 158, 207, 144, 1746, 133, 140, 430, 219, 308, 320, 192, 358, 102, 138, 128, 489, 108, 2361, 464, 101, 365, 454, 266, 101, 839, 193, 183, 201, 104, 110, 178, 124, 536, 254, 518, 736, 218, 248, 110, 369, 116, 250, 109, 202, 232, 136, 300, 159, 210, 207, 368, 310, 200, 223, 565, 143, 100, 111, 439, 329, 612, 117, 145, 159, 316, 163, 148, 222, 115, 898, 244, 228, 120, 143, 179, 125, 397, 142, 102, 140, 142, 126, 116, 145, 219, 128, 231, 188, 118, 110, 796, 134, 443, 100, 106, 228, 109, 112, 327, 180, 135, 203, 177, 139, 126, 116, 288, 181, 725, 284, 125, 100, 112, 188, 260, 167, 801, 192, 189, 116, 161, 280, 117, 123, 680, 458, 455, 518, 373, 119, 143, 256, 428, 129, 147, 197, 202, 155, 122, 429, 223, 237, 109, 100, 114, 131, 103, 112, 704, 388, 136, 276, 119, 118, 113, 105, 494, 144, 145, 131, 496, 100, 140, 254, 125, 138, 452, 126, 107, 137, 476, 777, 396, 210, 203, 106, 107, 274, 135, 247, 302, 413, 246, 340, 100, 201, 138]
    mean = torch.tensor([0.4758, 0.4603, 0.4248])
    std = torch.tensor([0.2360, 0.2343, 0.2471])
else:
    raise NotImplementedError()

normalizer = InputNormalize(mean, std).to(device)

if any(substring in DATASET for substring in ('cifar', 'svhn', 'sun397')):
    if ARGS.augment:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
else:
    raise NotImplementedError()

## Data Loader ##

N_SAMPLES_PER_CLASS_BASE = get_class_sample_counts(getattr(datasets, DATASET))
N_CLASSES = len(N_SAMPLES_PER_CLASS_BASE)

if all(count == N_SAMPLES_PER_CLASS_BASE[0] for count in N_SAMPLES_PER_CLASS_BASE):
    print(f"{DATASET} is a balanced dataset. Making it {ARGS.imb_type}-imbalanced.")
    if ARGS.imb_type == 'longtail':
        N_SAMPLES_PER_CLASS_BASE = make_longtailed_imb(N_SAMPLES_PER_CLASS_BASE[0], N_CLASSES, ARGS.ratio)

    elif ARGS.imb_type == 'step':
        for i in range(ARGS.imb_start, N_CLASSES):
            N_SAMPLES_PER_CLASS_BASE[i] = int(N_SAMPLES_PER_CLASS_BASE[0] * (1 / ARGS.ratio))

else:
    print(f"{DATASET} is naturally imbalanced. Ignoring the imb_type argument: {ARGS.imb_type}.")

N_SAMPLES_PER_CLASS_BASE = tuple(N_SAMPLES_PER_CLASS_BASE)
print(N_SAMPLES_PER_CLASS_BASE)

train_loader, val_loader, test_loader = get_imbalanced(DATASET, N_SAMPLES_PER_CLASS_BASE, BATCH_SIZE,
                                                       transform_train, transform_test)

## To apply effective number for over-sampling or cost-sensitive ##

if ARGS.over and ARGS.effect_over:
    _beta = ARGS.eff_beta
    effective_num = 1.0 - np.power(_beta, N_SAMPLES_PER_CLASS_BASE)
    N_SAMPLES_PER_CLASS = tuple(np.array(effective_num) / (1 - _beta))
    print(N_SAMPLES_PER_CLASS)
else:
    N_SAMPLES_PER_CLASS = N_SAMPLES_PER_CLASS_BASE
N_SAMPLES_PER_CLASS_T = torch.Tensor(N_SAMPLES_PER_CLASS).to(device)


def adjust_learning_rate(optimizer, lr_init, epoch):
    """decrease the learning rate at 160 and 180 epoch ( from LDAM-DRW, NeurIPS19 )"""
    lr = lr_init

    if epoch < 5:
        lr = (epoch + 1) * lr_init / 5
    else:
        if epoch >= 160:
            lr /= 100
        if epoch >= 180:
            lr /= 100

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def evaluate(net, dataloader, logger=None):
    is_training = net.training
    net.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct, total = 0.0, 0.0
    major_correct, neutral_correct, minor_correct = 0.0, 0.0, 0.0
    major_total, neutral_total, minor_total = 0.0, 0.0, 0.0

    class_correct = torch.zeros(N_CLASSES)
    class_total = torch.zeros(N_CLASSES)

    for inputs, targets in dataloader:
        batch_size = inputs.size(0)
        inputs, targets = inputs.to(device), targets.to(device)

        outputs, _ = net(normalizer(inputs))
        loss = criterion(outputs, targets)

        total_loss += loss.item() * batch_size
        predicted = outputs[:, :N_CLASSES].max(1)[1]
        total += batch_size
        correct_mask = (predicted == targets)
        correct += sum_t(correct_mask)

        # For accuracy of minority / majority classes.
        major_mask = targets < (N_CLASSES // 3)
        major_total += sum_t(major_mask)
        major_correct += sum_t(correct_mask * major_mask)

        minor_mask = targets >= (N_CLASSES - (N_CLASSES // 3))
        minor_total += sum_t(minor_mask)
        minor_correct += sum_t(correct_mask * minor_mask)

        neutral_mask = ~(major_mask + minor_mask)
        neutral_total += sum_t(neutral_mask)
        neutral_correct += sum_t(correct_mask * neutral_mask)

        for i in range(N_CLASSES):
            class_mask = (targets == i)
            class_total[i] += sum_t(class_mask)
            class_correct[i] += sum_t(correct_mask * class_mask)

    results = {
        'loss': total_loss / total,
        'acc': 100. * correct / total,
        'major_acc': 100. * major_correct / major_total,
        'neutral_acc': 100. * neutral_correct / neutral_total,
        'minor_acc': 100. * minor_correct / minor_total,
        'class_acc': 100. * class_correct / class_total,
    }

    msg = 'Loss: %.3f | Acc: %.3f%% (%d/%d) | Major_ACC: %.3f%% | Neutral_ACC: %.3f%% | Minor ACC: %.3f%% ' % \
          (
              results['loss'], results['acc'], correct, total,
              results['major_acc'], results['neutral_acc'], results['minor_acc']
          )
    if logger:
        logger.log(msg)
    else:
        print(msg)

    net.train(is_training)
    return results
