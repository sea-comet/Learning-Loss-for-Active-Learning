'''Active Learning Procedure in PyTorch.

Reference:
[Yoo et al. 2019] Learning Loss for Active Learning (https://arxiv.org/abs/1905.03677)
'''

# Python
import os
import random

# Torch
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler

# Torchvison
import torchvision.transforms as T
import torchvision.models as models
from torchvision.datasets import CIFAR100, CIFAR10  # 查查torch　API

# Utils
import visdom
from tqdm import tqdm  ##进度条 可以封装list 那样的

# Custom
import models.resnet as resnet  # 从自己的文件系统里引入的
import models.lossnet as lossnet
from config import *  # 各种可以调的参数
from data.sampler import SubsetSequentialSampler  # 从自己的文件系统里引入的

# Seed          #使一开始的随机数相同，使这个结果可以复现
random.seed("Inyoung Cho")  # ？？？ str的每个bit都会被用来随机使用
torch.manual_seed(0)  # ？？？Sets the seed for generating random numbers. Returns a torch.Generator object.
torch.backends.cudnn.deterministic = True  # True, Sets whether PyTorch operations must use “deterministic” algorithms
# That is, algorithms which, given the same input,
# and when run on the same software and hardware, always produce the same output.

##
# Data
train_transform = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomCrop(size=32, padding=4),
    T.ToTensor(),  # 查查 Normalize 怎么用
    T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    # T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)) # CIFAR-100
])

test_transform = T.Compose([
    T.ToTensor(),
    T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    # T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)) # CIFAR-100
])

cifar10_train = CIFAR10('../cifar10', train=True, download=True, transform=train_transform)
cifar10_unlabeled = CIFAR10('../cifar10', train=True, download=True, transform=test_transform)
cifar10_test = CIFAR10('../cifar10', train=False, download=True, transform=test_transform)


##
# Loss Prediction Loss
def LossPredLoss(input, target, margin=1.0, reduction='mean'):
    assert len(input) % 2 == 0, 'the batch size is not even.'
    assert input.shape == input.flip(0).shape  # flip　在０维度翻转？？
    # ＃待see 论文 !!!!! ？？？？？问？？？
    input = (input - input.flip(0))[
            :len(input) // 2]  # [l_1 - l_2B, l_2 - l_2B-1, ... , l_B - l_B+1], where batch_size = 2B
    target = (target - target.flip(0))[:len(target) // 2]
    target = target.detach()  # 将target从计算图中分离出来，使其不具备梯度？？？

    one = 2 * torch.sign(torch.clamp(target, min=0)) - 1  # 1 operation which is defined by the authors
    # torch.clamp 夹 使比0小的数都等于0，比0大的不变； torch.sign 把0变成0，大于0的变成1，小于0的变成-1
    if reduction == 'mean':  # 这个返回的好像是只含一个数的tensor
        loss = torch.sum(torch.clamp(margin - one * input, min=0))
        loss = loss / input.size(0)  # Note that the size of input is already halved;  ！！x.size(0)是得到batchsize的值
    elif reduction == 'none':  # 这个返回的好像是一大长串数的tensor
        loss = torch.clamp(margin - one * input, min=0)
    else:
        NotImplementedError()

    return loss


##
# Train Utils
iters = 0  # ?????这是干啥的？？


#
def train_epoch(models, criterion, optimizers, dataloaders, epoch, epoch_loss, vis=None, plot_data=None):
    models['backbone'].train()  # backbone 是resNet
    models['module'].train()  # module 是lossNet
    global iters
    # tqdm 是进度条
    for data in tqdm(dataloaders['train'], leave=False, total=len(dataloaders['train'])):
        inputs = data[0].cuda()  # image 图片
        labels = data[1].cuda()  # label 标签
        iters += 1

        optimizers['backbone'].zero_grad()
        optimizers['module'].zero_grad()

        scores, features = models['backbone'](inputs)  # 回去看resNet的 return值; scores 是最后预测值y'，features 是每一步输出
        target_loss = criterion(scores, labels)  # criterion 用的是crossEntropyLoss, 里面放y'和y

        if epoch > epoch_loss:
            # After 120 epochs, stop the gradient from the loss prediction module propagated to the target model.
            features[0] = features[0].detach()
            features[1] = features[1].detach()
            features[2] = features[2].detach()
            features[3] = features[3].detach()
        pred_loss = models['module'](features)  # ???????问？？？这是啥意思？？？？LossNet 是干啥的？？
        pred_loss = pred_loss.view(pred_loss.size(0))  # pred_loss.size(0)这玩意儿相当于batchSize

        m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)
        m_module_loss = LossPredLoss(pred_loss, target_loss, margin=MARGIN)  # 待see LossPredLoss; MARGIN == 1.0
        loss = m_backbone_loss + WEIGHT * m_module_loss  # WEIGHT == 1.0

        loss.backward()  # Computes the gradient of current tensor
        optimizers['backbone'].step()
        optimizers['module'].step()

        # Visualize
        if (iters % 100 == 0) and (vis != None) and (plot_data != None):
            plot_data['X'].append(iters)  # 横坐标是第几个iter
            plot_data['Y'].append([
                m_backbone_loss.item(),  # Tensor变标量
                m_module_loss.item(),
                loss.item()
            ])
            vis.line(
                X=np.stack([np.array(plot_data['X'])] * len(plot_data['legend']), 1),  # along维度1叠加
                Y=np.array(plot_data['Y']),
                opts={
                    'title': 'Loss over Time',
                    'legend': plot_data['legend'],
                    'xlabel': 'Iterations',
                    'ylabel': 'Loss',
                    'width': 1200,
                    'height': 390,
                },
                win=1
            )


#
def test(models, dataloaders, mode='val'):
    assert mode == 'val' or mode == 'test'
    models['backbone'].eval()
    models['module'].eval()

    total = 0
    correct = 0
    with torch.no_grad():
        for (inputs, labels) in dataloaders[mode]:
            inputs = inputs.cuda()
            labels = labels.cuda()

            scores, _ = models['backbone'](inputs)  # _ 表示不用返回这个， scores 是最后预测值
            _, preds = torch.max(scores.data, 1)  # .data 取出特征值，不要梯度
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    return 100 * correct / total  # 正确率


# train 其实才是外面的函数，train_epoch 在里面
def train(models, criterion, optimizers, schedulers, dataloaders, num_epochs, epoch_loss, vis, plot_data):
    print('>> Train a Model.')
    best_acc = 0.
    checkpoint_dir = os.path.join('./cifar10', 'train', 'weights')  # 看看 checkpoint 怎么用
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    for epoch in range(num_epochs):
        schedulers['backbone'].step()  # 都是 MultiStepLR
        schedulers['module'].step()  # 都是 MultiStepLR

        train_epoch(models, criterion, optimizers, dataloaders, epoch, epoch_loss, vis, plot_data)

        # Save a checkpoint  #See!!!!!
        if False and epoch % 5 == 4:
            acc = test(models, dataloaders, 'test')  # test 模式！！还有val 模式呢，待see !!!
            if best_acc < acc:
                best_acc = acc  # 不断更新best_acc
                torch.save({  # 保存模型
                    'epoch': epoch + 1,
                    'state_dict_backbone': models['backbone'].state_dict(),
                    'state_dict_module': models['module'].state_dict()
                },
                    '%s/active_resnet18_cifar10.pth' % (checkpoint_dir))  # 好的String 写法！！！pth文件存放路径！！
            print('Val Acc: {:.3f} \t Best Acc: {:.3f}'.format(acc, best_acc))  # 好的String 写法！！！
    print('>> Finished.')


# 判断数据的不确定性
def get_uncertainty(models, unlabeled_loader):  # 注意这是没label的数据！！
    models['backbone'].eval()  # eval 模式！！
    models['module'].eval()
    uncertainty = torch.tensor([]).cuda()  # 注意放到cuda里！！先是一个空的tensor

    with torch.no_grad():
        for (inputs, labels) in unlabeled_loader:
            inputs = inputs.cuda()
            # labels = labels.cuda()

            scores, features = models['backbone'](inputs)  # 这儿底下待See!!
            pred_loss = models['module'](features)  # pred_loss = criterion(scores, labels) # ground truth loss
            pred_loss = pred_loss.view(pred_loss.size(0))

            uncertainty = torch.cat((uncertainty, pred_loss), 0)  # 累加pred_loss， 在0 维度上增加！！

    return uncertainty.cpu()  # Returns a copy of this object in CPU memory. ????为啥要放到CPU一份？？


##
# Main
if __name__ == '__main__':
    vis = visdom.Visdom(server='http://localhost', port=9000)  # 用原本的port就得了呗
    plot_data = {'X': [], 'Y': [], 'legend': ['Backbone Loss', 'Module Loss', 'Total Loss']}  # 记住！！

    for trial in range(TRIALS):  # TRIALS == 3  # ？？？问？？？这底下是啥意思？？
        # Initialize a labeled dataset by randomly sampling K=ADDENDUM=1,000 data points from the entire dataset.
        indices = list(range(NUM_TRAIN))  # NUM_TRAIN == 50000 # 这是一串数字，相当于打乱了index
        random.shuffle(indices)  # 随机打乱
        labeled_set = indices[:ADDENDUM]  # ADDENDUM == 1000, 最开始的，不过好像后来逐渐每次label的data加1000
        unlabeled_set = indices[ADDENDUM:]

        train_loader = DataLoader(cifar10_train, batch_size=BATCH,  # BATCH == 128, 可以改改！！
                                  sampler=SubsetRandomSampler(labeled_set),  # 这是怎么弄得？？？好像只选了那1000个label数据来train??
                                  pin_memory=True)  # 这是干啥的？？
        test_loader = DataLoader(cifar10_test, batch_size=BATCH)
        dataloaders = {'train': train_loader, 'test': test_loader}

        # Model
        resnet18 = resnet.ResNet18(num_classes=10).cuda()  # 注意者利用的 resNet18, 还可以改成resnet很深的模型
        loss_module = lossnet.LossNet().cuda()
        models = {'backbone': resnet18, 'module': loss_module}
        torch.backends.cudnn.benchmark = False  # if True, causes cuDNN to benchmark multiple convolution algorithms
        # and select the fastest.

        # Active learning cycles # 主动学习
        for cycle in range(CYCLES):  # CYCLES:10
            # Loss, criterion and scheduler (re)initialization
            criterion = nn.CrossEntropyLoss(reduction='none')
            optim_backbone = optim.SGD(models['backbone'].parameters(), lr=LR,  # ??用Adam 可以嘛？？LR 才0.1, 可以改！！
                                       momentum=MOMENTUM, weight_decay=WDECAY)
            optim_module = optim.SGD(models['module'].parameters(), lr=LR,
                                     momentum=MOMENTUM, weight_decay=WDECAY)   # MOMENTUM = 0.9 WDECAY = 5e-4
            # 什么来？记得看，可以改！！

            sched_backbone = lr_scheduler.MultiStepLR(optim_backbone, milestones=MILESTONES)  # MILESTONE == [170]
            sched_module = lr_scheduler.MultiStepLR(optim_module, milestones=MILESTONES)  # ？？问？？MILESTONE 是干啥的？

            optimizers = {'backbone': optim_backbone, 'module': optim_module}
            schedulers = {'backbone': sched_backbone, 'module': sched_module}

            # Training and test
            train(models, criterion, optimizers, schedulers, dataloaders, EPOCH, EPOCHL, vis, plot_data)
            acc = test(models, dataloaders, mode='test')  # acc 用的 test 的acc, 用的test_dataloader!!!
            print('Trial {}/{} || Cycle {}/{} || Label set size {}: Test acc {}'.format(trial + 1, TRIALS, cycle + 1,
                                                                                        CYCLES, len(labeled_set), acc))

            ##
            #  Update the labeled dataset via loss prediction-based uncertainty measurement

            # Randomly sample 10000 unlabeled data points
            random.shuffle(unlabeled_set)
            subset = unlabeled_set[:SUBSET]  # SUBSET ==10000

            # Create unlabeled dataloader for the unlabeled subset
            unlabeled_loader = DataLoader(cifar10_unlabeled, batch_size=BATCH,
                                          sampler=SubsetSequentialSampler(subset),
                                          # more convenient if we maintain the order of subset
                                          pin_memory=True)

            # Measure uncertainty of each data points in the subset
            uncertainty = get_uncertainty(models, unlabeled_loader)

            # Index in ascending order
            arg = np.argsort(uncertainty)  # sort array, 升序排列index返回

            # Update the labeled dataset and the unlabeled dataset, respectively
            labeled_set += list(torch.tensor(subset)[arg][-ADDENDUM:].numpy())
            #  ！！！！！！！在10000 Subset 里挑出 uncertainty最大的，更新labeled_set
            unlabeled_set = list(torch.tensor(subset)[arg][:-ADDENDUM].numpy()) + unlabeled_set[SUBSET:]

            # Create a new dataloader for the updated labeled dataset
            dataloaders['train'] = DataLoader(cifar10_train, batch_size=BATCH,
                                              sampler=SubsetRandomSampler(labeled_set),
                                              pin_memory=True)

        # Save a checkpoint
        torch.save({      # ？？？？待See!! 为什么文件夹里没有？？？
            'trial': trial + 1,
            'state_dict_backbone': models['backbone'].state_dict(),
            'state_dict_module': models['module'].state_dict()
        },
            './cifar10/train/weights/active_resnet18_cifar10_trial{}.pth'.format(trial))
