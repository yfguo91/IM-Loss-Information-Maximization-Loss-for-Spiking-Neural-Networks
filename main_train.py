import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import argparse
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100
from data import CIFAR10Policy, Cutout
from data.sampler import DistributedSampler
import time
from models import *
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from models.spike_layer import SpikePool, LIFAct
import datetime
_seed_ = 2020
import random
import math
random.seed(2020)

torch.manual_seed(_seed_)  # use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA)
torch.cuda.manual_seed_all(_seed_)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def build_data(batch_size=128, cutout=False, workers=4, use_cifar10=False, auto_aug=False):

    aug = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    if auto_aug:
        aug.append(CIFAR10Policy())

    aug.append(transforms.ToTensor())

    if cutout:
        aug.append(Cutout(n_holes=1, length=16))

    if use_cifar10:
        aug.append(
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
        transform_train = transforms.Compose(aug)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = CIFAR10(root='./cifar10/',
                                train=True, download=True, transform=transform_train)
        val_dataset = CIFAR10(root='./cifar10/',
                              train=False, download=True, transform=transform_test)

    else:
        aug.append(
            transforms.Normalize(
                (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        )
        transform_train = transforms.Compose(aug)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        train_dataset = CIFAR100(root='./cifar100/',
                                 train=True, download=True, transform=transform_train)
        val_dataset = CIFAR100(root='./cifar100/',
                               train=False, download=False, transform=transform_test)

    #train_sampler = DistributedSampler(train_dataset)
    #val_sampler = DistributedSampler(val_dataset, round_up=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False,
                              num_workers=workers, pin_memory=True)

    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=workers, pin_memory=True)

    return train_loader, val_loader


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='model parameters',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset', default='CIFAR10', type=str, help='dataset name',
                        choices=['MNIST', 'CIFAR10', 'CIFAR100'])
    parser.add_argument('--arch', default='res20m', type=str, help='dataset name',
                        choices=['CIFARNet', 'VGG16', 'res19', 'res20', 'res20m', 'VGG11'])
    parser.add_argument('--batch_size', default=128, type=int, help='minibatch size')
    parser.add_argument('--learning_rate', default=1e-2, type=float, help='initial learning_rate')
    parser.add_argument('--epochs', default=100, type=int, help='number of training epochs')
    parser.add_argument('--thresh', default=1, type=int, help='snn threshold')
    parser.add_argument('--T', default=100, type=int, help='snn simulation length')
    parser.add_argument('--shift_snn', default=100, type=int, help='SNN left shift reference time')
    parser.add_argument('--step', default=4, type=int, help='snn step')
    parser.add_argument('--spike', action='store_true', help='use spiking network')
    parser.add_argument('--distribution', action='store_true', help='use distribution')
    parser.add_argument('--teacher', action='store_true', help='use teacher')
    parser.add_argument('--recon', action='store_true', help='use teacher')
    parser.add_argument('--seed', type=int, default=666, metavar='S',
                        help='random seed (default: 1)')
    writer = SummaryWriter('./summaries/resnet_2/'+datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))

    args = parser.parse_args()
    #torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    batch_size = args.batch_size
    learning_rate = args.learning_rate
    activation_save_name = args.arch + '_' + args.dataset + '_activation.npy'
    use_cifar10 = args.dataset == 'CIFAR10'

    train_loader, test_loader = build_data(cutout=False, use_cifar10=use_cifar10, auto_aug=False, batch_size=args.batch_size)
    best_acc = 0
    best_epoch = 0

    name = 'snn_T{}'.format(args.step) if args.spike is True else 'ann'
    model_save_name = 'raw/' + name + '_' + args.arch + '_' + str(args.batch_size)+'_'+ str(args.dataset) +'_wd1e4.pth'
    if args.distribution:
        model_save_name = model_save_name + '_dis_1' +'.pth'
    else:
        model_save_name = model_save_name +'.pth'
    if args.arch == 'CNN2':
        raise NotImplementedError
    elif args.arch == 'res20':
        ann = resnet20_cifar(num_classes=10 if use_cifar10 else 100)
        ann.load_state_dict(torch.load('raw/ann_res20wd5e4.pth', map_location='cpu'))
    elif args.arch == 'res19':
        ann = resnet19_cifar(num_classes=10 if use_cifar10 else 100)
        #ann.load_state_dict(torch.load('raw/snn_T2_res19_wd1e4.pth', map_location='cpu'))
    elif args.arch == 'res20m':
        ann = resnet20_cifar_modified(num_classes=10 if use_cifar10 else 100)
        #ann.load_state_dict(torch.load('raw/ann_res20m_wd5e4.pth', map_location='cpu'))
    elif args.arch == 'VGG16':
        ann = vgg16_bn(num_classes=10 if use_cifar10 else 100)
        #ann.load_state_dict(torch.load('raw/ann_res20m_wd5e4.pth', map_location='cpu'))
    elif args.arch == 'VGG11':
        ann = vgg11_bn(num_classes=10 if use_cifar10 else 100)
    else:
        raise NotImplementedError
    if args.spike is True:
        ann = SpikeModel(ann, args.step, args.distribution)
        ann.set_spike_state(True)

    if args.teacher is True:
        teacher = resnet20_cifar_modified(num_classes=10 if use_cifar10 else 100)
        teacher.to(device)
        teacher.eval()
        #teacher.load_state_dict(torch.load('raw/ann_res20m_wd5e4.pth'))
    else:
        teacher = None


    #ann.load_state_dict(torch.load('raw/snn_T2_res20m_wd1e4.pth', map_location='cpu'))
    ann.to(device)
    #device = next(ann.parameters()).device
    #ann.load_state_dict(torch.load('raw/snn_T3_res20m_wd1e4.pth'), strict=False)
    #ann.load_state_dict(torch.load('raw/snn_T2_res19_wd1e4.pth', map_location='cpu'), strict=False)
    #ann.load_state_dict(torch.load('raw/ann_res19_wd1e4.pth'), strict=False)snn_T4_res20m_wd5e4.pth
    
    num_epochs = 100
    criterion = nn.CrossEntropyLoss().to(device)
    # build optimizer
    optimizer = torch.optim.SGD(ann.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=num_epochs)

    #
    correct = torch.Tensor([0.]).to(device)
    total = torch.Tensor([0.]).to(device)
    acc = torch.Tensor([0.]).to(device)
    ann.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs,_ = ann(inputs)
            _, predicted = outputs.cpu().max(1)
            total += (targets.size(0))
            correct += (predicted.eq(targets.cpu()).sum().item())

    acc = 100 * correct / total
    print('Test Accuracy of the model on the 10000 test images: {}'.format(acc.item()))
    if best_acc < acc:
        best_acc = acc
        torch.save(ann.state_dict(), model_save_name)
    print('first_acc is: {}'.format(best_acc))
    writer.add_scalar('first Acc /epoch', 100. * correct / len(test_loader.dataset))

    for epoch in range(num_epochs):
        running_loss = 0
        start_time = time.time()
        ann.train()
        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            labels = labels.to(device)
            images = images.to(device)
            outputs, disloss = ann(images)
            if args.distribution is True:  
                #loss = criterion(outputs, labels) + 0.05*(1+math.cos(math.pi*epoch/num_epochs))*disloss
                loss = criterion(outputs, labels) + 0.005*disloss
                #loss = criterion(outputs, labels)
            else:
                loss = criterion(outputs, labels)
            if args.teacher is True:
                with torch.no_grad():
                    out_t = teacher(images)
                T = 3
                loss_kd = F.kl_div(F.log_softmax(outputs / T, dim=1), F.softmax(out_t / T, dim=1), reduction='batchmean') * T * T
                loss = loss_kd * 0.5 + loss * 0.5
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
            if (i + 1) % 80 == 0:
                print('Time elapsed: {}'.format(time.time() - start_time))
                writer.add_scalar('Train Loss /batchidx', loss, i + len(train_loader) * epoch)
                print(loss)
                print(disloss)
        scheduler.step()
        

        correct = torch.Tensor([0.]).to(device)
        total = torch.Tensor([0.]).to(device)
        acc = torch.Tensor([0.]).to(device)

        # start testing
        ann.eval()
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs,_ = ann(inputs)
                _, predicted = outputs.cpu().max(1)
                total += (targets.size(0))
                correct += (predicted.eq(targets.cpu()).sum().item())

        acc = 100 * correct / total
        print('Test Accuracy of the model on the 10000 test images: {}'.format(acc.item()))
        if best_acc < acc:
            best_acc = acc
            best_epoch = epoch + 1
            torch.save(ann.state_dict(), model_save_name)
        print('best_acc is: {}'.format(best_acc))
        print('best_iter: {}'.format(best_epoch))
        print('Iters: {}\n\n'.format(epoch))
        writer.add_scalar('Test Acc /epoch', 100. * correct / len(test_loader.dataset), epoch)
        #for name, m in ann.named_modules():
        #    if isinstance(m, LIFAct):
        #        print(m.temp)
    writer.close()