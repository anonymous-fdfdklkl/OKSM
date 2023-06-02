import os
import argparse
import datetime
import time
import csv
import pandas as pd
import importlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn

from models import gan
from models.models import classifier32, classifier32ABN,openset
from datasets.osr_dataloader import MNIST_OSR, kfw_loader, fiw_loader
from utils import Logger, save_networks, load_networks
from core import train, train_cs, test,test_while_train,test_fiw

parser = argparse.ArgumentParser("Training")

# Dataset
parser.add_argument('--dataset', type=str, default='fiw', help="mnist | kfw1 |kfw2 |fiw")
parser.add_argument('--dataroot', type=str, default='./data')
parser.add_argument('--outf', type=str, default='./log')
parser.add_argument('--out-num', type=int, default=10, help='For CIFAR100')

# optimization
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.1, help="learning rate for model")
parser.add_argument('--gan_lr', type=float, default=0.0002, help="learning rate for gan")
parser.add_argument('--max-epoch', type=int, default=200)
parser.add_argument('--stepsize', type=int, default=30)
parser.add_argument('--temp', type=float, default=1.0, help="temp")
parser.add_argument('--num-centers', type=int, default=1)
parser.add_argument('--name', type=str, default='fiw_7type')

# model
parser.add_argument('--weight-pl', type=float, default=0.1, help="weight for center loss")
parser.add_argument('--beta', type=float, default=0.1, help="weight for entropy loss")
parser.add_argument('--model', type=str, default='classifier32')

# misc
parser.add_argument('--nz', type=int, default=100)
parser.add_argument('--ns', type=int, default=1)
parser.add_argument('--eval-freq', type=int, default=10)
parser.add_argument('--print-freq', type=int, default=100)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--use-cpu', action='store_true')
parser.add_argument('--save-dir', type=str, default='../log')
parser.add_argument('--loss', type=str, default='GCPLoss')
parser.add_argument('--eval', action='store_true', help="Eval", default=True)
parser.add_argument('--cs', action='store_true', help="Confusing Sample", default=False)

parser.add_argument('--real_sn', default=False, type=bool, help='whether test real scenario')
parser.add_argument('--num_classes', type=int, default=7)
parser.add_argument('--feat_dim', type=int, default=128)
parser.add_argument('--cs_num', type=int, default=1)

def main_worker(options):
    torch.manual_seed(options['seed'])
    os.environ['CUDA_VISIBLE_DEVICES'] = options['gpu']
    use_gpu = torch.cuda.is_available()
    if options['use_cpu']: use_gpu = False

    if use_gpu:
        print("Currently using GPU: {}".format(options['gpu']))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(options['seed'])
    else:
        print("Currently using CPU")

    # Dataset
    print("{} Preparation".format(options['dataset']))
    if 'mnist' in options['dataset']:
        Data = MNIST_OSR(known=options['known'], dataroot=options['dataroot'], batch_size=options['batch_size'],
                         img_size=options['img_size'])
        trainloader, testloader, outloader = Data.train_loader, Data.test_loader, Data.out_loader
    elif 'kfw' in options['dataset']:
        trainloader, testloader = kfw_loader(options).get_train_loader()
        knownloader, unknownloader,testloader = kfw_loader(options).get_kfw_eval_loaders()
    else:
        trainloader, _ = fiw_loader(options).get_close_train_loader()
        _, _, testloader = fiw_loader(options).get_fiw_eval_loaders_new()

        _, _, testloader_eval = fiw_loader(options).get_fiw_eval_loaders_new_get()



    # Model
    print("Creating model: {}".format(options['model']))
    if options['cs']:
        net = classifier32ABN(num_classes=options['num_classes'])
    else:
        net = classifier32(num_classes=options['num_classes'])
        # net = openset(num_classes=options['num_classes'])

    feat_dim = options['feat_dim']

    if options['cs']:
        print("Creating GAN")
        nz, ns = options['nz'], 1
        if 'tiny_imagenet' in options['dataset']:
            netG = gan.Generator(1, nz, 64, 3)
            netD = gan.Discriminator(1, 3, 64)
        else:
            netG = gan.Generator32(1, nz, 64, 6)
            netD = gan.Discriminator32(1, 6, 64)
        fixed_noise = torch.FloatTensor(64, nz, 1, 1).normal_(0, 1)
        criterionD = nn.BCELoss()

    # Loss
    options.update(
        {
            'feat_dim': feat_dim,
            'use_gpu': use_gpu
        }
    )

    Loss = importlib.import_module('loss.' + options['loss'])
    criterion = getattr(Loss, options['loss'])(**options)

    if use_gpu:
        net = nn.DataParallel(net).cuda()
        criterion = criterion.cuda()
        if options['cs']:
            netG = nn.DataParallel(netG, device_ids=[i for i in range(len(options['gpu'].split(',')))]).cuda()
            netD = nn.DataParallel(netD, device_ids=[i for i in range(len(options['gpu'].split(',')))]).cuda()
            fixed_noise.cuda()

    model_path = os.path.join(options['outf'], 'models', options['dataset'])
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    if options['dataset'] == 'cifar100':
        model_path += '_50'
        file_name = '{}_{}_{}_{}_{}'.format(options['model'], options['loss'], 50, options['item'], options['cs'])
    else:
        file_name = '{}_{}_{}_{}_{}'.format(options['name'],options['model'], options['loss'], options['item'], options['cs'])

    if options['eval']:
        global print_result
        net, criterion = load_networks(net, model_path, file_name+'_'+str(options['max_epoch']), criterion=criterion)
        # results = test(net, criterion, testloader, outloader, epoch=0, **options)
        # print("Acc (%): {:.3f}\t AUROC (%): {:.3f}\t OSCR (%): {:.3f}\t".format(results['ACC'], results['AUROC'],
        #                                                                         results['OSCR']))
        name_model = options['model'] + '_' + options['dataset']
        if options['real_sn']:
            name_model += 'real'
        res = test_fiw(net, criterion,testloader_eval,testloader,name_model,**options)
        print_result.update(*res)

        return res

    params_list = [{'params': net.parameters()},
                   {'params': criterion.parameters()}]

    if options['dataset'] == 'tiny_imagenet':
        optimizer = torch.optim.Adam(params_list, lr=options['lr'])
    else:
        optimizer = torch.optim.SGD(params_list, lr=options['lr'], momentum=0.9, weight_decay=1e-4)
    if options['cs']:
        optimizerD = torch.optim.Adam(netD.parameters(), lr=options['gan_lr'], betas=(0.5, 0.999))
        optimizerG = torch.optim.Adam(netG.parameters(), lr=options['gan_lr'], betas=(0.5, 0.999))

    if options['stepsize'] > 0:
        # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90, 120])
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[ 60,  120])


    start_time = time.time()

    for epoch in range(options['max_epoch']):
        print("==> CROSS:{}-Epoch {}/{}".format(options['item'],epoch + 1, options['max_epoch']))

        if options['cs']:
            train_cs(net, netD, netG, criterion, criterionD,
                     optimizer, optimizerD, optimizerG,
                     trainloader, epoch=epoch, **options)

        train(net, criterion, optimizer, trainloader, epoch=epoch, **options)

        if options['eval_freq'] > 0 and (epoch + 1) % options['eval_freq'] == 0 or (epoch + 1) == options['max_epoch']:
            print("==> Test", options['loss'])

            # results = test(net, criterion, testloader, outloader, epoch=epoch, **options)
            # print("Acc (%): {:.3f}\t AUROC (%): {:.3f}\t OSCR (%): {:.3f}\t".format(results['ACC'], results['AUROC'],
            #                                                                         results['OSCR']))
            results = test_while_train(net,criterion,trainloader,**options)
        if (epoch + 1) % 10 == 0 or (epoch + 1) == options['max_epoch']:
            save_networks(net, model_path, file_name+'_'+str(epoch+1), criterion=criterion)

        if options['stepsize'] > 0: scheduler.step()

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))

    return [results]

class pr_res:
    def __init__(self):
        self.acc = []
        self.auroc_knun = []
        self.auroc_kinon = []
        self.F1 = []
        self.k_precision = []
        self.ukkin_precision = []
        self.unk_precision = []
        self.k_recall = []
        self.ukkin_recall = []
        self.unk_recall = []
        self.k_f1 = []
        self.ukkin_f1 = []
        self.unk_f1 = []
    def update(self,acc, auroc_knun, auroc_kinon, F1,
            k_precision,ukkin_precision,unk_precision,
            k_recall,ukkin_recall,unk_recall,
            k_f1, ukkin_f1, unk_f1):
        self.acc += [acc]
        self.auroc_knun += [auroc_knun]
        self.auroc_kinon += [auroc_kinon]
        self.F1 += [F1]
        self.k_precision = [k_precision]
        self.ukkin_precision = [ukkin_precision]
        self.unk_precision = [unk_precision]
        self.k_recall = [k_recall]
        self.ukkin_recall = [ukkin_recall]
        self.unk_recall = [unk_recall]
        self.k_f1 = [k_f1]
        self.ukkin_f1 = [ukkin_f1]
        self.unk_f1 = [unk_f1]


    def __str__(self):
        return 'Average acc:{:.4f}, auroc_knun:{:.4f}, auroc_kinon:{:.4f} f1:{:.4f} '.format(
            np.mean(self.acc), np.mean(self.auroc_knun), np.mean(self.auroc_kinon),
            np.mean(self.F1)) + '\n'+\
            'Primary:Precision:{:.4f}, Recall:{:.4f}, F1:{:.4f}'.format(np.mean(self.k_precision),np.mean(self.k_recall),np.mean(self.k_f1))+ '\n'+\
            'Secondary:Precision:{:.4f}, Recall:{:.4f}, F1:{:.4f}'.format(np.mean(self.ukkin_precision),np.mean(self.ukkin_recall),np.mean(self.ukkin_f1))+ '\n'+\
            'Non-kin:Precision:{:.4f}, Recall:{:.4f}, F1:{:.4f}'.format(np.mean(self.unk_precision),np.mean(self.unk_recall), np.mean(self.unk_f1))




if __name__ == '__main__':
    args = parser.parse_args()
    options = vars(args)
    options['dataroot'] = os.path.join(options['dataroot'], options['dataset'])
    img_size = 32
    results = dict()
    # print_result = {'acc':[],'auroc':[],'acc_neg':[],'F1':[],'F1_neg':[]}
    print_result = pr_res()
    from split import splits_2020 as splits


    print(args.name)
    # if '4_type' in args.name or '4type' in args.name:
    #     print('do not')
    #     options.update({
    #         'num_classes':4
    #     })
    # options.update({
    #         'num_classes':4})


    cs_num = options['cs_num']
    print("cross validation numer :{}".format(cs_num))
    # i = cs_num - 1
    for i in range(5):
        options['cs_num'] = i + 1
        train_5csvalid = splits[options['dataset']][i]
        get_thre = splits['fiw-thre'][i]

        print("cross validation numer :{}".format(i))
        options.update(
            {
                'item': i,
                'known': train_5csvalid,
                'unknown': [5-i],
                'eval_thre':get_thre,
                'img_size': img_size
            }
        )

        print(options)
        dir_name = '{}_{}_{}'.format(options['model'], options['loss'],options['dataset'])
        dir_path = os.path.join(options['outf'], 'results', dir_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)


        file_name = options['dataset'] + '.csv'

        res = main_worker(options)
        results[str(i)] = res
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(dir_path, file_name))

    if options['eval']:
        print('='*10,'5-cross-validation','='*10)
        print(print_result)