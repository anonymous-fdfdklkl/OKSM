import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets import MNIST, CIFAR10, CIFAR100, SVHN
from .datasets import kfw_caur, train_transform,test_transform,fiw_dataset,fiw_dataset_test,fiw_dataset_new
from torch.utils.data import Dataset,DataLoader



def get_dataloader(args,cfg):
    if 'cacur' in args.name:
        return kfw_loader(args,cfg).get_train_loader()
    elif 'class' in args.name:
        return kfw_loader(args,cfg).get_train_loader()



class kfw_loader():
    def __init__(self,options):
        self.options = options
        kfw = {'kfw1': 'I', 'kfw2': 'II'}[options['dataset']]
        self.train_ls = ['/home/wei/Documents/DATA/kinship/KinFaceW-{}/meta_data/fd_pairs.mat'.format(kfw),
                    '/home/wei/Documents/DATA/kinship/KinFaceW-{}/meta_data/fs_pairs.mat'.format(kfw),
                    '/home/wei/Documents/DATA/kinship/KinFaceW-{}/meta_data/md_pairs.mat'.format(kfw),
                    '/home/wei/Documents/DATA/kinship/KinFaceW-{}/meta_data/ms_pairs.mat'.format(kfw)
                    ]
        self.data_pth = ['/home/wei/Documents/DATA/kinship/KinFaceW-{}/images/father-dau'.format(kfw),
                    '/home/wei/Documents/DATA/kinship/KinFaceW-{}/images/father-son'.format(kfw),
                    '/home/wei/Documents/DATA/kinship/KinFaceW-{}/images/mother-dau'.format(kfw),
                    '/home/wei/Documents/DATA/kinship/KinFaceW-{}/images/mother-son'.format(kfw)
                    ]

    def get_train_loader(self):
        train_ls = self.train_ls
        data_pth = self.data_pth
        opt = self.options

        train_set = kfw_caur(train_ls, data_pth, opt['known'],
                             transform=train_transform,
                             cross_shuffle=False, sf_aln=False, real_sn=opt['real_sn'],get_pos=True)
        test_set = kfw_caur(train_ls, data_pth, opt['unknown'],
                            transform=test_transform,
                            test=True, real_sn=opt['real_sn'],get_pos=True)
        train_loader = DataLoader(train_set, batch_size=opt['batch_size'], shuffle=True)
        test_loader = DataLoader(test_set, batch_size=opt['batch_size'])
        return train_loader, test_loader

    def get_kfw_eval_loaders(self):
        train_ls = self.train_ls
        data_pth = self.data_pth
        opt = self.options
        test_set = kfw_caur(train_ls, data_pth, opt['unknown'],
                            transform=test_transform,
                            real_sn=opt['real_sn'])
        known_set = kfw_caur(train_ls, data_pth, opt['unknown'],
                             transform=test_transform,
                             real_sn=opt['real_sn'], get_pos=True)
        unknown_set = kfw_caur(train_ls, data_pth,opt['unknown'],
                               transform=test_transform,
                               real_sn=opt['real_sn'], get_neg=True)

        test_loader = DataLoader(test_set, batch_size=opt['batch_size'])

        known_set_loader = DataLoader(known_set, batch_size=opt['batch_size'])
        unknown_loader = DataLoader(unknown_set, batch_size=opt['batch_size'])

        return known_set_loader, unknown_loader, test_loader

    def get_vis_loader(self):
        train_ls = self.train_ls
        data_pth = self.data_pth
        opt = self.options

        train_pos = kfw_caur(train_ls, data_pth, opt['unknown'],
                             transform=test_transform,
                             real_sn=opt['real_sn'],get_pos=True)
        train_neg = kfw_caur(train_ls, data_pth, opt['unknown'],
                            transform=test_transform,
                             real_sn=opt['real_sn'],get_neg=True)
        train_loader = DataLoader(train_pos, batch_size=opt['batch_size'])
        test_loader = DataLoader(train_neg, batch_size=opt['batch_size'])
        return train_loader, test_loader



##################
class fiw_loader():
    def __init__(self,options):
        self.options = options
        if '4_type' in options['name'] or '4type' in options['name']:
            print("=" * 20 + "training 4 type")
            self.train_ls = [
                    '/var/scratch/wwang/DATA/FIW/origin/fitted_original_5split/fd.pkl',
                    '/var/scratch/wwang/DATA/FIW/origin/fitted_original_5split/fs.pkl',
                    '/var/scratch/wwang/DATA/FIW/origin/fitted_original_5split/md.pkl',
                    '/var/scratch/wwang/DATA/FIW/origin/fitted_original_5split/ms.pkl',
                    # '/var/scratch/wwang/DATA/FIW/origin/fitted_original_5split/bb.pkl',
                    # '/var/scratch/wwang/DATA/FIW/origin/fitted_original_5split/bs.pkl',
                    # '/var/scratch/wwang/DATA/FIW/origin/fitted_original_5split/ss.pkl',
                    ]
        else:
            print("=" * 20 + "training 7 type")
            self.train_ls = [
                    '/var/scratch/wwang/DATA/FIW/origin/fitted_original_5split/fd.pkl',
                    '/var/scratch/wwang/DATA/FIW/origin/fitted_original_5split/fs.pkl',
                    '/var/scratch/wwang/DATA/FIW/origin/fitted_original_5split/md.pkl',
                    '/var/scratch/wwang/DATA/FIW/origin/fitted_original_5split/ms.pkl',
                    '/var/scratch/wwang/DATA/FIW/origin/fitted_original_5split/bb.pkl',
                    '/var/scratch/wwang/DATA/FIW/origin/fitted_original_5split/bs.pkl',
                    '/var/scratch/wwang/DATA/FIW/origin/fitted_original_5split/ss.pkl',
                    ]
        self.train_ls_test = [
            '/var/scratch/wwang/DATA/FIW/origin/fitted_original_5split/fd.pkl',
            '/var/scratch/wwang/DATA/FIW/origin/fitted_original_5split/fs.pkl',
            '/var/scratch/wwang/DATA/FIW/origin/fitted_original_5split/md.pkl',
            '/var/scratch/wwang/DATA/FIW/origin/fitted_original_5split/ms.pkl',
            '/var/scratch/wwang/DATA/FIW/origin/fitted_original_5split/bb.pkl',
            '/var/scratch/wwang/DATA/FIW/origin/fitted_original_5split/bs.pkl',
            '/var/scratch/wwang/DATA/FIW/origin/fitted_original_5split/ss.pkl',
            '/var/scratch/wwang/DATA/FIW/origin/fitted_original_5split/gfgd.pkl',
            '/var/scratch/wwang/DATA/FIW/origin/fitted_original_5split/gfgs.pkl',
            '/var/scratch/wwang/DATA/FIW/origin/fitted_original_5split/gmgd.pkl',
            '/var/scratch/wwang/DATA/FIW/origin/fitted_original_5split/gmgs.pkl'
        ]
        self.data_pth = '/var/scratch/wwang/DATA/FIW/origin/train-faces'
        self.data_set = fiw_dataset

    def get_train_loader(self):
        train_ls = self.train_ls
        data_pth = self.data_pth
        opt = self.options

        train_set = self.data_set(train_ls, data_pth, opt['known'],
                             transform=train_transform,
                             cross_shuffle=True, sf_aln=False, real_sn=opt['real_sn'],get_pos=True)
        test_set = self.data_set(train_ls, data_pth, opt['unknown'],
                            transform=test_transform,
                            test=True, real_sn=opt['real_sn'],get_pos=True)
        train_loader = DataLoader(train_set, batch_size=opt['batch_size'], shuffle=True)
        test_loader = DataLoader(test_set, batch_size=opt['batch_size'])
        return train_loader, test_loader

    def get_close_train_loader(self):
        train_ls = self.train_ls
        data_pth = self.data_pth
        opt = self.options

        train_set = self.data_set(train_ls, data_pth, opt['known'],
                             transform=train_transform,cross_shuffle=False, sf_aln=False,
                             real_sn=opt['real_sn'],get_pos=True)
        test_set = self.data_set(train_ls, data_pth, opt['unknown'],
                            transform=test_transform,test=True, real_sn=opt['real_sn'],get_pos=True)
        train_loader = DataLoader(train_set, batch_size=opt['batch_size'], shuffle=True)
        test_loader = DataLoader(test_set, batch_size=opt['batch_size'])
        return train_loader, test_loader

    def get_fiw_eval_loaders(self):
        train_ls = self.train_ls
        data_pth = self.data_pth
        opt = self.options
        test_set = self.data_set(train_ls, data_pth, opt['unknown'],
                            transform=test_transform,
                            real_sn=opt['real_sn'])
        known_set = self.data_set(train_ls, data_pth, opt['unknown'],
                             transform=test_transform,
                             real_sn=opt['real_sn'], get_pos=True)
        unknown_set = self.data_set(train_ls, data_pth, opt['unknown'],
                               transform=test_transform,
                               real_sn=opt['real_sn'], get_neg=True)

        test_loader = DataLoader(test_set, batch_size=opt['batch_size'])

        known_set_loader = DataLoader(known_set, batch_size=opt['batch_size'])
        unknown_loader = DataLoader(unknown_set, batch_size=opt['batch_size'])

        return known_set_loader, unknown_loader, test_loader

    def get_vis_loader(self):
        train_ls = self.train_ls
        data_pth = self.data_pth
        opt = self.options

        train_pos = self.data_set(train_ls, data_pth, opt['known'],
                             transform=test_transform,
                             real_sn=opt['real_sn'],get_pos=True)
        train_neg = self.data_set(train_ls, data_pth, opt['known'],
                            transform=test_transform,
                             real_sn=opt['real_sn'],get_neg=True)
        train_loader = DataLoader(train_pos, batch_size=opt['batch_size'], shuffle=True)
        test_loader = DataLoader(train_neg, batch_size=opt['batch_size'])
        return train_loader, test_loader

    def get_fiw_eval_loaders_new(self):
        # train_ls = self.train_ls
        train_ls_test = self.train_ls_test
        data_pth = self.data_pth
        opt = self.options
        test_set = fiw_dataset_new(train_ls_test, data_pth, opt['unknown'],
                            transform=test_transform,
                            real_sn=opt['real_sn'])
        known_set = fiw_dataset_new(train_ls_test, data_pth, opt['unknown'],
                             transform=test_transform,
                             real_sn=opt['real_sn'], get_pos=True)
        unknown_set = fiw_dataset_new(train_ls_test, data_pth, opt['unknown'],
                               transform=test_transform,
                               real_sn=opt['real_sn'], get_neg=True)

        test_loader = DataLoader(test_set, batch_size=opt['batch_size'])

        known_set_loader = DataLoader(known_set, batch_size=opt['batch_size'])
        unknown_loader = DataLoader(unknown_set, batch_size=opt['batch_size'])

        return known_set_loader, unknown_loader, test_loader


# class fiw_loader():
#     def __init__(self,args,cfg):
#         self.args = args
#         self.cfg = cfg
#
#         self.train_ls_test = [
#                 '/var/scratch/wwang/DATA/FIW/origin/fitted_original_5split/fd.pkl',
#                 '/var/scratch/wwang/DATA/FIW/origin/fitted_original_5split/fs.pkl',
#                 '/var/scratch/wwang/DATA/FIW/origin/fitted_original_5split/md.pkl',
#                 '/var/scratch/wwang/DATA/FIW/origin/fitted_original_5split/ms.pkl',
#                 '/var/scratch/wwang/DATA/FIW/origin/fitted_original_5split/bb.pkl',
#                 '/var/scratch/wwang/DATA/FIW/origin/fitted_original_5split/bs.pkl',
#                 '/var/scratch/wwang/DATA/FIW/origin/fitted_original_5split/ss.pkl',
#                 '/var/scratch/wwang/DATA/FIW/origin/fitted_original_5split/gfgd.pkl',
#                 '/var/scratch/wwang/DATA/FIW/origin/fitted_original_5split/gfgs.pkl',
#                 '/var/scratch/wwang/DATA/FIW/origin/fitted_original_5split/gmgd.pkl',
#                 '/var/scratch/wwang/DATA/FIW/origin/fitted_original_5split/gmgs.pkl'
#                 ]
#
#         self.train_self_ls = [
#                 '/var/scratch/wwang/DATA/FIW/origin/fitted_original_5split/fd.pkl',
#                 '/var/scratch/wwang/DATA/FIW/origin/fitted_original_5split/fs.pkl',
#                 '/var/scratch/wwang/DATA/FIW/origin/fitted_original_5split/md.pkl',
#                 '/var/scratch/wwang/DATA/FIW/origin/fitted_original_5split/ms.pkl',
#                 '/var/scratch/wwang/DATA/FIW/origin/fitted_original_5split/bb.pkl',
#                 '/var/scratch/wwang/DATA/FIW/origin/fitted_original_5split/bs.pkl',
#                 '/var/scratch/wwang/DATA/FIW/origin/fitted_original_5split/ss.pkl',
#                 # '/var/scratch/wwang/DATA/FIW/origin/fitted_original_5split/gfgd.pkl',
#                 # '/var/scratch/wwang/DATA/FIW/origin/fitted_original_5split/gfgs.pkl',
#                 # '/var/scratch/wwang/DATA/FIW/origin/fitted_original_5split/gmgd.pkl',
#                 # '/var/scratch/wwang/DATA/FIW/origin/fitted_original_5split/gmgs.pkl'
#                 ]
#         if '4_type' in args.name or '4type' in args.name:
#             print("=" * 20 + "training 4 type")
#             self.train_ls = [
#                     '/var/scratch/wwang/DATA/FIW/origin/fitted_original_5split/fd.pkl',
#                     '/var/scratch/wwang/DATA/FIW/origin/fitted_original_5split/fs.pkl',
#                     '/var/scratch/wwang/DATA/FIW/origin/fitted_original_5split/md.pkl',
#                     '/var/scratch/wwang/DATA/FIW/origin/fitted_original_5split/ms.pkl',
#                     # '/var/scratch/wwang/DATA/FIW/origin/fitted_original_5split/bb.pkl',
#                     # '/var/scratch/wwang/DATA/FIW/origin/fitted_original_5split/bs.pkl',
#                     # '/var/scratch/wwang/DATA/FIW/origin/fitted_original_5split/ss.pkl',
#                     ]
#         else:
#             print("=" * 20 + "training 7 type")
#             self.train_ls = [
#                     '/var/scratch/wwang/DATA/FIW/origin/fitted_original_5split/fd.pkl',
#                     '/var/scratch/wwang/DATA/FIW/origin/fitted_original_5split/fs.pkl',
#                     '/var/scratch/wwang/DATA/FIW/origin/fitted_original_5split/md.pkl',
#                     '/var/scratch/wwang/DATA/FIW/origin/fitted_original_5split/ms.pkl',
#                     '/var/scratch/wwang/DATA/FIW/origin/fitted_original_5split/bb.pkl',
#                     '/var/scratch/wwang/DATA/FIW/origin/fitted_original_5split/bs.pkl',
#                     '/var/scratch/wwang/DATA/FIW/origin/fitted_original_5split/ss.pkl',
#                     ]
#         self.data_pth = '/home/wei/Documents/DATA/kinship/FIW/competition/rfiw2020/verification/train-faces'
#         self.data_set = fiw_dataset
#
#     def get_train_loader(self):
#         train_ls = self.train_ls
#         data_pth = self.data_pth
#         args =  self.args
#         cfg = self.cfg
#
#         train_set = self.data_set(train_ls, data_pth, cfg["cross_validation"][str(args.trial)]["train_cs"],
#                              transform=train_transform,
#                                   cross_shuffle=True, neg_ratio=1,
#                              sf_aln=False, real_sn=args.real_sn)
#         test_set = self.data_set(train_ls, data_pth, cfg["cross_validation"][str(args.trial)]["eval_cs"],
#                             transform=test_transform,
#                             test=True, real_sn=args.real_sn)
#         train_loader = DataLoader(train_set, batch_size=cfg['batch_size'], shuffle=True)
#         test_loader = DataLoader(test_set, batch_size=cfg['batch_size'])
#         return train_loader, test_loader
#
#
#
#     def get_close_train_loader(self):
#         train_ls = self.train_ls
#         data_pth = self.data_pth
#         args = self.args
#         cfg = self.cfg
#
#         train_set = self.data_set(train_ls, data_pth, cfg["cross_validation"][str(args.trial)]["train_cs"],
#                              transform=train_transform,cross_shuffle=False, sf_aln=False,
#                              real_sn=args.real_sn,get_pos=True)
#         test_set = self.data_set(train_ls, data_pth, cfg["cross_validation"][str(args.trial)]["eval_cs"],
#                             transform=test_transform,test=True, real_sn=args.real_sn,get_pos=True)
#         train_loader = DataLoader(train_set, batch_size=cfg['batch_size'], shuffle=True)
#         test_loader = DataLoader(test_set, batch_size=cfg['batch_size'])
#         return train_loader, test_loader
#
#     def get_fiw_eval_loaders(self):
#         # train_ls = self.train_ls
#         train_ls_test = self.train_ls_test
#         data_pth = self.data_pth
#         args =  self.args
#         cfg = self.cfg
#         test_set = self.data_set(train_ls_test, data_pth, cfg["cross_validation"][str(args.trial)]["eval_cs"],
#                             transform=test_transform,
#                             real_sn=args.real_sn)
#         known_set = fiw_dataset_test(train_ls_test, data_pth, cfg["cross_validation"][str(args.trial)]["eval_cs"],
#                              transform=test_transform,
#                              real_sn=args.real_sn, get_pos=True)
#         unknown_set = fiw_dataset_test(train_ls_test, data_pth, cfg["cross_validation"][str(args.trial)]["eval_cs"],
#                                transform=test_transform,
#                                real_sn=args.real_sn, get_neg=True)
#
#         test_loader = DataLoader(test_set, batch_size=cfg['batch_size'])
#
#         known_set_loader = DataLoader(known_set, batch_size=cfg['batch_size'])
#         unknown_loader = DataLoader(unknown_set, batch_size=cfg['batch_size'])
#
#         return known_set_loader, unknown_loader, test_loader
#
#     def get_fiw_eval_loaders_new(self):
#         # train_ls = self.train_ls
#         train_ls_test = self.train_ls_test
#         data_pth = self.data_pth
#         args =  self.args
#         cfg = self.cfg
#         test_set = fiw_dataset_new(train_ls_test, data_pth, cfg["cross_validation"][str(args.trial)]["eval_cs"],
#                             transform=test_transform,
#                             real_sn=args.real_sn)
#         known_set = fiw_dataset_new(train_ls_test, data_pth, cfg["cross_validation"][str(args.trial)]["eval_cs"],
#                              transform=test_transform,
#                              real_sn=args.real_sn, get_pos=True)
#         unknown_set = fiw_dataset_new(train_ls_test, data_pth, cfg["cross_validation"][str(args.trial)]["eval_cs"],
#                                transform=test_transform,
#                                real_sn=args.real_sn, get_neg=True)
#
#         test_loader = DataLoader(test_set, batch_size=cfg['batch_size'])
#
#         known_set_loader = DataLoader(known_set, batch_size=cfg['batch_size'])
#         unknown_loader = DataLoader(unknown_set, batch_size=cfg['batch_size'])
#
#         return known_set_loader, unknown_loader, test_loader
#
#     def get_vis_loader(self,cs = "train_cs"):
#         train_ls_test = self.train_ls_test
#         data_pth = self.data_pth
#         args = self.args
#         cfg = self.cfg
#
#         pos = self.data_set(train_ls_test, data_pth, cfg["cross_validation"][str(args.trial)][cs],
#                              transform=test_transform,
#                              real_sn=args.real_sn,get_pos=True)
#         neg = self.data_set(train_ls_test, data_pth, cfg["cross_validation"][str(args.trial)][cs],
#                             transform=test_transform,
#                              real_sn=args.real_sn,get_neg=True)
#         pos_loader = DataLoader(pos, batch_size=cfg['batch_size'], shuffle=True)
#         neg_loader = DataLoader(neg, batch_size=cfg['batch_size'])
#         return pos_loader, neg_loader
#
#
#     # def get_vis_loader_test(self):
#     #     train_ls_test = self.train_ls_test
#     #     data_pth = self.data_pth
#     #     args = self.args
#     #     cfg = self.cfg
#     #
#     #     known = self.data_set(train_ls_test, data_pth, cfg["cross_validation"][str(args.trial)]["eval_cs"],
#     #                          transform=test_transform,
#     #                          real_sn=args.real_sn,get_pos=True)
#     #     unknown = self.data_set(train_ls_test, data_pth, cfg["cross_validation"][str(args.trial)]["eval_cs"],
#     #                         transform=test_transform,
#     #                          real_sn=args.real_sn,get_neg=True)
#     #     known_loader = DataLoader(known, batch_size=cfg['batch_size'], shuffle=True)
#     #     unknown_loader = DataLoader(unknown, batch_size=cfg['batch_size'])
#     #     return known_loader, unknown_loader


class MNISTRGB(MNIST):
    """MNIST Dataset.
    """
    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy(), mode='L')
        img = img.convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class MNIST_Filter(MNISTRGB):
    """MNIST Dataset.
    """
    def __Filter__(self, known):
        targets = self.targets.data.numpy()
        mask, new_targets = [], []
        for i in range(len(targets)):
            if targets[i] in known:
                mask.append(i)
                new_targets.append(known.index(targets[i]))
        self.targets = np.array(new_targets)
        mask = torch.tensor(mask).long()
        self.data = torch.index_select(self.data, 0, mask)

class MNIST_OSR(object):
    def __init__(self, known, dataroot='./data/mnist', use_gpu=True, num_workers=8, batch_size=128, img_size=32):
        self.num_classes = len(known)
        self.known = known
        self.unknown = list(set(list(range(0, 10))) - set(known))

        print('Selected Labels: ', known)

        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

        pin_memory = True if use_gpu else False

        trainset = MNIST_Filter(root=dataroot, train=True, download=True, transform=train_transform)
        print('All Train Data:', len(trainset))
        trainset.__Filter__(known=self.known)
        
        self.train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        
        testset = MNIST_Filter(root=dataroot, train=False, download=True, transform=transform)
        print('All Test Data:', len(testset))
        testset.__Filter__(known=self.known)
        
        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        outset = MNIST_Filter(root=dataroot, train=False, download=True, transform=transform)
        outset.__Filter__(known=self.unknown)

        self.out_loader = torch.utils.data.DataLoader(
            outset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        print('Train: ', len(trainset), 'Test: ', len(testset), 'Out: ', len(outset))
        print('All Test: ', (len(testset) + len(outset)))

class CIFAR10_Filter(CIFAR10):
    """CIFAR10 Dataset.
    """
    def __Filter__(self, known):
        datas, targets = np.array(self.data), np.array(self.targets)
        mask, new_targets = [], []
        for i in range(len(targets)):
            if targets[i] in known:
                mask.append(i)
                new_targets.append(known.index(targets[i]))
        self.data, self.targets = np.squeeze(np.take(datas, mask, axis=0)), np.array(new_targets)


class CIFAR10_OSR(object):
    def __init__(self, known, dataroot='./data/cifar10', use_gpu=True, num_workers=8, batch_size=128, img_size=32):
        self.num_classes = len(known)
        self.known = known
        self.unknown = list(set(list(range(0, 10))) - set(known))

        print('Selected Labels: ', known)

        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomCrop(img_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

        pin_memory = True if use_gpu else False

        trainset = CIFAR10_Filter(root=dataroot, train=True, download=True, transform=train_transform)
        print('All Train Data:', len(trainset))
        trainset.__Filter__(known=self.known)
        
        self.train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        
        testset = CIFAR10_Filter(root=dataroot, train=False, download=True, transform=transform)
        print('All Test Data:', len(testset))
        testset.__Filter__(known=self.known)
        
        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        outset = CIFAR10_Filter(root=dataroot, train=False, download=True, transform=transform)
        outset.__Filter__(known=self.unknown)

        self.out_loader = torch.utils.data.DataLoader(
            outset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        print('Train: ', len(trainset), 'Test: ', len(testset), 'Out: ', len(outset))
        print('All Test: ', (len(testset) + len(outset)))

class CIFAR100_Filter(CIFAR100):
    """CIFAR100 Dataset.
    """
    def __Filter__(self, known):
        datas, targets = np.array(self.data), np.array(self.targets)
        mask, new_targets = [], []
        for i in range(len(targets)):
            if targets[i] in known:
                mask.append(i)
                new_targets.append(known.index(targets[i]))
        self.data, self.targets = np.squeeze(np.take(datas, mask, axis=0)), np.array(new_targets)

class CIFAR100_OSR(object):
    def __init__(self, known, dataroot='./data/cifar100', use_gpu=True, num_workers=8, batch_size=128, img_size=32):
        self.num_classes = len(known)
        self.known = known
        self.unknown = list(set(list(range(0, 100))) - set(known))

        print('Selected Labels: ', known)

        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

        pin_memory = True if use_gpu else False
        
        testset = CIFAR100_Filter(root=dataroot, train=False, download=True, transform=transform)
        testset.__Filter__(known=self.known)
        
        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )


class SVHN_Filter(SVHN):
    """SVHN Dataset.
    """
    def __Filter__(self, known):
        targets = np.array(self.labels)
        mask, new_targets = [], []
        for i in range(len(targets)):
            if targets[i] in known:
                mask.append(i)
                new_targets.append(known.index(targets[i]))
        self.data, self.labels = self.data[mask], np.array(new_targets)

class SVHN_OSR(object):
    def __init__(self, known, dataroot='./data/svhn', use_gpu=True, num_workers=8, batch_size=128, img_size=32):
        self.num_classes = len(known)
        self.known = known
        self.unknown = list(set(list(range(0, 10))) - set(known))

        print('Selected Labels: ', known)

        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomCrop(img_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

        pin_memory = True if use_gpu else False

        trainset = SVHN_Filter(root=dataroot, split='train', download=True, transform=train_transform)
        print('All Train Data:', len(trainset))
        trainset.__Filter__(known=self.known)
        
        self.train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        
        testset = SVHN_Filter(root=dataroot, split='test', download=True, transform=transform)
        print('All Test Data:', len(testset))
        testset.__Filter__(known=self.known)
        
        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        outset = SVHN_Filter(root=dataroot, split='test', download=True, transform=transform)
        outset.__Filter__(known=self.unknown)

        self.out_loader = torch.utils.data.DataLoader(
            outset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        print('Train: ', len(trainset), 'Test: ', len(testset), 'Out: ', len(outset))
        print('All Test: ', (len(testset) + len(outset)))

class Tiny_ImageNet_Filter(ImageFolder):
    """Tiny_ImageNet Dataset.
    """
    def __Filter__(self, known):
        datas, targets = self.imgs, self.targets
        new_datas, new_targets = [], []
        for i in range(len(datas)):
            if datas[i][1] in known:
                new_item = (datas[i][0], known.index(datas[i][1]))
                new_datas.append(new_item)
                # new_targets.append(targets[i])
                new_targets.append(known.index(targets[i]))
        datas, targets = new_datas, new_targets
        self.samples, self.imgs, self.targets = datas, datas, targets

class Tiny_ImageNet_OSR(object):
    def __init__(self, known, dataroot='./data/tiny_imagenet', use_gpu=True, num_workers=8, batch_size=128, img_size=64):
        self.num_classes = len(known)
        self.known = known
        self.unknown = list(set(list(range(0, 200))) - set(known))

        print('Selected Labels: ', known)

        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomCrop(img_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

        pin_memory = True if use_gpu else False

        trainset = Tiny_ImageNet_Filter(os.path.join(dataroot, 'tiny-imagenet-200', 'train'), train_transform)
        print('All Train Data:', len(trainset))
        trainset.__Filter__(known=self.known)
        
        self.train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        
        testset = Tiny_ImageNet_Filter(os.path.join(dataroot, 'tiny-imagenet-200', 'val'), transform)
        print('All Test Data:', len(testset))
        testset.__Filter__(known=self.known)
        
        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        outset = Tiny_ImageNet_Filter(os.path.join(dataroot, 'tiny-imagenet-200', 'val'), transform)
        outset.__Filter__(known=self.unknown)

        self.out_loader = torch.utils.data.DataLoader(
            outset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        print('Train: ', len(trainset), 'Test: ', len(testset), 'Out: ', len(outset))
        print('All Test: ', (len(testset) + len(outset)))