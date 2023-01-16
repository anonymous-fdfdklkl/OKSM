"""
	Functions useful for creating experiment datasets and dataloaders.

	Dimity Miller, 2020
"""

import torch
import torchvision
import torchvision.transforms as tf
import json
from torch.autograd import Variable
import numpy as np
from .datasets import *
import random

random.seed(1000)


class kfw_loader():
    def __init__(self,args,cfg):
        self.args = args
        self.cfg = cfg
        kfw = {'kfw1': 'I', 'kfw2': 'II'}[args.dataset]
        self.train_ls = ['/home//Documents/DATA/kinship/KinFaceW-{}/meta_data/fd_pairs.mat'.format(kfw),
                    '/home//Documents/DATA/kinship/KinFaceW-{}/meta_data/fs_pairs.mat'.format(kfw),
                    '/home//Documents/DATA/kinship/KinFaceW-{}/meta_data/md_pairs.mat'.format(kfw),
                    '/home//Documents/DATA/kinship/KinFaceW-{}/meta_data/ms_pairs.mat'.format(kfw)
                    ]
        self.data_pth = ['/home//Documents/DATA/kinship/KinFaceW-{}/images/father-dau'.format(kfw),
                    '/home//Documents/DATA/kinship/KinFaceW-{}/images/father-son'.format(kfw),
                    '/home//Documents/DATA/kinship/KinFaceW-{}/images/mother-dau'.format(kfw),
                    '/home//Documents/DATA/kinship/KinFaceW-{}/images/mother-son'.format(kfw)
                    ]
        self.data_set = kfw_caur

    def get_train_loader(self):
        train_ls = self.train_ls
        data_pth = self.data_pth
        args =  self.args
        cfg = self.cfg

        train_set = self.data_set(train_ls, data_pth, cfg["cross_validation"][str(args.trial)]["train_cs"],
                             transform=train_transform,
                             cross_shuffle=True,neg_ratio=3,
                             sf_aln=False, real_sn=args.real_sn)
        test_set = self.data_set(train_ls, data_pth, cfg["cross_validation"][str(args.trial)]["eval_cs"],
                            transform=test_transform,
                            test=True, real_sn=args.real_sn)
        train_loader = DataLoader(train_set, batch_size=cfg['batch_size'], shuffle=True)
        test_loader = DataLoader(test_set, batch_size=cfg['batch_size'])
        return train_loader, test_loader

    def get_close_train_loader(self):
        train_ls = self.train_ls
        data_pth = self.data_pth
        args = self.args
        cfg = self.cfg

        train_set = self.data_set(train_ls, data_pth, cfg["cross_validation"][str(args.trial)]["train_cs"],
                             transform=train_transform,cross_shuffle=False, sf_aln=False,
                             real_sn=args.real_sn,get_pos=True)
        test_set = self.data_set(train_ls, data_pth, cfg["cross_validation"][str(args.trial)]["eval_cs"],
                            transform=test_transform,test=True, real_sn=args.real_sn,get_pos=True)
        train_loader = DataLoader(train_set, batch_size=cfg['batch_size'], shuffle=True)
        test_loader = DataLoader(test_set, batch_size=cfg['batch_size'])
        return train_loader, test_loader

    def get_kfw_eval_loaders(self):
        train_ls = self.train_ls
        data_pth = self.data_pth
        args =  self.args
        cfg = self.cfg
        test_set = self.data_set(train_ls, data_pth, cfg["cross_validation"][str(args.trial)]["eval_cs"],
                            transform=test_transform,
                            real_sn=args.real_sn)
        known_set = self.data_set(train_ls, data_pth, cfg["cross_validation"][str(args.trial)]["eval_cs"],
                             transform=test_transform,
                             real_sn=args.real_sn, get_pos=True)
        unknown_set = self.data_set(train_ls, data_pth, cfg["cross_validation"][str(args.trial)]["eval_cs"],
                               transform=test_transform,
                               real_sn=args.real_sn, get_neg=True)

        test_loader = DataLoader(test_set, batch_size=cfg['batch_size'])

        known_set_loader = DataLoader(known_set, batch_size=cfg['batch_size'])
        unknown_loader = DataLoader(unknown_set, batch_size=cfg['batch_size'])

        return known_set_loader, unknown_loader, test_loader

    def get_vis_loader(self):
        train_ls = self.train_ls
        data_pth = self.data_pth
        args = self.args
        cfg = self.cfg

        train_pos = self.data_set(train_ls, data_pth, cfg["cross_validation"][str(args.trial)]["train_cs"],
                             transform=test_transform,
                             real_sn=args.real_sn,get_pos=True)
        train_neg = self.data_set(train_ls, data_pth, cfg["cross_validation"][str(args.trial)]["train_cs"],
                            transform=test_transform,
                             real_sn=args.real_sn,get_neg=True)
        train_loader = DataLoader(train_pos, batch_size=cfg['batch_size'], shuffle=True)
        test_loader = DataLoader(train_neg, batch_size=cfg['batch_size'])
        return train_loader, test_loader


##################
class fiw_loader():
    def __init__(self,args,cfg):
        self.args = args
        self.cfg = cfg

        self.train_ls_test = [
                ' datasets/fiw/fitted_original_5split/fd.pkl',
                ' datasets/fiw/fitted_original_5split/fs.pkl',
                ' datasets/fiw/fitted_original_5split/md.pkl',
                ' datasets/fiw/fitted_original_5split/ms.pkl',
                ' datasets/fiw/fitted_original_5split/bb.pkl',
                ' datasets/fiw/fitted_original_5split/bs.pkl',
                ' datasets/fiw/fitted_original_5split/ss.pkl',
                ' datasets/fiw/fitted_original_5split/gfgd.pkl',
                ' datasets/fiw/fitted_original_5split/gfgs.pkl',
                ' datasets/fiw/fitted_original_5split/gmgd.pkl',
                ' datasets/fiw/fitted_original_5split/gmgs.pkl'
                ]

        self.train_self_ls = [
                ' datasets/fiw/fitted_original_5split/fd.pkl',
                ' datasets/fiw/fitted_original_5split/fs.pkl',
                ' datasets/fiw/fitted_original_5split/md.pkl',
                ' datasets/fiw/fitted_original_5split/ms.pkl',
                ' datasets/fiw/fitted_original_5split/bb.pkl',
                ' datasets/fiw/fitted_original_5split/bs.pkl',
                ' datasets/fiw/fitted_original_5split/ss.pkl',
                # ' datasets/fiw/fitted_original_5split/gfgd.pkl',
                # ' datasets/fiw/fitted_original_5split/gfgs.pkl',
                # ' datasets/fiw/fitted_original_5split/gmgd.pkl',
                # ' datasets/fiw/fitted_original_5split/gmgs.pkl'
                ]
        if '4_type' in args.name or '4type' in args.name:
            print("=" * 20 + "training 4 type")
            self.train_ls = [
                    ' datasets/fiw/fitted_original_5split/fd.pkl',
                    ' datasets/fiw/fitted_original_5split/fs.pkl',
                    ' datasets/fiw/fitted_original_5split/md.pkl',
                    ' datasets/fiw/fitted_original_5split/ms.pkl',
                    # ' datasets/fiw/fitted_original_5split/bb.pkl',
                    # ' datasets/fiw/fitted_original_5split/bs.pkl',
                    # ' datasets/fiw/fitted_original_5split/ss.pkl',
                    ]
        else:
            print("=" * 20 + "training 7 type")
            self.train_ls = [
                    ' datasets/fiw/fitted_original_5split/fd.pkl',
                    ' datasets/fiw/fitted_original_5split/fs.pkl',
                    ' datasets/fiw/fitted_original_5split/md.pkl',
                    ' datasets/fiw/fitted_original_5split/ms.pkl',
                    ' datasets/fiw/fitted_original_5split/bb.pkl',
                    ' datasets/fiw/fitted_original_5split/bs.pkl',
                    ' datasets/fiw/fitted_original_5split/ss.pkl',
                    ]
        self.data_pth = ' datasets/fiw/train-faces'
        self.data_set = fiw_dataset

    def get_train_loader(self):
        train_ls = self.train_ls
        data_pth = self.data_pth
        args =  self.args
        cfg = self.cfg
        if 'FG' in args.name:
            train_set = self.data_set(train_ls, data_pth, cfg["cross_validation"][str(args.trial)]["train_cs"],
                                      transform=train_transform_fg,
                                      cross_shuffle=True, neg_ratio=1,
                                      sf_aln=False, real_sn=args.real_sn)
            test_set = self.data_set(train_ls, data_pth, cfg["cross_validation"][str(args.trial)]["eval_cs"],
                                     transform=test_transform_fg,
                                     test=True, real_sn=args.real_sn)
            train_loader = DataLoader(train_set, batch_size=cfg['batch_size'], shuffle=True)
            test_loader = DataLoader(test_set, batch_size=cfg['batch_size'])
            return train_loader, test_loader

        else:
            train_set = self.data_set(train_ls, data_pth, cfg["cross_validation"][str(args.trial)]["train_cs"],
                                 transform=train_transform,
                                      cross_shuffle=True, neg_ratio=1,
                                 sf_aln=False, real_sn=args.real_sn)
            test_set = self.data_set(train_ls, data_pth, cfg["cross_validation"][str(args.trial)]["eval_cs"],
                                transform=test_transform,
                                test=True, real_sn=args.real_sn)
            train_loader = DataLoader(train_set, batch_size=cfg['batch_size'], shuffle=True)
            test_loader = DataLoader(test_set, batch_size=cfg['batch_size'])
            return train_loader, test_loader

    def get_train_self_loader(self):
        train_ls = self.train_ls
        data_pth = self.data_pth
        args =  self.args
        cfg = self.cfg
        if 'Facenet' in args.name:
            train_set = fiw_dataset_self(train_ls, data_pth, cfg["cross_validation"][str(args.trial)]["train_cs"],
                                         transform=train_transform_vggface,
                                         cross_shuffle=True, neg_ratio=3,
                                         sf_aln=False, real_sn=args.real_sn)
            test_set = fiw_dataset_self(train_ls, data_pth, cfg["cross_validation"][str(args.trial)]["eval_cs"],
                                        transform=test_transform_vggface,
                                        test=True, real_sn=args.real_sn)
            train_loader = DataLoader(train_set, batch_size=cfg['batch_size'], shuffle=True)
            test_loader = DataLoader(test_set, batch_size=cfg['batch_size'])
            return train_loader, test_loader
        else:
            train_set = fiw_dataset_self(train_ls, data_pth, cfg["cross_validation"][str(args.trial)]["train_cs"],
                                 transform=train_transform,
                                      cross_shuffle=True, neg_ratio=3,
                                 sf_aln=False, real_sn=args.real_sn)
            test_set = fiw_dataset_self(train_ls, data_pth, cfg["cross_validation"][str(args.trial)]["eval_cs"],
                                transform=test_transform,
                                test=True, real_sn=args.real_sn)
            train_loader = DataLoader(train_set, batch_size=cfg['batch_size'], shuffle=True)
            test_loader = DataLoader(test_set, batch_size=cfg['batch_size'])
            return train_loader, test_loader

    def get_close_train_loader(self):
        train_ls = self.train_ls
        data_pth = self.data_pth
        args = self.args
        cfg = self.cfg

        train_set = self.data_set(train_ls, data_pth, cfg["cross_validation"][str(args.trial)]["train_cs"],
                             transform=train_transform,cross_shuffle=False, sf_aln=False,
                             real_sn=args.real_sn,get_pos=True)
        test_set = self.data_set(train_ls, data_pth, cfg["cross_validation"][str(args.trial)]["eval_cs"],
                            transform=test_transform,test=True, real_sn=args.real_sn,get_pos=True)
        train_loader = DataLoader(train_set, batch_size=cfg['batch_size'], shuffle=True)
        test_loader = DataLoader(test_set, batch_size=cfg['batch_size'])
        return train_loader, test_loader

    def get_fiw_eval_loaders(self):
        # train_ls = self.train_ls
        train_ls_test = self.train_ls_test
        data_pth = self.data_pth
        args =  self.args
        cfg = self.cfg
        test_set = self.data_set(train_ls_test, data_pth, cfg["cross_validation"][str(args.trial)]["eval_cs"],
                            transform=test_transform,
                            real_sn=args.real_sn)
        known_set = fiw_dataset_test(train_ls_test, data_pth, cfg["cross_validation"][str(args.trial)]["eval_cs"],
                             transform=test_transform,
                             real_sn=args.real_sn, get_pos=True)
        unknown_set = fiw_dataset_test(train_ls_test, data_pth, cfg["cross_validation"][str(args.trial)]["eval_cs"],
                               transform=test_transform,
                               real_sn=args.real_sn, get_neg=True)

        test_loader = DataLoader(test_set, batch_size=cfg['batch_size'])

        known_set_loader = DataLoader(known_set, batch_size=cfg['batch_size'])
        unknown_loader = DataLoader(unknown_set, batch_size=cfg['batch_size'])

        return known_set_loader, unknown_loader, test_loader

    def get_fiw_eval_loaders_new(self):
        # train_ls = self.train_ls
        train_ls_test = self.train_ls_test
        data_pth = self.data_pth
        args =  self.args
        cfg = self.cfg
        if 'Facenet' in args.name:
            test_set = fiw_dataset_new(train_ls_test, data_pth, cfg["cross_validation"][str(args.trial)]["eval_cs"],
                                       transform=test_transform_vggface,
                                       real_sn=args.real_sn)
            known_set = fiw_dataset_new(train_ls_test, data_pth, cfg["cross_validation"][str(args.trial)]["eval_cs"],
                                        transform=test_transform_vggface,
                                        real_sn=args.real_sn, get_pos=True)
            unknown_set = fiw_dataset_new(train_ls_test, data_pth, cfg["cross_validation"][str(args.trial)]["eval_cs"],
                                          transform=test_transform_vggface,
                                          real_sn=args.real_sn, get_neg=True)

            test_loader = DataLoader(test_set, batch_size=cfg['batch_size'])

            known_set_loader = DataLoader(known_set, batch_size=cfg['batch_size'])
            unknown_loader = DataLoader(unknown_set, batch_size=cfg['batch_size'])

            return known_set_loader, unknown_loader, test_loader
        else:
            test_set = fiw_dataset_new(train_ls_test, data_pth, cfg["cross_validation"][str(args.trial)]["eval_cs"],
                                transform=test_transform,
                                real_sn=args.real_sn)
            known_set = fiw_dataset_new(train_ls_test, data_pth, cfg["cross_validation"][str(args.trial)]["eval_cs"],
                                 transform=test_transform,
                                 real_sn=args.real_sn, get_pos=True)
            unknown_set = fiw_dataset_new(train_ls_test, data_pth, cfg["cross_validation"][str(args.trial)]["eval_cs"],
                                   transform=test_transform,
                                   real_sn=args.real_sn, get_neg=True)

            test_loader = DataLoader(test_set, batch_size=cfg['batch_size'])

            known_set_loader = DataLoader(known_set, batch_size=cfg['batch_size'])
            unknown_loader = DataLoader(unknown_set, batch_size=cfg['batch_size'])

            return known_set_loader, unknown_loader, test_loader

    def get_fiw_eval_loaders_img_new(self):
        # train_ls = self.train_ls
        train_ls_test = self.train_ls_test
        data_pth = self.data_pth
        args = self.args
        cfg = self.cfg
        if 'FG' in args.name:
            test_set = fiw_dataset_img_new(train_ls_test, data_pth, cfg["cross_validation"][str(args.trial)]["eval_cs"],
                                           transform=test_transform_fg,
                                           real_sn=args.real_sn)
            known_set = fiw_dataset_img_new(train_ls_test, data_pth,
                                            cfg["cross_validation"][str(args.trial)]["eval_cs"],
                                            transform=test_transform_fg,
                                            real_sn=args.real_sn, get_pos=True)
            unknown_set = fiw_dataset_img_new(train_ls_test, data_pth,
                                              cfg["cross_validation"][str(args.trial)]["eval_cs"],
                                              transform=test_transform_fg,
                                              real_sn=args.real_sn, get_neg=True)

            test_loader = DataLoader(test_set, batch_size=cfg['batch_size'])

            known_set_loader = DataLoader(known_set, batch_size=cfg['batch_size'])
            unknown_loader = DataLoader(unknown_set, batch_size=cfg['batch_size'])

            return known_set_loader, unknown_loader, test_loader
        elif 'Facenet' in args.name:
            test_set = fiw_dataset_img_new(train_ls_test, data_pth, cfg["cross_validation"][str(args.trial)]["eval_cs"],
                                           transform=test_transform_vggface,
                                           real_sn=args.real_sn)
            known_set = fiw_dataset_img_new(train_ls_test, data_pth,
                                            cfg["cross_validation"][str(args.trial)]["eval_cs"],
                                            transform=test_transform_vggface,
                                            real_sn=args.real_sn, get_pos=True)
            unknown_set = fiw_dataset_img_new(train_ls_test, data_pth,
                                              cfg["cross_validation"][str(args.trial)]["eval_cs"],
                                              transform=test_transform_vggface,
                                              real_sn=args.real_sn, get_neg=True)

            test_loader = DataLoader(test_set, batch_size=cfg['batch_size'])

            known_set_loader = DataLoader(known_set, batch_size=cfg['batch_size'])
            unknown_loader = DataLoader(unknown_set, batch_size=cfg['batch_size'])

            return known_set_loader, unknown_loader, test_loader
        else:
            test_set = fiw_dataset_img_new(train_ls_test, data_pth, cfg["cross_validation"][str(args.trial)]["eval_cs"],
                                           transform=test_transform,
                                           real_sn=args.real_sn)
            known_set = fiw_dataset_img_new(train_ls_test, data_pth,
                                            cfg["cross_validation"][str(args.trial)]["eval_cs"],
                                            transform=test_transform,
                                            real_sn=args.real_sn, get_pos=True)
            unknown_set = fiw_dataset_img_new(train_ls_test, data_pth,
                                              cfg["cross_validation"][str(args.trial)]["eval_cs"],
                                              transform=test_transform,
                                              real_sn=args.real_sn, get_neg=True)

            test_loader = DataLoader(test_set, batch_size=cfg['batch_size'])

            known_set_loader = DataLoader(known_set, batch_size=cfg['batch_size'])
            unknown_loader = DataLoader(unknown_set, batch_size=cfg['batch_size'])

            return known_set_loader, unknown_loader, test_loader

    def get_vis_loader(self,cs = "train_cs"):
        train_ls_test = self.train_ls_test
        data_pth = self.data_pth
        args = self.args
        cfg = self.cfg

        pos = self.data_set(train_ls_test, data_pth, cfg["cross_validation"][str(args.trial)][cs],
                             transform=test_transform,
                             real_sn=args.real_sn,get_pos=True)
        neg = self.data_set(train_ls_test, data_pth, cfg["cross_validation"][str(args.trial)][cs],
                            transform=test_transform,
                             real_sn=args.real_sn,get_neg=True)
        pos_loader = DataLoader(pos, batch_size=cfg['batch_size'], shuffle=True)
        neg_loader = DataLoader(neg, batch_size=cfg['batch_size'])
        return pos_loader, neg_loader


    # def get_vis_loader_test(self):
    #     train_ls_test = self.train_ls_test
    #     data_pth = self.data_pth
    #     args = self.args
    #     cfg = self.cfg
    #
    #     known = self.data_set(train_ls_test, data_pth, cfg["cross_validation"][str(args.trial)]["eval_cs"],
    #                          transform=test_transform,
    #                          real_sn=args.real_sn,get_pos=True)
    #     unknown = self.data_set(train_ls_test, data_pth, cfg["cross_validation"][str(args.trial)]["eval_cs"],
    #                         transform=test_transform,
    #                          real_sn=args.real_sn,get_neg=True)
    #     known_loader = DataLoader(known, batch_size=cfg['batch_size'], shuffle=True)
    #     unknown_loader = DataLoader(unknown, batch_size=cfg['batch_size'])
    #     return known_loader, unknown_loader




class fiw_triplet_loader():
    def __init__(self,args,cfg):
        self.args = args
        self.cfg = cfg

        self.train_ls_test = [
                ' datasets/fiw/fitted_original_5split/fd.pkl',
                ' datasets/fiw/fitted_original_5split/fs.pkl',
                ' datasets/fiw/fitted_original_5split/md.pkl',
                ' datasets/fiw/fitted_original_5split/ms.pkl',
                ' datasets/fiw/fitted_original_5split/bb.pkl',
                ' datasets/fiw/fitted_original_5split/bs.pkl',
                ' datasets/fiw/fitted_original_5split/ss.pkl',
                ' datasets/fiw/fitted_original_5split/gfgd.pkl',
                ' datasets/fiw/fitted_original_5split/gfgs.pkl',
                ' datasets/fiw/fitted_original_5split/gmgd.pkl',
                ' datasets/fiw/fitted_original_5split/gmgs.pkl'
                ]

        if '4_type' in args.name or '4type' in args.name:
            print("="*20 +"training 4 type")
            self.train_ls = [
                    ' datasets/fiw/fitted_original_5split/fd.pkl',
                    ' datasets/fiw/fitted_original_5split/fs.pkl',
                    ' datasets/fiw/fitted_original_5split/md.pkl',
                    ' datasets/fiw/fitted_original_5split/ms.pkl',
                    # ' datasets/fiw/fitted_original_5split/bb.pkl',
                    # ' datasets/fiw/fitted_original_5split/bs.pkl',
                    # ' datasets/fiw/fitted_original_5split/ss.pkl',
                    ]
        else:
            print("=" * 20 + "training 7 type")
            self.train_ls = [
                    ' datasets/fiw/fitted_original_5split/fd.pkl',
                    ' datasets/fiw/fitted_original_5split/fs.pkl',
                    ' datasets/fiw/fitted_original_5split/md.pkl',
                    ' datasets/fiw/fitted_original_5split/ms.pkl',
                    ' datasets/fiw/fitted_original_5split/bb.pkl',
                    ' datasets/fiw/fitted_original_5split/bs.pkl',
                    ' datasets/fiw/fitted_original_5split/ss.pkl',
                    ]

        self.data_pth = '/home//Documents/DATA/kinship/FIW/competition/rfiw2020/verification/train-faces'
        # self.data_set = fiw_dataset
        print('this is old split new path')

    def get_train_loader(self):
        train_ls = self.train_ls
        data_pth = self.data_pth
        args =  self.args
        cfg = self.cfg

        train_set = fiw_triplet_dataset(train_ls, data_pth, cfg["cross_validation"][str(args.trial)]["train_cs"],
                             transform=train_transform,
                                  cross_shuffle=True, neg_ratio=3,
                             sf_aln=False, real_sn=args.real_sn)
        test_set = fiw_triplet_dataset(train_ls, data_pth, cfg["cross_validation"][str(args.trial)]["eval_cs"],
                            transform=test_transform,
                            test=True, real_sn=args.real_sn)
        train_loader = DataLoader(train_set, batch_size=cfg['batch_size'], shuffle=True)
        test_loader = DataLoader(test_set, batch_size=cfg['batch_size'])

        return train_loader, test_loader

    def get_train_HK_loader(self):
        train_ls = self.train_ls
        data_pth = self.data_pth
        args = self.args
        cfg = self.cfg
        if 'FG' in args.name:
            train_set = fiw_triplet_HK_dataset(train_ls, data_pth, cfg["cross_validation"][str(args.trial)]["train_cs"],
                                               transform=train_transform_fg,
                                               cross_shuffle=True, neg_ratio=1,
                                               sf_aln=False, real_sn=args.real_sn)
            test_set = fiw_triplet_HK_dataset(train_ls, data_pth, cfg["cross_validation"][str(args.trial)]["eval_cs"],
                                              transform=test_transform_fg,
                                              test=True, real_sn=args.real_sn)
            train_loader = DataLoader(train_set, batch_size=cfg['batch_size'], shuffle=True)
            test_loader = DataLoader(test_set, batch_size=cfg['batch_size'])
        else:
            train_set = fiw_triplet_HK_dataset(train_ls, data_pth, cfg["cross_validation"][str(args.trial)]["train_cs"],
                                            transform=train_transform,
                                            cross_shuffle=True, neg_ratio=1,
                                            sf_aln=False, real_sn=args.real_sn)
            test_set = fiw_triplet_HK_dataset(train_ls, data_pth, cfg["cross_validation"][str(args.trial)]["eval_cs"],
                                           transform=test_transform,
                                           test=True, real_sn=args.real_sn)
            train_loader = DataLoader(train_set, batch_size=cfg['batch_size'], shuffle=True)
            test_loader = DataLoader(test_set, batch_size=cfg['batch_size'])

        return train_loader, test_loader


    def get_close_train_loader(self):
        train_ls = self.train_ls
        data_pth = self.data_pth
        args = self.args
        cfg = self.cfg

        train_set = fiw_dataset(train_ls, data_pth, cfg["cross_validation"][str(args.trial)]["train_cs"],
                             transform=train_transform,cross_shuffle=False, sf_aln=False,
                             real_sn=args.real_sn,get_pos=True)
        test_set = fiw_dataset(train_ls, data_pth, cfg["cross_validation"][str(args.trial)]["eval_cs"],
                            transform=test_transform,test=True, real_sn=args.real_sn,get_pos=True)
        train_loader = DataLoader(train_set, batch_size=cfg['batch_size'], shuffle=True)
        test_loader = DataLoader(test_set, batch_size=cfg['batch_size'])
        return train_loader, test_loader

    def get_fiw_eval_loaders(self):
        # train_ls = self.train_ls
        train_ls_test = self.train_ls_test
        data_pth = self.data_pth
        args =  self.args
        cfg = self.cfg
        test_set = fiw_dataset(train_ls_test, data_pth, cfg["cross_validation"][str(args.trial)]["eval_cs"],
                            transform=test_transform,
                            real_sn=args.real_sn)
        known_set = fiw_dataset_test(train_ls_test, data_pth, cfg["cross_validation"][str(args.trial)]["eval_cs"],
                             transform=test_transform,
                             real_sn=args.real_sn, get_pos=True)
        unknown_set = fiw_dataset_test(train_ls_test, data_pth, cfg["cross_validation"][str(args.trial)]["eval_cs"],
                               transform=test_transform,
                               real_sn=args.real_sn, get_neg=True)

        test_loader = DataLoader(test_set, batch_size=cfg['batch_size'])

        known_set_loader = DataLoader(known_set, batch_size=cfg['batch_size'])
        unknown_loader = DataLoader(unknown_set, batch_size=cfg['batch_size'])

        return known_set_loader, unknown_loader, test_loader

    def get_vis_loader(self):
        train_ls_test = self.train_ls_test
        data_pth = self.data_pth
        args = self.args
        cfg = self.cfg

        train_pos = fiw_dataset(train_ls_test, data_pth, cfg["cross_validation"][str(args.trial)]["train_cs"],
                             transform=test_transform,
                             real_sn=args.real_sn,get_pos=True)
        train_neg = fiw_dataset(train_ls_test, data_pth, cfg["cross_validation"][str(args.trial)]["train_cs"],
                            transform=test_transform,
                             real_sn=args.real_sn,get_neg=True)
        train_loader = DataLoader(train_pos, batch_size=cfg['batch_size'], shuffle=True)
        test_loader = DataLoader(train_neg, batch_size=cfg['batch_size'])
        return train_loader, test_loader

    def get_vis_loader_test(self):
        train_ls_test = self.train_ls_test
        data_pth = self.data_pth
        args = self.args
        cfg = self.cfg

        known = fiw_dataset(train_ls_test, data_pth, cfg["cross_validation"][str(args.trial)]["eval_cs"],
                             transform=test_transform,
                             real_sn=args.real_sn,get_pos=True)
        unknown = fiw_dataset(train_ls_test, data_pth, cfg["cross_validation"][str(args.trial)]["eval_cs"],
                            transform=test_transform,
                             real_sn=args.real_sn,get_neg=True)
        known_loader = DataLoader(known, batch_size=cfg['batch_size'], shuffle=True)
        unknown_loader = DataLoader(unknown, batch_size=cfg['batch_size'])
        return known_loader, unknown_loader





def get_train_loaders(datasetName, trial_num, cfg):
    """
        Create training dataloaders.

        datasetName: name of dataset
        trial_num: trial number dictating known/unknown class split
        cfg: config file

        returns trainloader, evalloader, testloader, mapping - changes labels from original to known class label
    """
    trainSet, valSet, testSet, _ = load_datasets(datasetName, cfg, trial_num)

    with open("datasets/{}/trainval_idxs.json".format(datasetName)) as f:
        trainValIdxs = json.load(f)
        train_idxs = trainValIdxs['Train']
        val_idxs = trainValIdxs['Val']

    with open("datasets/{}/class_splits/{}.json".format(datasetName, trial_num)) as f:
        class_splits = json.load(f)
        known_classes = class_splits['Known']

    trainSubset = create_dataSubsets(trainSet, known_classes, train_idxs)
    valSubset = create_dataSubsets(valSet, known_classes, val_idxs)
    testSubset = create_dataSubsets(testSet, known_classes)

    # create a mapping from dataset target class number to network known class number
    mapping = create_target_map(known_classes, cfg['num_classes'])

    batch_size = cfg['batch_size']

    trainloader = torch.utils.data.DataLoader(trainSubset, batch_size=batch_size, shuffle=True,
                                              num_workers=cfg['dataloader_workers'])
    valloader = torch.utils.data.DataLoader(valSubset, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(testSubset, batch_size=batch_size, shuffle=True)

    return trainloader, valloader, testloader, mapping


def get_cacur_eval_loaders(datasetName, trial_num, cfg):
    """
        Create evaluation dataloaders.

        datasetName: name of dataset
        trial_num: trial number dictating known/unknown class split
        cfg: config file

        returns knownloader, unknownloader, mapping - changes labels from original to known class label
    """
    if '+' in datasetName or 'All' in datasetName:
        _, _, testSet, unknownSet = load_datasets(datasetName, cfg, trial_num)
    else:
        _, _, testSet, _ = load_datasets(datasetName, cfg, trial_num)

    with open("datasets/{}/class_splits/{}.json".format(datasetName, trial_num)) as f:
        class_splits = json.load(f)
        known_classes = class_splits['Known']
        unknown_classes = class_splits['Unknown-Unknown']

    testSubset = create_dataSubsets(testSet, known_classes)

    if '+' in datasetName or 'All' in datasetName:
        unknownSubset = create_dataSubsets(unknownSet, unknown_classes)
    else:
        unknownSubset = create_dataSubsets(testSet, unknown_classes)

    # create a mapping from dataset target class number to network known class number
    mapping = create_target_map(known_classes, cfg['num_classes'])

    batch_size = cfg['batch_size']

    knownloader = torch.utils.data.DataLoader(testSubset, batch_size=batch_size, shuffle=False)
    unknownloader = torch.utils.data.DataLoader(unknownSubset, batch_size=batch_size, shuffle=False)

    return knownloader, unknownloader, mapping


def get_eval_loaders(datasetName, trial_num, cfg):
    """
        Create evaluation dataloaders.

        datasetName: name of dataset
        trial_num: trial number dictating known/unknown class split
        cfg: config file

        returns knownloader, unknownloader, mapping - changes labels from original to known class label
    """
    if '+' in datasetName or 'All' in datasetName:
        _, _, testSet, unknownSet = load_datasets(datasetName, cfg, trial_num)
    else:
        _, _, testSet, _ = load_datasets(datasetName, cfg, trial_num)

    with open("datasets/{}/class_splits/{}.json".format(datasetName, trial_num)) as f:
        class_splits = json.load(f)
        known_classes = class_splits['Known']
        unknown_classes = class_splits['Unknown']

    testSubset = create_dataSubsets(testSet, known_classes)

    if '+' in datasetName or 'All' in datasetName:
        unknownSubset = create_dataSubsets(unknownSet, unknown_classes)
    else:
        unknownSubset = create_dataSubsets(testSet, unknown_classes)

    # create a mapping from dataset target class number to network known class number
    mapping = create_target_map(known_classes, cfg['num_classes'])

    batch_size = cfg['batch_size']

    knownloader = torch.utils.data.DataLoader(testSubset, batch_size=batch_size, shuffle=False)
    unknownloader = torch.utils.data.DataLoader(unknownSubset, batch_size=batch_size, shuffle=False)

    return knownloader, unknownloader, mapping


def get_data_stats(dataset, known_classes):
    """
        Calculates mean and std of data in a dataset.

        dataset: dataset to calculate mean and std of
        known_classes: what classes are known and should be included

        returns means and stds of data, across each colour channel
    """
    try:
        ims = np.asarray(dataset.data)
        try:
            labels = np.asarray(dataset.targets)
        except:
            labels = np.asarray(dataset.labels)

        mask = labels == 1000
        for cl in known_classes:
            mask = mask | (labels == cl)
        known_ims = ims[mask]

        means = []
        stds = []
        if len(np.shape(known_ims)) < 4:
            means += [known_ims.mean() / 255]
            stds += [known_ims.std() / 255]
        else:
            for i in range():
                means += [known_ims[:, :, :, i].mean() / 255]
                stds += [known_ims[:, :, :, i].std() / 255]
    except:
        imloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)

        r_data = []
        g_data = []
        b_data = []
        for i, data in enumerate(imloader):
            im, labels = data
            mask = labels == 1000
            for cl in known_classes:
                mask = mask | (labels == cl)
            if torch.sum(mask) == 0:
                continue
            im = im[mask]
            r_data += im[:, 0].detach().tolist()
            g_data += im[:, 1].detach().tolist()
            b_data += im[:, 2].detach().tolist()
        means = [np.mean(r_data), np.mean(g_data), np.mean(b_data)]
        stds = [np.std(r_data), np.std(g_data), np.std(b_data)]
    return means, stds


def load_datasets(datasetName, cfg, trial_num):
    """
        Load all datasets for training/evaluation.

        datasetName: name of dataset
        cfg: config file
        trial_num: trial number dictating known/unknown class split

        returns trainset, valset, knownset, unknownset
    """
    with open("datasets/{}/class_splits/{}.json".format(datasetName, trial_num)) as f:
        class_splits = json.load(f)
        known_classes = class_splits['Known']

    # controls data transforms
    means = cfg['data_mean'][trial_num]
    stds = cfg['data_std'][trial_num]
    flip = cfg['data_transforms']['flip']
    rotate = cfg['data_transforms']['rotate']
    scale_min = cfg['data_transforms']['scale_min']

    transforms = {
        'train': tf.Compose([
            tf.Resize(cfg['im_size']),
            tf.RandomResizedCrop(cfg['im_size'], scale=(scale_min, 1.0)),
            tf.RandomHorizontalFlip(flip),
            tf.RandomRotation(rotate),
            tf.ToTensor(),
            tf.Normalize(means, stds)
        ]),
        'val': tf.Compose([
            tf.Resize(cfg['im_size']),
            tf.ToTensor(),
            tf.Normalize(means, stds)
        ]),
        'test': tf.Compose([
            tf.Resize(cfg['im_size']),
            tf.ToTensor(),
            tf.Normalize(means, stds)
        ])
    }

    unknownSet = None
    if datasetName == "MNIST":
        trainSet = torchvision.datasets.MNIST('datasets/data', transform=transforms['train'], download=True)
        valSet = torchvision.datasets.MNIST('datasets/data', transform=transforms['val'])
        testSet = torchvision.datasets.MNIST('datasets/data', train=False, transform=transforms['test'])
    elif "CIFAR" in datasetName:
        trainSet = torchvision.datasets.CIFAR10('datasets/data', transform=transforms['train'], download=True)
        valSet = torchvision.datasets.CIFAR10('datasets/data', transform=transforms['val'])
        testSet = torchvision.datasets.CIFAR10('datasets/data', train=False, transform=transforms['test'])
        if '+' in datasetName:
            unknownSet = torchvision.datasets.CIFAR100('datasets/data', train=False, transform=transforms['test'],
                                                       download=True)
    elif datasetName == "SVHN":
        trainSet = torchvision.datasets.SVHN('datasets/data', transform=transforms['train'], download=True)
        valSet = torchvision.datasets.SVHN('datasets/data', transform=transforms['val'])
        testSet = torchvision.datasets.SVHN('datasets/data', split='test', transform=transforms['test'])
    elif datasetName == "TinyImageNet":
        trainSet = torchvision.datasets.ImageFolder('datasets/data/tiny-imagenet-200/train',
                                                    transform=transforms['train'])
        valSet = torchvision.datasets.ImageFolder('datasets/data/tiny-imagenet-200/train', transform=transforms['val'])
        testSet = torchvision.datasets.ImageFolder('datasets/data/tiny-imagenet-200/val', transform=transforms['test'])
    else:
        print("Sorry, that dataset has not been implemented.")
        exit()

    return trainSet, valSet, testSet, unknownSet


def get_anchor_loaders(datasetName, trial_num, cfg):
    """
        Supply trainloaders, with no extra rotate/crop data augmentation, for calculating anchor class centres.

        datasetName: name of dataset
        trial_num: trial number dictating known/unknown class split
        cfg: config file

        returns trainloader and trainloaderFlipped (horizontally)
    """
    trainSet, trainSetFlipped = load_anchor_datasets(datasetName, cfg, trial_num)

    with open("datasets/{}/trainval_idxs.json".format(datasetName)) as f:
        trainValIdxs = json.load(f)
        train_idxs = trainValIdxs['Train']
        val_idxs = trainValIdxs['Val']

    with open("datasets/{}/class_splits/{}.json".format(datasetName, trial_num)) as f:
        class_splits = json.load(f)
        known_classes = class_splits['Known']

    trainSubset = create_dataSubsets(trainSet, known_classes, train_idxs)
    trainSubsetFlipped = create_dataSubsets(trainSetFlipped, known_classes, train_idxs)

    trainloader = torch.utils.data.DataLoader(trainSubset, batch_size=128)
    trainloaderFlipped = torch.utils.data.DataLoader(trainSubsetFlipped, batch_size=128)

    return trainloader, trainloaderFlipped


def load_anchor_datasets(datasetName, cfg, trial_num):
    """
        Load train datasets, with no extra rotate/crop data augmentation, for calculating anchor class centres.

        datasetName: name of dataset
        cfg: config file
        trial_num: trial number dictating known/unknown class split

        returns trainset and trainsetFlipped (horizontally)
    """
    with open("datasets/{}/class_splits/{}.json".format(datasetName, trial_num)) as f:
        class_splits = json.load(f)
        known_classes = class_splits['Known']

    means = cfg['data_mean'][trial_num]
    stds = cfg['data_std'][trial_num]

    # for digit datasets, we don't want to provide a flip dataset
    if datasetName == "MNIST" or datasetName == "SVHN":
        flip = 0
    else:
        flip = 1

    transforms = {
        'train': tf.Compose([
            tf.Resize(cfg['im_size']), tf.CenterCrop(cfg['im_size']),
            tf.ToTensor(),
            tf.Normalize(means, stds)
        ]),
        'trainFlip': tf.Compose([
            tf.Resize(cfg['im_size']), tf.CenterCrop(cfg['im_size']),
            tf.RandomHorizontalFlip(flip),
            tf.ToTensor(),
            tf.Normalize(means, stds)
        ])
    }
    if datasetName == "MNIST":
        trainSet = torchvision.datasets.MNIST('datasets/data', transform=transforms['train'])
        trainSetFlip = torchvision.datasets.MNIST('datasets/data', transform=transforms['trainFlip'])
    elif "CIFAR" in datasetName:
        trainSet = torchvision.datasets.CIFAR10('datasets/data', transform=transforms['train'])
        trainSetFlip = torchvision.datasets.CIFAR10('datasets/data', transform=transforms['trainFlip'])
    elif datasetName == "SVHN":
        trainSet = torchvision.datasets.SVHN('datasets/data', transform=transforms['train'])
        trainSetFlip = torchvision.datasets.SVHN('datasets/data', transform=transforms['trainFlip'])
    elif datasetName == "TinyImageNet":
        trainSet = torchvision.datasets.ImageFolder('datasets/data/tiny-imagenet-200/train',
                                                    transform=transforms['train'])
        trainSetFlip = torchvision.datasets.ImageFolder('datasets/data/tiny-imagenet-200/train',
                                                        transform=transforms['trainFlip'])
    else:
        print("Sorry, that dataset has not been implemented.")
        exit()

    return trainSet, trainSetFlip


def create_dataSubsets(dataset, classes_to_use, idxs_to_use=None):
    """
        Returns dataset subset that satisfies class and idx restraints.
        dataset: torchvision dataset
        classes_to_use: classes that are allowed in the subset (known vs unknown)
        idxs_to_use: image indexes that are allowed in the subset (train vs val, not relevant for test)

        returns torch Subset
    """
    import torch

    # get class label for dataset images. svhn has different syntax as .labels
    try:
        targets = dataset.targets
    except:
        targets = dataset.labels

    subset_idxs = []
    if idxs_to_use == None:
        for i, lbl in enumerate(targets):
            if lbl in classes_to_use:
                subset_idxs += [i]
    else:
        for class_num in idxs_to_use.keys():
            if int(class_num) in classes_to_use:
                subset_idxs += idxs_to_use[class_num]

    dataSubset = torch.utils.data.Subset(dataset, subset_idxs)
    return dataSubset


def create_target_map(known_classes, num_classes):
    """
        Creates a mapping from original dataset labels to new 'known class' training label
        known_classes: classes that will be trained with
        num_classes: number of classes the dataset typically has

        returns mapping - a dictionary where mapping[original_class_label] = known_class_label
    """
    mapping = [None for i in range(num_classes)]

    known_classes.sort()
    for i, num in enumerate(known_classes):
        mapping[num] = i

    return mapping