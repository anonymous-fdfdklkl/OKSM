import scipy.io
import os
import torch
from torch.utils.data import Dataset,DataLoader
import random
from random import shuffle
import pickle
# from utils.transform import facenet_trans,crop_full_transform
from PIL import Image
import copy
import torchvision.transforms as transforms
import numpy as np
import torch


#################################### kinFaceW transform
train_transform = transforms.Compose(
    [
     transforms.Resize((64,64)),
     # transforms.ColorJitter(brightness=0.3,
     #                        contrast=0.3,
     #                        saturation=0.3,
     #                        hue=0.3
     #                        ),

     transforms.RandomGrayscale(),
     transforms.RandomHorizontalFlip(p=0.4),
     # transforms.RandomPerspective(distortion_scale=0.1,p=0.3),
     # transforms.RandomResizedCrop(size=(64,64), scale=(0.9, 1.05), ratio=(0.97, 1.05), interpolation=2),
     transforms.ToTensor(),
     # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
     # transforms.RandomErasing()
     ])


train_transform_vggface = transforms.Compose(
    [
     transforms.Resize((160,160)),
     # transforms.ColorJitter(brightness=0.3,
     #                        contrast=0.3,
     #                        saturation=0.3,
     #                        hue=0.3
     #                        ),

     transforms.RandomGrayscale(),
     transforms.RandomHorizontalFlip(p=0.4),
     # transforms.RandomPerspective(distortion_scale=0.1,p=0.3),
     # transforms.RandomResizedCrop(size=(64,64), scale=(0.9, 1.05), ratio=(0.97, 1.05), interpolation=2),
     transforms.ToTensor(),
     # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
     # transforms.RandomErasing()
     ])

train_transform_fg = transforms.Compose(
    [
     transforms.Resize((224,224)),
     transforms.RandomGrayscale(),
     transforms.RandomHorizontalFlip(p=0.4),
     transforms.ToTensor(),
     transforms.Normalize((91.4953, 103.8827, 131.0912), (1, 1, 1))
     ])

test_transform = transforms.Compose(
    [
     transforms.Resize((64,64)),
     # transforms.Resize((73,73)),
     # transforms.CenterCrop((64, 64)),
     # transforms.Grayscale(num_output_channels=3),
     transforms.ToTensor(),
     # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

test_transform_vggface = transforms.Compose(
    [
     transforms.Resize((160,160)),
     # transforms.Resize((73,73)),
     # transforms.CenterCrop((64, 64)),
     # transforms.Grayscale(num_output_channels=3),
     transforms.ToTensor(),
     # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])


test_transform_fg = transforms.Compose(
    [
     transforms.Resize((224,224)),
     transforms.ToTensor(),
     transforms.Normalize((91.4953, 103.8827, 131.0912), (1, 1, 1))
    ])
######################################

class Kin_dataset(Dataset):
    def __init__(self,mat_path,im_root, cross_vali = None,transform = None,
                 cross_shuffle = False,sf_aln = False,
                 test = False,test_each = False,real_sn = False,read_all_imgs = False,
                 get_neg=False,get_pos=False,neg_ratio=1):
        """
        THE dataset of KinfaceW-I/ KinfaceW-II
        :param mat_path:
        :param im_root:
        :param cross_vali:    cross validation's folds: e.g. [1,2,4,5]
        :param transform:     add data augmentation
        :param cross_shuffle: shuffle names among pair list
        :param sf_aln:        whether shuffle all names or only img2s' names
        out: imgs(1x6x64x64),kin(0/1),img1_n(imag1_name),img2_n(image2_name)
        """
        self.neg_ratio = neg_ratio
        self.get_neg = get_neg
        self.get_pos = get_pos
        self.read_all_imgs = read_all_imgs
        self.neg_flag = 0
        self.test = test
        self.im_root = im_root
        # kin_list is the whole 1,2,3,4,5 folds from mat
        if not test_each:
            self.kin_list = self.read_mat(mat_path,im_root)
        else:
            self.kin_list = self.read_test_mat(mat_path,im_root)

        self.transform = transform
        self.cout = 0
        if cross_vali is not None:
            #extract matched folds e.g. [1,2,4,5]
            self.kin_list = self._get_cross(cross_vali)
        # if cross_shuffle:
        self.crf = cross_shuffle  # cross shuffle

        # store all img2/ all_(img1+img2) list for cross shuffle,
        self.img2_list = [i[3] for i in self.kin_list]
        self.alln_list = self.get_alln(self.kin_list)
        self.sf_aln = sf_aln
        # self.crs_insect = 0
        # self.kin_list_bak= copy.deepcopy(self.kin_list)
        if real_sn:
            self.kin_list = self.get_real_sn(self.kin_list)

        self.kin_list = self._init_list(self.kin_list)

        self.lth = len(self.kin_list)

    def _init_list(self,ls):
        return ls

    def get_real_sn(self,lis):
        new_kin_ls = []

        return new_kin_ls

    def get_alln(self,ls):
        all_name = []
        for i in ls:
            all_name.append(i[2])
            all_name.append(i[3])
        return all_name

    def __len__(self):
        return len(self.kin_list)

    def __getitem__(self, item):

        if torch.is_tensor(item):
            item = item.tolist()
        # extract img1
        img1_p = self.kin_list[item][2]
        img1 = Image.open(img1_p)
        # extract img2
        img2_p = self.kin_list[item][3]
        img2 = Image.open(img2_p)
        # get kin label 0/1
        kin = self.kin_list[item][1]
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        imgs = torch.cat((img1,img2))
        # after each epoch, shuffle once
        if self.crf:
            self.cout +=1
            if self.cout == self.lth:
                self.cout = 0
                self._cross_shuffle(1)


        return imgs,kin

    def _cross_shuffle(self,neg_ratio):
        """
        shuffle the second images name after each epoch
        :return:
        """
        if self.sf_aln:
            im2_ls = self.alln_list
        else:
            im2_ls = self.img2_list
        rand_lth = len(im2_ls)
        new_pair_list = []

        for pair_l in self.kin_list:
            if pair_l[1] == -1:
                for neg_iter in range(neg_ratio):
                    new_img2 = im2_ls[random.randint(0, rand_lth-1)]
                    while pair_l[2][-12:-6] == new_img2[-12:-6]:
                        new_img2 = im2_ls[random.randint(0, rand_lth-1)]
                    pair_l[3] = new_img2
                    new_pair_list.append(pair_l)
            else:
                new_pair_list.append(pair_l)

        self.kin_list = new_pair_list
        return self.kin_list

    def read_mat(self,mat_paths,im_root):
        new_kin_list = []
        return new_kin_list

    def read_test_mat(self,mat_path,im_root):
        new_kin_list = []
        return new_kin_list

    def _read_mat(self,mat_path):
        nemo_ls = []
        return nemo_ls

    def get_img_name(self,ims_ls):
        new_ls = []
        return new_ls

    def _get_cross(self,cross):

        return [i for i in self.kin_list if i[0] in cross]




class kfw_caur(Kin_dataset):

    def _init_list(self, ls):
        if self.crf:
            ls = self._cross_shuffle(self.neg_ratio)
        if self.get_neg:
            new= [item for item in ls if item[1]==-1]
            return  new
        elif self.get_pos:
            new=[item for item in ls if item[1]!=-1]
            return  new
        else:
            return ls

    def read_mat(self,mat_paths,im_roots):
            self.neg_flag = -1
            ls_dict = {}
            for i,(mat_path,im_root) in enumerate(zip(mat_paths,im_roots)):
                kin_list = self._read_mat(mat_path)
                for kl in kin_list:
                    if not kl[1]==0:
                        kl[1]=kl[1]+i-1
                    else:
                        kl[1] = -1
                    kl[2]=os.path.join(im_root,kl[2])
                    kl[3]=os.path.join(im_root,kl[3])
                ls_dict[i]=kin_list
            new_kin_list = []
            for cr_num in range(1,6):
                for kn in ls_dict:
                    for ls in ls_dict[kn]:
                        if ls[0] == cr_num:
                            new_kin_list.append(ls)
            return new_kin_list

    def _read_mat(self,mat_path):
        mat = scipy.io.loadmat(mat_path)
        conv_type = lambda ls: [int(ls[0][0]), int(ls[1][0]), str(ls[2][0]), str(ls[3][0])]
        pair_list = [conv_type(ls) for ls in mat['pairs']]

        return pair_list

    def get_real_sn(self,lis):

        """
        get real senario list
        :param ls:
        :return:
        """
        # if self.neg_flag == 0:
        #     lb_dict = {'fd':1,'fs':2,'md':3,'ms':4}
        #     neg = 0
        # elif self.neg_flag == -1:
        #     lb_dict = {'fd': 0, 'fs': 1, 'md': 2, 'ms': 3}
        #     neg = -1
        lb_dict = {'fd': 0, 'fs': 1, 'md': 2, 'ms': 3}
        new_kin_ls = []
        ls_pool = []
        for ls in lis:
            if ls[2] not in ls_pool:
                ls_pool.append(ls[2])
            if ls[3] not in ls_pool:
                ls_pool.append(ls[3])
        for lsn1 in ls_pool:
            for lsn2 in ls_pool:
                if lsn1[:-6] == lsn2[:-6]:
                    if lsn1.split('_')[2] == lsn2.split('_')[2]:
                        pass
                    elif lsn1.split('_')[-1] == '2.jpg':
                        pass
                    else:
                        lb = lb_dict[lsn1.split('_')[0].split('/')[-1]]
                        new_kin_ls.append([0,lb,lsn1,lsn2])
                else:
                    new_kin_ls.append([0, self.neg_flag, lsn1, lsn2])

        return new_kin_ls


class fiw_dataset(Kin_dataset):


    def _init_list(self, ls):
        if self.crf:
            self._cross_shuffle(self.neg_ratio)
        if self.get_neg:
            new = [item for item in ls if item[1] == -1]
            return new
        elif self.get_pos:
            new = [item for item in ls if item[1] != -1]
            return new
        else:
            return ls

    def read_mat(self,mat_paths,im_roots):
            self.neg_flag = -1
            ls_dict = {}
            for i,mat_path in enumerate(mat_paths):
                kin_list = self._read_mat(mat_path)
                for kl in kin_list:
                    if not kl[1]==0:
                        kl[1]=kl[1]+i-1
                    else:
                        kl[1] = -1
                    kl[2]=os.path.join(im_roots,kl[2])
                    kl[3]=os.path.join(im_roots,kl[3])
                ls_dict[i]=kin_list
            new_kin_list = []
            for cr_num in range(1,6):
                for kn in ls_dict:
                    for ls in ls_dict[kn]:
                        if ls[0] == cr_num:
                            new_kin_list.append(ls)
            return new_kin_list

    def _read_mat(self,mat_path):
        with open (mat_path, 'rb') as fp:
            nemo_ls = pickle.load(fp)

        nemo_ls = self.get_img_name(nemo_ls)
        return nemo_ls

    def get_img_name(self,ims_ls):
        new_ls = []
        for im in ims_ls:
            im1_pth = os.path.join(self.im_root,im[2])
            im2_pth = os.path.join(self.im_root,im[3])
            if self.read_all_imgs and (not self.test):
                im1ls = sorted(os.listdir(im1_pth))
                im2ls = sorted(os.listdir(im2_pth))
                for im1 in im1ls:
                    for im2 in im2ls:
                        new_ls.append([im[0], im[1], os.path.join(im[2], im1), os.path.join(im[3], im2)])

            else:
                # im1_nm = sorted(os.listdir(im1_pth))[0]
                # im2_nm = sorted(os.listdir(im2_pth))[0]
                # new_ls.append([im[0],im[1],os.path.join(im[2],im1_nm),os.path.join(im[3],im2_nm)])
                im1_nm = sorted(os.listdir(im1_pth))
                im2_nm = sorted(os.listdir(im2_pth))
                # lenth = zip(im1_nm,im2_nm)
                for i, (im1, im2) in enumerate(zip(im1_nm, im2_nm)):
                    new_ls.append([im[0], im[1], os.path.join(im[2], im1), os.path.join(im[3], im2)])


        return new_ls

    def get_real_sn(self,lis):

        """
        get real senario list
        :param ls:
        :return:
        """

        new_kin_ls = []
        ls_pool = []
        pos_ls_pool = []
        for ls in lis:
            if ls[2] not in ls_pool:
                ls_pool.append(ls[2])

            if ls[3] not in ls_pool:
                ls_pool.append(ls[3])

            if ls[1]!=-1:
                pos_ls_pool.append(ls[2])

        pos_lth = len(pos_ls_pool)

        while len(new_kin_ls) < 10*pos_lth:
            for lsn1 in pos_ls_pool:
                for lsn2 in ls_pool:
                    if lsn1.split('/')[-3] == lsn2.split('/')[-3]:
                        continue
                    elif [0, self.neg_flag, lsn1, lsn2] in new_kin_ls:
                        continue
                    else:
                        new_kin_ls.append([0, self.neg_flag, lsn1, lsn2])

                    break
        #
        # for lsn1 in ls_pool:
        #     for lsn2 in ls_pool:
        #         if lsn1.split('/')[-3] == lsn2.split('/')[-3]:
        #             pass
        #         else:
        #             new_kin_ls.append([0, self.neg_flag, lsn1, lsn2])

        for ls in lis:
            if ls[1] != -1:
                new_kin_ls.append(ls)

        return new_kin_ls

    def _cross_shuffle(self,neg_ratio):
        """
        shuffle the second images name after each epoch
        :return:
        """
        if self.sf_aln:
            im2_ls = self.alln_list
        else:
            im2_ls = self.img2_list
        rand_lth = len(im2_ls)
        new_pair_list = []

        for pair_l in self.kin_list:
            if pair_l[1] == -1:
                for neg_iter in range(neg_ratio):
                    new_img2 = im2_ls[random.randint(0, rand_lth-1)]
                    while pair_l[2].split('/')[-3] == new_img2.split('/')[-3]:
                        new_img2 = im2_ls[random.randint(0, rand_lth-1)]
                    pair_l[3] = new_img2
                    new_pair_list.append(pair_l)
            else:
                new_pair_list.append(pair_l)

        self.kin_list = new_pair_list



class fiw_dataset_new(Kin_dataset):


    def _init_list(self, ls):
        if self.crf:
            self._cross_shuffle(self.neg_ratio)
        if self.get_neg:
            new = [item for item in ls if item[1] == -1]
            return new
        elif self.get_pos:
            new = [item for item in ls if item[1] != -1]
            return new
        else:
            return ls

    def read_mat(self,mat_paths,im_roots):
            self.neg_flag = -1
            ls_dict = {}
            for i,mat_path in enumerate(mat_paths):
                if i in [7,8,9,10]:
                    kin_list = self._read_mat(mat_path,digdata=True)
                else:
                    kin_list = self._read_mat(mat_path)
                for kl in kin_list:
                    if not kl[1]==0:
                        kl[1]=kl[1]+i-1


                    else:
                        kl[1] = -1
                    kl[2]=os.path.join(im_roots,kl[2])
                    kl[3]=os.path.join(im_roots,kl[3])
                ls_dict[i]=kin_list
            new_kin_list = []

            #### allocating cross valids: (try to utilize all the "U_k" samples)
            # for i in [7,8,9,10]:
            #     temp = ls_dict[i]
            #     temp = [[5, item[1],item[2],item[3]] for item in temp ]
            #     ls_dict[i]=temp
            ###

            for cr_num in range(1,6):
                for kn in ls_dict:
                    for ls in ls_dict[kn]:
                        if ls[0] == cr_num:
                            new_kin_list.append(ls)
            return new_kin_list

    def _read_mat(self,mat_path,digdata = False):
        with open (mat_path, 'rb') as fp:
            nemo_ls = pickle.load(fp)

        nemo_ls = self.get_img_name(nemo_ls,digdata)
        return nemo_ls

    def get_img_name(self,ims_ls,digdata):
        new_ls = []
        for im in ims_ls:
            im1_pth = os.path.join(self.im_root,im[2])
            im2_pth = os.path.join(self.im_root,im[3])
            if self.read_all_imgs and (not self.test):
                im1ls = sorted(os.listdir(im1_pth))
                im2ls = sorted(os.listdir(im2_pth))
                for im1 in im1ls:
                    for im2 in im2ls:
                        new_ls.append([im[0], im[1], os.path.join(im[2], im1), os.path.join(im[3], im2)])

            else:

                if digdata:
                    im1ls = sorted(os.listdir(im1_pth))
                    im2ls = sorted(os.listdir(im2_pth))
                    for im1 in im1ls:
                        for im2 in im2ls:
                            new_ls.append([im[0], im[1], os.path.join(im[2], im1), os.path.join(im[3], im2)])
                    # im1_nm = sorted(os.listdir(im1_pth))
                    # im2_nm = sorted(os.listdir(im2_pth))
                    # # lenth = zip(im1_nm,im2_nm)
                    # for i, (im1, im2) in enumerate(zip(im1_nm, im2_nm)):
                    #     new_ls.append([im[0], im[1], os.path.join(im[2], im1), os.path.join(im[3], im2)])

                else:
                    im1_nm = sorted(os.listdir(im1_pth))[0]
                    im2_nm = sorted(os.listdir(im2_pth))[0]
                    new_ls.append([im[0],im[1],os.path.join(im[2],im1_nm),os.path.join(im[3],im2_nm)])
                    # im1_nm = sorted(os.listdir(im1_pth))
                    # im2_nm = sorted(os.listdir(im2_pth))
                    # # lenth = zip(im1_nm,im2_nm)
                    # for i, (im1, im2) in enumerate(zip(im1_nm, im2_nm)):
                    #     new_ls.append([im[0], im[1], os.path.join(im[2], im1), os.path.join(im[3], im2)])


        return new_ls

    def get_real_sn(self,lis):

        """
        get real senario list
        :param ls:
        :return:
        """

        new_kin_ls = []
        ls_pool = []
        pos_ls_pool = []
        for ls in lis:
            if ls[2] not in ls_pool:
                ls_pool.append(ls[2])

            if ls[3] not in ls_pool:
                ls_pool.append(ls[3])

            if ls[1]!=-1:
                pos_ls_pool.append(ls[2])

        pos_lth = len(pos_ls_pool)

        while len(new_kin_ls) < 10*pos_lth:
            for lsn1 in pos_ls_pool:
                for lsn2 in ls_pool:
                    if lsn1.split('/')[-3] == lsn2.split('/')[-3]:
                        continue
                    elif [0, self.neg_flag, lsn1, lsn2] in new_kin_ls:
                        continue
                    else:
                        new_kin_ls.append([0, self.neg_flag, lsn1, lsn2])

                    break
        #
        # for lsn1 in ls_pool:
        #     for lsn2 in ls_pool:
        #         if lsn1.split('/')[-3] == lsn2.split('/')[-3]:
        #             pass
        #         else:
        #             new_kin_ls.append([0, self.neg_flag, lsn1, lsn2])

        for ls in lis:
            if ls[1] != -1:
                new_kin_ls.append(ls)

        return new_kin_ls

    def _cross_shuffle(self,neg_ratio):
        """
        shuffle the second images name after each epoch
        :return:
        """
        if self.sf_aln:
            im2_ls = self.alln_list
        else:
            im2_ls = self.img2_list
        rand_lth = len(im2_ls)
        new_pair_list = []

        for pair_l in self.kin_list:
            if pair_l[1] == -1:
                for neg_iter in range(neg_ratio):
                    new_img2 = im2_ls[random.randint(0, rand_lth-1)]
                    while pair_l[2].split('/')[-3] == new_img2.split('/')[-3]:
                        new_img2 = im2_ls[random.randint(0, rand_lth-1)]
                    pair_l[3] = new_img2
                    new_pair_list.append(pair_l)
            else:
                new_pair_list.append(pair_l)

        self.kin_list = new_pair_list

class fiw_dataset_img_new(Kin_dataset):


    def _init_list(self, ls):
        if self.crf:
            self._cross_shuffle(self.neg_ratio)
        if self.get_neg:
            new = [item for item in ls if item[1] == -1]
            return new
        elif self.get_pos:
            new = [item for item in ls if item[1] != -1]
            return new
        else:
            return ls

    def read_mat(self,mat_paths,im_roots):
            self.neg_flag = -1
            ls_dict = {}
            for i,mat_path in enumerate(mat_paths):
                if i in [7,8,9,10]:
                    kin_list = self._read_mat(mat_path,digdata=True)
                else:
                    kin_list = self._read_mat(mat_path)
                for kl in kin_list:
                    if not kl[1]==0:
                        kl[1]=kl[1]+i-1


                    else:
                        kl[1] = -1
                    kl[2]=os.path.join(im_roots,kl[2])
                    kl[3]=os.path.join(im_roots,kl[3])
                ls_dict[i]=kin_list
            new_kin_list = []

            #### allocating cross valids: (try to utilize all the "U_k" samples)
            # for i in [7,8,9,10]:
            #     temp = ls_dict[i]
            #     temp = [[5, item[1],item[2],item[3]] for item in temp ]
            #     ls_dict[i]=temp
            ###

            for cr_num in range(1,6):
                for kn in ls_dict:
                    for ls in ls_dict[kn]:
                        if ls[0] == cr_num:
                            new_kin_list.append(ls)
            return new_kin_list

    def _read_mat(self,mat_path,digdata = False):
        with open (mat_path, 'rb') as fp:
            nemo_ls = pickle.load(fp)


        ############### test check 2022.3.1
        # total_ls = []
        # for mm in range(5):
        #
        #     temp = [item for item in nemo_ls if item[0] == mm + 1]
        #     pos_temp = [item for item in temp if item[1] != 0]
        #
        #     item_1 = []
        #     item_2 = []
        #     for itm_pos in pos_temp:
        #         item_1.append(itm_pos[2])
        #         item_2.append(itm_pos[3])
        #
        #     while self._check(item_1, item_2):
        #         shuffle(item_2)
        #
        #     neg_temp = [[mm + 1, 0, it1, it2] for it1, it2 in zip(item_1, item_2)]
        #     total_ls +=pos_temp
        #     total_ls +=neg_temp
        ##############


        nemo_ls = self.get_img_name(nemo_ls,digdata)
        return nemo_ls

    def _check(self,a, b):
        """
        if there is more than one pair matched, return True
        :param a:
        :param b:
        :return:
        """
        for i, j in zip(a, b):
            if i.split('/')[0] == j.split('/')[0]:
                return True
        return False
    def __getitem__(self, item):

        if torch.is_tensor(item):
            item = item.tolist()
        # extract img1
        img1_p = self.kin_list[item][2]
        img1 = Image.open(img1_p)
        # extract img2
        img2_p = self.kin_list[item][3]
        img2 = Image.open(img2_p)
        # get kin label 0/1
        kin = self.kin_list[item][1]
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        imgs = torch.cat((img1,img2))
        # after each epoch, shuffle once
        if self.crf:
            self.cout +=1
            if self.cout == self.lth:
                self.cout = 0
                self._cross_shuffle(1)


        return imgs,kin,img1_p, img2_p

    def get_img_name(self,ims_ls,digdata):
        new_ls = []
        for im in ims_ls:
            im1_pth = os.path.join(self.im_root,im[2])
            im2_pth = os.path.join(self.im_root,im[3])
            if self.read_all_imgs and (not self.test):
                im1ls = sorted(os.listdir(im1_pth))
                im2ls = sorted(os.listdir(im2_pth))
                for im1 in im1ls:
                    for im2 in im2ls:
                        new_ls.append([im[0], im[1], os.path.join(im[2], im1), os.path.join(im[3], im2)])

            else:

                if digdata:
                    im1ls = sorted(os.listdir(im1_pth))
                    im2ls = sorted(os.listdir(im2_pth))
                    for im1 in im1ls:
                        for im2 in im2ls:
                            new_ls.append([im[0], im[1], os.path.join(im[2], im1), os.path.join(im[3], im2)])
                    # im1_nm = sorted(os.listdir(im1_pth))
                    # im2_nm = sorted(os.listdir(im2_pth))
                    # # lenth = zip(im1_nm,im2_nm)
                    # for i, (im1, im2) in enumerate(zip(im1_nm, im2_nm)):
                    #     new_ls.append([im[0], im[1], os.path.join(im[2], im1), os.path.join(im[3], im2)])

                else:
                    im1_nm = sorted(os.listdir(im1_pth))[0]
                    im2_nm = sorted(os.listdir(im2_pth))[0]
                    new_ls.append([im[0],im[1],os.path.join(im[2],im1_nm),os.path.join(im[3],im2_nm)])
                    # im1_nm = sorted(os.listdir(im1_pth))
                    # im2_nm = sorted(os.listdir(im2_pth))
                    # # lenth = zip(im1_nm,im2_nm)
                    # for i, (im1, im2) in enumerate(zip(im1_nm, im2_nm)):
                    #     new_ls.append([im[0], im[1], os.path.join(im[2], im1), os.path.join(im[3], im2)])


        return new_ls

    def get_real_sn(self,lis):

        """
        get real senario list
        :param ls:
        :return:
        """

        new_kin_ls = []
        ls_pool = []
        pos_ls_pool = []
        for ls in lis:
            if ls[2] not in ls_pool:
                ls_pool.append(ls[2])

            if ls[3] not in ls_pool:
                ls_pool.append(ls[3])

            if ls[1]!=-1:
                pos_ls_pool.append(ls[2])

        pos_lth = len(pos_ls_pool)

        while len(new_kin_ls) < 10*pos_lth:
            for lsn1 in pos_ls_pool:
                for lsn2 in ls_pool:
                    if lsn1.split('/')[-3] == lsn2.split('/')[-3]:
                        continue
                    elif [0, self.neg_flag, lsn1, lsn2] in new_kin_ls:
                        continue
                    else:
                        new_kin_ls.append([0, self.neg_flag, lsn1, lsn2])

                    break
        #
        # for lsn1 in ls_pool:
        #     for lsn2 in ls_pool:
        #         if lsn1.split('/')[-3] == lsn2.split('/')[-3]:
        #             pass
        #         else:
        #             new_kin_ls.append([0, self.neg_flag, lsn1, lsn2])

        for ls in lis:
            if ls[1] != -1:
                new_kin_ls.append(ls)

        return new_kin_ls

    def _cross_shuffle(self,neg_ratio):
        """
        shuffle the second images name after each epoch
        :return:
        """
        if self.sf_aln:
            im2_ls = self.alln_list
        else:
            im2_ls = self.img2_list
        rand_lth = len(im2_ls)
        new_pair_list = []

        for pair_l in self.kin_list:
            if pair_l[1] == -1:
                for neg_iter in range(neg_ratio):
                    new_img2 = im2_ls[random.randint(0, rand_lth-1)]
                    while pair_l[2].split('/')[-3] == new_img2.split('/')[-3]:
                        new_img2 = im2_ls[random.randint(0, rand_lth-1)]
                    pair_l[3] = new_img2
                    new_pair_list.append(pair_l)
            else:
                new_pair_list.append(pair_l)

        self.kin_list = new_pair_list


class fiw_dataset_self(fiw_dataset):
    def read_mat(self, mat_paths, im_roots):
        self.neg_flag = -1
        ls_dict = {}
        for i, mat_path in enumerate(mat_paths):
            kin_list = self._read_mat(mat_path)
            for kl in kin_list:
                if kl[1] ==-2:
                    kl[1]=-2
                elif not kl[1] == 0:
                    kl[1] = kl[1] + i - 1
                else:
                    kl[1] = -1

                kl[2] = os.path.join(im_roots, kl[2])
                kl[3] = os.path.join(im_roots, kl[3])
            ls_dict[i] = kin_list
        new_kin_list = []
        for cr_num in range(1, 6):
            for kn in ls_dict:
                for ls in ls_dict[kn]:
                    if ls[0] == cr_num:
                        new_kin_list.append(ls)
        return new_kin_list

    def _read_mat(self, mat_path):
        with open(mat_path, 'rb') as fp:
            nemo_ls = pickle.load(fp)

        nemo_ls = self.get_img_name(nemo_ls)
        return nemo_ls

    def get_img_name(self, ims_ls):
        new_ls = []
        for im in ims_ls:
            im1_pth = os.path.join(self.im_root, im[2])
            im2_pth = os.path.join(self.im_root, im[3])
            if self.read_all_imgs and (not self.test):
                im1ls = sorted(os.listdir(im1_pth))
                im2ls = sorted(os.listdir(im2_pth))
                for im1 in im1ls:
                    for im2 in im2ls:
                        new_ls.append([im[0], im[1], os.path.join(im[2], im1), os.path.join(im[3], im2)])

            else:
                # im1_nm = sorted(os.listdir(im1_pth))[0]
                # im2_nm = sorted(os.listdir(im2_pth))[0]
                # new_ls.append([im[0],im[1],os.path.join(im[2],im1_nm),os.path.join(im[3],im2_nm)])
                im1_nm = sorted(os.listdir(im1_pth))
                im2_nm = sorted(os.listdir(im2_pth))
                # lenth = zip(im1_nm,im2_nm)
                if im[1] == 0:
                    for i, (im1, im2) in enumerate(zip(im1_nm, im2_nm)):
                        new_ls.append([im[0], im[1], os.path.join(im[2], im1), os.path.join(im[3], im2)])
                else:
                    for i, (im1, im2) in enumerate(zip(im1_nm, im2_nm)):
                        new_ls.append([im[0], im[1], os.path.join(im[2], im1), os.path.join(im[3], im2)])
                        new_ls.append([im[0], -2, os.path.join(im[2], im1),
                                   os.path.join(im1_pth, im1_nm[int((i + 1) % len(im1_nm))])])


                # ### self pair
                #
                # new_ls.append([im[0], -2, os.path.join(im[2], random.sample(im1_nm,1)[0]),
                #                os.path.join(im[2], random.sample(im1_nm,1)[0])])
                # new_ls.append([im[0], -2, os.path.join(im[3], random.sample(im2_nm, 1)[0]),
                #                os.path.join(im[3], random.sample(im2_nm, 1)[0])])


        return new_ls



class fiw_triplet_dataset(Dataset):

    def __init__(self, mat_path, im_root, cross_vali=None, transform=None,
                 cross_shuffle=False, sf_aln=False,
                 test=False, test_each=False, real_sn=False, read_all_imgs=False,
                 get_neg=False, get_pos=False, neg_ratio=1):
        """
        THE dataset of KinfaceW-I/ KinfaceW-II
        :param mat_path:
        :param im_root:
        :param cross_vali:    cross validation's folds: e.g. [1,2,4,5]
        :param transform:     add data augmentation
        :param cross_shuffle: shuffle names among pair list
        :param sf_aln:        whether shuffle all names or only img2s' names
        out: imgs(1x6x64x64),kin(0/1),img1_n(imag1_name),img2_n(image2_name)
        """
        self.neg_ratio = neg_ratio
        self.get_neg = get_neg
        self.get_pos = get_pos
        self.read_all_imgs = read_all_imgs
        self.neg_flag = 0
        self.test = test
        self.im_root = im_root
        # kin_list is the whole 1,2,3,4,5 folds from mat
        if not test_each:
            self.kin_list = self.read_mat(mat_path, im_root)
        else:
            self.kin_list = self.read_test_mat(mat_path, im_root)

        self.transform = transform
        self.cout = 0
        if cross_vali is not None:
            # extract matched folds e.g. [1,2,4,5]
            self.kin_list = self._get_cross(cross_vali)
        # if cross_shuffle:
        self.crf = cross_shuffle  # cross shuffle

        # store all img2/ all_(img1+img2) list for cross shuffle,
        self.img2_list = list(set([i[3] for i in self.kin_list] + [i[4] for i in self.kin_list]))
        self.alln_list = list(set(self.get_alln(self.kin_list)))
        self.sf_aln = sf_aln
        # self.crs_insect = 0
        # self.kin_list_bak= copy.deepcopy(self.kin_list)
        if real_sn:
            self.kin_list = self.get_real_sn(self.kin_list)

        self.kin_list = self._init_list(self.kin_list)

        self.lth = len(self.kin_list)

    def get_alln(self, ls):
        all_name = []
        for i in ls:
            all_name.append(i[2])
            all_name.append(i[3])
            all_name.append(i[4])

        return all_name

    def __len__(self):
        return len(self.kin_list)

    def __getitem__(self, item):

        if torch.is_tensor(item):
            item = item.tolist()
        # extract img1
        img1_p = self.kin_list[item][2]
        img1 = Image.open(img1_p)

        # extract img2
        img2_p = self.kin_list[item][3]
        img2 = Image.open(img2_p)

        img3_neg = self.kin_list[item][4]
        img3 = Image.open(img3_neg)
        # get kin label 0/1
        kin = self.kin_list[item][1]
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)

        # imgs = torch.cat((img1,img2))
        # after each epoch, shuffle once
        if self.crf:
            self.cout += 1
            if self.cout == self.lth:
                self.cout = 0
                self._cross_shuffle(1)

        return (img1, img2, img3), kin

    def read_test_mat(self, mat_path, im_root):
        new_kin_list = []
        return new_kin_list

    def _get_cross(self, cross):

        return [i for i in self.kin_list if i[0] in cross]

    def _init_list(self, ls):
        if self.crf:
            self._cross_shuffle(self.neg_ratio)
        if self.get_neg:
            new = [item for item in ls if item[1] == -1]
            return new
        elif self.get_pos:
            new = [item for item in ls if item[1] != -1]
            return new
        else:
            return ls

    def read_mat(self, mat_paths, im_roots):
        self.neg_flag = -1
        ls_dict = {}
        for i, mat_path in enumerate(mat_paths):
            kin_list = self._read_mat(mat_path)
            temp_all_ls = []
            for mm in range(5):
                new_small_ls = []
                # cs_len = int(len(kin_list) / 5)
                # temp = kin_list[mm * cs_len:(mm + 1) * cs_len]

                temp = [item for item in kin_list if item[0] == mm + 1]

                pos_temp = temp[0:int(len(temp) / 2)]
                neg_temp = temp[int(len(temp) / 2):len(temp)]
                # assert  [item[2] for item in pos_temp]==[item[2] for item in neg_temp]
                assert len(pos_temp) == len(neg_temp)

                neg_imgsls = [item[3] for item in neg_temp]
                for inum, smls in enumerate(pos_temp):
                    if smls[2].split('/')[0] != neg_imgsls[inum].split('/')[0]:
                        new_small_ls.append([smls[0], smls[1] + i - 1,
                                             smls[2],
                                             smls[3],
                                             neg_imgsls[inum]])
                    else:
                        select_neg = neg_imgsls[inum]
                        while smls[2].split('/')[0] != select_neg.split('/')[0]:
                            select_neg = neg_imgsls[random.randint(0, len(neg_imgsls) - 1)]
                        new_small_ls.append([smls[0], smls[1] + i - 1,
                                             smls[2],
                                             smls[3],
                                             select_neg])
                temp_all_ls += new_small_ls

            ls_dict[i] = self.get_img_name(temp_all_ls)
        new_kin_list = []
        for cr_num in range(1, 6):
            for kn in ls_dict:
                for ls in ls_dict[kn]:
                    if ls[0] == cr_num:
                        new_kin_list.append(ls)
        return new_kin_list

    def _read_mat(self, mat_path):
        with open(mat_path, 'rb') as fp:
            nemo_ls = pickle.load(fp)
        # nemo_ls = self.get_img_name(nemo_ls)
        return nemo_ls

    def get_img_name(self, ims_ls):
        new_ls = []
        for im in ims_ls:
            im1_pth = os.path.join(self.im_root, im[2])  # anchor
            im2_pth = os.path.join(self.im_root, im[3])  # pos
            im3_pth = os.path.join(self.im_root, im[4])  # neg

            # less
            #
            # im1_nm = sorted(os.listdir(im1_pth))[0]
            # im2_nm = sorted(os.listdir(im2_pth))[0]
            # im3_nm = sorted(os.listdir(im3_pth))[0]
            # new_ls.append([im[0], im[1], os.path.join(im1_pth, im1_nm), os.path.join(im2_pth, im2_nm),
            #                os.path.join(im3_pth, im3_nm)])
            #
            # more
            im1_nm = sorted(os.listdir(im1_pth))
            im2_nm = sorted(os.listdir(im2_pth))
            im3_nm = sorted(os.listdir(im3_pth))
            # lenth = zip(im1_nm,im2_nm)
            for i, (im1, im2) in enumerate(zip(im1_nm, im2_nm)):
                anchor = os.path.join(im1_pth, im1)
                pos = os.path.join(im2_pth, im2)

                neg = os.path.join(im3_pth, im3_nm[int(i % len(im3_nm))])

                new_ls.append([im[0], im[1], anchor, pos, neg])

        return new_ls

    def get_real_sn(self, lis):

        """
        get real senario list
        :param ls:
        :return:
        """

        new_kin_ls = []
        ls_pool = []
        pos_ls_pool = []
        for ls in lis:
            if ls[2] not in ls_pool:
                ls_pool.append(ls[2])

            if ls[3] not in ls_pool:
                ls_pool.append(ls[3])

            if ls[1] != -1:
                pos_ls_pool.append(ls[2])

        pos_lth = len(pos_ls_pool)

        while len(new_kin_ls) < 10 * pos_lth:
            for lsn1 in pos_ls_pool:
                for lsn2 in ls_pool:
                    if lsn1.split('/')[-3] == lsn2.split('/')[-3]:
                        continue
                    elif [0, self.neg_flag, lsn1, lsn2] in new_kin_ls:
                        continue
                    else:
                        new_kin_ls.append([0, self.neg_flag, lsn1, lsn2])

                    break

        for ls in lis:
            if ls[1] != -1:
                new_kin_ls.append(ls)

        return new_kin_ls

    def _cross_shuffle(self, neg_ratio):
        """
        shuffle the second images name after each epoch
        :return:
        """
        if self.sf_aln:
            im2_ls = self.alln_list
        else:
            im2_ls = self.img2_list
        rand_lth = len(im2_ls)
        new_pair_list = []

        for pair_l in self.kin_list:

            # for neg_iter in range(neg_ratio):
            new_img2 = im2_ls[random.randint(0, rand_lth - 1)]
            while pair_l[2].split('/')[-3] == new_img2.split('/')[-3]:
                new_img2 = im2_ls[random.randint(0, rand_lth - 1)]
            pair_l[4] = new_img2


            new_pair_list.append(pair_l)

        self.kin_list = new_pair_list



class fiw_triplet_HK_dataset(fiw_triplet_dataset):



    def __getitem__(self, item):

        if torch.is_tensor(item):
            item = item.tolist()
        # extract img1
        img1_p = self.kin_list[item][2]
        img1 = Image.open(img1_p)

        # extract img2
        img2_p = self.kin_list[item][3]
        img2 = Image.open(img2_p)

        img3_neg = self.kin_list[item][4]
        img3 = Image.open(img3_neg)

        img4_self = self.kin_list[item][5]
        img4 = Image.open(img4_self)


        kin = self.kin_list[item][1]
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
            img4 = self.transform(img4)

        # imgs = torch.cat((img1,img2))
        # after each epoch, shuffle once
        if self.crf:
            self.cout += 1
            if self.cout == self.lth:
                self.cout = 0
                self._cross_shuffle(1)

        return (img1,img4,img2, img3), kin

    def read_test_mat(self, mat_path, im_root):
        new_kin_list = []
        return new_kin_list

    def _get_cross(self, cross):

        return [i for i in self.kin_list if i[0] in cross]

    def _init_list(self, ls):
        if self.crf:
            self._cross_shuffle(self.neg_ratio)
        if self.get_neg:
            new = [item for item in ls if item[1] == -1]
            return new
        elif self.get_pos:
            new = [item for item in ls if item[1] != -1]
            return new
        else:
            return ls

    def _check(self,a, b):
        """
        if there is more than one pair matched, return True
        :param a:
        :param b:
        :return:
        """
        for i, j in zip(a, b):
            if i.split('/')[0] == j.split('/')[0]:
                return True
        return False

    def read_mat(self, mat_paths, im_roots):
        self.label2img_dict = {}
        self.neg_flag = -1
        ls_dict = {}
        for i, mat_path in enumerate(mat_paths):
            kin_list = self._read_mat(mat_path)
            temp_all_ls = []
            for mm in range(5):
                new_small_ls = []


                temp = [item for item in kin_list if item[0]==mm+1]


                pos_temp = [item for item in temp if item[1]!=0]

                neg_imgsls = [item[3] for item in pos_temp]+[item[2] for item in pos_temp]
                for inum, smls in enumerate(pos_temp):
                    if smls[2].split('/')[0] != neg_imgsls[inum].split('/')[0]:
                        new_small_ls.append([smls[0], smls[1] + i - 1,
                                             smls[2],
                                             smls[3],
                                             neg_imgsls[inum]])
                    else:
                        select_neg = neg_imgsls[inum]
                        while smls[2].split('/')[0] != select_neg.split('/')[0]:
                            select_neg = neg_imgsls[random.randint(0, len(neg_imgsls) - 1)]
                        new_small_ls.append([smls[0], smls[1] + i - 1,
                                             smls[2],
                                             smls[3],
                                             select_neg])
                temp_all_ls += new_small_ls

            ls_dict[i] = self.get_img_name(temp_all_ls)
        new_kin_list = []
        for cr_num in range(1, 6):
            for kn in ls_dict:
                for ls in ls_dict[kn]:
                    if ls[0] == cr_num:
                        new_kin_list.append(ls)
        return new_kin_list

    def _read_mat(self, mat_path):
        with open(mat_path, 'rb') as fp:
            nemo_ls = pickle.load(fp)
        # nemo_ls = self.get_img_name(nemo_ls)
        return nemo_ls

    def get_img_name(self, ims_ls):
        new_ls = []

        for im in ims_ls:
            im1_pth = os.path.join(self.im_root, im[2])#anchor
            im2_pth = os.path.join(self.im_root, im[3])#pos
            im3_pth = os.path.join(self.im_root, im[4])#neg


            im1_nm = sorted(os.listdir(im1_pth))
            im2_nm = sorted(os.listdir(im2_pth))
            im3_nm = sorted(os.listdir(im3_pth))

            ##### 2022.2.28
            self.label2img_dict[im[2]] = im1_nm
            self.label2img_dict[im[3]] = im2_nm
            self.label2img_dict[im[4]] = im3_nm


            # lenth = zip(im1_nm,im2_nm)
            for i, (im1, im2) in enumerate(zip(im1_nm, im2_nm)):
                anchor = os.path.join(im1_pth, im1)
                self_im = os.path.join(im1_pth, im1_nm[int((i+1)%len(im1_nm))])

                pos = os.path.join(im2_pth, im2)

                neg = os.path.join(im3_pth, im3_nm[int(i%len(im3_nm))])

                new_ls.append([im[0], im[1], anchor,pos,neg,self_im])

        return new_ls

    def _cross_shuffle(self, neg_ratio):
        """
        shuffle the second images name after each epoch
        :return:
        """
        if self.sf_aln:
            im2_ls = self.alln_list
        else:
            im2_ls = self.img2_list
        rand_lth = len(im2_ls)
        new_pair_list = []

        for pair_l in self.kin_list:

            # for neg_iter in range(neg_ratio):
            new_img2 = im2_ls[random.randint(0, rand_lth - 1)]
            while pair_l[2].split('/')[-3] == new_img2.split('/')[-3]:
                new_img2 = im2_ls[random.randint(0, rand_lth - 1)]
            pair_l[4] = new_img2  ############## checked fault
            ########### 2022.2.28
            # self.label2img_dict[]
            ### pos pair_l[3]
            pos_tag = pair_l[3].split('/')[-3]+'/'+pair_l[3].split('/')[-2]

            pair_l[3]=os.path.join(self.im_root,pos_tag,random.choice(self.label2img_dict[pos_tag]))
            ### self pair_l[5]
            self_tag = pair_l[2].split('/')[-3] + '/' + pair_l[2].split('/')[-2]
            pair_l[5] = os.path.join(self.im_root,self_tag,random.choice(self.label2img_dict[self_tag]))

            new_pair_list.append(pair_l)

        self.kin_list = new_pair_list



class fiw_dataset_test(fiw_dataset):
    def read_mat(self,mat_paths,im_roots):
            self.neg_flag = -1
            ls_dict = {}
            for i,mat_path in enumerate(mat_paths):
                kin_list = self._read_mat(mat_path)
                for kl in kin_list:
                    if i <= 3:
                        if not kl[1]==0:
                            kl[1]=kl[1]+i-1
                        else:
                            kl[1] = -1
                    else:
                        kl[1]=-1
                    kl[2]=os.path.join(im_roots,kl[2])
                    kl[3]=os.path.join(im_roots,kl[3])
                ls_dict[i]=kin_list
            new_kin_list = []
            for cr_num in range(1,6):
                for kn in ls_dict:
                    for ls in ls_dict[kn]:
                        if ls[0] == cr_num:
                            new_kin_list.append(ls)
            return new_kin_list