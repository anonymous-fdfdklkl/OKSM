"""
    Network definition for our proposed CAC open set classifier.

    Dimity Miller, 2020
"""


import torch
import torchvision
import torch.nn as nn
import  torch.nn.functional as F
from math import sqrt
import numpy as np
from networks.transformers import KAT,KAT_tanh,Transformer,Transformer_tanh,KAT_single,KAT_single_2,KAT_small,KAT_smalldeep,KAT_final,KAT_single_final
from facenet_pytorch import InceptionResnetV1

class cnn(nn.Module):
    def __init__(self, num_classes=4, num_channels=3, im_size=64, init_weights=False, dropout=0.3, **kwargs):
        super(cnn, self).__init__()

        self.num_classes = num_classes
        self.encoder = attenNet_basic()
        # self.encoder = CNN()
        if im_size == 32:
            self.classify = nn.Linear(512, num_classes)
        elif im_size == 64:
            self.classify = nn.Linear(512, num_classes)
        else:
            print('That image size has not been implemented, sorry.')
            exit()

        if init_weights:
            self._initialize_weights()

        self.cuda()

    def forward(self, x):
        batch_size = len(x)

        x = self.encoder(x)
        x = x.view(batch_size, -1)

        outLinear = self.classify(x)

        return outLinear

    def similarity(self,x,close = True):
        logits = self.forward(x)
        if close:
            scores = F.softmax(logits,dim=1)
            pred = -torch.max(scores,dim=1)[0]
        else:
            pred = F.softmax(logits,dim=1)[:,0]
        return pred


    def feature(self, x ):
        batch_size = len(x)
        x = self.encoder(x)
        x = x.view(batch_size, -1)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class openset(nn.Module):
    def __init__(self, num_classes=7, num_channels=3, im_size=64, init_weights=False, dropout=0.3, **kwargs):
        super(openset, self).__init__()

        self.num_classes = num_classes
        self.encoder = attenNet_basic()
        # self.encoder = CNN()
        if im_size == 32:
            self.classify = nn.Linear(512, num_classes)
        elif im_size == 64:
            self.classify = nn.Linear(512, num_classes)
        else:
            print('That image size has not been implemented, sorry.')
            exit()

        self.anchors = nn.Parameter(torch.zeros(self.num_classes, self.num_classes).double(), requires_grad=False)

        if init_weights:
            self._initialize_weights()


        self._init_dis()
        self.cuda()

    def _init_dis(self):
        pass

    def forward(self, x, skip_distance=False):
        batch_size = len(x)

        x = self.encoder(x)
        x = x.view(batch_size, -1)

        outLinear = self.classify(x)

        if skip_distance:
            return outLinear, None

        outDistance = self.distance_classifier(outLinear)

        return outLinear, outDistance

    def similarity(self,x):
        outputs = self.forward(x)
        # logits = outputs[0]
        distances = outputs[1]
        softmin = F.softmax(-distances,dim=1)
        invScores = 1 - softmin
        scores = distances * invScores
        predicted = torch.min(scores, axis=1)[0]
        return predicted

    def feature(self, x ):
        batch_size = len(x)
        x = self.encoder(x)
        x = x.view(batch_size, -1)
        # x = self.classify(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def set_anchors(self, means):
        self.anchors = nn.Parameter(means.double(), requires_grad=False)
        self.cuda()

    def distance_classifier(self, x):
        ''' Calculates euclidean distance from x to each class anchor
            Returns n x m array of distance from input of batch_size n to anchors of size m
        '''

        n = x.size(0)
        m = self.num_classes
        d = self.num_classes

        x = x.unsqueeze(1).expand(n, m, d).double()
        anchors = self.anchors.unsqueeze(0).expand(n, m, d)
        dists = torch.norm(x - anchors, 2, 2)

        return dists



class HK_trip_tanhNet_c(nn.Module):
    """
     Hierarchical Kinship Triplet network
    """
    def __init__(self, num_classes=4, num_channels=3, im_size=64, init_weights=False, dropout=0.3, **kwargs):
        super(HK_trip_tanhNet_c, self).__init__()

        self.num_classes = num_classes

        self.encoder = attenNet_basic(input_channel=3)

        # model = SoftIntroVAE(cdim=3, zdim=128, channels=[64, 128, 256, 512], image_size=64)
        # load_model(model, '/home/wei/Documents/CODE/CVPR2021/openset-kinship/datasets/fiw_soft_intro_betas_0.1_1.0_1.0_model_epoch_100_iter_43400.pth', 'cuda')
        # self.encoder = model.encoder
        # for param in self.encoder.parameters():
        #     param.requires_grad = False

        self.fea = 64
        # self.fc = Self_Attention(128,self.fea,self.fea)
        self.fc = nn.Sequential(nn.LayerNorm(512),
                                nn.Linear(512,self.fea),
                                nn.Tanh()
                                )

        # self.classify = nn.Linear(self.fea, num_classes)
        # self.fc_center = nn.Linear(self.fea, num_classes)
        # self.dis_center = [torch.zeros([1]).cuda(),
        #                    nn.Parameter(torch.randn(1), requires_grad=True).cuda(),
        #                    nn.Parameter(torch.randn(1), requires_grad=True).cuda()]
        self.dis_center = [torch.zeros([1]).cuda(),
                           torch.ones([1]).cuda(),
                           2*torch.ones([1]).cuda()]
        # self.centers = nn.Parameter(0.1 * torch.randn(num_classes, self.fea),requires_grad=True)
        if init_weights:
            self._initialize_weights()

        self._init_dis()
        self.cuda()

    def _init_dis(self):
        # self.neg_axis = nn.Parameter(-torch.ones(1, self.anchors.shape[0]).double(), requires_grad=False)
        # self.cos = nn.CosineSimilarity(1)
        # self.cos_pos = nn.CosineSimilarity(2)
        pass

    def train_forward(self,x1,x2,x3,x4,skip_distance=False):
        batch_size = len(x1)
        x1_f = self.encoder(x1)
        x2_f = self.encoder(x2)
        x3_f= self.encoder(x3)
        x4_f = self.encoder(x4)
        x1_f = x1_f.view(batch_size, -1)
        x2_f = x2_f.view(batch_size, -1)
        x3_f = x3_f.view(batch_size, -1)
        x4_f = x4_f.view(batch_size, -1)
        x1_f = self.fc(x1_f)
        x2_f = self.fc(x2_f)
        x3_f = self.fc(x3_f)
        x4_f = self.fc(x4_f)

        outLinear = 0

        if skip_distance:
            return outLinear, None

        return outLinear, x1_f, x2_f,x3_f,x4_f,self.dis_center


    def forward(self, x,skip_distance=False):
        pass

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def similarity(self,x):
        batch_size = len(x)
        x1 = x[:, 0:3, :, :]
        x2 = x[:, 3:6, :, :]
        x1_f = self.encoder(x1)
        x2_f = self.encoder(x2)
        x1_f = x1_f.view(batch_size, -1)
        x2_f = x2_f.view(batch_size, -1)

        x1_f = self.fc(x1_f)
        x2_f = self.fc(x2_f)
        return  nn.PairwiseDistance()(x1_f,x2_f)

    def feature(self,x):
        batch_size = len(x)
        x1 = x[:, 0:3, :, :]
        x2 = x[:, 3:6, :, :]
        x1_f = self.encoder(x1)
        x2_f = self.encoder(x2)
        x1_f = x1_f.view(batch_size, -1)
        x2_f = x2_f.view(batch_size, -1)

        x1_f = self.fc(x1_f)
        x2_f = self.fc(x2_f)
        return  x1_f-x2_f

class HK_trip_tanhNet(nn.Module):
    """
     Hierarchical Kinship Triplet network
    """
    def __init__(self, num_classes=4, num_channels=3, im_size=64, init_weights=False, dropout=0.3, **kwargs):
        super(HK_trip_tanhNet, self).__init__()

        self.num_classes = num_classes

        self.encoder = attenNet_basic(input_channel=3)

        # model = SoftIntroVAE(cdim=3, zdim=128, channels=[64, 128, 256, 512], image_size=64)
        # load_model(model, '/home/wei/Documents/CODE/CVPR2021/openset-kinship/datasets/fiw_soft_intro_betas_0.1_1.0_1.0_model_epoch_100_iter_43400.pth', 'cuda')
        # self.encoder = model.encoder
        # for param in self.encoder.parameters():
        #     param.requires_grad = False

        self.fea = 64
        # self.fc = Self_Attention(128,self.fea,self.fea)
        self.fc = nn.Sequential(nn.LayerNorm(512),
                                nn.Linear(512,self.fea),
                                nn.Tanh()
                                )

        # self.classify = nn.Linear(self.fea, num_classes)
        # self.fc_center = nn.Linear(self.fea, num_classes)
        # self.dis_center = [torch.zeros([1]).cuda(),
        #                    nn.Parameter(torch.randn(1), requires_grad=True).cuda(),
        #                    nn.Parameter(torch.randn(1), requires_grad=True).cuda()]
        self.dis_center = [torch.zeros([1]).cuda(),
                           torch.ones([1]).cuda(),
                           4*torch.ones([1]).cuda()]
        # self.centers = nn.Parameter(0.1 * torch.randn(num_classes, self.fea),requires_grad=True)
        if init_weights:
            self._initialize_weights()

        self._init_dis()
        self.cuda()

    def _init_dis(self):
        # self.neg_axis = nn.Parameter(-torch.ones(1, self.anchors.shape[0]).double(), requires_grad=False)
        # self.cos = nn.CosineSimilarity(1)
        # self.cos_pos = nn.CosineSimilarity(2)
        pass

    def train_forward(self,x1,x2,x3,x4,skip_distance=False):
        batch_size = len(x1)
        x1_f = self.encoder(x1)
        x2_f = self.encoder(x2)
        x3_f= self.encoder(x3)
        x4_f = self.encoder(x4)
        x1_f = x1_f.view(batch_size, -1)
        x2_f = x2_f.view(batch_size, -1)
        x3_f = x3_f.view(batch_size, -1)
        x4_f = x4_f.view(batch_size, -1)
        x1_f = self.fc(x1_f)
        x2_f = self.fc(x2_f)
        x3_f = self.fc(x3_f)
        x4_f = self.fc(x4_f)

        outLinear = 0

        if skip_distance:
            return outLinear, None

        return outLinear, x1_f, x2_f,x3_f,x4_f,self.dis_center


    def forward(self, x,skip_distance=False):
        pass

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def similarity(self,x):
        batch_size = len(x)
        x1 = x[:, 0:3, :, :]
        x2 = x[:, 3:6, :, :]
        x1_f = self.encoder(x1)
        x2_f = self.encoder(x2)
        x1_f = x1_f.view(batch_size, -1)
        x2_f = x2_f.view(batch_size, -1)

        x1_f = self.fc(x1_f)
        x2_f = self.fc(x2_f)
        return  nn.PairwiseDistance()(x1_f,x2_f)

    def feature(self,x):
        batch_size = len(x)
        x1 = x[:, 0:3, :, :]
        x2 = x[:, 3:6, :, :]
        x1_f = self.encoder(x1)
        x2_f = self.encoder(x2)
        x1_f = x1_f.view(batch_size, -1)
        x2_f = x2_f.view(batch_size, -1)

        x1_f = self.fc(x1_f)
        x2_f = self.fc(x2_f)
        return  x1_f-x2_f

class HK_trip_tanhNet_different(nn.Module):
    """
     Hierarchical Kinship Triplet network
    """
    def __init__(self, num_classes=4, num_channels=3, im_size=64, init_weights=False, dropout=0.3, **kwargs):
        super(HK_trip_tanhNet_different, self).__init__()

        self.num_classes = num_classes

        self.encoder = attenNet_basic(input_channel=3)

        # model = SoftIntroVAE(cdim=3, zdim=128, channels=[64, 128, 256, 512], image_size=64)
        # load_model(model, '/home/wei/Documents/CODE/CVPR2021/openset-kinship/datasets/fiw_soft_intro_betas_0.1_1.0_1.0_model_epoch_100_iter_43400.pth', 'cuda')
        # self.encoder = model.encoder
        # for param in self.encoder.parameters():
        #     param.requires_grad = False

        self.fea = 64
        # self.fc = Self_Attention(128,self.fea,self.fea)
        self.fc1 = nn.Sequential(nn.LayerNorm(512),
                                nn.Linear(512,self.fea),
                                nn.Tanh()
                                )

        self.fc2 = nn.Sequential(nn.LayerNorm(512),
                                 nn.Linear(512, self.fea),
                                 nn.Tanh()
                                 )

        # self.classify = nn.Linear(self.fea, num_classes)
        # self.fc_center = nn.Linear(self.fea, num_classes)
        # self.dis_center = [torch.zeros([1]).cuda(),
        #                    nn.Parameter(torch.randn(1), requires_grad=True).cuda(),
        #                    nn.Parameter(torch.randn(1), requires_grad=True).cuda()]
        self.dis_center = [torch.zeros([1]).cuda(),
                           torch.ones([1]).cuda(),
                           4*torch.ones([1]).cuda()]
        # self.centers = nn.Parameter(0.1 * torch.randn(num_classes, self.fea),requires_grad=True)
        if init_weights:
            self._initialize_weights()

        self._init_dis()
        self.cuda()

    def _init_dis(self):
        # self.neg_axis = nn.Parameter(-torch.ones(1, self.anchors.shape[0]).double(), requires_grad=False)
        # self.cos = nn.CosineSimilarity(1)
        # self.cos_pos = nn.CosineSimilarity(2)
        pass

    def train_forward(self,x1,x2,x3,x4,skip_distance=False):
        batch_size = len(x1)
        x1_f = self.encoder(x1)
        x2_f = self.encoder(x2)
        x3_f= self.encoder(x3)
        x4_f = self.encoder(x4)
        x1_f = x1_f.view(batch_size, -1)
        x2_f = x2_f.view(batch_size, -1)
        x3_f = x3_f.view(batch_size, -1)
        x4_f = x4_f.view(batch_size, -1)
        x1_f = self.fc1(x1_f)
        x2_f = self.fc2(x2_f)
        x3_f = self.fc2(x3_f)
        x4_f = self.fc2(x4_f)

        outLinear = 0

        if skip_distance:
            return outLinear, None

        return outLinear, x1_f, x2_f,x3_f,x4_f,self.dis_center


    def forward(self, x,skip_distance=False):
        pass

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def similarity(self,x):
        batch_size = len(x)
        x1 = x[:, 0:3, :, :]
        x2 = x[:, 3:6, :, :]
        x1_f = self.encoder(x1)
        x2_f = self.encoder(x2)
        x1_f = x1_f.view(batch_size, -1)
        x2_f = x2_f.view(batch_size, -1)

        x1_f = self.fc1(x1_f)
        x2_f = self.fc2(x2_f)
        return  nn.PairwiseDistance()(x1_f,x2_f)

    def feature(self,x):
        batch_size = len(x)
        x1 = x[:, 0:3, :, :]
        x2 = x[:, 3:6, :, :]
        x1_f = self.encoder(x1)
        x2_f = self.encoder(x2)
        x1_f = x1_f.view(batch_size, -1)
        x2_f = x2_f.view(batch_size, -1)

        x1_f = self.fc(x1_f)
        x2_f = self.fc(x2_f)
        return  x1_f-x2_f

class HK_trip_tanhNet_notanh(nn.Module):
    """
     Hierarchical Kinship Triplet network
    """
    def __init__(self, num_classes=4, num_channels=3, im_size=64, init_weights=False, dropout=0.3, **kwargs):
        super(HK_trip_tanhNet_notanh, self).__init__()

        self.num_classes = num_classes

        self.encoder = attenNet_basic(input_channel=3)

        # model = SoftIntroVAE(cdim=3, zdim=128, channels=[64, 128, 256, 512], image_size=64)
        # load_model(model, '/home/wei/Documents/CODE/CVPR2021/openset-kinship/datasets/fiw_soft_intro_betas_0.1_1.0_1.0_model_epoch_100_iter_43400.pth', 'cuda')
        # self.encoder = model.encoder
        # for param in self.encoder.parameters():
        #     param.requires_grad = False

        self.fea = 64
        # self.fc = Self_Attention(128,self.fea,self.fea)
        self.fc = nn.Sequential(nn.LayerNorm(512),
                                nn.Linear(512,self.fea),
                                # nn.Tanh()
                                )

        # self.classify = nn.Linear(self.fea, num_classes)
        # self.fc_center = nn.Linear(self.fea, num_classes)
        # self.dis_center = [torch.zeros([1]).cuda(),
        #                    nn.Parameter(torch.randn(1), requires_grad=True).cuda(),
        #                    nn.Parameter(torch.randn(1), requires_grad=True).cuda()]
        self.dis_center = [torch.zeros([1]).cuda(),
                           torch.ones([1]).cuda(),
                           4*torch.ones([1]).cuda()]
        # self.centers = nn.Parameter(0.1 * torch.randn(num_classes, self.fea),requires_grad=True)
        if init_weights:
            self._initialize_weights()

        self._init_dis()
        self.cuda()

    def _init_dis(self):
        # self.neg_axis = nn.Parameter(-torch.ones(1, self.anchors.shape[0]).double(), requires_grad=False)
        # self.cos = nn.CosineSimilarity(1)
        # self.cos_pos = nn.CosineSimilarity(2)
        pass

    def train_forward(self,x1,x2,x3,x4,skip_distance=False):
        batch_size = len(x1)
        x1_f = self.encoder(x1)
        x2_f = self.encoder(x2)
        x3_f= self.encoder(x3)
        x4_f = self.encoder(x4)
        x1_f = x1_f.view(batch_size, -1)
        x2_f = x2_f.view(batch_size, -1)
        x3_f = x3_f.view(batch_size, -1)
        x4_f = x4_f.view(batch_size, -1)
        x1_f = self.fc(x1_f)
        x2_f = self.fc(x2_f)
        x3_f = self.fc(x3_f)
        x4_f = self.fc(x4_f)

        outLinear = 0

        if skip_distance:
            return outLinear, None

        return outLinear, x1_f, x2_f,x3_f,x4_f,self.dis_center


    def forward(self, x,skip_distance=False):
        pass

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def similarity(self,x):
        batch_size = len(x)
        x1 = x[:, 0:3, :, :]
        x2 = x[:, 3:6, :, :]
        x1_f = self.encoder(x1)
        x2_f = self.encoder(x2)
        x1_f = x1_f.view(batch_size, -1)
        x2_f = x2_f.view(batch_size, -1)

        x1_f = self.fc(x1_f)
        x2_f = self.fc(x2_f)
        return  nn.PairwiseDistance()(x1_f,x2_f)

    def feature(self,x):
        batch_size = len(x)
        x1 = x[:, 0:3, :, :]
        x2 = x[:, 3:6, :, :]
        x1_f = self.encoder(x1)
        x2_f = self.encoder(x2)
        x1_f = x1_f.view(batch_size, -1)
        x2_f = x2_f.view(batch_size, -1)

        x1_f = self.fc(x1_f)
        x2_f = self.fc(x2_f)
        return  x1_f-x2_f


from networks import resnet
import pickle

def resnet50_load(model, fname):
    """
    Set parameters converted from Caffe models authors of VGGFace2 provide.
    See https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/.

    Arguments:
        model: model
        fname: file name of parameters converted from a Caffe model, assuming the file format is Pickle.
    """
    with open(fname, 'rb') as f:
        weights = pickle.load(f, encoding='latin1')

    own_state = model.state_dict()
    for name, param in weights.items():
        if name in own_state:

            try:
                own_state[name].copy_(torch.from_numpy(param))
            except Exception:
                # raise RuntimeError('While copying the parameter named {}, whose dimensions in the model are {} and whose '\
                #                    'dimensions in the checkpoint are {}.'.format(name, own_state[name].size(), param.size()))
                continue
        else:
            continue
            # raise KeyError('unexpected key "{}" in state_dict'.format(name))

class FG(nn.Module):
    def __init__(self, num_classes=2, num_channels=3, im_size=64, init_weights=False, dropout=0.3, **kwargs):
        super(FG, self).__init__()

        self.num_classes = 2
        self.encoder = resnet.resnet50(include_top=False)
        resnet50_load(self.encoder,'./resnet50_scratch_weight.pkl')
        self.globalmaxpool = nn.MaxPool2d(1)
        self.globalavgpool = nn.MaxPool2d(1)
        self.relu = nn.ReLU()
        self.linear= nn.Linear(6144,128)
        self.dropout = nn.Dropout(0.02)
        self.linear2 = nn.Linear(128,1)
        self.sigmoid = nn.Sigmoid()

        self.cuda()

    def fg_forward(self,x1,x2):
        x1 = self.encoder(x1)
        x2 = self.encoder(x2)
        x1 = self.globalmaxpool(x1).squeeze()
        x2 = self.globalavgpool(x2).squeeze()

        x3 = torch.sub(x1,x2)
        x3 = torch.mul(x3,x3)

        x1_ = torch.mul(x1,x2)
        x2_ = torch.mul(x2,x2)
        x4 = torch.sub(x1_,x2_)

        x5 = torch.mul(x1,x2)

        x = torch.cat((x3,x4,x5),dim=1)

        x = self.relu(self.linear(x))
        x = self.dropout(x)
        x = self.sigmoid(self.linear2(x))
        # x = self.linear2(x)
        return x



    def forward(self, x):
        # batch_size = len(x)
        x1 = x[:, 0:3, :, :]
        x2 = x[:, 3:6, :, :]

        x = self.fg_forward(x1,x2)

        return x.squeeze()

    def similarity(self,x,close = True):
        # logits = self.forward(x)
        # if close:
        #     scores = F.softmax(logits,dim=1)
        #     pred = -torch.max(scores,dim=1)[0]
        # else:
        #     pred = F.softmax(logits,dim=1)[:,0]
        x1 = x[:, 0:3, :, :]
        x2 = x[:, 3:6, :, :]

        x = self.fg_forward(x1, x2).squeeze()
        # x = self.sigmoid(x)
        return -x


class HK_trip_FG(nn.Module):
    """
     Hierarchical Kinship Triplet network
    """
    def __init__(self, num_classes=4, num_channels=3, im_size=64, init_weights=False, dropout=0.3, **kwargs):
        super(HK_trip_FG, self).__init__()

        self.num_classes = num_classes

        self.encoder = resnet.resnet50(include_top=False)
        resnet50_load(self.encoder, './resnet50_ft_weight.pkl')

        self.fea = 64

        self.fc = nn.Sequential(nn.LayerNorm(2048),
                                nn.Linear(2048,self.fea),
                                nn.Tanh()
                                )

        self.dis_center = [torch.zeros([1]).cuda(),
                           torch.ones([1]).cuda(),
                           4*torch.ones([1]).cuda()]


        self._init_dis()
        self.cuda()

    def _init_dis(self):

        pass

    def train_forward(self,x1,x2,x3,x4,skip_distance=False):
        batch_size = len(x1)
        x1_f = self.encoder(x1)
        x2_f = self.encoder(x2)
        x3_f= self.encoder(x3)
        x4_f = self.encoder(x4)
        x1_f = x1_f.view(batch_size, -1)
        x2_f = x2_f.view(batch_size, -1)
        x3_f = x3_f.view(batch_size, -1)
        x4_f = x4_f.view(batch_size, -1)
        x1_f = self.fc(x1_f)
        x2_f = self.fc(x2_f)
        x3_f = self.fc(x3_f)
        x4_f = self.fc(x4_f)

        outLinear = 0

        if skip_distance:
            return outLinear, None

        return outLinear, x1_f, x2_f,x3_f,x4_f,self.dis_center


    def forward(self, x,skip_distance=False):
        pass


    def similarity(self,x):
        batch_size = len(x)
        x1 = x[:, 0:3, :, :]
        x2 = x[:, 3:6, :, :]
        x1_f = self.encoder(x1)
        x2_f = self.encoder(x2)
        x1_f = x1_f.view(batch_size, -1)
        x2_f = x2_f.view(batch_size, -1)

        x1_f = self.fc(x1_f)
        x2_f = self.fc(x2_f)
        return  nn.PairwiseDistance()(x1_f,x2_f)

    def feature(self,x):
        batch_size = len(x)
        x1 = x[:, 0:3, :, :]
        x2 = x[:, 3:6, :, :]
        x1_f = self.encoder(x1)
        x2_f = self.encoder(x2)
        x1_f = x1_f.view(batch_size, -1)
        x2_f = x2_f.view(batch_size, -1)

        x1_f = self.fc(x1_f)
        x2_f = self.fc(x2_f)
        return  x1_f-x2_f



# class HK_trip_ALTGVT(nn.Module):
#     """
#      Hierarchical Kinship Triplet network
#     """
#     def __init__(self, num_classes=4, num_channels=3, im_size=64, init_weights=False, dropout=0.3, **kwargs):
#         super(HK_trip_ALTGVT, self).__init__()
#
#         self.num_classes = num_classes
#
#         self.encoder = ALTGVT(img_size=64,patch_size = 4)
#
#         # model = SoftIntroVAE(cdim=3, zdim=128, channels=[64, 128, 256, 512], image_size=64)
#         # load_model(model, '/home/wei/Documents/CODE/CVPR2021/openset-kinship/datasets/fiw_soft_intro_betas_0.1_1.0_1.0_model_epoch_100_iter_43400.pth', 'cuda')
#         # self.encoder = model.encoder
#         # for param in self.encoder.parameters():
#         #     param.requires_grad = False
#
#         self.fea = 64
#         # self.fc = Self_Attention(128,self.fea,self.fea)
#         self.fc = nn.Sequential(
#                                 nn.Linear(1000,self.fea),
#                                 nn.Tanh()
#                                 )
#
#         # self.classify = nn.Linear(self.fea, num_classes)
#         # self.fc_center = nn.Linear(self.fea, num_classes)
#         # self.dis_center = [torch.zeros([1]).cuda(),
#         #                    nn.Parameter(torch.randn(1), requires_grad=True).cuda(),
#         #                    nn.Parameter(torch.randn(1), requires_grad=True).cuda()]
#         self.dis_center = [torch.zeros([1]).cuda(),
#                            torch.ones([1]).cuda(),
#                            4*torch.ones([1]).cuda()]
#         # self.centers = nn.Parameter(0.1 * torch.randn(num_classes, self.fea),requires_grad=True)
#         # if init_weights:
#         #     self._initialize_weights()
#
#         self._init_dis()
#         self.cuda()
#
#     def _init_dis(self):
#         # self.neg_axis = nn.Parameter(-torch.ones(1, self.anchors.shape[0]).double(), requires_grad=False)
#         # self.cos = nn.CosineSimilarity(1)
#         # self.cos_pos = nn.CosineSimilarity(2)
#         pass
#
#     def train_forward(self,x1,x2,x3,x4,skip_distance=False):
#         batch_size = len(x1)
#         x1_f = self.encoder(x1)
#         x2_f = self.encoder(x2)
#         x3_f= self.encoder(x3)
#         x4_f = self.encoder(x4)
#         x1_f = x1_f.view(batch_size, -1)
#         x2_f = x2_f.view(batch_size, -1)
#         x3_f = x3_f.view(batch_size, -1)
#         x4_f = x4_f.view(batch_size, -1)
#         x1_f = self.fc(x1_f)
#         x2_f = self.fc(x2_f)
#         x3_f = self.fc(x3_f)
#         x4_f = self.fc(x4_f)
#
#         outLinear = 0
#
#         if skip_distance:
#             return outLinear, None
#
#         return outLinear, x1_f, x2_f,x3_f,x4_f,self.dis_center
#
#
#     def forward(self, x,skip_distance=False):
#         pass
#
#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, 0, 0.01)
#                 nn.init.constant_(m.bias, 0)
#
#     def similarity(self,x):
#         batch_size = len(x)
#         x1 = x[:, 0:3, :, :]
#         x2 = x[:, 3:6, :, :]
#         x1_f = self.encoder(x1)
#         x2_f = self.encoder(x2)
#         x1_f = x1_f.view(batch_size, -1)
#         x2_f = x2_f.view(batch_size, -1)
#
#         x1_f = self.fc(x1_f)
#         x2_f = self.fc(x2_f)
#         return  nn.PairwiseDistance()(x1_f,x2_f)
#
#     def feature(self,x):
#         batch_size = len(x)
#         x1 = x[:, 0:3, :, :]
#         x2 = x[:, 3:6, :, :]
#         x1_f = self.encoder(x1)
#         x2_f = self.encoder(x2)
#         x1_f = x1_f.view(batch_size, -1)
#         x2_f = x2_f.view(batch_size, -1)
#
#         x1_f = self.fc(x1_f)
#         x2_f = self.fc(x2_f)
#         return  x1_f-x2_f
#



class HK_trip_KATsmall(nn.Module):
    """
     Hierarchical Kinship Triplet network
    """
    def __init__(self, num_classes=4, num_channels=3, im_size=64, init_weights=False, dropout=0.3, **kwargs):
        super(HK_trip_KATsmall, self).__init__()

        self.num_classes = num_classes

        self.encoder = attenNet_basic(input_channel=3)

        self.KAT = KAT_small( dim=64, depth=1, heads=8, dim_head=16, mlp_dim=32, dropout=0.)

        self.fea = 64

        self.fc = nn.Sequential(nn.LayerNorm(512),
                                nn.Linear(512,self.fea),
                                nn.Tanh() #
                                # nn.GELU()
                                )

        self.tanh = nn.Tanh()

        self.dis_center = [torch.zeros([1]).cuda(),
                           torch.ones([1]).cuda(),
                           4*torch.ones([1]).cuda()]

        if init_weights:
            self._initialize_weights()

        self._init_dis()
        self.cuda()

    def _init_dis(self):

        pass

    def train_forward(self,x1,x2,x3,x4,skip_distance=False):
        # batch_size = len(x1)
        a_f = self.encoder(x1)
        s_f = self.encoder(x2)
        p_f= self.encoder(x3)
        n_f = self.encoder(x4)
        # x1_f = x1_f.view(batch_size, -1)
        # x2_f = x2_f.view(batch_size, -1)
        # x3_f = x3_f.view(batch_size, -1)
        # x4_f = x4_f.view(batch_size, -1)
        a_f = torch.unsqueeze(self.fc(a_f),1)
        s_f = torch.unsqueeze(self.fc(s_f),1)
        p_f = torch.unsqueeze(self.fc(p_f),1)
        n_f = torch.unsqueeze(self.fc(n_f),1)

        (a_asf, s_f1) =self.KAT(a_f,s_f)
        (a_apf, p_f2) = self.KAT(a_f,p_f)
        (a_anf, n_f3) =self.KAT(a_f,n_f)


        outLinear = 0

        if skip_distance:
            return outLinear, None

        return outLinear, (a_asf.squeeze(), s_f1.squeeze()), \
               (a_apf.squeeze(), p_f2.squeeze()), (a_anf.squeeze(), n_f3.squeeze()),self.dis_center


    def forward(self, x,skip_distance=False):
        pass

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            # elif isinstance(m, nn.Linear):
            #     nn.init.normal_(m.weight, 0, 0.01)
            #     nn.init.constant_(m.bias, 0)

    def similarity(self,x):
        batch_size = len(x)
        x1 = x[:, 0:3, :, :]
        x2 = x[:, 3:6, :, :]
        x1_f = self.encoder(x1)
        x2_f = self.encoder(x2)
        x1_f = x1_f.view(batch_size, -1)
        x2_f = x2_f.view(batch_size, -1)
        x1_f = torch.unsqueeze(self.fc(x1_f), 1)
        x2_f = torch.unsqueeze(self.fc(x2_f), 1)
        x1_f,x2_f= self.KAT(x1_f, x2_f)

        return  nn.PairwiseDistance()(x1_f.squeeze(),x2_f.squeeze())

    def feature(self,x):
        batch_size = len(x)
        x1 = x[:, 0:3, :, :]
        x2 = x[:, 3:6, :, :]
        x1_f = self.encoder(x1)
        x2_f = self.encoder(x2)
        x1_f = x1_f.view(batch_size, -1)
        x2_f = x2_f.view(batch_size, -1)
        x1_f = torch.unsqueeze(self.fc(x1_f), 1)
        x2_f = torch.unsqueeze(self.fc(x2_f), 1)
        return  x1_f-x2_f

class HK_trip_KATdouble(nn.Module):
    """
     Hierarchical Kinship Triplet network
    """
    def __init__(self, num_classes=4, num_channels=3, im_size=64, init_weights=False, dropout=0.3, **kwargs):
        super(HK_trip_KATdouble, self).__init__()

        self.num_classes = num_classes

        self.encoder = attenNet_basic(input_channel=3)

        self.KAT = KAT_tanh( dim=64, depth=1, heads=8, dim_head=16, mlp_dim=32, dropout=0.)

        self.fea = 64

        self.fc = nn.Sequential(nn.LayerNorm(512),
                                nn.Linear(512,self.fea),
                                nn.Tanh() #
                                # nn.GELU()
                                )

        self.tanh = nn.Tanh()

        self.dis_center = [torch.zeros([1]).cuda(),
                           torch.ones([1]).cuda(),
                           4*torch.ones([1]).cuda()]

        if init_weights:
            self._initialize_weights()

        self._init_dis()
        self.cuda()

    def _init_dis(self):

        pass

    def train_forward(self,x1,x2,x3,x4,skip_distance=False):
        # batch_size = len(x1)
        a_f = self.encoder(x1)
        s_f = self.encoder(x2)
        p_f= self.encoder(x3)
        n_f = self.encoder(x4)
        # x1_f = x1_f.view(batch_size, -1)
        # x2_f = x2_f.view(batch_size, -1)
        # x3_f = x3_f.view(batch_size, -1)
        # x4_f = x4_f.view(batch_size, -1)
        a_f = torch.unsqueeze(self.fc(a_f),1)
        s_f = torch.unsqueeze(self.fc(s_f),1)
        p_f = torch.unsqueeze(self.fc(p_f),1)
        n_f = torch.unsqueeze(self.fc(n_f),1)

        (a_asf, s_f1) =self.KAT(a_f,s_f)
        (a_apf, p_f2) = self.KAT(a_f,p_f)
        (a_anf, n_f3) =self.KAT(a_f,n_f)


        outLinear = 0

        if skip_distance:
            return outLinear, None

        return outLinear, (a_asf.squeeze(), s_f1.squeeze()), \
               (a_apf.squeeze(), p_f2.squeeze()), (a_anf.squeeze(), n_f3.squeeze()),self.dis_center


    def forward(self, x,skip_distance=False):
        pass

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            # elif isinstance(m, nn.Linear):
            #     nn.init.normal_(m.weight, 0, 0.01)
            #     nn.init.constant_(m.bias, 0)

    def similarity(self,x):
        batch_size = len(x)
        x1 = x[:, 0:3, :, :]
        x2 = x[:, 3:6, :, :]
        x1_f = self.encoder(x1)
        x2_f = self.encoder(x2)
        x1_f = x1_f.view(batch_size, -1)
        x2_f = x2_f.view(batch_size, -1)
        x1_f = torch.unsqueeze(self.fc(x1_f), 1)
        x2_f = torch.unsqueeze(self.fc(x2_f), 1)
        x1_f,x2_f= self.KAT(x1_f, x2_f)

        return  nn.PairwiseDistance()(x1_f.squeeze(),x2_f.squeeze())

    def feature(self,x):
        batch_size = len(x)
        x1 = x[:, 0:3, :, :]
        x2 = x[:, 3:6, :, :]
        x1_f = self.encoder(x1)
        x2_f = self.encoder(x2)
        x1_f = x1_f.view(batch_size, -1)
        x2_f = x2_f.view(batch_size, -1)
        x1_f = torch.unsqueeze(self.fc(x1_f), 1)
        x2_f = torch.unsqueeze(self.fc(x2_f), 1)
        return  x1_f-x2_f
class HK_trip_KATsmalldeep(nn.Module):
    """
     Hierarchical Kinship Triplet network
    """
    def __init__(self, num_classes=4, num_channels=3, im_size=64, init_weights=False, dropout=0.3, **kwargs):
        super(HK_trip_KATsmalldeep, self).__init__()

        self.num_classes = num_classes

        self.encoder = attenNet_basic(input_channel=3)

        self.KAT = KAT_small( dim=64, depth=2, heads=8, dim_head=16, mlp_dim=32, dropout=0.)

        self.fea = 64

        self.fc = nn.Sequential(nn.LayerNorm(512),
                                nn.Linear(512,self.fea),
                                nn.Tanh() #
                                # nn.GELU()
                                )

        self.tanh = nn.Tanh()

        self.dis_center = [torch.zeros([1]).cuda(),
                           torch.ones([1]).cuda(),
                           4*torch.ones([1]).cuda()]

        if init_weights:
            self._initialize_weights()

        self._init_dis()
        self.cuda()

    def _init_dis(self):

        pass

    def train_forward(self,x1,x2,x3,x4,skip_distance=False):
        # batch_size = len(x1)
        a_f = self.encoder(x1)
        s_f = self.encoder(x2)
        p_f= self.encoder(x3)
        n_f = self.encoder(x4)
        # x1_f = x1_f.view(batch_size, -1)
        # x2_f = x2_f.view(batch_size, -1)
        # x3_f = x3_f.view(batch_size, -1)
        # x4_f = x4_f.view(batch_size, -1)
        a_f = torch.unsqueeze(self.fc(a_f),1)
        s_f = torch.unsqueeze(self.fc(s_f),1)
        p_f = torch.unsqueeze(self.fc(p_f),1)
        n_f = torch.unsqueeze(self.fc(n_f),1)

        (a_asf, s_f1) =self.KAT(a_f,s_f)
        (a_apf, p_f2) = self.KAT(a_f,p_f)
        (a_anf, n_f3) =self.KAT(a_f,n_f)


        outLinear = 0

        if skip_distance:
            return outLinear, None

        return outLinear, (a_asf.squeeze(), s_f1.squeeze()), \
               (a_apf.squeeze(), p_f2.squeeze()), (a_anf.squeeze(), n_f3.squeeze()),self.dis_center


    def forward(self, x,skip_distance=False):
        pass

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            # elif isinstance(m, nn.Linear):
            #     nn.init.normal_(m.weight, 0, 0.01)
            #     nn.init.constant_(m.bias, 0)

    def similarity(self,x):
        batch_size = len(x)
        x1 = x[:, 0:3, :, :]
        x2 = x[:, 3:6, :, :]
        x1_f = self.encoder(x1)
        x2_f = self.encoder(x2)
        x1_f = x1_f.view(batch_size, -1)
        x2_f = x2_f.view(batch_size, -1)
        x1_f = torch.unsqueeze(self.fc(x1_f), 1)
        x2_f = torch.unsqueeze(self.fc(x2_f), 1)
        x1_f,x2_f= self.KAT(x1_f, x2_f)

        return  nn.PairwiseDistance()(x1_f.squeeze(),x2_f.squeeze())

    def feature(self,x):
        batch_size = len(x)
        x1 = x[:, 0:3, :, :]
        x2 = x[:, 3:6, :, :]
        x1_f = self.encoder(x1)
        x2_f = self.encoder(x2)
        x1_f = x1_f.view(batch_size, -1)
        x2_f = x2_f.view(batch_size, -1)
        x1_f = torch.unsqueeze(self.fc(x1_f), 1)
        x2_f = torch.unsqueeze(self.fc(x2_f), 1)
        return  x1_f-x2_f

class HK_trip_KAT(nn.Module):
    """
     Hierarchical Kinship Triplet network
    """
    def __init__(self, num_classes=4, num_channels=3, im_size=64, init_weights=False, dropout=0.3, **kwargs):
        super(HK_trip_KAT, self).__init__()

        self.num_classes = num_classes

        self.encoder = attenNet_basic(input_channel=3)

        # self.KAT = KAT_single( dim=64, depth=1, heads=8, dim_head=64, mlp_dim=32, dropout=0.)
        self.KAT = KAT_single(dim=64, depth=1, heads=8, dim_head=32, mlp_dim=32, dropout=0.)

        self.fea = 64

        self.fc = nn.Sequential(nn.LayerNorm(512),
                                nn.Linear(512,self.fea),
                                nn.Tanh() #
                                # nn.GELU()
                                )

        self.tanh = nn.Tanh()

        self.dis_center = [torch.zeros([1]).cuda(),
                           torch.ones([1]).cuda(),
                           4*torch.ones([1]).cuda()]

        if init_weights:
            self._initialize_weights()

        self._init_dis()
        self.cuda()

    def _init_dis(self):

        pass

    def train_forward(self,x1,x2,x3,x4,skip_distance=False):
        # batch_size = len(x1)
        a_f = self.encoder(x1)
        s_f = self.encoder(x2)
        p_f= self.encoder(x3)
        n_f = self.encoder(x4)
        # x1_f = x1_f.view(batch_size, -1)
        # x2_f = x2_f.view(batch_size, -1)
        # x3_f = x3_f.view(batch_size, -1)
        # x4_f = x4_f.view(batch_size, -1)
        a_f = torch.unsqueeze(self.fc(a_f),1)
        s_f = torch.unsqueeze(self.fc(s_f),1)
        p_f = torch.unsqueeze(self.fc(p_f),1)
        n_f = torch.unsqueeze(self.fc(n_f),1)

        (a_asf, s_f1) =self.KAT(a_f,s_f)
        (a_apf, p_f2) = self.KAT(a_f,p_f)
        (a_anf, n_f3) =self.KAT(a_f,n_f)


        outLinear = 0

        if skip_distance:
            return outLinear, None

        return outLinear, (a_asf.squeeze(), s_f1.squeeze()), \
               (a_apf.squeeze(), p_f2.squeeze()), (a_anf.squeeze(), n_f3.squeeze()),self.dis_center


    def forward(self, x,skip_distance=False):
        pass

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            # elif isinstance(m, nn.Linear):
            #     nn.init.normal_(m.weight, 0, 0.01)
            #     nn.init.constant_(m.bias, 0)

    def similarity(self,x):
        batch_size = len(x)
        x1 = x[:, 0:3, :, :]
        x2 = x[:, 3:6, :, :]
        x1_f = self.encoder(x1)
        x2_f = self.encoder(x2)
        x1_f = x1_f.view(batch_size, -1)
        x2_f = x2_f.view(batch_size, -1)
        x1_f = torch.unsqueeze(self.fc(x1_f), 1)
        x2_f = torch.unsqueeze(self.fc(x2_f), 1)
        x1_f,x2_f= self.KAT(x1_f, x2_f)

        return  nn.PairwiseDistance()(x1_f.squeeze(),x2_f.squeeze())

    def feature(self,x):
        batch_size = len(x)
        x1 = x[:, 0:3, :, :]
        x2 = x[:, 3:6, :, :]
        x1_f = self.encoder(x1)
        x2_f = self.encoder(x2)
        x1_f = x1_f.view(batch_size, -1)
        x2_f = x2_f.view(batch_size, -1)
        x1_f = torch.unsqueeze(self.fc(x1_f), 1)
        x2_f = torch.unsqueeze(self.fc(x2_f), 1)
        return  x1_f-x2_f

class HK_trip_KATfinal(nn.Module):
    """
     Hierarchical Kinship Triplet network
    """
    def __init__(self, num_classes=4, num_channels=3, im_size=64, init_weights=False, dropout=0.3, **kwargs):
        super(HK_trip_KATfinal, self).__init__()

        self.num_classes = num_classes

        self.encoder = attenNet_basic(input_channel=3)

        self.KAT = KAT_single_final( dim=64, depth=1, heads=8, dim_head=64, mlp_dim=32, dropout=0.)
        # self.KAT = KAT_final(dim=64, depth=1, heads=8, dim_head=32, mlp_dim=32, dropout=0.)
        # self.KAT = KAT(dim=64, depth=1, heads=8, dim_head=32, mlp_dim=32, dropout=0.)
        self.fea = 64

        self.fc = nn.Sequential(nn.LayerNorm(512),
                                nn.Linear(512,self.fea),
                                nn.Tanh() #
                                # nn.GELU()
                                )

        self.tanh = nn.Tanh()

        self.dis_center = [torch.zeros([1]).cuda(),
                           torch.ones([1]).cuda(),
                           4*torch.ones([1]).cuda()]

        if init_weights:
            self._initialize_weights()

        self._init_dis()
        self.cuda()

    def _init_dis(self):

        pass

    def train_forward(self,x1,x2,x3,x4,skip_distance=False):
        # batch_size = len(x1)
        a_f = self.encoder(x1)
        s_f = self.encoder(x2)
        p_f= self.encoder(x3)
        n_f = self.encoder(x4)
        # x1_f = x1_f.view(batch_size, -1)
        # x2_f = x2_f.view(batch_size, -1)
        # x3_f = x3_f.view(batch_size, -1)
        # x4_f = x4_f.view(batch_size, -1)
        a_f = torch.unsqueeze(self.fc(a_f),1)
        s_f = torch.unsqueeze(self.fc(s_f),1)
        p_f = torch.unsqueeze(self.fc(p_f),1)
        n_f = torch.unsqueeze(self.fc(n_f),1)

        s_f =self.KAT(a_f,s_f)
        p_f = self.KAT(a_f,p_f)
        n_f =self.KAT(a_f,n_f)


        outLinear = 0

        if skip_distance:
            return outLinear, None

        return outLinear,\
               a_f.squeeze(), s_f.squeeze(), p_f.squeeze(), n_f.squeeze(),self.dis_center


    def forward(self, x,skip_distance=False):
        pass

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            # elif isinstance(m, nn.Linear):
            #     nn.init.normal_(m.weight, 0, 0.01)
            #     nn.init.constant_(m.bias, 0)

    def similarity(self,x):
        batch_size = len(x)
        x1 = x[:, 0:3, :, :]
        x2 = x[:, 3:6, :, :]
        x1_f = self.encoder(x1)
        x2_f = self.encoder(x2)
        x1_f = x1_f.view(batch_size, -1)
        x2_f = x2_f.view(batch_size, -1)
        x1_f = torch.unsqueeze(self.fc(x1_f), 1)
        x2_f = torch.unsqueeze(self.fc(x2_f), 1)
        x2_f= self.KAT(x1_f, x2_f)

        return  nn.PairwiseDistance()(x1_f.squeeze(),x2_f.squeeze())

    def feature(self,x):
        batch_size = len(x)
        x1 = x[:, 0:3, :, :]
        x2 = x[:, 3:6, :, :]
        x1_f = self.encoder(x1)
        x2_f = self.encoder(x2)
        x1_f = x1_f.view(batch_size, -1)
        x2_f = x2_f.view(batch_size, -1)
        x1_f = torch.unsqueeze(self.fc(x1_f), 1)
        x2_f = torch.unsqueeze(self.fc(x2_f), 1)
        return  x1_f-x2_f


class HK_trip_KATsimple(nn.Module):
    """
     Hierarchical Kinship Triplet network
    """
    def __init__(self, num_classes=4, num_channels=3, im_size=64, init_weights=False, dropout=0.3, **kwargs):
        super(HK_trip_KATsimple, self).__init__()

        self.num_classes = num_classes

        self.encoder = attenNet_basic(input_channel=3)

        self.KAT = KAT_single_final( dim=64, depth=1, heads=2, dim_head=16, mlp_dim=16, dropout=0.)
        # self.KAT = KAT_final(dim=64, depth=1, heads=8, dim_head=32, mlp_dim=32, dropout=0.)
        # self.KAT = KAT(dim=64, depth=1, heads=8, dim_head=32, mlp_dim=32, dropout=0.)
        self.fea = 64

        self.fc = nn.Sequential(nn.LayerNorm(512),
                                nn.Linear(512,self.fea),
                                nn.Tanh() #
                                # nn.GELU()
                                )

        self.tanh = nn.Tanh()

        self.dis_center = [torch.zeros([1]).cuda(),
                           torch.ones([1]).cuda(),
                           4*torch.ones([1]).cuda()]

        if init_weights:
            self._initialize_weights()

        self._init_dis()
        self.cuda()

    def _init_dis(self):

        pass

    def train_forward(self,x1,x2,x3,x4,skip_distance=False):
        # batch_size = len(x1)
        a_f = self.encoder(x1)
        s_f = self.encoder(x2)
        p_f= self.encoder(x3)
        n_f = self.encoder(x4)
        # x1_f = x1_f.view(batch_size, -1)
        # x2_f = x2_f.view(batch_size, -1)
        # x3_f = x3_f.view(batch_size, -1)
        # x4_f = x4_f.view(batch_size, -1)
        a_f = torch.unsqueeze(self.fc(a_f),1)
        s_f = torch.unsqueeze(self.fc(s_f),1)
        p_f = torch.unsqueeze(self.fc(p_f),1)
        n_f = torch.unsqueeze(self.fc(n_f),1)

        s_f =self.KAT(a_f,s_f)
        p_f = self.KAT(a_f,p_f)
        n_f =self.KAT(a_f,n_f)


        outLinear = 0

        if skip_distance:
            return outLinear, None

        return outLinear,\
               a_f.squeeze(), s_f.squeeze(), p_f.squeeze(), n_f.squeeze(),self.dis_center


    def forward(self, x,skip_distance=False):
        pass

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            # elif isinstance(m, nn.Linear):
            #     nn.init.normal_(m.weight, 0, 0.01)
            #     nn.init.constant_(m.bias, 0)

    def similarity(self,x):
        batch_size = len(x)
        x1 = x[:, 0:3, :, :]
        x2 = x[:, 3:6, :, :]
        x1_f = self.encoder(x1)
        x2_f = self.encoder(x2)
        x1_f = x1_f.view(batch_size, -1)
        x2_f = x2_f.view(batch_size, -1)
        x1_f = torch.unsqueeze(self.fc(x1_f), 1)
        x2_f = torch.unsqueeze(self.fc(x2_f), 1)
        x2_f= self.KAT(x1_f, x2_f)

        return  nn.PairwiseDistance()(x1_f.squeeze(),x2_f.squeeze())

    def feature(self,x):
        batch_size = len(x)
        x1 = x[:, 0:3, :, :]
        x2 = x[:, 3:6, :, :]
        x1_f = self.encoder(x1)
        x2_f = self.encoder(x2)
        x1_f = x1_f.view(batch_size, -1)
        x2_f = x2_f.view(batch_size, -1)
        x1_f = torch.unsqueeze(self.fc(x1_f), 1)
        x2_f = torch.unsqueeze(self.fc(x2_f), 1)
        return  x1_f-x2_f

class HK_trip_KAT_deep(nn.Module):
    """
     Hierarchical Kinship Triplet network
    """
    def __init__(self, num_classes=4, num_channels=3, im_size=64, init_weights=False, dropout=0.3, **kwargs):
        super(HK_trip_KAT_deep, self).__init__()

        self.num_classes = num_classes

        self.encoder = attenNet_basic(input_channel=3)

        self.KAT = KAT_single( dim=64, depth=2, heads=8, dim_head=64, mlp_dim=32, dropout=0.)

        self.fea = 64

        self.fc = nn.Sequential(nn.LayerNorm(512),
                                nn.Linear(512,self.fea),
                                nn.Tanh() #
                                # nn.GELU()
                                )

        self.tanh = nn.Tanh()

        self.dis_center = [torch.zeros([1]).cuda(),
                           torch.ones([1]).cuda(),
                           4*torch.ones([1]).cuda()]

        if init_weights:
            self._initialize_weights()

        self._init_dis()
        self.cuda()

    def _init_dis(self):

        pass

    def train_forward(self,x1,x2,x3,x4,skip_distance=False):
        # batch_size = len(x1)
        a_f = self.encoder(x1)
        s_f = self.encoder(x2)
        p_f= self.encoder(x3)
        n_f = self.encoder(x4)
        # x1_f = x1_f.view(batch_size, -1)
        # x2_f = x2_f.view(batch_size, -1)
        # x3_f = x3_f.view(batch_size, -1)
        # x4_f = x4_f.view(batch_size, -1)
        a_f = torch.unsqueeze(self.fc(a_f),1)
        s_f = torch.unsqueeze(self.fc(s_f),1)
        p_f = torch.unsqueeze(self.fc(p_f),1)
        n_f = torch.unsqueeze(self.fc(n_f),1)

        (a_asf, s_f1) =self.KAT(a_f,s_f)
        (a_apf, p_f2) = self.KAT(a_f,p_f)
        (a_anf, n_f3) =self.KAT(a_f,n_f)


        outLinear = 0

        if skip_distance:
            return outLinear, None

        return outLinear, (a_asf.squeeze(), s_f1.squeeze()), \
               (a_apf.squeeze(), p_f2.squeeze()), (a_anf.squeeze(), n_f3.squeeze()),self.dis_center


    def forward(self, x,skip_distance=False):
        pass

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            # elif isinstance(m, nn.Linear):
            #     nn.init.normal_(m.weight, 0, 0.01)
            #     nn.init.constant_(m.bias, 0)

    def similarity(self,x):
        batch_size = len(x)
        x1 = x[:, 0:3, :, :]
        x2 = x[:, 3:6, :, :]
        x1_f = self.encoder(x1)
        x2_f = self.encoder(x2)
        x1_f = x1_f.view(batch_size, -1)
        x2_f = x2_f.view(batch_size, -1)
        x1_f = torch.unsqueeze(self.fc(x1_f), 1)
        x2_f = torch.unsqueeze(self.fc(x2_f), 1)
        x1_f,x2_f= self.KAT(x1_f, x2_f)

        return  nn.PairwiseDistance()(x1_f.squeeze(),x2_f.squeeze())

    def feature(self,x):
        batch_size = len(x)
        x1 = x[:, 0:3, :, :]
        x2 = x[:, 3:6, :, :]
        x1_f = self.encoder(x1)
        x2_f = self.encoder(x2)
        x1_f = x1_f.view(batch_size, -1)
        x2_f = x2_f.view(batch_size, -1)
        x1_f = torch.unsqueeze(self.fc(x1_f), 1)
        x2_f = torch.unsqueeze(self.fc(x2_f), 1)
        return  x1_f-x2_f

class HK_trip_KAT_deep_new(nn.Module):
    """
     Hierarchical Kinship Triplet network
    """
    def __init__(self, num_classes=4, num_channels=3, im_size=64, init_weights=False, dropout=0.3, **kwargs):
        super(HK_trip_KAT_deep_new, self).__init__()

        self.num_classes = num_classes

        self.encoder = attenNet_basic(input_channel=3)

        self.KAT = KAT_single_2( dim=64, depth=2, heads=8, dim_head=64, mlp_dim=32, dropout=0.)

        self.fea = 64

        self.fc = nn.Sequential(nn.LayerNorm(512),
                                nn.Linear(512,self.fea),
                                nn.Tanh() #
                                # nn.GELU()
                                )

        self.tanh = nn.Tanh()

        self.dis_center = [torch.zeros([1]).cuda(),
                           torch.ones([1]).cuda(),
                           4*torch.ones([1]).cuda()]

        if init_weights:
            self._initialize_weights()

        self._init_dis()
        self.cuda()

    def _init_dis(self):

        pass

    def train_forward(self,x1,x2,x3,x4,skip_distance=False):
        # batch_size = len(x1)
        a_f = self.encoder(x1)
        s_f = self.encoder(x2)
        p_f= self.encoder(x3)
        n_f = self.encoder(x4)
        # x1_f = x1_f.view(batch_size, -1)
        # x2_f = x2_f.view(batch_size, -1)
        # x3_f = x3_f.view(batch_size, -1)
        # x4_f = x4_f.view(batch_size, -1)
        a_f = torch.unsqueeze(self.fc(a_f),1)
        s_f = torch.unsqueeze(self.fc(s_f),1)
        p_f = torch.unsqueeze(self.fc(p_f),1)
        n_f = torch.unsqueeze(self.fc(n_f),1)

        (a_asf, s_f1) =self.KAT(a_f,s_f)
        (a_apf, p_f2) = self.KAT(a_f,p_f)
        (a_anf, n_f3) =self.KAT(a_f,n_f)


        outLinear = 0

        if skip_distance:
            return outLinear, None

        return outLinear, (a_asf.squeeze(), s_f1.squeeze()), \
               (a_apf.squeeze(), p_f2.squeeze()), (a_anf.squeeze(), n_f3.squeeze()),self.dis_center


    def forward(self, x,skip_distance=False):
        pass

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            # elif isinstance(m, nn.Linear):
            #     nn.init.normal_(m.weight, 0, 0.01)
            #     nn.init.constant_(m.bias, 0)

    def similarity(self,x):
        batch_size = len(x)
        x1 = x[:, 0:3, :, :]
        x2 = x[:, 3:6, :, :]
        x1_f = self.encoder(x1)
        x2_f = self.encoder(x2)
        x1_f = x1_f.view(batch_size, -1)
        x2_f = x2_f.view(batch_size, -1)
        x1_f = torch.unsqueeze(self.fc(x1_f), 1)
        x2_f = torch.unsqueeze(self.fc(x2_f), 1)
        x1_f,x2_f= self.KAT(x1_f, x2_f)

        return  nn.PairwiseDistance()(x1_f.squeeze(),x2_f.squeeze())

    def feature(self,x):
        batch_size = len(x)
        x1 = x[:, 0:3, :, :]
        x2 = x[:, 3:6, :, :]
        x1_f = self.encoder(x1)
        x2_f = self.encoder(x2)
        x1_f = x1_f.view(batch_size, -1)
        x2_f = x2_f.view(batch_size, -1)
        x1_f = torch.unsqueeze(self.fc(x1_f), 1)
        x2_f = torch.unsqueeze(self.fc(x2_f), 1)
        return  x1_f-x2_f

class HK_trip_trans(nn.Module):
    """
     Hierarchical Kinship Triplet network
    """
    def __init__(self, num_classes=4, num_channels=3, im_size=64, init_weights=False, dropout=0.3, **kwargs):
        super(HK_trip_trans, self).__init__()

        self.num_classes = num_classes

        self.encoder = attenNet_basic(input_channel=3)

        self.KAT = Transformer( dim=64, depth=1, heads=4, dim_head=64, mlp_dim=32, dropout=0.)

        self.fea = 64

        self.fc = nn.Sequential(nn.LayerNorm(512),
                                nn.Linear(512,self.fea),
                                nn.Tanh() #
                                # nn.GELU()
                                )

        self.tanh = nn.Tanh()

        self.dis_center = [torch.zeros([1]).cuda(),
                           torch.ones([1]).cuda(),
                           4*torch.ones([1]).cuda()]

        if init_weights:
            self._initialize_weights()

        self._init_dis()
        self.cuda()

    def _init_dis(self):

        pass

    def train_forward(self, x1, x2, x3, x4, skip_distance=False):
        # batch_size = len(x1)
        a_f = self.encoder(x1)
        s_f = self.encoder(x2)
        p_f = self.encoder(x3)
        n_f = self.encoder(x4)
        # x1_f = x1_f.view(batch_size, -1)
        # x2_f = x2_f.view(batch_size, -1)
        # x3_f = x3_f.view(batch_size, -1)
        # x4_f = x4_f.view(batch_size, -1)
        a_f = torch.unsqueeze(self.fc(a_f), 1)
        s_f = torch.unsqueeze(self.fc(s_f), 1)
        p_f = torch.unsqueeze(self.fc(p_f), 1)
        n_f = torch.unsqueeze(self.fc(n_f), 1)

        a_f = self.KAT(a_f)
        s_f = self.KAT(s_f)
        p_f = self.KAT(p_f)
        n_f = self.KAT(n_f)

        outLinear = 0

        if skip_distance:
            return outLinear, None

        return outLinear, a_f.squeeze(), s_f.squeeze(), p_f.squeeze(), n_f.squeeze(), self.dis_center


    def forward(self, x,skip_distance=False):
        pass

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            # elif isinstance(m, nn.Linear):
            #     nn.init.normal_(m.weight, 0, 0.01)
            #     nn.init.constant_(m.bias, 0)

    def similarity(self, x):
        batch_size = len(x)
        x1 = x[:, 0:3, :, :]
        x2 = x[:, 3:6, :, :]
        x1_f = self.encoder(x1)
        x2_f = self.encoder(x2)
        x1_f = x1_f.view(batch_size, -1)
        x2_f = x2_f.view(batch_size, -1)
        x1_f = torch.unsqueeze(self.fc(x1_f), 1)
        x2_f = torch.unsqueeze(self.fc(x2_f), 1)
        x1_f = self.KAT(x1_f)
        x2_f = self.KAT(x2_f)

        return nn.PairwiseDistance()(x1_f.squeeze(), x2_f.squeeze())

    def feature(self,x):
        batch_size = len(x)
        x1 = x[:, 0:3, :, :]
        x2 = x[:, 3:6, :, :]
        x1_f = self.encoder(x1)
        x2_f = self.encoder(x2)
        x1_f = x1_f.view(batch_size, -1)
        x2_f = x2_f.view(batch_size, -1)
        x1_f = torch.unsqueeze(self.fc(x1_f), 1)
        x2_f = torch.unsqueeze(self.fc(x2_f), 1)
        return  x1_f-x2_f


# class HK_Facenet_fc(nn.Module):
#     """
#      Hierarchical Kinship Triplet network
#     """
#
#     def __init__(self, num_classes=4, num_channels=3, im_size=64, init_weights=False, dropout=0.3, **kwargs):
#         super(HK_Facenet_fc, self).__init__()
#
#         self.num_classes = num_classes
#
#         self.encoder = InceptionResnetV1(pretrained='vggface2')
#
#         self.freeze(self.encoder)
#         # self.encoder.eval()
#
#         # self.KAT = KAT( dim=64, depth=1, heads=4, dim_head=64, mlp_dim=32, dropout=0.)
#         # self.KAT = Transformer( dim=64, depth=1, heads=4, dim_head=64, mlp_dim=32, dropout=0.)
#
#         self.fea = 64
#
#         self.fc = nn.Sequential(
#             # nn.LayerNorm(512),
#             nn.Linear(512, self.fea),
#             nn.Tanh(),  #
#             nn.LayerNorm(self.fea),
#             nn.Linear(self.fea, self.fea),
#             nn.Tanh()
#             # nn.GELU()
#         )
#
#         self.tanh = nn.Tanh()
#
#         self.dis_center = [torch.zeros([1]).cuda(),
#                            torch.ones([1]).cuda(),
#                            4 * torch.ones([1]).cuda()]
#
#         if init_weights:
#             self._initialize_weights()
#
#         self._init_dis()
#         self.cuda()
#
#     def freeze(self, feat):
#         for name, child in feat.named_children():
#             # if name == 'repeat_3':
#             #     return
#             for param in child.parameters():
#                 param.requires_grad = False
#
#     def _init_dis(self):
#
#         pass
#
#     def train_forward(self, x1, x2, x3, x4, skip_distance=False):
#         # batch_size = len(x1)
#         # self.encoder.eval()
#         a_f = self.encoder(x1)
#         # print(a_f)
#         s_f = self.encoder(x2)
#         p_f = self.encoder(x3)
#         n_f = self.encoder(x4)
#         # x1_f = x1_f.view(batch_size, -1)
#         # x2_f = x2_f.view(batch_size, -1)
#         # x3_f = x3_f.view(batch_size, -1)
#         # x4_f = x4_f.view(batch_size, -1)
#         a_f = self.fc(a_f)
#         s_f = self.fc(s_f)
#         p_f = self.fc(p_f)
#         n_f = self.fc(n_f)
#
#         outLinear = 0
#
#         if skip_distance:
#             return outLinear, None
#
#         return outLinear, a_f, s_f, p_f, n_f, self.dis_center
#
#     def forward(self, x, skip_distance=False):
#         pass
#
#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#             # elif isinstance(m, nn.Linear):
#             #     nn.init.normal_(m.weight, 0, 0.01)
#             #     nn.init.constant_(m.bias, 0)
#
#     def similarity(self, x):
#         self.encoder.eval()
#         batch_size = len(x)
#         x1 = x[:, 0:3, :, :]
#         x2 = x[:, 3:6, :, :]
#         x1_f = self.encoder(x1)
#         x2_f = self.encoder(x2)
#         x1_f = x1_f.view(batch_size, -1)
#         x2_f = x2_f.view(batch_size, -1)
#         x1_f = self.fc(x1_f)
#         x2_f = self.fc(x2_f)
#
#         return nn.PairwiseDistance()(x1_f, x2_f)
#
#     def feature(self, x):
#         batch_size = len(x)
#         x1 = x[:, 0:3, :, :]
#         x2 = x[:, 3:6, :, :]
#         x1_f = self.encoder(x1)
#         x2_f = self.encoder(x2)
#         x1_f = x1_f.view(batch_size, -1)
#         x2_f = x2_f.view(batch_size, -1)
#         x1_f = torch.unsqueeze(self.fc(x1_f), 1)
#         x2_f = torch.unsqueeze(self.fc(x2_f), 1)
#         return x1_f - x2_f

class HK_Facenet_CAT(nn.Module):
    """
     Hierarchical Kinship Triplet network
    """
    def __init__(self, num_classes=4, num_channels=3, im_size=64, init_weights=False, dropout=0.3, **kwargs):
        super(HK_Facenet_CAT, self).__init__()

        self.num_classes = num_classes

        # self.encoder = attenNet_basic(input_channel=3)
        self.encoder = InceptionResnetV1(pretrained='vggface2')

        self.freeze(self.encoder)
        self.encoder.eval()

        self.KAT = KAT_small( dim=64, depth=2, heads=8, dim_head=32, mlp_dim=32, dropout=0.)

        self.fea = 64

        self.fc = nn.Sequential(nn.LayerNorm(512),
                                nn.Linear(512,self.fea),
                                nn.Tanh() #
                                # nn.GELU()
                                )

        self.tanh = nn.Tanh()

        self.dis_center = [torch.zeros([1]).cuda(),
                           torch.ones([1]).cuda(),
                           4*torch.ones([1]).cuda()]

        # if init_weights:
        #     self._initialize_weights()

        self._init_dis()
        self.cuda()

    def _init_dis(self):

        pass

    def freeze(self, feat):
        for name, child in feat.named_children():
            # if name == 'repeat_3':
            #     return
            for param in child.parameters():
                param.requires_grad = False

    def train_forward(self,x1,x2,x3,x4,skip_distance=False):
        # batch_size = len(x1)
        a_f = self.encoder(x1)
        s_f = self.encoder(x2)
        p_f= self.encoder(x3)
        n_f = self.encoder(x4)
        # x1_f = x1_f.view(batch_size, -1)
        # x2_f = x2_f.view(batch_size, -1)
        # x3_f = x3_f.view(batch_size, -1)
        # x4_f = x4_f.view(batch_size, -1)
        a_f = torch.unsqueeze(self.fc(a_f),1)
        s_f = torch.unsqueeze(self.fc(s_f),1)
        p_f = torch.unsqueeze(self.fc(p_f),1)
        n_f = torch.unsqueeze(self.fc(n_f),1)

        (a_asf, s_f1) =self.KAT(a_f,s_f)
        (a_apf, p_f2) = self.KAT(a_f,p_f)
        (a_anf, n_f3) =self.KAT(a_f,n_f)


        outLinear = 0

        if skip_distance:
            return outLinear, None

        return outLinear, (a_asf.squeeze(), s_f1.squeeze()), \
               (a_apf.squeeze(), p_f2.squeeze()), (a_anf.squeeze(), n_f3.squeeze()),self.dis_center


    def forward(self, x,skip_distance=False):
        pass

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            # elif isinstance(m, nn.Linear):
            #     nn.init.normal_(m.weight, 0, 0.01)
            #     nn.init.constant_(m.bias, 0)

    def similarity(self,x):
        batch_size = len(x)
        x1 = x[:, 0:3, :, :]
        x2 = x[:, 3:6, :, :]
        x1_f = self.encoder(x1)
        x2_f = self.encoder(x2)
        x1_f = x1_f.view(batch_size, -1)
        x2_f = x2_f.view(batch_size, -1)
        x1_f = torch.unsqueeze(self.fc(x1_f), 1)
        x2_f = torch.unsqueeze(self.fc(x2_f), 1)
        x1_f,x2_f= self.KAT(x1_f, x2_f)

        return  nn.PairwiseDistance()(x1_f.squeeze(),x2_f.squeeze())

    def feature(self,x):
        batch_size = len(x)
        x1 = x[:, 0:3, :, :]
        x2 = x[:, 3:6, :, :]
        x1_f = self.encoder(x1)
        x2_f = self.encoder(x2)
        x1_f = x1_f.view(batch_size, -1)
        x2_f = x2_f.view(batch_size, -1)
        x1_f = torch.unsqueeze(self.fc(x1_f), 1)
        x2_f = torch.unsqueeze(self.fc(x2_f), 1)
        return  x1_f-x2_f
class HK_Facenet_trans(nn.Module):
    """
     Hierarchical Kinship Triplet network
    """
    def __init__(self, num_classes=4, num_channels=3, im_size=64, init_weights=False, dropout=0.3, **kwargs):
        super(HK_Facenet_trans, self).__init__()

        self.num_classes = num_classes

        self.encoder = InceptionResnetV1(pretrained='vggface2')

        self.freeze(self.encoder)
        # self.encoder.eval()


        # self.KAT = KAT( dim=64, depth=1, heads=4, dim_head=64, mlp_dim=32, dropout=0.)
        self.KAT = Transformer( dim=64, depth=2, heads=8, dim_head=64, mlp_dim=32, dropout=0.)


        self.fea = 64

        self.fc = nn.Sequential(nn.Linear(512,self.fea),
                                nn.Tanh() #
                                # nn.GELU()
                                )

        self.tanh = nn.Tanh()

        self.dis_center = [torch.zeros([1]).cuda(),
                           torch.ones([1]).cuda(),
                           4*torch.ones([1]).cuda()]

        if init_weights:
            self._initialize_weights()

        self._init_dis()
        self.cuda()

    def freeze(self, feat):
        for name, child in feat.named_children():
            # if name == 'repeat_3':
            #     return
            for param in child.parameters():
                param.requires_grad = False

    def _init_dis(self):

        pass

    def train_forward(self,x1,x2,x3,x4,skip_distance=False):
        # batch_size = len(x1)
        a_f = self.encoder(x1)
        s_f = self.encoder(x2)
        p_f= self.encoder(x3)
        n_f = self.encoder(x4)
        # x1_f = x1_f.view(batch_size, -1)
        # x2_f = x2_f.view(batch_size, -1)
        # x3_f = x3_f.view(batch_size, -1)
        # x4_f = x4_f.view(batch_size, -1)
        a_f = torch.unsqueeze(self.fc(a_f),1)
        s_f = torch.unsqueeze(self.fc(s_f),1)
        p_f = torch.unsqueeze(self.fc(p_f),1)
        n_f = torch.unsqueeze(self.fc(n_f),1)

        a_f=self.KAT(a_f)
        s_f = self.KAT(s_f)
        p_f = self.KAT(p_f)
        n_f = self.KAT(n_f)



        outLinear = 0

        if skip_distance:
            return outLinear, None

        return outLinear,a_f.squeeze(), s_f.squeeze(), p_f.squeeze(), n_f.squeeze(),self.dis_center


    def forward(self, x,skip_distance=False):
        pass

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            # elif isinstance(m, nn.Linear):
            #     nn.init.normal_(m.weight, 0, 0.01)
            #     nn.init.constant_(m.bias, 0)

    def similarity(self,x):
        batch_size = len(x)
        x1 = x[:, 0:3, :, :]
        x2 = x[:, 3:6, :, :]
        x1_f = self.encoder(x1)
        x2_f = self.encoder(x2)
        x1_f = x1_f.view(batch_size, -1)
        x2_f = x2_f.view(batch_size, -1)
        x1_f = torch.unsqueeze(self.fc(x1_f), 1)
        x2_f = torch.unsqueeze(self.fc(x2_f), 1)
        x1_f= self.KAT(x1_f)
        x2_f= self.KAT(x2_f)

        return  nn.PairwiseDistance()(x1_f.squeeze(),x2_f.squeeze())

    def feature(self,x):
        batch_size = len(x)
        x1 = x[:, 0:3, :, :]
        x2 = x[:, 3:6, :, :]
        x1_f = self.encoder(x1)
        x2_f = self.encoder(x2)
        x1_f = x1_f.view(batch_size, -1)
        x2_f = x2_f.view(batch_size, -1)
        x1_f = torch.unsqueeze(self.fc(x1_f), 1)
        x2_f = torch.unsqueeze(self.fc(x2_f), 1)
        return  x1_f-x2_f







class HK_trip_CNN_Net(nn.Module):
    """
     Hierarchical Kinship Triplet network
    """
    def __init__(self, num_classes=4, num_channels=3, im_size=64, init_weights=False, dropout=0.3, **kwargs):
        super(HK_trip_CNN_Net, self).__init__()

        self.num_classes = num_classes

        self.encoder = attenNet_basic(input_channel=6)

        self.fea = 64

        self.fc = nn.Sequential(nn.Linear(512,self.fea),
                                nn.ReLU(),
                                nn.Linear(self.fea,3)
                                )

        if init_weights:
            self._initialize_weights()

        self._init_dis()
        self.cuda()

    def _init_dis(self):

        pass

    def train_forward(self,x1,x2,x3,x4,skip_distance=False):
        x_as = torch.cat((x1,x2),dim=1)
        x_ap = torch.cat((x1,x3),dim=1)
        x_an = torch.cat((x1,x4), dim=1)


        batch_size = len(x1)
        x_as = self.encoder(x_as)
        x_ap = self.encoder(x_ap)
        x_an= self.encoder(x_an)
        x_as = x_as.view(batch_size, -1)
        x_ap = x_ap.view(batch_size, -1)
        x_an = x_an.view(batch_size, -1)

        p_as = self.fc(x_as)
        p_ap = self.fc(x_ap)
        p_an = self.fc(x_an)


        outLinear = 0

        if skip_distance:
            return outLinear, None

        return outLinear, p_as, p_ap,p_an


    def forward(self, x,skip_distance=False):
        pass

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def similarity(self,x):
        batch_size = len(x)

        x = self.encoder(x)

        x = x.view(batch_size, -1)


        p = self.fc(x)
        ped = F.softmax(p,dim=1)
        ped = ped[:,2]
        return  ped


class HK_trip_CNN_point_Net(nn.Module):
    """
     Hierarchical Kinship Triplet network
    """
    def __init__(self, num_classes=4, num_channels=3, im_size=64, init_weights=False, dropout=0.3, **kwargs):
        super(HK_trip_CNN_point_Net, self).__init__()

        self.num_classes = num_classes

        self.encoder = CNN_basic()

        self.fea = 3

        self.fc = nn.Sequential(nn.Linear(640,self.fea))

        if init_weights:
            self._initialize_weights()

        self._init_dis()
        self.cuda()

    def _init_dis(self):

        pass

    def train_forward(self,x1,x2,x3,x4,skip_distance=False):
        x_as = torch.cat((x1,x2),dim=1)
        x_ap = torch.cat((x1,x3),dim=1)
        x_an = torch.cat((x1,x4), dim=1)


        batch_size = len(x1)
        x_as = self.encoder(x_as)
        x_ap = self.encoder(x_ap)
        x_an= self.encoder(x_an)
        x_as = x_as.view(batch_size, -1)
        x_ap = x_ap.view(batch_size, -1)
        x_an = x_an.view(batch_size, -1)

        p_as = self.fc(x_as)
        p_ap = self.fc(x_ap)
        p_an = self.fc(x_an)


        outLinear = 0

        if skip_distance:
            return outLinear, None

        return outLinear, p_as, p_ap,p_an


    def forward(self, x,skip_distance=False):
        pass

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def similarity(self,x):
        batch_size = len(x)

        x = self.encoder(x)

        x = x.view(batch_size, -1)


        p = self.fc(x)
        ped = F.softmax(p,dim=1)
        ped = ped[:,2]
        return  ped



class HK_trip_CNN_basic_Net(nn.Module):
    """
     Hierarchical Kinship Triplet network
    """
    def __init__(self, num_classes=4, num_channels=3, im_size=64, init_weights=False, dropout=0.3, **kwargs):
        super(HK_trip_CNN_basic_Net, self).__init__()

        self.num_classes = num_classes

        self.encoder = CNN_point()

        self.fea = 3

        self.fc = nn.Sequential(nn.Linear(6400,self.fea))

        if init_weights:
            self._initialize_weights()

        self._init_dis()
        self.cuda()

    def _init_dis(self):

        pass

    def train_forward(self,x1,x2,x3,x4,skip_distance=False):
        x_as = torch.cat((x1,x2),dim=1)
        x_ap = torch.cat((x1,x3),dim=1)
        x_an = torch.cat((x1,x4), dim=1)


        batch_size = len(x1)
        x_as = self.encoder(x_as)
        x_ap = self.encoder(x_ap)
        x_an= self.encoder(x_an)
        x_as = x_as.view(batch_size, -1)
        x_ap = x_ap.view(batch_size, -1)
        x_an = x_an.view(batch_size, -1)

        p_as = self.fc(x_as)
        p_ap = self.fc(x_ap)
        p_an = self.fc(x_an)


        outLinear = 0

        if skip_distance:
            return outLinear, None

        return outLinear, p_as, p_ap,p_an


    def forward(self, x,skip_distance=False):
        pass

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def similarity(self,x):
        batch_size = len(x)

        x = self.encoder(x)

        x = x.view(batch_size, -1)


        p = self.fc(x)
        ped = F.softmax(p,dim=1)
        ped = ped[:,2]
        return  ped



import math



class res_unit(nn.Module):
    """
    this is the attention module before Residual structure
    """
    def __init__(self,channel,up_size = None):
        """

        :param channel: channels of input feature map
        :param up_size: upsample size
        """
        super(res_unit,self).__init__()
        self.pool = nn.MaxPool2d(2,2)
        self.conv = nn.Conv2d(channel,channel,3,padding=1)
        if up_size == None:
            self.upsample = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False)
        else:
            self.upsample = nn.Upsample(size=(up_size,up_size), mode='bilinear', align_corners=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        identity = x
        x = self.pool(x)
        x = self.conv(x)
        x = self.upsample(x)
        x = self.sigmoid(x)
        x = torch.mul(identity,x)
        return x



class attenNet_basic(nn.Module):
    """
    the attention Module in <Learning part-aware attention networks for kinship verification>
    """
    def __init__(self,input_channel = 6):
        super(attenNet_basic,self).__init__()
        self.conv1 = nn.Conv2d(input_channel,32,5)
        self.conv2 = nn.Conv2d(32,64,5)
        self.conv3 = nn.Conv2d(64,128,5)
        self.at1 = res_unit(32)
        self.at2 = res_unit(64)
        self.at3 = res_unit(128,up_size=9)
        self.pool = nn.MaxPool2d(2,2)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear((9*9*128),512)
        # self.dp  = nn.Dropout()
        # self.fc2 = nn.Linear(512,2)

    def forward(self,x):
        """
        :param x: 6x64x64
        :return:
        """
        x = self.conv1(x)
        identity1 = x
        x = self.at1(x)
        x = identity1+x
        x = self.bn1(x)
        x = self.pool(F.relu(x))
        x = self.conv2(x)
        identity2 = x
        x = self.at2(x)
        x = identity2 + x
        x = self.bn2(x)
        x = self.pool(F.relu((x)))
        x = self.conv3(x)
        identity3 = x
        x = self.at3(x)
        x = identity3 + x
        x = self.bn3(x)
        x = F.relu(x)
        x = x.view(-1, 9*9*128)
        x = F.relu(self.fc1(x))
        # x = self.fc1(x)
        # x = self.dp(x)
        # x = self.fc2(x)
        return x

class CNN_basic(nn.Module):

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(16, 64, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64, 128, kernel_size=5),
        )
        # self.avgpool = nn.AdaptiveAvgPool2d((9, 9))
        self.classifier = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(128 * 9 * 9, 640),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.features(x)
        # x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

import torchvision.transforms as transforms
class CNN_point(nn.Module):

    def __init__(self):
        super().__init__()
        self.net1 = CNN_basic()
        self.net2 = CNN_basic()
        self.net3 = CNN_basic()
        self.net4 = CNN_basic()
        self.net5 = CNN_basic()
        self.net6 = CNN_basic()
        self.net7 = CNN_basic()
        self.net8 = CNN_basic()
        self.net9 = CNN_basic()
        self.net10 = CNN_basic()
        self.fc1 = nn.Linear(10 * 640, 6400)
        self.bn = nn.BatchNorm2d(6400)
        self.act1_f = nn.Sigmoid()
        # self.fc2 = nn.Linear(6400, 2)
        self.tencrop = transforms.TenCrop(39)

    def tencrop(self,x):
        pass


    def forward(self, x):
        x0 = self.net1(x[:, 0, :, :])
        x1 = self.net2(x[:, 1, :, :])
        x2 = self.net3(x[:, 2, :, :])
        x3 = self.net4(x[:, 3, :, :])
        x4 = self.net5(x[:, 4, :, :])
        x5 = self.net6(x[:, 5, :, :])
        x6 = self.net7(x[:, 6, :, :])
        x7 = self.net8(x[:, 7, :, :])
        x8 = self.net9(x[:, 8, :, :])
        x9 = self.net10(x[:, 9, :, :])

        x_cat = torch.cat((x0, x1, x2, x3, x4, x5, x6, x7, x8, x9), dim=-1)
        # print(x_cat.data.shape)
        x_cat = self.act1_f(self.bn(self.fc1(x_cat)))
        # x_cat = self.fc2(x_cat)
        return x_cat


import torchvision.models as models

class CNN(nn.Module):
    """
    this is the attention module before Residual structure
    """
    def __init__(self,input_channel=6):
        """

        :param channel: channels of input feature map
        :param up_size: upsample size
        """
        super(CNN,self).__init__()
        self.conv = nn.Conv2d(input_channel,3,1)
        self.resnet = models.resnet18(pretrained=True)
    def forward(self,x):
        x = self.conv(x)
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        return x


class CosFace(nn.Module):
    def __init__(self, s=64.0, m=0.40):
        super(CosFace, self).__init__()
        self.s = s
        self.m = m

    def forward(self, cosine, label):
        index = torch.where(label != -1)[0]
        m_hot = torch.zeros(index.size()[0], cosine.size()[1], device=cosine.device)
        m_hot.scatter_(1, label[index, None], self.m)
        cosine[index] -= m_hot
        ret = cosine * self.s
        return ret


class ArcFace(nn.Module):
    def __init__(self, s=64.0, m=0.5):
        super(ArcFace, self).__init__()
        self.s = s
        self.m = m

    def forward(self, cosine: torch.Tensor, label):
        index = torch.where(label != -1)[0]
        m_hot = torch.zeros(index.size()[0], cosine.size()[1], device=cosine.device)
        m_hot.scatter_(1, label[index, None], self.m)
        cosine.acos_()
        cosine[index] += m_hot
        cosine.cos_().mul_(self.s)
        return cosine

############################################### added VAE
def load_model(model, pretrained, device):
    weights = torch.load(pretrained, map_location=device)
    model.load_state_dict(weights['model'], strict=False)


class ResidualBlock(nn.Module):
    """
    https://github.com/hhb072/IntroVAE
    Difference: self.bn2 on output and not on (output + identity)
    """

    def __init__(self, inc=64, outc=64, groups=1, scale=1.0):
        super(ResidualBlock, self).__init__()

        midc = int(outc * scale)

        if inc is not outc:
            self.conv_expand = nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=1, stride=1, padding=0,
                                         groups=1, bias=False)
        else:
            self.conv_expand = None

        self.conv1 = nn.Conv2d(in_channels=inc, out_channels=midc, kernel_size=3, stride=1, padding=1, groups=groups,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(midc)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=midc, out_channels=outc, kernel_size=3, stride=1, padding=1, groups=groups,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(outc)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        if self.conv_expand is not None:
            identity_data = self.conv_expand(x)
        else:
            identity_data = x

        output = self.relu1(self.bn1(self.conv1(x)))
        output = self.conv2(output)
        output = self.bn2(output)
        output = self.relu2(torch.add(output, identity_data))
        return output

class Encoder(nn.Module):
    def __init__(self, cdim=3, zdim=512, channels=(64, 128, 256, 512, 512, 512), image_size=256, conditional=False,
                 cond_dim=10):
        super(Encoder, self).__init__()
        self.zdim = zdim
        self.cdim = cdim
        self.image_size = image_size
        self.conditional = conditional
        self.cond_dim = cond_dim
        cc = channels[0]
        self.main = nn.Sequential(
            nn.Conv2d(cdim, cc, 5, 1, 2, bias=False),
            nn.BatchNorm2d(cc),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(2),
        )

        sz = image_size // 2
        for ch in channels[1:]:
            self.main.add_module('res_in_{}'.format(sz), ResidualBlock(cc, ch, scale=1.0))
            self.main.add_module('down_to_{}'.format(sz // 2), nn.AvgPool2d(2))
            cc, sz = ch, sz // 2

        self.main.add_module('res_in_{}'.format(sz), ResidualBlock(cc, cc, scale=1.0))
        self.conv_output_size = self.calc_conv_output_size()
        num_fc_features = torch.zeros(self.conv_output_size).view(-1).shape[0]
        print("conv shape: ", self.conv_output_size)
        print("num fc features: ", num_fc_features)
        if self.conditional:
            self.fc = nn.Linear(num_fc_features + self.cond_dim, 2 * zdim)
        else:
            self.fc = nn.Linear(num_fc_features, 2 * zdim)

    def calc_conv_output_size(self):
        dummy_input = torch.zeros(1, self.cdim, self.image_size, self.image_size)
        dummy_input = self.main(dummy_input)
        return dummy_input[0].shape

    def forward(self, x, o_cond=None):
        y = self.main(x).view(x.size(0), -1)
        if self.conditional and o_cond is not None:
            y = torch.cat([y, o_cond], dim=1)
        y = self.fc(y)
        mu, logvar = y.chunk(2, dim=1)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, cdim=3, zdim=512, channels=(64, 128, 256, 512, 512, 512), image_size=256, conditional=False,
                 conv_input_size=None, cond_dim=10):
        super(Decoder, self).__init__()
        self.cdim = cdim
        self.image_size = image_size
        self.conditional = conditional
        cc = channels[-1]
        self.conv_input_size = conv_input_size
        if conv_input_size is None:
            num_fc_features = cc * 4 * 4
        else:
            num_fc_features = torch.zeros(self.conv_input_size).view(-1).shape[0]
        self.cond_dim = cond_dim
        if self.conditional:
            self.fc = nn.Sequential(
                nn.Linear(zdim + self.cond_dim, num_fc_features),
                nn.ReLU(True),
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(zdim, num_fc_features),
                nn.ReLU(True),
            )

        sz = 4

        self.main = nn.Sequential()
        for ch in channels[::-1]:
            self.main.add_module('res_in_{}'.format(sz), ResidualBlock(cc, ch, scale=1.0))
            self.main.add_module('up_to_{}'.format(sz * 2), nn.Upsample(scale_factor=2, mode='nearest'))
            cc, sz = ch, sz * 2

        self.main.add_module('res_in_{}'.format(sz), ResidualBlock(cc, cc, scale=1.0))
        self.main.add_module('predict', nn.Conv2d(cc, cdim, 5, 1, 2))

    def forward(self, z, y_cond=None):
        z = z.view(z.size(0), -1)
        if self.conditional and y_cond is not None:
            y_cond = y_cond.view(y_cond.size(0), -1)
            z = torch.cat([z, y_cond], dim=1)
        y = self.fc(z)
        y = y.view(z.size(0), *self.conv_input_size)
        y = self.main(y)
        return y

def reparameterize(mu, logvar):
    """
    This function applies the reparameterization trick:
    z = mu(X) + sigma(X)^0.5 * epsilon, where epsilon ~ N(0,I)
    :param mu: mean of x
    :param logvar: log variaance of x
    :return z: the sampled latent variable
    """
    device = mu.device
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std).to(device)
    return mu + eps * std


############################################## loss

class SoftIntroVAE(nn.Module):
    def __init__(self, cdim=3, zdim=128, channels=(64, 128, 256, 512), image_size=64, conditional=False,
                 cond_dim=10):
        super(SoftIntroVAE, self).__init__()

        self.zdim = zdim
        self.conditional = conditional
        self.cond_dim = cond_dim

        self.encoder = Encoder(cdim, zdim, channels, image_size, conditional=conditional, cond_dim=cond_dim)

        self.decoder = Decoder(cdim, zdim, channels, image_size, conditional=conditional,
                               conv_input_size=self.encoder.conv_output_size, cond_dim=cond_dim)

        self.fc = nn.Sequential(
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,64)
        )

    def train_triplet(self,x1,x2,x3):
        x1_f, _ = self.encoder(x1)
        x2_f, _ = self.encoder(x2)
        x3_f, _ = self.encoder(x3)
        x1_f = self.fc(x1_f)
        x2_f = self.fc(x2_f)
        x3_f = self.fc(x3_f)
        return x1_f, x2_f, x3_f

    def similarity(self,x):

        x1 = x[:, 0:3, :, :]
        x2 = x[:, 3:6, :, :]
        x1_f,_ = self.encoder(x1)
        x2_f,_ = self.encoder(x2)

        x1_f = self.fc(x1_f)
        x2_f = self.fc(x2_f)
        return  torch.norm(x1_f-x2_f, 2, 1)

    def forward(self, x, o_cond=None, deterministic=False):
        if self.conditional and o_cond is not None:
            mu, logvar = self.encode(x, o_cond=o_cond)
            if deterministic:
                z = mu
            else:
                z = reparameterize(mu, logvar)
            y = self.decode(z, y_cond=o_cond)
        else:
            mu, logvar = self.encode(x)
            if deterministic:
                z = mu
            else:
                z = reparameterize(mu, logvar)
            y = self.decode(z)
        return mu, logvar, z, y

    def sample(self, z, y_cond=None):
        y = self.decode(z, y_cond=y_cond)
        return y

    def sample_with_noise(self, num_samples=1, device=torch.device("cpu"), y_cond=None):
        z = torch.randn(num_samples, self.z_dim).to(device)
        return self.decode(z, y_cond=y_cond)

    def encode(self, x, o_cond=None):
        if self.conditional and o_cond is not None:
            mu, logvar = self.encoder(x, o_cond=o_cond)
        else:
            mu, logvar = self.encoder(x)
        return mu, logvar

    def decode(self, z, y_cond=None):
        if self.conditional and y_cond is not None:
            y = self.decoder(z, y_cond=y_cond)
        else:
            y = self.decoder(z)
        return y



############################# cac
class openSetClassifier(nn.Module):
    def __init__(self, num_classes=20, num_channels=6, im_size=64, init_weights=False, dropout=0.3, **kwargs):
        super(openSetClassifier, self).__init__()

        self.num_classes = num_classes
        self.encoder = BaseEncoder(num_channels, init_weights, dropout)

        if im_size == 32:
            self.classify = nn.Linear(128 * 4 * 4, num_classes)
        elif im_size == 64:
            self.classify = nn.Linear(128 * 8 * 8, num_classes)
        else:
            print('That image size has not been implemented, sorry.')
            exit()

        self.anchors = nn.Parameter(torch.zeros(self.num_classes, self.num_classes).double(), requires_grad=False)

        if init_weights:
            self._initialize_weights()

        self.cuda()

    def similarity(self,x):
        outputs = self.forward(x)
        # logits = outputs[0]
        distances = outputs[1]
        softmin = F.softmax(-distances,dim=1)
        invScores = 1 - softmin
        scores = distances * invScores
        predicted = torch.min(scores, axis=1)[0]
        return predicted

    def forward(self, x, skip_distance=False):
        batch_size = len(x)

        x = self.encoder(x)
        x = x.view(batch_size, -1)

        outLinear = self.classify(x)

        if skip_distance:
            return outLinear, None

        outDistance = self.distance_classifier(outLinear)

        return outLinear, outDistance

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def set_anchors(self, means):
        self.anchors = nn.Parameter(means.double(), requires_grad=False)
        self.cuda()

    def distance_classifier(self, x):
        ''' Calculates euclidean distance from x to each class anchor
            Returns n x m array of distance from input of batch_size n to anchors of size m
        '''

        n = x.size(0)
        m = self.num_classes
        d = self.num_classes

        x = x.unsqueeze(1).expand(n, m, d).double()
        anchors = self.anchors.unsqueeze(0).expand(n, m, d)
        dists = torch.norm(x - anchors, 2, 2)

        return dists


class BaseEncoder(nn.Module):
    def __init__(self, num_channels, init_weights, dropout=0.3, **kwargs):
        super().__init__()
        self.dropout = nn.Dropout2d(dropout)
        self.relu = nn.LeakyReLU(0.2)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(128)
        self.bn6 = nn.BatchNorm2d(128)

        self.bn7 = nn.BatchNorm2d(128)
        self.bn8 = nn.BatchNorm2d(128)
        self.bn9 = nn.BatchNorm2d(128)

        self.conv1 = nn.Conv2d(num_channels, 64, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.conv3 = nn.Conv2d(64, 128, 3, 2, 1, bias=False)

        self.conv4 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.conv5 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.conv6 = nn.Conv2d(128, 128, 3, 2, 1, bias=False)

        self.conv7 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.conv8 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.conv9 = nn.Conv2d(128, 128, 3, 2, 1, bias=False)

        self.encoder1 = nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu,
            self.conv2,
            self.bn2,
            self.relu,
            self.conv3,
            self.bn3,
            self.relu,
            self.dropout,
        )

        self.encoder2 = nn.Sequential(
            self.conv4,
            self.bn4,
            self.relu,
            self.conv5,
            self.bn5,
            self.relu,
            self.conv6,
            self.bn6,
            self.relu,
            self.dropout,
        )

        self.encoder3 = nn.Sequential(
            self.conv7,
            self.bn7,
            self.relu,
            self.conv8,
            self.bn8,
            self.relu,
            self.conv9,
            self.bn9,
            self.relu,
            self.dropout,

        )

        if init_weights:
            self._initialize_weights()

        self.cuda()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        return x3





# """
#     Network definition for our proposed CAC open set classifier.
#
#     Dimity Miller, 2020
# """
#
#
# import torch
# import torchvision
# import torch.nn as nn
# import  torch.nn.functional as F
# from math import sqrt
# import numpy as np
# class cnn(nn.Module):
#     def __init__(self, num_classes=4, num_channels=3, im_size=64, init_weights=False, dropout=0.3, **kwargs):
#         super(cnn, self).__init__()
#
#         self.num_classes = num_classes
#         self.encoder = attenNet_basic()
#         # self.encoder = CNN()
#         if im_size == 32:
#             self.classify = nn.Linear(512, num_classes)
#         elif im_size == 64:
#             self.classify = nn.Linear(512, num_classes)
#         else:
#             print('That image size has not been implemented, sorry.')
#             exit()
#
#         if init_weights:
#             self._initialize_weights()
#
#         self.cuda()
#
#     def forward(self, x):
#         batch_size = len(x)
#
#         x = self.encoder(x)
#         x = x.view(batch_size, -1)
#
#         outLinear = self.classify(x)
#
#         return outLinear
#
#     def similarity(self,x,close = True):
#         logits = self.forward(x)
#         if close:
#             scores = F.softmax(logits,dim=1)
#             pred = -torch.max(scores,dim=1)[0]
#         else:
#             pred = F.softmax(logits,dim=1)[:,0]
#         return pred
#
#
#     def feature(self, x ):
#         batch_size = len(x)
#         x = self.encoder(x)
#         x = x.view(batch_size, -1)
#         return x
#
#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, 0, 0.01)
#                 nn.init.constant_(m.bias, 0)
#
#
# class openset(nn.Module):
#     def __init__(self, num_classes=7, num_channels=3, im_size=64, init_weights=False, dropout=0.3, **kwargs):
#         super(openset, self).__init__()
#
#         self.num_classes = num_classes
#         self.encoder = attenNet_basic()
#         # self.encoder = CNN()
#         if im_size == 32:
#             self.classify = nn.Linear(512, num_classes)
#         elif im_size == 64:
#             self.classify = nn.Linear(512, num_classes)
#         else:
#             print('That image size has not been implemented, sorry.')
#             exit()
#
#         self.anchors = nn.Parameter(torch.zeros(self.num_classes, self.num_classes).double(), requires_grad=False)
#
#         if init_weights:
#             self._initialize_weights()
#
#
#         self._init_dis()
#         self.cuda()
#
#     def _init_dis(self):
#         pass
#
#     def forward(self, x, skip_distance=False):
#         batch_size = len(x)
#
#         x = self.encoder(x)
#         x = x.view(batch_size, -1)
#
#         outLinear = self.classify(x)
#
#         if skip_distance:
#             return outLinear, None
#
#         outDistance = self.distance_classifier(outLinear)
#
#         return outLinear, outDistance
#
#     def similarity(self,x):
#         outputs = self.forward(x)
#         # logits = outputs[0]
#         distances = outputs[1]
#         softmin = F.softmax(-distances,dim=1)
#         invScores = 1 - softmin
#         scores = distances * invScores
#         predicted = torch.min(scores, axis=1)[0]
#         return predicted
#
#     def feature(self, x ):
#         batch_size = len(x)
#         x = self.encoder(x)
#         x = x.view(batch_size, -1)
#         # x = self.classify(x)
#         return x
#
#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, 0, 0.01)
#                 nn.init.constant_(m.bias, 0)
#
#     def set_anchors(self, means):
#         self.anchors = nn.Parameter(means.double(), requires_grad=False)
#         self.cuda()
#
#     def distance_classifier(self, x):
#         ''' Calculates euclidean distance from x to each class anchor
#             Returns n x m array of distance from input of batch_size n to anchors of size m
#         '''
#
#         n = x.size(0)
#         m = self.num_classes
#         d = self.num_classes
#
#         x = x.unsqueeze(1).expand(n, m, d).double()
#         anchors = self.anchors.unsqueeze(0).expand(n, m, d)
#         dists = torch.norm(x - anchors, 2, 2)
#
#         return dists
#
#
# class HK_trip_tanhNet(nn.Module):
#     """
#      Hierarchical Kinship Triplet network
#     """
#     def __init__(self, num_classes=4, num_channels=3, im_size=64, init_weights=False, dropout=0.3, **kwargs):
#         super(HK_trip_tanhNet, self).__init__()
#
#         self.num_classes = num_classes
#
#         self.encoder = attenNet_basic(input_channel=3)
#
#         # model = SoftIntroVAE(cdim=3, zdim=128, channels=[64, 128, 256, 512], image_size=64)
#         # load_model(model, '/home/wei/Documents/CODE/CVPR2021/openset-kinship/datasets/fiw_soft_intro_betas_0.1_1.0_1.0_model_epoch_100_iter_43400.pth', 'cuda')
#         # self.encoder = model.encoder
#         # for param in self.encoder.parameters():
#         #     param.requires_grad = False
#
#         self.fea = 64
#         # self.fc = Self_Attention(128,self.fea,self.fea)
#         self.fc = nn.Sequential(nn.Linear(512,self.fea),
#                                 nn.Tanh()
#                                 )
#
#         self.classify = nn.Linear(self.fea, num_classes)
#         self.fc_center = nn.Linear(self.fea, num_classes)
#         # self.dis_center = [torch.zeros([1]).cuda(),
#         #                    nn.Parameter(torch.randn(1), requires_grad=True).cuda(),
#         #                    nn.Parameter(torch.randn(1), requires_grad=True).cuda()]
#         self.dis_center = [torch.zeros([1]).cuda(),
#                            torch.ones([1]).cuda(),
#                            2*torch.ones([1]).cuda()]
#         self.centers = nn.Parameter(0.1 * torch.randn(num_classes, self.fea),requires_grad=True)
#         if init_weights:
#             self._initialize_weights()
#
#         self._init_dis()
#         self.cuda()
#
#     def _init_dis(self):
#         # self.neg_axis = nn.Parameter(-torch.ones(1, self.anchors.shape[0]).double(), requires_grad=False)
#         # self.cos = nn.CosineSimilarity(1)
#         # self.cos_pos = nn.CosineSimilarity(2)
#         pass
#
#     def train_forward(self,x1,x2,x3,x4,skip_distance=False):
#         batch_size = len(x1)
#         x1_f = self.encoder(x1)
#         x2_f = self.encoder(x2)
#         x3_f= self.encoder(x3)
#         x4_f = self.encoder(x4)
#         x1_f = x1_f.view(batch_size, -1)
#         x2_f = x2_f.view(batch_size, -1)
#         x3_f = x3_f.view(batch_size, -1)
#         x4_f = x4_f.view(batch_size, -1)
#         x1_f = self.fc(x1_f)
#         x2_f = self.fc(x2_f)
#         x3_f = self.fc(x3_f)
#         x4_f = self.fc(x4_f)
#
#         outLinear = 0
#
#         if skip_distance:
#             return outLinear, None
#
#         return outLinear, x1_f, x2_f,x3_f,x4_f,self.dis_center,self.centers
#
#
#     def forward(self, x,skip_distance=False):
#         pass
#
#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, 0, 0.01)
#                 nn.init.constant_(m.bias, 0)
#
#     def similarity(self,x):
#         batch_size = len(x)
#         x1 = x[:, 0:3, :, :]
#         x2 = x[:, 3:6, :, :]
#         x1_f = self.encoder(x1)
#         x2_f = self.encoder(x2)
#         x1_f = x1_f.view(batch_size, -1)
#         x2_f = x2_f.view(batch_size, -1)
#
#         x1_f = self.fc(x1_f)
#         x2_f = self.fc(x2_f)
#         return  nn.PairwiseDistance()(x1_f,x2_f)
#
# class HK_trip_tanh_deep_Net(nn.Module):
#     """
#      Hierarchical Kinship Triplet network
#     """
#     def __init__(self, num_classes=4, num_channels=3, im_size=64, init_weights=False, dropout=0.3, **kwargs):
#         super(HK_trip_tanh_deep_Net, self).__init__()
#
#         self.num_classes = num_classes
#
#         self.encoder = attenNet_basic(input_channel=3)
#
#         # model = SoftIntroVAE(cdim=3, zdim=128, channels=[64, 128, 256, 512], image_size=64)
#         # load_model(model, '/home/wei/Documents/CODE/CVPR2021/openset-kinship/datasets/fiw_soft_intro_betas_0.1_1.0_1.0_model_epoch_100_iter_43400.pth', 'cuda')
#         # self.encoder = model.encoder
#         # for param in self.encoder.parameters():
#         #     param.requires_grad = False
#
#         self.fea = 64
#         # self.fc = Self_Attention(128,self.fea,self.fea)
#         self.fea = 64
#         self.q = nn.Linear(2, 2)
#         self.k = nn.Linear(2, 2)
#         self.v = nn.Linear(512, 64)
#
#         self.classify = nn.Linear(self.fea, num_classes)
#         self.fc_center = nn.Linear(self.fea, num_classes)
#         # self.dis_center = [torch.zeros([1]).cuda(),
#         #                    nn.Parameter(torch.randn(1), requires_grad=True).cuda(),
#         #                    nn.Parameter(torch.randn(1), requires_grad=True).cuda()]
#         self.dis_center = [torch.zeros([1]).cuda(),
#                            torch.ones([1]).cuda(),
#                            2*torch.ones([1]).cuda()]
#         # self.centers = nn.Parameter(0.1 * torch.randn(num_classes, self.fea),requires_grad=True)
#         if init_weights:
#             self._initialize_weights()
#
#         self._init_dis()
#         self.cuda()
#
#     def _init_dis(self):
#         pass
#
#     def transform(self, x1, x2):
#         # x1 batch*64*1
#         x1 = self.v(x1).unsqueeze(dim=2)  # batch*64*1
#         x2 = self.v(x2).unsqueeze(dim=2)  # batch*64*1
#
#         com = torch.cat((x1, x2), dim=2)  # batch*64*2
#
#         Q = self.q(com)  # batch*64
#         K = self.k(com)
#
#         atten = nn.Softmax(dim=-1)(torch.bmm(Q, K.permute(0, 2, 1)))  # batch*64*64
#
#         x1 = torch.bmm(atten, x1)
#         x2 = torch.bmm(atten, x2)
#         return x1.squeeze(), x2.squeeze()
#
#     def train_forward(self,x1,x2,x3,x4,skip_distance=False):
#         batch_size = len(x1)
#         a = self.encoder(x1)
#         s = self.encoder(x2)
#         p= self.encoder(x3)
#         n = self.encoder(x4)
#
#         a = a.view(batch_size, -1)
#         s = s.view(batch_size, -1)
#         p = p.view(batch_size, -1)
#         n = n.view(batch_size, -1)
#
#         a = nn.Tanh()(a)
#         s = nn.Tanh()(s)
#         p = nn.Tanh()(p)
#         n = nn.Tanh()(n)
#         a1,s1 = self.transform(a,s)
#         a2,p2 = self.transform(a,p)
#         a3,n3 = self.transform(a,n)
#
#         outLinear = 0
#
#         if skip_distance:
#             return outLinear, None
#
#         return outLinear, a1,s1, a2,p2,a3,n3,self.dis_center
#
#
#     def forward(self, x,skip_distance=False):
#         pass
#
#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, 0, 0.01)
#                 nn.init.constant_(m.bias, 0)
#
#     def similarity(self,x):
#         batch_size = len(x)
#         x1 = x[:, 0:3, :, :]
#         x2 = x[:, 3:6, :, :]
#         x1_f = self.encoder(x1)
#         x2_f = self.encoder(x2)
#         x1_f = x1_f.view(batch_size, -1)
#         x2_f = x2_f.view(batch_size, -1)
#         # x1_f = self.fc(x1_f)
#         # x2_f = self.fc(x2_f)
#         Px1, Px2 = self.transform(x1_f, x2_f)
#         return  nn.PairwiseDistance()(Px1,Px2)
#
# class HK_trip_CNN_Net(nn.Module):
#     """
#      Hierarchical Kinship Triplet network
#     """
#     def __init__(self, num_classes=4, num_channels=3, im_size=64, init_weights=False, dropout=0.3, **kwargs):
#         super(HK_trip_CNN_Net, self).__init__()
#
#         self.num_classes = num_classes
#
#         self.encoder = attenNet_basic(input_channel=6)
#
#         self.fea = 64
#
#         self.fc = nn.Sequential(nn.Linear(512,self.fea),
#                                 nn.ReLU(),
#                                 nn.Linear(self.fea,3)
#                                 )
#
#         if init_weights:
#             self._initialize_weights()
#
#         self._init_dis()
#         self.cuda()
#
#     def _init_dis(self):
#
#         pass
#
#     def train_forward(self,x1,x2,x3,x4,skip_distance=False):
#         x_as = torch.cat((x1,x2),dim=1)
#         x_ap = torch.cat((x1,x3),dim=1)
#         x_an = torch.cat((x1,x4), dim=1)
#
#
#         batch_size = len(x1)
#         x_as = self.encoder(x_as)
#         x_ap = self.encoder(x_ap)
#         x_an= self.encoder(x_an)
#         x_as = x_as.view(batch_size, -1)
#         x_ap = x_ap.view(batch_size, -1)
#         x_an = x_an.view(batch_size, -1)
#
#         p_as = self.fc(x_as)
#         p_ap = self.fc(x_ap)
#         p_an = self.fc(x_an)
#
#
#         outLinear = 0
#
#         if skip_distance:
#             return outLinear, None
#
#         return outLinear, p_as, p_ap,p_an
#
#
#     def forward(self, x,skip_distance=False):
#         pass
#
#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, 0, 0.01)
#                 nn.init.constant_(m.bias, 0)
#
#     def similarity(self,x):
#         batch_size = len(x)
#
#         x = self.encoder(x)
#
#         x = x.view(batch_size, -1)
#
#
#         p = self.fc(x)
#         ped = F.softmax(p,dim=1)
#         ped = ped[:,2]
#         return  ped
#
# class HK_trip_tanh_vae_Net(nn.Module):
#     """
#      Hierarchical Kinship Triplet network
#     """
#     def __init__(self, num_classes=4, num_channels=3, im_size=64, init_weights=False, dropout=0.3, **kwargs):
#         super(HK_trip_tanh_vae_Net, self).__init__()
#
#         self.num_classes = num_classes
#
#         # self.encoder = attenNet_basic(input_channel=3)
#
#         model = SoftIntroVAE(cdim=3, zdim=128, channels=[64, 128, 256, 512], image_size=64)
#         load_model(model, '/home/wei/Documents/CODE/CVPR2021/openset-kinship/datasets/fiw_soft_intro_betas_0.1_1.0_1.0_model_epoch_100_iter_43400.pth', 'cuda')
#         self.encoder = model.encoder
#         for param in self.encoder.parameters():
#             param.requires_grad = False
#
#         self.fea = 64
#         # self.fc = Self_Attention(128,self.fea,self.fea)
#         self.fc = nn.Sequential(nn.Linear(128,self.fea),
#                                 nn.ReLU(),
#                                 nn.Linear(self.fea, self.fea),
#                                 nn.ReLU(),
#                                 nn.Linear(self.fea, self.fea),
#                                 nn.Tanh()
#                                 )
#
#         self.classify = nn.Linear(self.fea, num_classes)
#         self.fc_center = nn.Linear(self.fea, num_classes)
#         # self.dis_center = [torch.zeros([1]).cuda(),
#         #                    nn.Parameter(torch.randn(1), requires_grad=True).cuda(),
#         #                    nn.Parameter(torch.randn(1), requires_grad=True).cuda()]
#         self.dis_center = [torch.zeros([1]).cuda(),
#                            torch.ones([1]).cuda(),
#                            2*torch.ones([1]).cuda()]
#         self.centers = nn.Parameter(0.1 * torch.randn(num_classes, self.fea),requires_grad=True)
#         if init_weights:
#             self._initialize_weights()
#
#         self._init_dis()
#         self.cuda()
#
#     def _init_dis(self):
#         # self.neg_axis = nn.Parameter(-torch.ones(1, self.anchors.shape[0]).double(), requires_grad=False)
#         # self.cos = nn.CosineSimilarity(1)
#         # self.cos_pos = nn.CosineSimilarity(2)
#         pass
#
#     def train_forward(self,x1,x2,x3,x4,skip_distance=False):
#         batch_size = len(x1)
#         x1_f,_ = self.encoder(x1)
#         x2_f,_ = self.encoder(x2)
#         x3_f,_= self.encoder(x3)
#         x4_f,_ = self.encoder(x4)
#         x1_f = x1_f.view(batch_size, -1)
#         x2_f = x2_f.view(batch_size, -1)
#         x3_f = x3_f.view(batch_size, -1)
#         x4_f = x4_f.view(batch_size, -1)
#         x1_f = self.fc(x1_f)
#         x2_f = self.fc(x2_f)
#         x3_f = self.fc(x3_f)
#         x4_f = self.fc(x4_f)
#
#         outLinear = 0
#
#         if skip_distance:
#             return outLinear, None
#
#         return outLinear, x1_f, x2_f,x3_f,x4_f,self.dis_center,self.centers
#
#
#     def forward(self, x,skip_distance=False):
#         pass
#
#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, 0, 0.01)
#                 nn.init.constant_(m.bias, 0)
#
#     def similarity(self,x):
#         batch_size = len(x)
#         x1 = x[:, 0:3, :, :]
#         x2 = x[:, 3:6, :, :]
#         x1_f,_ = self.encoder(x1)
#         x2_f,_ = self.encoder(x2)
#         x1_f = x1_f.view(batch_size, -1)
#         x2_f = x2_f.view(batch_size, -1)
#
#         x1_f = self.fc(x1_f)
#         x2_f = self.fc(x2_f)
#         return  nn.PairwiseDistance()(x1_f,x2_f)
#
#
# import math
# class arc_HK_tripNet(nn.Module):
#     """
#      Hierachical Kinship Triplet network
#     """
#     def __init__(self, num_classes=4, num_channels=3, im_size=64, init_weights=False, dropout=0.3, **kwargs):
#         super(arc_HK_tripNet, self).__init__()
#
#         self.num_classes = num_classes
#
#         self.encoder = attenNet_basic(input_channel=3)
#
#         # model = SoftIntroVAE(cdim=3, zdim=128, channels=[64, 128, 256, 512], image_size=64)
#         # load_model(model, '/home/wei/Documents/CODE/CVPR2021/openset-kinship/datasets/fiw_soft_intro_betas_0.1_1.0_1.0_model_epoch_100_iter_43400.pth', 'cuda')
#         # self.encoder = model.encoder
#         self.fea = 64
#         # self.fc = Self_Attention(128,self.fea,self.fea)
#         self.fc = nn.Sequential(nn.Linear(512,self.fea),
#                                 nn.Tanh()
#                                 )
#
#
#         self.classify = nn.Linear(self.fea, num_classes)
#         self.fc_center = nn.Linear(self.fea, num_classes)
#
#         self.dis_center = [torch.zeros([1]).cuda(),
#                            torch.ones([1]).cuda(),
#                            2*torch.ones([1]).cuda()]
#         self.centers = nn.Parameter(0.1 * torch.randn(num_classes, self.fea), requires_grad=True)
#         ######## arc
#
#         in_features = 128
#         out_features = 7
#         self.s = 30.0
#         m = 0.5
#         self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
#         nn.init.xavier_uniform_(self.weight)
#
#         self.easy_margin = False
#         self.cos_m = math.cos(m)
#         self.sin_m = math.sin(m)
#         self.th = math.cos(math.pi - m)
#         self.mm = math.sin(math.pi - m) * m
#
#
#         if init_weights:
#             self._initialize_weights()
#
#         self._init_dis()
#         self.cuda()
#
#
#     def arc(self,input, label):
#         # --------------------------- cos(theta) & phi(theta) ---------------------------
#         cosine = F.linear(F.normalize(input), F.normalize(self.weight))
#         sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
#         phi = cosine * self.cos_m - sine * self.sin_m
#         if self.easy_margin:
#             phi = torch.where(cosine > 0, phi, cosine)
#         else:
#             phi = torch.where(cosine > self.th, phi, cosine - self.mm)
#         # --------------------------- convert label to one-hot ---------------------------
#         # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
#         one_hot = torch.zeros(cosine.size(), device='cuda')
#         one_hot.scatter_(1, label.view(-1, 1).long(), 1)
#         # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
#         output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
#         output *= self.s
#         # print(output)
#
#         return output
#
#     def _init_dis(self):
#         # self.neg_axis = nn.Parameter(-torch.ones(1, self.anchors.shape[0]).double(), requires_grad=False)
#         # self.cos = nn.CosineSimilarity(1)
#         # self.cos_pos = nn.CosineSimilarity(2)
#         pass
#
#     def train_forward(self,x1,x2,x3,x4,lb):
#         batch_size = len(x1)
#         x1_f = self.encoder(x1)
#         x2_f = self.encoder(x2)
#         x3_f= self.encoder(x3)
#         x4_f = self.encoder(x4)
#         x1_f = x1_f.view(batch_size, -1)
#         x2_f = x2_f.view(batch_size, -1)
#         x3_f = x3_f.view(batch_size, -1)
#         x4_f = x4_f.view(batch_size, -1)
#         x1_f = self.fc(x1_f)
#         x2_f = self.fc(x2_f)
#         x3_f = self.fc(x3_f)
#         x4_f = self.fc(x4_f)
#
#
#         ########## arc
#         kin_feature = torch.cat((x1_f,x3_f),dim=1)
#         logit = self.arc(kin_feature,lb)
#
#
#         outLinear = 0
#
#         return outLinear, x1_f, x2_f,x3_f,x4_f,self.dis_center,logit
#
#
#     def forward(self, x,skip_distance=False):
#         pass
#
#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, 0, 0.01)
#                 nn.init.constant_(m.bias, 0)
#
#     def similarity(self,x):
#         batch_size = len(x)
#         x1 = x[:, 0:3, :, :]
#         x2 = x[:, 3:6, :, :]
#         x1_f = self.encoder(x1)
#         x2_f = self.encoder(x2)
#         x1_f = x1_f.view(batch_size, -1)
#         x2_f = x2_f.view(batch_size, -1)
#
#         x1_f = self.fc(x1_f)
#         x2_f = self.fc(x2_f)
#         return  nn.PairwiseDistance()(x1_f,x2_f)
#
#
#
# class res_unit(nn.Module):
#     """
#     this is the attention module before Residual structure
#     """
#     def __init__(self,channel,up_size = None):
#         """
#
#         :param channel: channels of input feature map
#         :param up_size: upsample size
#         """
#         super(res_unit,self).__init__()
#         self.pool = nn.MaxPool2d(2,2)
#         self.conv = nn.Conv2d(channel,channel,3,padding=1)
#         if up_size == None:
#             self.upsample = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False)
#         else:
#             self.upsample = nn.Upsample(size=(up_size,up_size), mode='bilinear', align_corners=False)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self,x):
#         identity = x
#         x = self.pool(x)
#         x = self.conv(x)
#         x = self.upsample(x)
#         x = self.sigmoid(x)
#         x = torch.mul(identity,x)
#         return x
#
#
#
# class attenNet_basic(nn.Module):
#     """
#     the attention Module in <Learning part-aware attention networks for kinship verification>
#     """
#     def __init__(self,input_channel = 6):
#         super(attenNet_basic,self).__init__()
#         self.conv1 = nn.Conv2d(input_channel,32,5)
#         self.conv2 = nn.Conv2d(32,64,5)
#         self.conv3 = nn.Conv2d(64,128,5)
#         self.at1 = res_unit(32)
#         self.at2 = res_unit(64)
#         self.at3 = res_unit(128,up_size=9)
#         self.pool = nn.MaxPool2d(2,2)
#         self.bn1 = nn.BatchNorm2d(32)
#         self.bn2 = nn.BatchNorm2d(64)
#         self.bn3 = nn.BatchNorm2d(128)
#         self.fc1 = nn.Linear((9*9*128),512)
#         # self.dp  = nn.Dropout()
#         # self.fc2 = nn.Linear(512,2)
#
#     def forward(self,x):
#         """
#         :param x: 6x64x64
#         :return:
#         """
#         x = self.conv1(x)
#         identity1 = x
#         x = self.at1(x)
#         x = identity1+x
#         x = self.bn1(x)
#         x = self.pool(F.relu(x))
#         x = self.conv2(x)
#         identity2 = x
#         x = self.at2(x)
#         x = identity2 + x
#         x = self.bn2(x)
#         x = self.pool(F.relu((x)))
#         x = self.conv3(x)
#         identity3 = x
#         x = self.at3(x)
#         x = identity3 + x
#         x = self.bn3(x)
#         x = F.relu(x)
#         x = x.view(-1, 9*9*128)
#         x = F.relu(self.fc1(x))
#         # x = self.fc1(x)
#         # x = self.dp(x)
#         # x = self.fc2(x)
#         return x
#
#
#
# import torchvision.models as models
#
# class CNN(nn.Module):
#     """
#     this is the attention module before Residual structure
#     """
#     def __init__(self,input_channel=6):
#         """
#
#         :param channel: channels of input feature map
#         :param up_size: upsample size
#         """
#         super(CNN,self).__init__()
#         self.conv = nn.Conv2d(input_channel,3,1)
#         self.resnet = models.resnet18(pretrained=True)
#     def forward(self,x):
#         x = self.conv(x)
#         x = self.resnet.conv1(x)
#         x = self.resnet.bn1(x)
#         x = self.resnet.relu(x)
#         x = self.resnet.maxpool(x)
#
#         x = self.resnet.layer1(x)
#         x = self.resnet.layer2(x)
#         x = self.resnet.layer3(x)
#         x = self.resnet.layer4(x)
#
#         x = self.resnet.avgpool(x)
#         x = torch.flatten(x, 1)
#         return x
#
#
# class CosFace(nn.Module):
#     def __init__(self, s=64.0, m=0.40):
#         super(CosFace, self).__init__()
#         self.s = s
#         self.m = m
#
#     def forward(self, cosine, label):
#         index = torch.where(label != -1)[0]
#         m_hot = torch.zeros(index.size()[0], cosine.size()[1], device=cosine.device)
#         m_hot.scatter_(1, label[index, None], self.m)
#         cosine[index] -= m_hot
#         ret = cosine * self.s
#         return ret
#
#
# class ArcFace(nn.Module):
#     def __init__(self, s=64.0, m=0.5):
#         super(ArcFace, self).__init__()
#         self.s = s
#         self.m = m
#
#     def forward(self, cosine: torch.Tensor, label):
#         index = torch.where(label != -1)[0]
#         m_hot = torch.zeros(index.size()[0], cosine.size()[1], device=cosine.device)
#         m_hot.scatter_(1, label[index, None], self.m)
#         cosine.acos_()
#         cosine[index] += m_hot
#         cosine.cos_().mul_(self.s)
#         return cosine
#
# ############################################### added VAE
# def load_model(model, pretrained, device):
#     weights = torch.load(pretrained, map_location=device)
#     model.load_state_dict(weights['model'], strict=False)
#
#
# class ResidualBlock(nn.Module):
#     """
#     https://github.com/hhb072/IntroVAE
#     Difference: self.bn2 on output and not on (output + identity)
#     """
#
#     def __init__(self, inc=64, outc=64, groups=1, scale=1.0):
#         super(ResidualBlock, self).__init__()
#
#         midc = int(outc * scale)
#
#         if inc is not outc:
#             self.conv_expand = nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=1, stride=1, padding=0,
#                                          groups=1, bias=False)
#         else:
#             self.conv_expand = None
#
#         self.conv1 = nn.Conv2d(in_channels=inc, out_channels=midc, kernel_size=3, stride=1, padding=1, groups=groups,
#                                bias=False)
#         self.bn1 = nn.BatchNorm2d(midc)
#         self.relu1 = nn.LeakyReLU(0.2, inplace=True)
#         self.conv2 = nn.Conv2d(in_channels=midc, out_channels=outc, kernel_size=3, stride=1, padding=1, groups=groups,
#                                bias=False)
#         self.bn2 = nn.BatchNorm2d(outc)
#         self.relu2 = nn.LeakyReLU(0.2, inplace=True)
#
#     def forward(self, x):
#         if self.conv_expand is not None:
#             identity_data = self.conv_expand(x)
#         else:
#             identity_data = x
#
#         output = self.relu1(self.bn1(self.conv1(x)))
#         output = self.conv2(output)
#         output = self.bn2(output)
#         output = self.relu2(torch.add(output, identity_data))
#         return output
#
# class Encoder(nn.Module):
#     def __init__(self, cdim=3, zdim=512, channels=(64, 128, 256, 512, 512, 512), image_size=256, conditional=False,
#                  cond_dim=10):
#         super(Encoder, self).__init__()
#         self.zdim = zdim
#         self.cdim = cdim
#         self.image_size = image_size
#         self.conditional = conditional
#         self.cond_dim = cond_dim
#         cc = channels[0]
#         self.main = nn.Sequential(
#             nn.Conv2d(cdim, cc, 5, 1, 2, bias=False),
#             nn.BatchNorm2d(cc),
#             nn.LeakyReLU(0.2),
#             nn.AvgPool2d(2),
#         )
#
#         sz = image_size // 2
#         for ch in channels[1:]:
#             self.main.add_module('res_in_{}'.format(sz), ResidualBlock(cc, ch, scale=1.0))
#             self.main.add_module('down_to_{}'.format(sz // 2), nn.AvgPool2d(2))
#             cc, sz = ch, sz // 2
#
#         self.main.add_module('res_in_{}'.format(sz), ResidualBlock(cc, cc, scale=1.0))
#         self.conv_output_size = self.calc_conv_output_size()
#         num_fc_features = torch.zeros(self.conv_output_size).view(-1).shape[0]
#         print("conv shape: ", self.conv_output_size)
#         print("num fc features: ", num_fc_features)
#         if self.conditional:
#             self.fc = nn.Linear(num_fc_features + self.cond_dim, 2 * zdim)
#         else:
#             self.fc = nn.Linear(num_fc_features, 2 * zdim)
#
#     def calc_conv_output_size(self):
#         dummy_input = torch.zeros(1, self.cdim, self.image_size, self.image_size)
#         dummy_input = self.main(dummy_input)
#         return dummy_input[0].shape
#
#     def forward(self, x, o_cond=None):
#         y = self.main(x).view(x.size(0), -1)
#         if self.conditional and o_cond is not None:
#             y = torch.cat([y, o_cond], dim=1)
#         y = self.fc(y)
#         mu, logvar = y.chunk(2, dim=1)
#         return mu, logvar
#
#
# class Decoder(nn.Module):
#     def __init__(self, cdim=3, zdim=512, channels=(64, 128, 256, 512, 512, 512), image_size=256, conditional=False,
#                  conv_input_size=None, cond_dim=10):
#         super(Decoder, self).__init__()
#         self.cdim = cdim
#         self.image_size = image_size
#         self.conditional = conditional
#         cc = channels[-1]
#         self.conv_input_size = conv_input_size
#         if conv_input_size is None:
#             num_fc_features = cc * 4 * 4
#         else:
#             num_fc_features = torch.zeros(self.conv_input_size).view(-1).shape[0]
#         self.cond_dim = cond_dim
#         if self.conditional:
#             self.fc = nn.Sequential(
#                 nn.Linear(zdim + self.cond_dim, num_fc_features),
#                 nn.ReLU(True),
#             )
#         else:
#             self.fc = nn.Sequential(
#                 nn.Linear(zdim, num_fc_features),
#                 nn.ReLU(True),
#             )
#
#         sz = 4
#
#         self.main = nn.Sequential()
#         for ch in channels[::-1]:
#             self.main.add_module('res_in_{}'.format(sz), ResidualBlock(cc, ch, scale=1.0))
#             self.main.add_module('up_to_{}'.format(sz * 2), nn.Upsample(scale_factor=2, mode='nearest'))
#             cc, sz = ch, sz * 2
#
#         self.main.add_module('res_in_{}'.format(sz), ResidualBlock(cc, cc, scale=1.0))
#         self.main.add_module('predict', nn.Conv2d(cc, cdim, 5, 1, 2))
#
#     def forward(self, z, y_cond=None):
#         z = z.view(z.size(0), -1)
#         if self.conditional and y_cond is not None:
#             y_cond = y_cond.view(y_cond.size(0), -1)
#             z = torch.cat([z, y_cond], dim=1)
#         y = self.fc(z)
#         y = y.view(z.size(0), *self.conv_input_size)
#         y = self.main(y)
#         return y
#
# def reparameterize(mu, logvar):
#     """
#     This function applies the reparameterization trick:
#     z = mu(X) + sigma(X)^0.5 * epsilon, where epsilon ~ N(0,I)
#     :param mu: mean of x
#     :param logvar: log variaance of x
#     :return z: the sampled latent variable
#     """
#     device = mu.device
#     std = torch.exp(0.5 * logvar)
#     eps = torch.randn_like(std).to(device)
#     return mu + eps * std
#
#
# ############################################## loss
#
# class SoftIntroVAE(nn.Module):
#     def __init__(self, cdim=3, zdim=128, channels=(64, 128, 256, 512), image_size=64, conditional=False,
#                  cond_dim=10):
#         super(SoftIntroVAE, self).__init__()
#
#         self.zdim = zdim
#         self.conditional = conditional
#         self.cond_dim = cond_dim
#
#         self.encoder = Encoder(cdim, zdim, channels, image_size, conditional=conditional, cond_dim=cond_dim)
#
#         self.decoder = Decoder(cdim, zdim, channels, image_size, conditional=conditional,
#                                conv_input_size=self.encoder.conv_output_size, cond_dim=cond_dim)
#
#         self.fc = nn.Sequential(
#             nn.Linear(128,64),
#             nn.ReLU(),
#             nn.Linear(64,64)
#         )
#
#     def train_triplet(self,x1,x2,x3):
#         x1_f, _ = self.encoder(x1)
#         x2_f, _ = self.encoder(x2)
#         x3_f, _ = self.encoder(x3)
#         x1_f = self.fc(x1_f)
#         x2_f = self.fc(x2_f)
#         x3_f = self.fc(x3_f)
#         return x1_f, x2_f, x3_f
#
#     def similarity(self,x):
#
#         x1 = x[:, 0:3, :, :]
#         x2 = x[:, 3:6, :, :]
#         x1_f,_ = self.encoder(x1)
#         x2_f,_ = self.encoder(x2)
#
#         x1_f = self.fc(x1_f)
#         x2_f = self.fc(x2_f)
#         return  torch.norm(x1_f-x2_f, 2, 1)
#
#     def forward(self, x, o_cond=None, deterministic=False):
#         if self.conditional and o_cond is not None:
#             mu, logvar = self.encode(x, o_cond=o_cond)
#             if deterministic:
#                 z = mu
#             else:
#                 z = reparameterize(mu, logvar)
#             y = self.decode(z, y_cond=o_cond)
#         else:
#             mu, logvar = self.encode(x)
#             if deterministic:
#                 z = mu
#             else:
#                 z = reparameterize(mu, logvar)
#             y = self.decode(z)
#         return mu, logvar, z, y
#
#     def sample(self, z, y_cond=None):
#         y = self.decode(z, y_cond=y_cond)
#         return y
#
#     def sample_with_noise(self, num_samples=1, device=torch.device("cpu"), y_cond=None):
#         z = torch.randn(num_samples, self.z_dim).to(device)
#         return self.decode(z, y_cond=y_cond)
#
#     def encode(self, x, o_cond=None):
#         if self.conditional and o_cond is not None:
#             mu, logvar = self.encoder(x, o_cond=o_cond)
#         else:
#             mu, logvar = self.encoder(x)
#         return mu, logvar
#
#     def decode(self, z, y_cond=None):
#         if self.conditional and y_cond is not None:
#             y = self.decoder(z, y_cond=y_cond)
#         else:
#             y = self.decoder(z)
#         return y
#
#
#
# ############################# cac
# class openSetClassifier(nn.Module):
#     def __init__(self, num_classes=20, num_channels=6, im_size=64, init_weights=False, dropout=0.3, **kwargs):
#         super(openSetClassifier, self).__init__()
#
#         self.num_classes = num_classes
#         self.encoder = BaseEncoder(num_channels, init_weights, dropout)
#
#         if im_size == 32:
#             self.classify = nn.Linear(128 * 4 * 4, num_classes)
#         elif im_size == 64:
#             self.classify = nn.Linear(128 * 8 * 8, num_classes)
#         else:
#             print('That image size has not been implemented, sorry.')
#             exit()
#
#         self.anchors = nn.Parameter(torch.zeros(self.num_classes, self.num_classes).double(), requires_grad=False)
#
#         if init_weights:
#             self._initialize_weights()
#
#         self.cuda()
#
#     def forward(self, x, skip_distance=False):
#         batch_size = len(x)
#
#         x = self.encoder(x)
#         x = x.view(batch_size, -1)
#
#         outLinear = self.classify(x)
#
#         if skip_distance:
#             return outLinear, None
#
#         outDistance = self.distance_classifier(outLinear)
#
#         return outLinear, outDistance
#
#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, 0, 0.01)
#                 nn.init.constant_(m.bias, 0)
#
#     def set_anchors(self, means):
#         self.anchors = nn.Parameter(means.double(), requires_grad=False)
#         self.cuda()
#
#     def distance_classifier(self, x):
#         ''' Calculates euclidean distance from x to each class anchor
#             Returns n x m array of distance from input of batch_size n to anchors of size m
#         '''
#
#         n = x.size(0)
#         m = self.num_classes
#         d = self.num_classes
#
#         x = x.unsqueeze(1).expand(n, m, d).double()
#         anchors = self.anchors.unsqueeze(0).expand(n, m, d)
#         dists = torch.norm(x - anchors, 2, 2)
#
#         return dists
#
#
# class BaseEncoder(nn.Module):
#     def __init__(self, num_channels, init_weights, dropout=0.3, **kwargs):
#         super().__init__()
#         self.dropout = nn.Dropout2d(dropout)
#         self.relu = nn.LeakyReLU(0.2)
#
#         self.bn1 = nn.BatchNorm2d(64)
#         self.bn2 = nn.BatchNorm2d(64)
#         self.bn3 = nn.BatchNorm2d(128)
#
#         self.bn4 = nn.BatchNorm2d(128)
#         self.bn5 = nn.BatchNorm2d(128)
#         self.bn6 = nn.BatchNorm2d(128)
#
#         self.bn7 = nn.BatchNorm2d(128)
#         self.bn8 = nn.BatchNorm2d(128)
#         self.bn9 = nn.BatchNorm2d(128)
#
#         self.conv1 = nn.Conv2d(num_channels, 64, 3, 1, 1, bias=False)
#         self.conv2 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
#         self.conv3 = nn.Conv2d(64, 128, 3, 2, 1, bias=False)
#
#         self.conv4 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
#         self.conv5 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
#         self.conv6 = nn.Conv2d(128, 128, 3, 2, 1, bias=False)
#
#         self.conv7 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
#         self.conv8 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
#         self.conv9 = nn.Conv2d(128, 128, 3, 2, 1, bias=False)
#
#         self.encoder1 = nn.Sequential(
#             self.conv1,
#             self.bn1,
#             self.relu,
#             self.conv2,
#             self.bn2,
#             self.relu,
#             self.conv3,
#             self.bn3,
#             self.relu,
#             self.dropout,
#         )
#
#         self.encoder2 = nn.Sequential(
#             self.conv4,
#             self.bn4,
#             self.relu,
#             self.conv5,
#             self.bn5,
#             self.relu,
#             self.conv6,
#             self.bn6,
#             self.relu,
#             self.dropout,
#         )
#
#         self.encoder3 = nn.Sequential(
#             self.conv7,
#             self.bn7,
#             self.relu,
#             self.conv8,
#             self.bn8,
#             self.relu,
#             self.conv9,
#             self.bn9,
#             self.relu,
#             self.dropout,
#
#         )
#
#         if init_weights:
#             self._initialize_weights()
#
#         self.cuda()
#
#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, 0, 0.01)
#                 nn.init.constant_(m.bias, 0)
#
#     def forward(self, x):
#         x1 = self.encoder1(x)
#         x2 = self.encoder2(x1)
#         x3 = self.encoder3(x2)
#         return x3
