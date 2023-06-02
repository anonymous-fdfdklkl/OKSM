import torch
import torch.nn as nn
from torch.nn import functional as F
from models.ABN import MultiBatchNorm

class ConvNet(nn.Module):
    """LeNet++ as described in the Center Loss paper."""
    def __init__(self, num_classes):
        super(ConvNet, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 32, 5, stride=1, padding=2)
        self.prelu1_1 = nn.PReLU()
        self.conv1_2 = nn.Conv2d(32, 32, 5, stride=1, padding=2)
        self.prelu1_2 = nn.PReLU()
        
        self.conv2_1 = nn.Conv2d(32, 64, 5, stride=1, padding=2)
        self.prelu2_1 = nn.PReLU()
        self.conv2_2 = nn.Conv2d(64, 64, 5, stride=1, padding=2)
        self.prelu2_2 = nn.PReLU()
        
        self.conv3_1 = nn.Conv2d(64, 128, 5, stride=1, padding=2)
        self.prelu3_1 = nn.PReLU()
        self.conv3_2 = nn.Conv2d(128, 128, 5, stride=1, padding=2)
        self.prelu3_2 = nn.PReLU()
        
        self.fc1 = nn.Linear(128*4*4, 2)
        self.prelu_fc1 = nn.PReLU()
        self.fc2 = nn.Linear(2, num_classes)

    def forward(self, x, rf=False):
        x = self.prelu1_1(self.conv1_1(x))
        x = self.prelu1_2(self.conv1_2(x))
        x = F.max_pool2d(x, 2)
        
        x = self.prelu2_1(self.conv2_1(x))
        x = self.prelu2_2(self.conv2_2(x))
        x = F.max_pool2d(x, 2)
        
        x = self.prelu3_1(self.conv3_1(x))
        x = self.prelu3_2(self.conv3_2(x))
        x = F.max_pool2d(x, 2)
        
        x = x.view(-1, 128*4*4)
        x = self.prelu_fc1(self.fc1(x))
        y = self.fc2(x)
        
        if rf:
            return x, y
        return y

def weights_init(m):
    classname = m.__class__.__name__
    # TODO: what about fully-connected layers?
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.05)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class classifier32(nn.Module):
    def __init__(self, num_classes=10):
        super(self.__class__, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(6,       64,     3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(64,      64,     3, 1, 1, bias=False)
        self.conv3 = nn.Conv2d(64,     128,     3, 2, 1, bias=False)

        self.conv4 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv5 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv6 = nn.Conv2d(128,    128,     3, 2, 1, bias=False)

        self.conv7 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv8 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv9 = nn.Conv2d(128,    128,     3, 2, 1, bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(128)
        self.bn6 = nn.BatchNorm2d(128)

        self.bn7 = nn.BatchNorm2d(128)
        self.bn8 = nn.BatchNorm2d(128)
        self.bn9 = nn.BatchNorm2d(128)
        self.bn10 = nn.BatchNorm2d(128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes, bias=False)
        self.dr1 = nn.Dropout2d(0.2)
        self.dr2 = nn.Dropout2d(0.2)
        self.dr3 = nn.Dropout2d(0.2)

        self.apply(weights_init)
        self.cuda()

    def forward(self, x, return_feature=False):
        x = self.dr1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = nn.LeakyReLU(0.2)(x)

        x = self.dr2(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = nn.LeakyReLU(0.2)(x)

        x = self.dr3(x)
        x = self.conv7(x)
        x = self.bn7(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv8(x)
        x = self.bn8(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv9(x)
        x = self.bn9(x)
        x = nn.LeakyReLU(0.2)(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        y = self.fc(x)
        if return_feature:
            return x, y
        else:
            return y

def weights_init_ABN(m):
    classname = m.__class__.__name__
    # TODO: what about fully-connected layers?
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.05)
    elif classname.find('MultiBatchNorm') != -1:
        m.bns[0].weight.data.normal_(1.0, 0.02)
        m.bns[0].bias.data.fill_(0)
        m.bns[1].weight.data.normal_(1.0, 0.02)
        m.bns[1].bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class classifier32ABN(nn.Module):
    def __init__(self, num_classes=10, num_ABN=2):
        super(self.__class__, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(6,       64,     3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(64,      64,     3, 1, 1, bias=False)
        self.conv3 = nn.Conv2d(64,     128,     3, 2, 1, bias=False)

        self.conv4 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv5 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv6 = nn.Conv2d(128,    128,     3, 2, 1, bias=False)

        self.conv7 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv8 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv9 = nn.Conv2d(128,    128,     3, 2, 1, bias=False)

        self.bn1 = MultiBatchNorm(64, num_ABN)
        self.bn2 = MultiBatchNorm(64, num_ABN)
        self.bn3 = MultiBatchNorm(128, num_ABN)

        self.bn4 = MultiBatchNorm(128, num_ABN)
        self.bn5 = MultiBatchNorm(128, num_ABN)
        self.bn6 = MultiBatchNorm(128, num_ABN)

        self.bn7 = MultiBatchNorm(128, num_ABN)
        self.bn8 = MultiBatchNorm(128, num_ABN)
        self.bn9 = MultiBatchNorm(128, num_ABN)
        self.bn10 = MultiBatchNorm(128, num_ABN)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes, bias=False)
        self.dr1 = nn.Dropout2d(0.2)
        self.dr2 = nn.Dropout2d(0.2)
        self.dr3 = nn.Dropout2d(0.2)

        self.apply(weights_init_ABN)
        self.cuda()

    def forward(self, x, return_feature=False, bn_label=None):
        if bn_label is None:
            bn_label = 0 * torch.ones(x.shape[0], dtype=torch.long).cuda()
        x = self.dr1(x)
        x = self.conv1(x)
        x, _ = self.bn1(x, bn_label)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv2(x)
        x, _ = self.bn2(x, bn_label)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv3(x)
        x, _ = self.bn3(x, bn_label)
        x = nn.LeakyReLU(0.2)(x)

        x = self.dr2(x)
        x = self.conv4(x)
        x, _ = self.bn4(x, bn_label)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv5(x)
        x, _ = self.bn5(x, bn_label)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv6(x)
        x, _ = self.bn6(x, bn_label)
        x = nn.LeakyReLU(0.2)(x)

        x = self.dr3(x)
        x = self.conv7(x)
        x, _ = self.bn7(x, bn_label)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv8(x)
        x, _ = self.bn8(x, bn_label)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv9(x)
        x, _ = self.bn9(x, bn_label)
        x = nn.LeakyReLU(0.2)(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        y = self.fc(x)
        if return_feature:
            return x, y
        else:
            return y



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
    def __init__(self):
        super(attenNet_basic,self).__init__()
        self.conv1 = nn.Conv2d(6,32,5)
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

class openset(nn.Module):
    def __init__(self, num_classes=7, num_channels=3, im_size=64, init_weights=True, dropout=0.3, **kwargs):
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

    def forward(self, x, return_feature=False):
        batch_size = len(x)

        x = self.encoder(x)
        x = x.view(batch_size, -1)

        y = self.classify(x)
        if return_feature:
            return x, y
        else:
            return y


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

