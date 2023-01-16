import torch
import  torch.nn.functional as F

import torch.nn as nn

class CAC_Loss(nn.Module):
    def __init__(self, args,cfg):
        super(CAC_Loss, self).__init__()
        self.args = args
        self.cfg = cfg
    def forward(self,outputs,gt):
        '''Returns CAC_ loss, as well as the Anchor and Tuplet loss components separately for visualisation.'''
        cfg = self.cfg
        args = self.args
        distances = outputs[1]
        true = torch.gather(distances, 1, gt.view(-1, 1)).view(-1)
        non_gt = torch.Tensor(
            [[i for i in range(cfg['num_known_classes']) if gt[x] != i] for x in range(len(distances))]).long().cuda()
        others = torch.gather(distances, 1, non_gt)

        anchor = torch.mean(true)

        tuplet = torch.exp(-others + true.unsqueeze(1))
        tuplet = torch.mean(torch.log(1 + torch.sum(tuplet, dim=1)))

        total = args.lbda * anchor + tuplet

        return total, (anchor, tuplet)


class ce(nn.Module):
    def __init__(self, args,cfg):
        super(ce, self).__init__()
        self.loss = nn.CrossEntropyLoss()
    def forward(self,outputs,gt):
        total = self.loss(outputs[0],gt)
        return total, 0



class similarity_metric(nn.Module):
    def __init__(self):
        super(similarity_metric, self).__init__()
        self.cos = nn.CosineSimilarity(dim=1)

    def forward(self,x1,x2):
        return 1-self.cos(x1,x2)


class final_loss2(nn.Module):
    """
    Hierachical Kinship Triplet loss
    """
    def __init__(self, args,cfg):
        super(final_loss2, self).__init__()
        self.args = args
        self.cfg = cfg

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.triplet_margin1 = nn.TripletMarginWithDistanceLoss(distance_function=nn.PairwiseDistance(), margin=0)
        self.triplet_margin = nn.TripletMarginWithDistanceLoss(distance_function=nn.PairwiseDistance(),margin=3)
        self.dis = nn.PairwiseDistance()
        # self.marginloss = nn.MarginRankingLoss()
        self.mse = nn.MSELoss()


    def forward(self,outputs,lb):
        a = outputs[1]
        s = outputs[2]
        p = outputs[3]
        n = outputs[4]
        dis_center = outputs[5]

        c_loss = self.mse(dis_center[0].expand(a.size()[0]),self.dis(a,s))+\
                 self.mse(dis_center[1].expand(a.size()[0]),self.dis(a,p))+ \
                 self.mse(dis_center[2].expand(a.size()[0]),self.dis(a,n))
        trip_loss = self.triplet_margin1(a,s,p)+self.triplet_margin(a,p,n)

        loss  = c_loss+trip_loss
        return  loss, (c_loss,trip_loss)


class final_loss(nn.Module):
    """
    Hierachical Kinship Triplet loss
    """
    def __init__(self, args,cfg):
        super(final_loss, self).__init__()
        self.args = args
        self.cfg = cfg

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.triplet_margin1 = nn.TripletMarginWithDistanceLoss(distance_function=nn.PairwiseDistance(), margin=0.5)
        self.triplet_margin = nn.TripletMarginWithDistanceLoss(distance_function=nn.PairwiseDistance(),margin=3)
        self.dis = nn.PairwiseDistance()
        # self.marginloss = nn.MarginRankingLoss()
        self.mse = nn.MSELoss()


    def forward(self,outputs,lb):
        a = outputs[1]
        s = outputs[2]
        p = outputs[3]
        n = outputs[4]
        dis_center = outputs[5]

        c_loss = self.mse(dis_center[0].expand(a.size()[0]),self.dis(a,s))+\
                 self.mse(dis_center[1].expand(a.size()[0]),self.dis(a,p))+ \
                 self.mse(dis_center[2].expand(a.size()[0]),self.dis(a,n))
        trip_loss = self.triplet_margin1(a,s,p)+self.triplet_margin(a,p,n)

        loss  = 2*c_loss+trip_loss
        return  loss, (c_loss,trip_loss)


class final_loss3(nn.Module):
    """
    Hierachical Kinship Triplet loss
    """
    def __init__(self, args,cfg):
        super(final_loss3, self).__init__()
        self.args = args
        self.cfg = cfg

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.triplet_margin1 = nn.TripletMarginWithDistanceLoss(distance_function=nn.PairwiseDistance(), margin=0.5)
        self.triplet_margin = nn.TripletMarginWithDistanceLoss(distance_function=nn.PairwiseDistance(),margin=3)
        self.dis = nn.PairwiseDistance()
        # self.marginloss = nn.MarginRankingLoss()
        self.mse = nn.MSELoss()


    def forward(self,outputs,lb):
        a = outputs[1]
        s = outputs[2]
        p = outputs[3]
        n = outputs[4]
        dis_center = outputs[5]

        c_loss = self.mse(dis_center[0].expand(a.size()[0]),self.dis(a,s))+\
                 self.mse(dis_center[1].expand(a.size()[0]),self.dis(a,p))+ \
                 self.mse(dis_center[2].expand(a.size()[0]),self.dis(a,n))
        trip_loss = self.triplet_margin1(a,s,p)+self.triplet_margin(a,p,n)

        loss  = c_loss+2*trip_loss
        return  loss, (c_loss,trip_loss)



class hktrans_loss(nn.Module):
    """
    Hierachical Kinship Triplet loss -transformer version
    """
    def __init__(self, args,cfg):
        super(hktrans_loss, self).__init__()
        self.args = args
        self.cfg = cfg

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.triplet_margin1 = nn.TripletMarginWithDistanceLoss(distance_function=nn.PairwiseDistance(), margin=0.5)
        self.triplet_margin = nn.TripletMarginWithDistanceLoss(distance_function=nn.PairwiseDistance(),margin=3)
        self.dis = nn.PairwiseDistance()
        # self.marginloss = nn.MarginRankingLoss()
        self.mse = nn.MSELoss()


    def forward(self,outputs,lb):
        (a1, s) = outputs[1]
        (a2, p) = outputs[2]
        (a3, n) = outputs[3]
        dis_center = outputs[4]

        c_loss = self.mse(dis_center[0].expand(a1.size()[0]),self.dis(a1,s))+\
                 self.mse(dis_center[1].expand(a1.size()[0]),self.dis(a2,p))+ \
                 self.mse(dis_center[2].expand(a1.size()[0]),self.dis(a3,n))
        trip_loss = self.triplet_margin1(a1,s,p)+self.triplet_margin(a1,p,n)+ \
                    self.triplet_margin1(a2,s,p)+self.triplet_margin(a2,p,n)+\
                    self.triplet_margin1(a3,s,p)+self.triplet_margin(a3,p,n)

        loss  = c_loss+1/3*trip_loss
        return  loss, (c_loss,trip_loss)


class final_hardloss(nn.Module):
    """
    Hierachical Kinship Triplet loss
    """
    def __init__(self, args,cfg):
        super(final_hardloss, self).__init__()
        self.args = args
        self.cfg = cfg

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.triplet_margin1 = nn.TripletMarginWithDistanceLoss(distance_function=nn.PairwiseDistance(), margin=0)
        self.triplet_margin = nn.TripletMarginWithDistanceLoss(distance_function=nn.PairwiseDistance(),margin=3)
        self.dis = nn.PairwiseDistance()
        # self.marginloss = nn.MarginRankingLoss()
        self.mse = nn.MSELoss()


    def forward(self,outputs,lb):
        a = outputs[1]
        s = outputs[2]
        p = outputs[3]
        n = outputs[4]
        dis_center = outputs[5]

        c_loss = self.mse(dis_center[0].expand(a.size()[0]),self.dis(a,s))+\
                 self.mse(dis_center[1].expand(a.size()[0]),self.dis(a,p))+ \
                 self.mse(dis_center[2].expand(a.size()[0]),self.dis(a,n))
        trip_loss = self.triplet_margin1(a,s,p)+self.triplet_margin(a,p,n)

        loss  = c_loss+trip_loss
        return  loss, (c_loss,trip_loss)



class tripCNNLoss(nn.Module):
    """
    triplet cross entropy loss
    """
    def __init__(self, args,cfg):
        super(tripCNNLoss, self).__init__()
        self.args = args
        self.cfg = cfg

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.ce = nn.CrossEntropyLoss()


    def forward(self,outputs,lb):
        x_as = outputs[1]
        x_ap = outputs[2]
        x_an = outputs[3]
        as_target = torch.zeros(x_as.size()[0]).cuda().type(torch.int64)
        ap_target = torch.ones(x_as.size()[0]).cuda().type(torch.int64)
        an_target = 2*torch.ones(x_as.size()[0]).cuda().type(torch.int64)

        loss = self.ce(x_as,as_target)+self.ce(x_ap,ap_target)+self.ce(x_an,an_target)

        c_loss = 0
        trip_loss = 0
        return  loss, (c_loss,trip_loss)



import math
class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
         3 years ago: • arcface, cosine face
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)

        return output


class CenterLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss


############################

