import os
import sys
import time
import math
import numpy as np
import torch
from utils import loader3 as dataHelper
from networks import model
from utils.loader3 import kfw_loader,fiw_loader,get_train_loaders,fiw_triplet_loader
import torch.nn.functional as F
import torch.optim as optim

try:
    _, term_width = os.popen('stty size', 'r').read().split()
    term_width = int(term_width)
except:
    term_width = 84

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time


########################################## get
def get_dataloader(args,cfg):
    if 'kfw' in args.dataset:
        if 'cacur' in args.name or 'class' in args.name:
            return kfw_loader(args,cfg).get_train_loader()
        elif 'cac' in args.name or 'close' in args.name:
            return kfw_loader(args,cfg).get_close_train_loader()
    elif 'fiw' in args.dataset:
        if 'cacur' in args.name or 'class' in args.name:
            return fiw_loader(args,cfg).get_train_loader()
        elif 'cac' in args.name or 'close' in args.name:
            return fiw_loader(args,cfg).get_close_train_loader()
        elif 'HK' in args.name:
            return fiw_triplet_loader(args, cfg).get_train_HK_loader()
        elif 'trip' in args.name:
            return fiw_triplet_loader(args, cfg).get_train_loader()
        elif 'self' in args.name:
            return fiw_loader(args, cfg).get_train_self_loader()

        else:
            return fiw_loader(args, cfg).get_train_loader()
    else:
        return get_train_loaders(args.dataset, args.trial, cfg)


def get_test_dataloader(args,cfg):
    if 'kfw' in args.dataset:
        return kfw_loader(args, cfg).get_kfw_eval_loaders()
    elif 'fiw' in args.dataset:
        return fiw_loader(args,cfg).get_fiw_eval_loaders()
    # return fiw_loader(args, cfg).get_fiw_eval_loaders()

def get_network(args,cfg):
    if 'cac' in args.name:
        return model.openSetClassifier(cfg['num_known_classes'], cfg['im_channels'], cfg['im_size'],
                                      init_weights=True, dropout=cfg['dropout'])
    elif 'HK_trip_FG' in args.name:
        return model.HK_trip_FG(cfg['num_known_classes'], cfg['im_channels'], cfg['im_size'],
                             init_weights=True, dropout=cfg['dropout'])
    elif 'FG' in args.name:
        return model.FG(cfg['num_known_classes'], cfg['im_channels'], cfg['im_size'],
                                      init_weights=True, dropout=cfg['dropout'])


    elif 'class' in args.name:
        return model.cnn(cfg['num_known_classes']+1, cfg['im_channels'], cfg['im_size'],
                                      init_weights=True, dropout=cfg['dropout'])
    elif 'close' in args.name:

        return model.cnn(cfg['num_known_classes'], cfg['im_channels'], cfg['im_size'],
                                      init_weights=True, dropout=cfg['dropout'])

    elif 'HK_trip_tanhNet_different' in args.name:
        return model.HK_trip_tanhNet_different(cfg['num_known_classes'], cfg['im_channels'], cfg['im_size'],
                                 init_weights=True, dropout=cfg['dropout'])

    elif 'HK_trip_tanhNet_notanh' in args.name:
        return model.HK_trip_tanhNet_notanh(cfg['num_known_classes'], cfg['im_channels'], cfg['im_size'],
                                 init_weights=True, dropout=cfg['dropout'])

    elif 'HK_trip_KATsimple' in args.name:
        return model.HK_trip_KATsimple(cfg['num_known_classes'], cfg['im_channels'], cfg['im_size'],
                                 init_weights=True, dropout=cfg['dropout'])
    elif 'HK_trip_KATfinal' in args.name:
        return model.HK_trip_KATfinal(cfg['num_known_classes'], cfg['im_channels'], cfg['im_size'],
                                 init_weights=True, dropout=cfg['dropout'])


    elif 'HK_trip_ALTGVT' in args.name:
        return model.HK_trip_ALTGVT(cfg['num_known_classes'], cfg['im_channels'], cfg['im_size'],
                                 init_weights=True, dropout=cfg['dropout'])

    elif 'HK_trip_KATdouble' in args.name:
        return model.HK_trip_KATdouble(cfg['num_known_classes'], cfg['im_channels'], cfg['im_size'],
                                 init_weights=True, dropout=cfg['dropout'])
    elif 'HK_trip_KATsmalldeep' in args.name:
        return model.HK_trip_KATsmalldeep(cfg['num_known_classes'], cfg['im_channels'], cfg['im_size'],
                                 init_weights=True, dropout=cfg['dropout'])
    elif 'HK_trip_KATsmall' in args.name:
        return model.HK_trip_KATsmall(cfg['num_known_classes'], cfg['im_channels'], cfg['im_size'],
                                 init_weights=True, dropout=cfg['dropout'])
    elif 'HK_Facenet_CAT' in args.name:
        return model.HK_Facenet_CAT(cfg['num_known_classes'], cfg['im_channels'], cfg['im_size'],
                                 init_weights=True, dropout=cfg['dropout'])
    elif 'HK_trip_KAT_deep_new' in args.name:
        return model.HK_trip_KAT_deep_new(cfg['num_known_classes'], cfg['im_channels'], cfg['im_size'],
                                 init_weights=True, dropout=cfg['dropout'])
    elif 'HK_trip_KAT_deep' in args.name:
        return model.HK_trip_KAT_deep(cfg['num_known_classes'], cfg['im_channels'], cfg['im_size'],
                                 init_weights=True, dropout=cfg['dropout'])

    elif 'HK_Facenet_fc' in args.name:
        return model.HK_Facenet_fc(cfg['num_known_classes'], cfg['im_channels'], cfg['im_size'],
                                 init_weights=True, dropout=cfg['dropout'])

    elif 'HK_Facenet_trans' in args.name:
        return model.HK_Facenet_trans(cfg['num_known_classes'], cfg['im_channels'], cfg['im_size'],
                                 init_weights=True, dropout=cfg['dropout'])
    elif 'HK_trip_KAT' in args.name:
        return model.HK_trip_KAT(cfg['num_known_classes'], cfg['im_channels'], cfg['im_size'],
                                   init_weights=True, dropout=cfg['dropout'])
    elif 'HK_trip_trans' in args.name:
        return model.HK_trip_trans(cfg['num_known_classes'], cfg['im_channels'], cfg['im_size'],
                                     init_weights=True, dropout=cfg['dropout'])
    elif 'HK_trip_tanhNet' in args.name:
        return model.HK_trip_tanhNet(cfg['num_known_classes'], cfg['im_channels'], cfg['im_size'],
                             init_weights=True, dropout=cfg['dropout'])

    elif 'HK_trip_CNN_Net' in args.name:
        return model.HK_trip_CNN_Net(cfg['num_known_classes'], cfg['im_channels'], cfg['im_size'],
                             init_weights=True, dropout=cfg['dropout'])
    elif 'HK_trip_CNN_point_Net' in args.name:
        return model.HK_trip_CNN_point_Net(cfg['num_known_classes'], cfg['im_channels'], cfg['im_size'],
                             init_weights=True, dropout=cfg['dropout'])
    elif 'SoftIntroVAE' in args.name:
        return model.SoftIntroVAE()
    else:
        return None


def get_anchors(args,cfg):
    if 'cacur' in args.name:
        anchors = torch.diag(torch.Tensor([args.alpha for i in range(cfg['num_known_classes'])]))
        return anchors
    elif 'cac' in args.name:
        anchors = torch.diag(torch.Tensor([args.alpha for i in range(cfg['num_known_classes'])]))
        return anchors


def get_optim(args,cfg,net):
    if 'cacur' in args.name :
        if 'anchor' in args.name or 'anxis' in args.name:
            params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] in "anchors", net.named_parameters()))))
            base_params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] not in "anchors", net.named_parameters()))))
            learning_rate = cfg['training']['learning_rate']
            optimizer = optim.SGD([{'params': base_params}, {'params': params, 'lr': learning_rate * 1}],
                                  lr=learning_rate,
                                  momentum=0.9, weight_decay=cfg['training']['weight_decay'])
            return optimizer

        else:
            return optim.SGD(net.parameters(), lr=cfg['training']['learning_rate'],
                              momentum=0.9, weight_decay=cfg['training']['weight_decay'])

    elif 'class' in args.name or 'close' in args.name or 'cac' in args.name:
        return optim.SGD(net.parameters(), lr=cfg['training']['learning_rate'],
                      momentum=0.9, weight_decay=cfg['training']['weight_decay'])
    else:
        if "adam" in args.name :
            return optim.Adam(net.parameters(), lr=cfg['training']['learning_rate'],
                          weight_decay=cfg['training']['weight_decay'])
        else:
            params = [p for p in net.parameters() if p.requires_grad]
            return optim.SGD(params, lr=cfg['training']['learning_rate'],
                         momentum=0.9, weight_decay=cfg['training']['weight_decay'])






#############################################
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width - int(TOTAL_BAR_LENGTH / 2) + 2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current + 1, total))

    if current < total - 1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def find_anchor_means(net, mapping, datasetName, trial_num, args,cfg, only_correct=False):
    ''' Tests data and fits a multivariate gaussian to each class' logits.
        If dataloaderFlip is not None, also test with flipped images.
        Returns means and covariances for each class. '''
    # find gaussians for each class
    if datasetName == 'MNIST' or datasetName == "SVHN":
        loader, _ = dataHelper.get_anchor_loaders(datasetName, trial_num, cfg)
        logits, labels = gather_outputs(net, mapping, loader, only_correct=only_correct)
    elif 'kfw' in datasetName:
        loader, _ = kfw_loader(args,cfg).get_train_loader()
        logits, labels = gather_outputs(net, mapping, loader, only_correct=only_correct)
    else:
        loader, loaderFlipped = dataHelper.get_anchor_loaders(datasetName, trial_num, cfg)
        logits, labels = gather_outputs(net, mapping, loader, loaderFlipped, only_correct=only_correct)

    num_classes = cfg['num_known_classes']
    means = [None for i in range(num_classes)]

    for cl in range(num_classes):
        x = logits[labels == cl]
        x = np.squeeze(x)
        means[cl] = np.mean(x, axis=0)

    return means


def SoftmaxTemp(logits, T=1):
    num = torch.exp(logits / T)
    denom = torch.sum(torch.exp(logits / T), 1).unsqueeze(1)
    return num / denom


def gather_outputs(net, mapping, dataloader, data_idx=0,
                   calculate_scores=False, unknown=False, only_correct=False):
    ''' Tests data and returns outputs and their ground truth labels.
        data_idx        0 returns logits, 1 returns distances to anchors
        use_softmax     True to apply softmax
        unknown         True if an unknown dataset
        only_correct    True to filter for correct classifications as per logits
    '''
    X = []
    y = []

    if calculate_scores:
        softmax = torch.nn.Softmax(dim=1)

    for i, data in enumerate(dataloader):
        images, labels = data
        images = images.cuda()

        if unknown:
            targets = labels
        else:
            targets = torch.Tensor([mapping[x] for x in labels]).long().cuda()

        outputs = net(images)
        logits = outputs[0]
        distances = outputs[1]

        if only_correct:
            if data_idx == 0:
                _, predicted = torch.max(logits, 1)
            else:
                _, predicted = torch.min(distances, 1)

            mask = predicted == targets
            logits = logits[mask]
            distances = distances[mask]
            targets = targets[mask]

        if calculate_scores:
            softmin = softmax(-distances)
            invScores = 1 - softmin
            scores = distances * invScores

            if len(outputs) == 4:

                # scores = scores+(10*outputs[2].unsqueeze(1))
                scores = scores + 10*(outputs[2].unsqueeze(1))
                # scores = distances + (10 * outputs[2].unsqueeze(1))
                # scores = invScores
                # scores = (distances-10*outputs[3])
                # scores = invScores*(-outputs[3]+1)/2
                # scores = scores

        else:
            if data_idx == 0:
                scores = logits
            if data_idx == 1:
                scores = distances

        X += scores.cpu().detach().tolist()
        y += targets.cpu().tolist()

    X = np.asarray(X)
    y = np.asarray(y)

    return X, y


def gather_feature(net, mapping, dataloader, unknown=False):
    ''' Tests data and returns outputs and their ground truth labels.
        data_idx        0 returns logits, 1 returns distances to anchors
        use_softmax     True to apply softmax
        unknown         True if an unknown dataset
    '''
    X = []
    y = []


    for i, data in enumerate(dataloader):
        images, labels = data
        images = images.cuda()

        if unknown:
            targets = labels
        else:
            targets = torch.Tensor([mapping[x] for x in labels]).long().cuda()

        features = net.feature(images)

        # features = net(images)[0]



        X += features.cpu().detach().tolist()
        y += targets.cpu().tolist()

    X = np.asarray(X)
    y = np.asarray(y)

    return X, y

def gather_dis(net, mapping, dataloader, unknown=False):
    ''' Tests data and returns outputs and their ground truth labels.
        data_idx        0 returns logits, 1 returns distances to anchors
        use_softmax     True to apply softmax
        unknown         True if an unknown dataset
    '''
    X = []
    y = []


    for i, data in enumerate(dataloader):
        images, labels = data
        images = images.cuda()

        if unknown:
            targets = labels
        else:
            targets = torch.Tensor([mapping[x] for x in labels]).long().cuda()

        features = net.similarity(images)

        # features = net(images)[0]



        X += features.cpu().detach().tolist()
        y += targets.cpu().tolist()

    X = np.asarray(X)
    y = np.asarray(y)

    return X, y


def gather_feature2d(net, mapping, dataloader, unknown=False):
    ''' Tests data and returns outputs and their ground truth labels.
        data_idx        0 returns logits, 1 returns distances to anchors
        use_softmax     True to apply softmax
        unknown         True if an unknown dataset
    '''
    X = []
    y = []

    for i, data in enumerate(dataloader):
        images, labels = data
        images = images.cuda()

        if unknown:
            targets = labels
        else:
            targets = torch.Tensor([mapping[x] for x in labels]).long().cuda()

        features = net.feature_2d(images)

        # features = net(images)[0]



        X += features.cpu().detach().tolist()
        y += targets.cpu().tolist()

    X = np.asarray(X)
    y = np.asarray(y)

    return X, y


def gather_vis(net, mapping, dataloader, dataloaderFlip=None, data_idx=0, calculate_scores=False, unknown=False,
               only_correct=False):
    ''' Tests data and returns outputs and their ground truth labels.
        data_idx        0 returns logits, 1 returns distances to anchors
        use_softmax     True to apply softmax
        unknown         True if an unknown dataset
        only_correct    True to filter for correct classifications as per logits
    '''
    X = []
    y = []

    if calculate_scores:
        softmax = torch.nn.Softmax(dim=1)

    for i, data in enumerate(dataloader):
        images, labels = data
        images = images.cuda()

        if unknown:
            targets = labels
        else:
            targets = torch.Tensor([mapping[x] for x in labels]).long().cuda()

        outputs = net(images)
        if isinstance(outputs, tuple):
            logits = outputs[0]
            distances = outputs[1]
            softmin = softmax(-distances)
            invScores = 1 - softmin
            scores = distances * invScores
            #
            if len(outputs) == 4:
                scores = scores + 10*(outputs[2].unsqueeze(1))
                # scores = scores-10*outputs[3]
            logits = scores
        else:
            # logits = F.softmax(outputs, 1)
            logits = outputs
        # distances = outputs[1]

        X += logits.cpu().detach().tolist()
        y += targets.cpu().tolist()

    X = np.asarray(X)
    y = np.asarray(y)

    return X, y