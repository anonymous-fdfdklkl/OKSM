
from matplotlib import pyplot as plt

import pandas as pd
import seaborn as sn

import numpy as np
from sklearn.metrics import confusion_matrix
import torch

import core.metrics as metrics
from core import evaluation

def test(net, criterion, testloader, outloader, epoch=None, **options):
    net.eval()
    correct, total = 0, 0

    torch.cuda.empty_cache()

    _pred_k, _pred_u, _labels = [], [], []

    with torch.no_grad():
        for data, labels in testloader:
            if options['use_gpu']:
                data, labels = data.cuda(), labels.cuda()
            
            with torch.set_grad_enabled(False):
                x, y = net(data, True)
                logits, _ = criterion(x, y)
                predictions = logits.data.max(1)[1]
                total += labels.size(0)
                correct += (predictions == labels.data).sum()
            
                _pred_k.append(logits.data.cpu().numpy())
                _labels.append(labels.data.cpu().numpy())

        for batch_idx, (data, labels) in enumerate(outloader):
            if options['use_gpu']:
                data, labels = data.cuda(), labels.cuda()
            
            with torch.set_grad_enabled(False):
                x, y = net(data, True)
                # x, y = net(data, return_feature=True)
                logits, _ = criterion(x, y)
                _pred_u.append(logits.data.cpu().numpy())

    # Accuracy
    acc = float(correct) * 100. / float(total)
    print('Acc: {:.5f}'.format(acc))

    _pred_k = np.concatenate(_pred_k, 0)
    _pred_u = np.concatenate(_pred_u, 0)
    _labels = np.concatenate(_labels, 0)
    
    # Out-of-Distribution detction evaluation
    x1, x2 = np.max(_pred_k, axis=1), np.max(_pred_u, axis=1)
    results = evaluation.metric_ood(x1, x2)['Bas']
    
    # OSCR
    _oscr_socre = evaluation.compute_oscr(_pred_k, _pred_u, _labels)

    results['ACC'] = acc
    results['OSCR'] = _oscr_socre * 100.

    return results


def test_while_train(net, criterion, trainloader, **options):
    net.eval()
    correct, total = 0, 0
    torch.cuda.empty_cache()
    _pred_k, _pred_u, _labels = [], [], []

    with torch.no_grad():
        for data, labels in trainloader:
            if options['use_gpu']:
                data, labels = data.cuda(), labels.cuda()

            with torch.set_grad_enabled(False):
                x, y = net(data, True)
                logits, _ = criterion(x, y)
                predictions = logits.data.max(1)[1]
                total += labels.size(0)
                correct += (predictions == labels.data).sum()

    # Accuracy
    acc = float(correct) * 100. / float(total)
    print('Acc: {:.5f}'.format(acc))
    return acc


def test_kfw(net, criterion, knownloader,unknownloader,testloader,name, **options):
    print('==> Evaluating open set network accuracy')
    x, y = gather_outputs(net, criterion, knownloader,options)
    accuracy = metrics.accuracy(x, y)


    print('==> Evaluating open set network AUROC')
    xK, yK = gather_outputs(net, criterion, knownloader,options)
    xU, yU = gather_outputs(net, criterion, unknownloader, options)

    auroc,op_thresh = metrics.auroc(xK, xU,True,True,name=name)


    print('==> Evaluating open set network accuracy with negative samples')
    pre, label = gather_outputs(net, criterion, testloader, options)
    accuracy_with_neg,f1_neginclude,f1_negexclude = metrics.accuracy_with_neg(pre, label,op_thresh,sv_cm=True,name=name)


    return [accuracy,auroc,f1_neginclude,f1_negexclude,accuracy_with_neg]





# def save_confusion_metric(gt, pred, name, label=[0,1,2]):
#     plt.figure()
#
#     confu_m = confusion_matrix(gt, pred, labels=label, normalize='true')
#     df_cm = pd.DataFrame(confu_m, ['Primary-Kin', 'Secondary-Kin', 'Non-Kin'], ['Primary-Kin', 'Secondary-Kin', 'Non-Kin'])
#
#     sn.set(font_scale=1)  # for label size
#     sn.heatmap(df_cm, vmin=0, vmax=1, cmap='Blues', annot=True, annot_kws={"size": 26})  # font size
#     # plt.show()
#     # plt.savefig('stage3-{}_test1{}_hm{}.png'.format(number,stage3_joint_config.kin_config.model_name, '_avg'))
#     plt.savefig('hm-{}.png'.format(name))

def save_confusion_metric(gt, pred, name, label=[0,1,2]):
    plt.figure()

    confu_m = confusion_matrix(gt, pred, labels=label, normalize='true')
    # df_cm = pd.DataFrame(confu_m, ['Primary-Kin', 'Secondary-Kin', 'Non-Kin'], ['Primary-Kin', 'Secondary-Kin', 'Non-Kin'])
    df_cm = pd.DataFrame(confu_m, ['P', 'S', 'N'],
                         ['P', 'S', 'N'])
    sn.set(font_scale=2.5)  # for label size
    sn.heatmap(df_cm, vmin=0, vmax=1, cmap='Blues', annot=True, annot_kws={"size": 26})  # font size
    # plt.show()
    # plt.savefig('stage3-{}_test1{}_hm{}.png'.format(number,stage3_joint_config.kin_config.model_name, '_avg'))
    plt.savefig('hm-{}.png'.format(name))

from sklearn.metrics import f1_score, fbeta_score


def test_fiw(net, criterion, testloader_eval, testloader, model_name, **options):

    net.eval()

    ################################################################################## get thresh

    X = []
    y = []
    # neg_val_pos = []
    # softmax = torch.nn.Softmax(dim=1)

    mapping = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, -1]

    for i, data in enumerate(testloader):
        images, labels = data
        # targets = labels.cuda()
        targets = torch.Tensor([mapping[x] for x in labels]).long().cuda()

        images = images.cuda()

        # dis = net.similarity(images)
        # scores = softmax(logits)

        with torch.set_grad_enabled(False):
            x, zz = net(images, True)
            logits, _ = criterion(x, zz)
            dis = logits.data.max(1)[0]

        X += dis.cpu().detach().tolist()
        y += targets.cpu().tolist()

    X = np.asarray(X)
    y = np.asarray(y)
    X = X




    if ('4_type' in options['name']) or ('4type' in options['name']):
        # if '4_type'  in options['name'] or '4type' in options['name']:
        ## todo: solve bugs
        known_label = [True if i in [0, 1, 2, 3] else False for i in y]
        unknown_label = [True if i in [7, 8, 9, 10, -1] else False for i in y]
        kin_related_label = [True if i in [7, 8, 9, 10] else False for i in y]
        unknon_nonkin_label = [True if i in [-1] else False for i in y]
        name_cm = 'ARPL_4_type_{}'.format(options['cs_num'])
    else:
        known_label = [True if i in [0, 1, 2, 3, 4, 5, 6] else False for i in y]
        unknown_label = [True if i in [7, 8, 9, 10, -1] else False for i in y]
        kin_related_label = [True if i in [7, 8, 9, 10] else False for i in y]
        unknon_nonkin_label = [True if i in [-1] else False for i in y]
        name_cm = 'ARPL_7_type_{}'.format(options['cs_num'])
    known_set = X[known_label]
    unk_set = X[unknown_label]
    ###
    unk_nonkin_set = X[unknon_nonkin_label]
    unk_kin_set = X[kin_related_label]

    inlow = False

    #### 1) known vs unknown
    _, op_shr_knun = metrics.auroc(known_set, unk_set, in_low=inlow, show=False,
                                            name=[model_name, 'known', 'unknown'], trial_num=options['cs_num'] - 1)

    #### kin vs nonkin (within unknon sets)

    _, op_shr_kinnon = metrics.auroc(unk_kin_set, unk_nonkin_set, in_low=inlow, show=False,
                                               name=[model_name, 'kin', 'non-kin'], trial_num=options['cs_num'] - 1)


    ################################################################################

    X = []
    y = []
    # neg_val_pos = []
    # softmax = torch.nn.Softmax(dim=1)

    mapping = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, -1]

    for i, data in enumerate(testloader):
        images, labels = data
        # targets = labels.cuda()
        targets = torch.Tensor([mapping[x] for x in labels]).long().cuda()

        images = images.cuda()

        # dis = net.similarity(images)
        # scores = softmax(logits)

        with torch.set_grad_enabled(False):
            x, zz = net(images, True)
            logits, _ = criterion(x, zz)
            dis = logits.data.max(1)[0]

        X += dis.cpu().detach().tolist()
        y += targets.cpu().tolist()

    X = np.asarray(X)
    y = np.asarray(y)
    X = X

    if ('4_type' in options['name']) or ('4type' in options['name']):
        # if '4_type'  in options['name'] or '4type' in options['name']:
        ## todo: solve bugs
        known_label = [True if i in [0, 1, 2, 3] else False for i in y]
        unknown_label = [True if i in [7, 8, 9, 10, -1] else False for i in y]
        kin_related_label = [True if i in [7, 8, 9, 10] else False for i in y]
        unknon_nonkin_label = [True if i in [-1] else False for i in y]
        name_cm = 'ARPL_4_type_{}'.format(options['cs_num'])
    else:
        known_label = [True if i in [0, 1, 2, 3, 4, 5, 6] else False for i in y]
        unknown_label = [True if i in [7, 8, 9, 10, -1] else False for i in y]
        kin_related_label = [True if i in [7, 8, 9, 10] else False for i in y]
        unknon_nonkin_label = [True if i in [-1] else False for i in y]
        name_cm = 'ARPL_7_type_{}'.format(options['cs_num'])


    print('==='*30,name_cm)

    known_set = X[known_label]
    unk_set = X[unknown_label]
    ###
    unk_nonkin_set = X[unknon_nonkin_label]
    unk_kin_set = X[kin_related_label]

    inlow = False

    #### 1) known vs unknown
    auroc_knun, _ = metrics.auroc(known_set, unk_set, in_low=inlow, show=True,
                                            name=[model_name, 'known', 'unknown'], trial_num=options['cs_num'] - 1)

    #### kin vs nonkin (within unknon sets)

    auroc_kinon, _ = metrics.auroc(unk_kin_set, unk_nonkin_set, in_low=inlow, show=True,
                                               name=[model_name, 'kin', 'non-kin'], trial_num=options['cs_num'] - 1)

    GT = np.zeros(len(y))
    GT[kin_related_label] = 1
    GT[unknon_nonkin_label] = 2
    pred = np.ones(len(y))

    pred[X >= op_shr_knun] = 0
    pred[X <= op_shr_kinnon] = 2

    acc = np.sum(GT == pred) / len(y)

    print("should save cm .......cs_num:{}".format(options['cs_num'] - 1))

    save_confusion_metric(GT, pred, name_cm + '{}'.format(options['cs_num'] - 1), label=[0, 1, 2])

    import pickle
    with open('{}_acc_{}.pickle'.format('ARPL', options['cs_num'] - 1), 'wb') as ff:
        pickle.dump(pred, ff)

    F1 = f1_score(GT, pred, average='macro')

    print('acc:{}'.format(acc))
    gt_lb = GT
    pred_lb = pred

    k_tp = np.sum((gt_lb == pred_lb)[gt_lb == 0])
    k_fp = len(pred_lb[pred_lb == 0]) - k_tp
    k_fn = len(gt_lb[gt_lb == 0]) - k_tp

    ukkin_tp = np.sum((gt_lb == pred_lb)[gt_lb == 1])
    ukkin_fp = len(pred_lb[pred_lb == 1]) - ukkin_tp
    ukkin_fn = len(gt_lb[gt_lb == 1]) - ukkin_tp

    unk_tp = np.sum((gt_lb == pred_lb)[gt_lb == 2])
    unk_fp = len(pred_lb[pred_lb == 2]) - unk_tp
    unk_fn = len(gt_lb[gt_lb == 2]) - unk_tp

    ### precision = tp/(tp+fp) : correct/predict
    k_precision = k_tp / (k_tp + k_fp + 1e-16)
    ukkin_precision = ukkin_tp / (ukkin_tp + ukkin_fp + 1e-16)
    unk_precision = unk_tp / (unk_tp + unk_fp + 1e-16)

    ### recall = tp/(tp+fn): correct/total_target

    k_recall = k_tp / (k_tp + k_fn + 1e-16)
    ukkin_recall = ukkin_tp / (ukkin_tp + ukkin_fn + 1e-16)
    unk_recall = unk_tp / (unk_tp + unk_fn + 1e-16)

    #####F1 : 2*precision*recall/(precision+recall)
    k_f1 = 2 * k_precision * k_recall / (k_precision + k_recall + 1e-16)
    ukkin_f1 = 2 * ukkin_precision * ukkin_recall / (ukkin_precision + ukkin_recall + 1e-16)
    unk_f1 = 2 * unk_precision * unk_recall / (unk_precision + unk_recall + 1e-16)

    print('Precision:=> known:{:.4f}, unknown_kin-related:{:.4f}, unknown-non-kin:{:.4f}'.
          format(k_precision, ukkin_precision, unk_precision))
    print('Recall:=> known:{:.4f}, unknown_kin-related:{:.4f}, unknown-non-kin:{:.4f}'.
          format(k_recall, ukkin_recall, unk_recall))
    print('F1:=> known:{:.4f}, unknown_kin-related:{:.4f}, unknown-non-kin:{:.4f}'.
          format(k_f1, ukkin_f1, unk_f1))
    print(
        'ACC:{:.4f},AUC-known-unknon:{:.4f},AUC-kin-nonkin:{:.4f}, F1:{:.4f}'.format(acc, auroc_knun, auroc_kinon, F1))

    return [acc, auroc_knun, auroc_kinon, F1,
            k_precision, ukkin_precision, unk_precision,
            k_recall, ukkin_recall, unk_recall,
            k_f1, ukkin_f1, unk_f1]

def gather_outputs(net, criterion, dataloader, options):
    ''' Tests data and returns outputs and their ground truth labels.
        data_idx        0 returns logits, 1 returns distances to anchors
        use_softmax     True to apply softmax
        unknown         True if an unknown dataset
        only_correct    True to filter for correct classifications as per logits
    '''
    X = []
    y = []
    with torch.no_grad():
        for data, labels in dataloader:
            if options['use_gpu']:
                data, labels = data.cuda(), labels.cuda()

            with torch.set_grad_enabled(False):
                a, b = net(data, True)
                logits, _ = criterion(a, b)

                X += logits.cpu().detach().tolist()
                y += labels.cpu().tolist()

    X = np.asarray(X)
    y = np.asarray(y)

    return X, y