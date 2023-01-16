import numpy as np
import pandas as pd
import seaborn as sns
from random import shuffle
import pickle
import os
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve


image_pth = '/FIW/Kinship Verification (T-1)/TRAIN/train-faces'



for kin_type in ['fd','fs','md','ms','bb','sibs','ss','gfgs','gfgd','gmgd','gmgs']:
    ls = kin_type

    ######## read csv
    file = open('/FIW/folds_5splits-csv/folds_5splits-csv/{}.csv'.format(kin_type))
    csvreader = csv.reader(file)
    # to list
    header = next(csvreader)
    print(header)
    ori_train_ls = []
    for row in csvreader:
        ori_train_ls.append(row)


    ######## check image and update folds
    train_ls = []
    temp23 = []
    temp_neg = []
    for row in ori_train_ls:

        temp2 = row[2].split('/')[0]+'/'+row[2].split('/')[1]
        temp3 = row[3].split('/')[0]+'/'+row[3].split('/')[1]

        if int(row[1]) != 0:
            if [temp2,temp3] == temp23:
                continue
            else:
                temp23 = [temp2,temp3]
                if os.path.exists(os.path.join(image_pth,temp2)):
                    train_ls.append([int(row[0]),int(row[1]),temp2,temp3])
        else:
            if temp2 == temp_neg:
                continue
            else:
                temp_neg = temp2
                if os.path.exists(os.path.join(image_pth, temp2)):
                    train_ls.append([int(row[0]), int(row[1]), temp2, temp3])


    se_train_ls = []
    for i in range(1,6):
        for row in train_ls:
            if row[0] == i:
                se_train_ls.append(row)

    ######## save

    with open('./{}.pkl'.format(kin_type), 'wb') as fp:
        pickle.dump(se_train_ls, fp)


    print('=='*10,"Finish","=="*10)

