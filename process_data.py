# -*- coding: utf-8 -*-
import numpy as np
import csv
import os
from itertools import groupby
from operator import itemgetter
import pandas as pd
import glob
import sys
defaultencoding = 'utf-8'
if sys.getdefaultencoding() != defaultencoding:
    reload(sys)
    sys.setdefaultencoding(defaultencoding)

def process_row(valuelist):
    """
    process each row of stock information
    """
    assert len(valuelist) == 16
    valuelist[6] = float(valuelist[6])*1e-7
    valuelist[9] = float(valuelist[9])*1e-9
    valuelist[10:14] = map(lambda x: float(x) * 1e-2, valuelist[10:14])
    valuelist[14] = float(valuelist[14])*1e-9

    return valuelist

def _load_data(refer=True):
    """
    for process raw data into data folder. ture flag　to refer everyday overal stock information
    """

    REFER = []
    if refer:
        with open('index.csv','rb') as f:
            spamreader = csv.reader(f, delimiter=' ', quotechar='|')
            for row in list(spamreader):
                row = ','.join(row)
                index = row.find(',') + 1
                row =  row[index:]
                REFER.append(row.split(','))

    #scan all raw files
    filelist = os.listdir('./raw')
    filelist = sorted(filelist)
    count = 1
    data = []
    folder = set()
    for filename in filelist:
        with open(''.join(['raw/',filename]),'rb') as f:
            spamreader = csv.reader(f, delimiter=' ', quotechar='|')
            for row in list(spamreader)[1:]:
                row = ','.join(row)
                index = row.find(',') + 1
                row =  (row[:index] + str(count) + ',' + row[index:]).decode("gb2312")
                rindex = row.rfind(',') + 1
                row =  (row[:rindex] + ','.join(REFER[count-1]) + ',' + row[rindex:])
                row = row.split(',')
                row = process_row(row)
                data.append(row)
                folder.add(row[-1])
        count = count + 1
    data.sort(key = itemgetter(0))
    groups = groupby(data, itemgetter(0))
    result = [[item for item in value] for (key, value) in groups]
    for item  in folder:
        directory = ''.join(['data/',item])
        if not os.path.exists(directory):
            os.makedirs(directory)
    print len(folder)
    for items  in result:
        for item  in items:
            item[-1] = item[-1].encode('utf-8')
        my_df = pd.DataFrame(items)
        my_df.to_csv(''.join(['data/',items[-1][-1],'/',items[-1][0],'.csv']), index=False, header=False)

def generate_data(glob_param,length=30):
    """
    This just prepare data and splits data to training and testing parts
    """
    filelist = glob.glob(glob_param)
    post_fix = glob_param.split("/")[2]
    x_data = None
    y_data = None
    LENGTH = len(filelist)
    count = 0
    sample = 1500
    x_data = np.zeros((sample*LENGTH,30,13))
    y_data = np.zeros((sample*LENGTH,1))
    x_data_test = np.zeros((sample*LENGTH,30,13))
    y_data_test = np.zeros((sample*LENGTH,1))

    #ratio for train/test split
    ratio = 0.9
    #predict days after the lastest day of training data
    lasting_day = 7

    print x_data.shape,y_data.shape
    index = 0
    index_test = 0
    for filename in filelist:
        with open(filename,'rb') as f:
            spamreader = csv.reader(f, delimiter=' ', quotechar='|')
            mylist = []
            for row in list(spamreader):
                row = ','.join(row)
                row = row.split(',')
                mylist.append(row)
            mydf = pd.DataFrame(mylist)
            myarray = np.array(mydf)
            tt_index = int(ratio*min(myarray.shape[0] - length,sample))
            #,\1 if myarray[i+length,5] > myarray[i+length-1,5] else 0)
            tx_data = [myarray[i:i+length,2:15]\
                for i in range(0,tt_index)]
            ty_data = [1 if myarray[i+length,5] > myarray[i+length-1,5] else 0\
                for i in range(0,tt_index)]
            tx_data_test = [myarray[i:i+length,2:15]\
                for i in range(tt_index,min(tt_index+lasting_day,myarray.shape[0] - length))]
            ty_data_test = [1 if myarray[i+length-1,5] > myarray[i+length-2,5] else 0\
                for i in range(tt_index,min(tt_index+lasting_day,myarray.shape[0] - length))]
            if len(tx_data)==0:
                continue
            tx_data = np.dstack(tx_data)
            tx_data = np.rollaxis(tx_data,-1)
            tx_data_test = np.dstack(tx_data_test)
            tx_data_test = np.rollaxis(tx_data_test,-1)
            ty_data = np.dstack(ty_data)
            ty_data = np.rollaxis(ty_data,-1)
            ty_data = ty_data.reshape(ty_data.shape[0],1)
            ty_data_test = np.dstack(ty_data_test)
            ty_data_test = np.rollaxis(ty_data_test,-1)
            ty_data_test = ty_data_test.reshape(ty_data_test.shape[0],1)
            x_data[index:index+tx_data.shape[0]] = tx_data
            y_data[index:index+ty_data.shape[0]] = ty_data
            x_data_test[index_test:index_test+tx_data_test.shape[0]] = tx_data_test
            y_data_test[index_test:index_test+ty_data_test.shape[0]] = ty_data_test
            index = index+tx_data.shape[0]
            index_test = index_test+tx_data_test.shape[0]
        count = count + 1

    print y_data[:index].shape
    print x_data[:index].shape
    print x_data_test[:index_test].shape
    print y_data_test[:index_test].shape
    mean = np.mean(np.reshape(x_data[:index],(-1,13)), axis=0)
    std  = np.std(np.reshape(x_data[:index],(-1,13)), axis=0)
    print mean
    print std

    #np.savez('mat/mean_std', mean=mean, std=std)
    #np.savez('mat/train_data_all', data=x_data[:index], label=y_data[:index])
    np.savez('mat/test_data_{}'.format(lasting_day), data=x_data_test[:index_test], label=y_data_test[:index_test])
    print 'end'

def train_test_split(dataset,test_size=0.1):

    index = int(dataset['label'].shape[0]*(1-test_size))
    return (dataset['data'][:index],dataset['label'][:index],\
    dataset['data'][index:],dataset['label'][index:])

if __name__ == "__main__":
    #_load_data(True)     #uncomment
    generate_data("./data/*/*.csv")
