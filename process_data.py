# -*- coding: utf-8 -*-
import numpy as np
import csv
import os
from itertools import groupby
from operator import itemgetter
import pandas as pd

import sys
defaultencoding = 'utf-8'
if sys.getdefaultencoding() != defaultencoding:
    reload(sys)
    sys.setdefaultencoding(defaultencoding)
def process_row(valuelist):
    assert len(valuelist) == 16
    valuelist[6] = float(valuelist[6])*1e-7
    valuelist[9] = float(valuelist[9])*1e-9
    valuelist[10:14] = map(lambda x: float(x) * 1e-2, valuelist[10:14])
    valuelist[14] = float(valuelist[14])*1e-9

    return valuelist
def _load_data(refer=True):
    REFER = []
    if refer:
        with open('index.csv','rb') as f:
            spamreader = csv.reader(f, delimiter=' ', quotechar='|')
            for row in list(spamreader):
                row = ','.join(row)
                index = row.find(',') + 1
                row =  row[index:]
                REFER.append(row.split(','))
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
def generate_data(root, folder,length=30):
    """
    This just prepare data and splits data to training and testing parts
    """
    filelist = os.listdir('/'.join([root,folder]))
    x_data = None
    y_data = None
    count = 1
    for filename in filelist:
        with open('/'.join([root,folder,filename]),'rb') as f:
            spamreader = csv.reader(f, delimiter=' ', quotechar='|')
            mylist = []
            for row in list(spamreader):
                row = ','.join(row)
                row = row.split(',')
                mylist.append(row)
            mydf = pd.DataFrame(mylist)
            myarray = np.array(mydf)
            #,\1 if myarray[i+length,5] > myarray[i+length-1,5] else 0)
            tx_data = [myarray[i:i+length,2:15]\
                for i in range(0,myarray.shape[0] - length)]
            ty_data = [1 if myarray[i+length,5] > myarray[i+length-1,5] else 0\
                for i in range(0,myarray.shape[0] - length)]
            tx_data = np.dstack(tx_data)
            tx_data = np.rollaxis(tx_data,-1)
            ty_data = np.dstack(ty_data)
            ty_data = np.rollaxis(ty_data,-1)
            ty_data = ty_data.reshape(ty_data.shape[0],1)
            if count == 1:
                x_data = tx_data
                y_data = ty_data
            x_data = np.vstack((x_data,tx_data))
            y_data = np.vstack((y_data,ty_data))
        count = count + 1

    print y_data.shape
    print x_data.shape
    np.savez('temp/train_data', data=x_data, label=y_data)
    #npzfile = np.load('temp/train_data.npz')
    #print npzfile['data'].shape
    #print npzfile['label'].shape
    print 'end'
    return (x_data,y_data)

def train_test_split(dataset,test_size=0.1):

    index = int(dataset['label'].shape[0]*(1-test_size))
    return (dataset['data'][:index],dataset['label'][:index],\
    dataset['data'][index:],dataset['label'][index:])

#generate_data("data","计算机")
#_load_data(True)
