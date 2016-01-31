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
def generate_data(glob_param,length=30):
    """
    This just prepare data and splits data to training and testing parts
    """
    filelist = glob.glob(glob_param)
    post_fix = glob_param.split("/")[2]
    #filelist = os.listdir('/'.join([root,folder]))
    x_data = None
    y_data = None
    LENGTH = len(filelist)
    count = 0
    sample = 2000
    x_data = np.zeros((sample*LENGTH,30,13))
    y_data = np.zeros((sample*LENGTH,1))
    y_data_test = np.zeros((sample*LENGTH,1))
    print x_data.shape,y_data.shape
    index = 0
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
            #,\1 if myarray[i+length,5] > myarray[i+length-1,5] else 0)
            tx_data = [myarray[i:i+length,2:15]\
                for i in range(0,min(myarray.shape[0] - length,sample))]
            ty_data = [1 if myarray[i+length,5] > myarray[i+length-1,5] else 0\
                for i in range(0,min(sample,myarray.shape[0] - length))]
            ty_data_test = [1 if myarray[i+length-1,5] > myarray[i+length-2,5] else 0\
                for i in range(0,min(sample,myarray.shape[0] - length))]
            print len(tx_data),min(sample,myarray.shape[0] - length)
            if len(tx_data)==0:
                continue
            tx_data = np.dstack(tx_data)
            tx_data = np.rollaxis(tx_data,-1)
            ty_data = np.dstack(ty_data)
            ty_data = np.rollaxis(ty_data,-1)
            ty_data = ty_data.reshape(ty_data.shape[0],1)
            ty_data_test = np.dstack(ty_data_test)
            ty_data_test = np.rollaxis(ty_data_test,-1)
            ty_data_test = ty_data_test.reshape(ty_data_test.shape[0],1)
            x_data[index:index+tx_data.shape[0]] = tx_data
            y_data[index:index+ty_data.shape[0]] = ty_data
            y_data_test[index:index+ty_data_test.shape[0]] = ty_data_test
            index = index+tx_data.shape[0]
            #x_data = np.vstack((x_data,tx_data))
            #y_data = np.vstack((y_data,ty_data))
        count = count + 1
        print count,index,LENGTH

    print y_data[:index].shape
    print x_data[:index].shape
    sti = 0
    for i in range(y_data[:index].shape[0]):
        sti += int(y_data[i,0] == y_data_test[i,0])
    print sti*1.0 / y_data[:index].shape[0]
    x = x_data[:index]
    y = y_data[:index]
    print x.shape
    print y.shape
    split = int(x.shape[0]*(1-0.1))
    print x[:split].shape

    np.savez('mat/train_data_all', data=x[:split], label=y[:split])
    np.savez('mat/test_data_all', data=x[split:], label=y[split:])

    #npzfile = np.savez('mat/train_data_{}.npz'.format(post_fix),data=x_data, label=y_data)
    #print npzfile['data'].shape
    #print npzfile['label'].shape
    print 'end'
    #return (x_data,y_data)

def train_test_split(dataset,test_size=0.1):

    index = int(dataset['label'].shape[0]*(1-test_size))
    return (dataset['data'][:index],dataset['label'][:index],\
    dataset['data'][index:],dataset['label'][index:])

if __name__ == "__main__":
    generate_data("./data/*/*.csv")
#_load_data(True)
