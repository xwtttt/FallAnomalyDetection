'''
Description: 
Author: liguang-ops
Github: https://github.com/liguang-ops
Date: 2020-12-14 11:43:51
LastEditors: liguang-ops
LastEditTime: 2021-04-22 15:55:47
'''
import re
import pandas as pd
import numpy as np
import os
import sys
def compute(filename):
    print(filename)
    time = []
    device = []
    value = []
    path = os.getcwd()
    dir = os.path.join(path,filename)
    with open(dir,"r") as f:
        lines = f.readlines()
        i = 1
        for line in lines:
            data = re.match(r'\w\t(.*)\tNotification received from (.*?), value: \(0x\)\s(.*)',line)
            if data:
                #print('matched  i:{}'.format(i),end=' ')
                #print(data.group(1),end=' ')
                #print(data.group(2),end=' ')
                #print(data.group(3))
                time.append(data.group(1))
                device.append(data.group(2))
                value.append(data.group(3))


    print(len(time),len(device),len(value))
    #value_replaced = [str.replace('-','') for str in value]
    #value_done = [[].append(str[4*i:4(i+1)] for i in range(length/4)) for str in value_replaced]
    
    all = {
        'time':time,
        'device':device,
        'value':value,
    }
    df = pd.DataFrame(all)
    
    def split(x):
        x = x.replace('-','')
        if len(x) == 12:
            return str(int(x[0:4],16)) + ',' + str(int(x[4:],16))
        elif len(x) == 24:
            return ','.join([str(int(x[4*i:4*(i+1)],16)) for i in range(6)])

    df['value_split']= df['value'].apply(split)
    df[['X','Y','Z','RX','RY','RZ']] = df['value_split'].str.split(',',expand=True).astype(np.float32)

    group = df.groupby('device')
    for key,value in group:
        value.to_excel(key[0:8] + '.xlsx',index=False)
    #df.to_excel('Joe.xlsx',index=False)

if __name__=="__main__":
    compute(sys.argv[1])