# -*- coding: utf-8 -*-
"""
Created on Tue May 12 10:37:09 2020

@author: AANUJ JAIN
"""


import pandas as pd
import numpy as np

creditData = pd.read_csv('BNG(credit-g).csv')
categoricalFeatures = list(creditData.select_dtypes('object').columns)
del categoricalFeatures[categoricalFeatures.index('class')]

#%%
'''Encode target '''
target = creditData['class']
target = target.replace({'good':0,'bad':1})
creditData['class'] = target

#%%
'''Mean encoding implementation'''

creditValues = pd.DataFrame()
for feature in categoricalFeatures:
    mean_encode = creditData.groupby(feature)['class'].mean()
    creditValues['encoded_'+feature] = creditData[feature].map(mean_encode)


#%%
'''Binary Encoding'''

import category_encoders as ce
creditValues = pd.DataFrame()

bin_encoder = ce.BinaryEncoder()    
dataFrame = bin_encoder.fit_transform(creditData)
creditValues = dataFrame.copy()


#%%
'''Hashing Encoder'''

creditValues = pd.DataFrame()
hash_encoder = ce.HashingEncoder()
dataFrame = hash_encoder.fit_transform(creditData)
creditHashData = dataFrame.copy()



#%%+[]
'''log(odds) to calculate WoE'''

for feature in categoricalFeatures:
    dataGroup = creditData.groupby(feature)['class'].mean()
    dataGroup = pd.DataFrame(dataGroup)
    dataGroup = dataGroup.rename(columns={'class':'bad'})
    
    dataGroup['good'] = np.where(1-dataGroup['bad']==0,0.0001,1-dataGroup['bad'])
    dataGroup['WoE'] = np.log(dataGroup.bad/dataGroup.good)
    print(dataGroup)
    new_name = 'encoded_'+str(feature)
    creditData[new_name] = creditData[feature].map(dataGroup['WoE'])

#%%
creditData = creditData.drop([i for i in categoricalFeatures],axis=1)








