# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 19:55:39 2020

@author: AANUJ JAIN
"""



import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler,OneHotEncoder
#from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,roc_auc_score,roc_curve
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from scipy.stats import norm
import category_encoders as ce


#%%


cat_1 = ['checking_status','credit_history','purpose','savings_status']
cat_2 = ['property_magnitude','housing','employment','other_payment_plans','personal_status']
cat_3 = ['other_parties','foreign_worker','job','own_telephone']
num_1 = ['duration']
num_2 = ['credit_amount','age','installment_commitment']
num_3 = ['existing_credits','num_dependents','residence_since']


creditData = pd.read_csv('BNG(credit-g).csv')
target = creditData['class'].replace({'good':0,'bad':1})

catCols = list(creditData.select_dtypes('object').columns)
catCols.remove('class')

Y = target
X = creditData.drop(['class'],axis=1)
#%%
for ix1 in range(len(catCols)-1):
    item1 = catCols[ix1]
    new1 = X.loc[:,item1].copy()
    for ix2 in range(ix1+1,len(catCols)):
           item2 = catCols[ix2]
           new2 = X.loc[:,item2].copy()
           X[item1+"_"+item2] = new1+new2
#%%
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=13)

tarEncoder = ce.TargetEncoder()
cat_cols = list(X_train.select_dtypes('object').columns)
tar_cols_train = tarEncoder.fit_transform(X_train[cat_cols],Y_train)
tar_cols_test = tarEncoder.transform(X_test[cat_cols])

enc_X_train = tar_cols_train.join(X_train.select_dtypes(['int64','float64']))
enc_X_test = tar_cols_test.join(X_test.select_dtypes(['int64','float64']))
X_train = enc_X_train
X_test = enc_X_test
print("ENCODING COMPLETED")   

#%%

#%%
corrDf = X_train.corr()
#%%
corrDf.to_csv('Correlation.csv')























