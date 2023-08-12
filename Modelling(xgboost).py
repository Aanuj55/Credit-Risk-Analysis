# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 16:17:09 2020

@author: AANUJ JAIN
"""


import pandas as pd
from sklearn import tree
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,roc_auc_score,roc_curve
import matplotlib.pyplot as plt
import category_encoders as ce
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO 
from IPython.display import Image
import pydotplus
import graphviz
from xgboost import XGBClassifier

# file = open("RandomForestModel_13.csv",'w')
# file.write("Total features,max_depth,min_samples_split,min_samples_leaf,max_leaf_node,min_impurity_decrease,AUC test,AUC train\n")
# file.close()

#%%
'''MODEL 8 '''
cat_1 = ['checking_status','credit_history','purpose','savings_status']
cat_2 = ['property_magnitude','housing','employment','other_payment_plans','personal_status']
cat_3 = ['other_parties','foreign_worker','job','own_telephone']
num_1 = ['duration']
num_2 = ['credit_amount','age','installment_commitment']
num_3 = ['existing_credits','num_dependents','residence_since']

#num_2.remove('credit_amount')
selectedFeatures = num_1+cat_1+num_2+cat_2+cat_3+num_3

selectedList = selectedFeatures
creditData = pd.read_csv('BNG(credit-g).csv')
y = creditData['class'].replace({'bad':1,'good':0})
X = creditData[selectedList] 

X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size=0.2,random_state=13)

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

model = XGBClassifier(random_state=55,n_estimators=80,max_depth=8,learning_rate=0.25,subsample=0.8,reg_lambda=25,min_child_weight=100)
model.fit(X_train,Y_train)
predictions = model.predict(X_test)

accuracy_test = accuracy_score(Y_test,predictions)*100
accuracy_train = accuracy_score(Y_train,model.predict(X_train))*100
print ("Accuracy Test : ",accuracy_test)
print("Accuracy Training: ",accuracy_train)
auc_test = roc_auc_score(Y_test, model.predict_proba(X_test)[:,1])
auc_train = roc_auc_score(Y_train, model.predict_proba(X_train)[:,1])
test_auc = (round(auc_test,4))*100
train_auc = (round(auc_train,4))*100
print("test auc: ",test_auc)
print("train auc: ",train_auc)
importance = list(model.feature_importances_)
print("{0:<20}{1:>10}".format("Feature","importance"))
#print("Leaves",model.get_n_leaves())
params = model.get_params()
print(params)
for val in zip(X_train.columns,importance):
    print("{0:<30}{1:>10}%".format(val[0],round(val[1]*100,2)))


#%%
pairsToMerge = [('checking_status','savings_status'),('property_magnitude','housing'),('credit_history','other_payment_plans'),('purpose','personal_status'),('foreign_worker','own_telephone'),('job','own_telephone'),('purpose','job'),('property_magnitude','other_parties'),
                ('property_magnitude','job'),('employment','job'),('personal_status','foreign_worker')]
addedFeature = ['duration','age']

creditData = pd.read_csv('BNG(credit-g).csv')
y = creditData['class'].replace({'bad':1,'good':0})

X = pd.DataFrame()
for pair in pairsToMerge:
    X[pair[0]+"_"+pair[1]] = creditData[pair[0]] + creditData[pair[1]]

for val in addedFeature:
    X[val] = creditData[val]
    
X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size=0.2,random_state=13)

tarEncoder = ce.TargetEncoder(smoothing=51)
cat_cols = list(X_train.select_dtypes('object').columns)
tar_cols_train = tarEncoder.fit_transform(X_train[cat_cols],Y_train)
tar_cols_test = tarEncoder.transform(X_test[cat_cols])

enc_X_train = tar_cols_train.join(X_train.select_dtypes(['int64','float64']))
enc_X_test = tar_cols_test.join(X_test.select_dtypes(['int64','float64']))
X_train = enc_X_train
X_test = enc_X_test
print("ENCODING COMPLETED")   
