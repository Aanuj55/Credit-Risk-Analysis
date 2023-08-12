# -*- coding: utf-8 -*-
"""
Created on Fri May 29 20:40:50 2020

@author: AANUJ JAIN
"""


import pandas as pd
from sklearn import tree
import numpy as np
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score,roc_auc_score,roc_curve
import matplotlib.pyplot as plt
import category_encoders as ce
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO 
from IPython.display import Image
import pydotplus
import graphviz
from sklearn.ensemble import RandomForestClassifier
# file = open("decisionTreeModel_8.csv",'w')
# file.write("Total features,max_depth,min_samples_split,min_samples_leaf,Leaves in model,AUC test,AUC train\n")
#file.close()
#%%
cat_1 = ['checking_status','credit_history','purpose','savings_status']
cat_2 = ['property_magnitude','housing','employment','other_payment_plans','personal_status']
cat_3 = ['other_parties','foreign_worker','job','own_telephone']
num_1 = ['duration']
num_2 = ['credit_amount','age','installment_commitment']
num_3 = ['existing_credits','num_dependents','residence_since']

num_2.remove('credit_amount')
selectedFeatures = num_1+cat_1+num_2+cat_2+cat_3

selectedList = selectedFeatures
creditData = pd.read_csv('BNG(credit-g).csv')
y = creditData['class'].replace({'bad':1,'good':0})
X = creditData[selectedList] 

X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size=0.2,random_state=13)
    
tarEncoder = ce.TargetEncoder(min_samples_leaf=5)
cat_cols = list(X_train.select_dtypes('object').columns)
tar_cols_train = tarEncoder.fit_transform(X_train[cat_cols],Y_train)
tar_cols_test = tarEncoder.transform(X_test[cat_cols])

enc_X_train = tar_cols_train.join(X_train.select_dtypes(['int64','float64']))
enc_X_test = tar_cols_test.join(X_test.select_dtypes(['int64','float64']))
X_train = enc_X_train
X_test = enc_X_test
print("ENCODING COMPLETED")   

#%%

#file = open("decisionTreeModel_8.csv",'a')

model = tree.DecisionTreeClassifier(criterion = "gini",class_weight='balanced',random_state=41,max_depth=18,min_samples_split=40,min_samples_leaf=30,max_leaf_nodes=1100,min_impurity_decrease=0.0000001*5)
 
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
print("Leaves",model.get_n_leaves())
params = model.get_params()
print(params)
for val in zip(X_train.columns,importance):
    print("{0:<30}{1:>10}%".format(val[0],round(val[1]*100,2)))

# dot_data = tree.export_graphviz(model, out_file=None, 
#                     feature_names=list(X_train.columns),    
#                     filled=True, rounded=True,  
#                      special_characters=True)  
# graph = graphviz.Source(dot_data)
# print(graph) 
# file.write("{0},{1},{2},{3},{4},{5},{6}\n".format(len(X_train.columns),params['max_depth'],
#                                                             params['min_samples_split'],params['min_samples_leaf'],
#                                                             model.get_n_leaves(),test_auc,train_auc))
    
# file.close()
#%%

pairsToMerge = [('checking_status','savings_status'),('property_magnitude','housing'),('credit_history','other_payment_plans'),('purpose','personal_status'),('foreign_worker','own_telephone'),('job','own_telephone'),('purpose','job'),('property_magnitude','other_parties'),
                ('property_magnitude','job'),('employment','job'),('personal_status','foreign_worker')]
addedFeature = ['duration','age','installment_commitment']

creditData = pd.read_csv('BNG(credit-g).csv')
y = creditData['class'].replace({'bad':1,'good':0})

X = pd.DataFrame()
for pair in pairsToMerge:
    X[pair[0]+"_"+pair[1]] = creditData[pair[0]] + creditData[pair[1]]

for val in addedFeature:
    X[val] = creditData[val]
    
X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size=0.2,random_state=13)
    
tarEncoder = ce.TargetEncoder(smoothing=280,min_samples_leaf=940)
cat_cols = list(X_train.select_dtypes('object').columns)
tar_cols_train = tarEncoder.fit_transform(X_train[cat_cols],Y_train)
tar_cols_test = tarEncoder.transform(X_test[cat_cols])

enc_X_train = tar_cols_train.join(X_train.select_dtypes(['int64','float64']))
enc_X_test = tar_cols_test.join(X_test.select_dtypes(['int64','float64']))
X_train = enc_X_train
X_test = enc_X_test
print("ENCODING COMPLETED")   


#%%
# =============================================================================
# file = open("decisionTreeModel_23.csv",'w')
# file.write("Total features,max_depth,min_samples_split,min_samples_leaf,Leaves in model,AUC test,AUC train\n")
# file.close()
# =============================================================================


#%%
'''Model 23'''
#file = open("decisionTreeModel_23.csv",'a')
# para = {'max_leaf_nodes':[800,1000,1050],
#           'max_depth':[7,8,10],
#           'min_samples_split': [200,500,1200], 
#           'min_samples_leaf':[1500,1000,1250],
#           'random_state':[123]}
model = tree.DecisionTreeClassifier(criterion='gini',max_depth=17,random_state=41,min_samples_split=600,min_samples_leaf=400,max_leaf_nodes=1200,min_impurity_decrease=0.000001)
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
print("Leaves",model.get_n_leaves())
params = model.get_params()
print(params)
for val in zip(X_train.columns,importance):
    print("{0:<30}{1:>10}%".format(val[0],round(val[1]*100,2)))

# dot_data = tree.export_graphviz(model, out_file=None, 
#                     feature_names=list(X_train.columns),    
#                     filled=True, rounded=True,  
#                      special_characters=True)  
# graph = graphviz.Source(dot_data)
# print(graph) 
# file.write("{0},{1},{2},{3},{4},{5},{6}\n".format(len(X_train.columns),params['max_depth'],
#                                                             params['min_samples_split'],params['min_samples_leaf'],
#                                                             model.get_n_leaves(),test_auc,train_auc))
    
# file.close()
    


