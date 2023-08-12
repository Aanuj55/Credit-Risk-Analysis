# -*- coding: utf-8 -*-
"""
Created on Thu May  7 16:35:01 2020

@author: AANUJ JAIN
"""


import pandas as pd
import numpy as np

creditData = pd.read_csv('BNG(credit-g).csv')
X = creditData.select_dtypes(['int64','float64'])
y = creditData['class']
#%%

categoryColumns = creditData.select_dtypes('object')

for name in categoryColumns:
    #uniqueLabels = list(creditData[name].unique())
    if name != 'class':
        df1 = creditData.groupby([name,'class'])['class'].agg('count')
        
        print(df1.unstack().reset_index())
        
#%%
'''T test analysis'''
for name in categoryColumns:
    goodPer = []
    if name != 'class':
        df1 = creditData.groupby([name,'class'])['class'].agg('count')
        df1 = df1.unstack().reset_index()
        for ix,label in enumerate(df1[name]):
            lab = label
            totalCount = df1['bad'][ix] + df1['good'][ix]
            labelPercent = totalCount / len(creditData[name]) * 100
            goodCount = df1['good'][ix]
            badCount = df1['bad'][ix]
            goodPercent = goodCount / totalCount * 100
            goodPer.append(goodPercent)
            badPercent = badCount/totalCount *100
        avg = sum(goodPer)/len(goodPer)
        sqrsum = 0
        for val in goodPer:
            sqrsum += ((val-avg)**2)/(len(goodPer)-1)
        std = (sqrsum**0.5)/((len(goodPer))**0.5)
        tscore = (avg-70)/(std)
        print(name, abs(tscore))
        
#%%
''' Chi square test for categorical independence'''
        
catData = {} 
chiValues = {}
categoricalFeatures = list(categoryColumns.columns)
for name in categoricalFeatures:  
    if name != 'class':
        observed = []
        expected = []
        catData[name] = {}
        df1 = creditData.groupby([name,'class'])['class'].agg('count')
        df1 = df1.unstack().reset_index()
        for ix,label in enumerate(df1[name]):
            lab = label
            totalCount = df1['bad'][ix] + df1['good'][ix]
            labelPercent = totalCount / len(creditData[name]) * 100
            goodCount = df1['good'][ix]
            badCount = df1['bad'][ix]
            observed.append(goodCount)
            expected.append(0.699774*totalCount)
            observed.append(badCount)
            expected.append(0.300226*totalCount)
            goodPercent = goodCount / totalCount * 100
            badPercent = badCount/totalCount *100    
            catData[name][label] = [labelPercent,goodPercent,totalCount]
            
        chiScore = 0
        obs = np.array(observed)
        expect = np.array(expected)
        chiScore = sum(((obs-expect)**2)/expect)
        chiValues[name] = chiScore
        
featureByChi = sorted(chiValues.items(), key=lambda x:x[1], reverse=True)
print(featureByChi)
        
#%%

'''MULTICOLLINEARITY b/w dependent variables using chisquare'''
count =0
for i in range(len(categoricalFeatures)-1):
    name1 = categoricalFeatures[i]
    chiValues = {}
    for j in range(i+1,len(categoricalFeatures)):
        name = categoricalFeatures[j]
        observed = []
        expected = []
        df1 = creditData.groupby([name,name1])['class'].agg('count')
        df1 = df1.unstack()
        for label in list(df1.index):
            rowSum = df1.loc[label].sum()
            for otherlabel in list(df1.columns):
                observed.append(df1.loc[label][otherlabel])
                colSum = df1[otherlabel].sum()
                ext = colSum/1000000
                expected.append(ext*rowSum)
            
        chiScore = 0
        obs = np.array(observed)
        expect = np.array(expected)
        chiScore = sum(((obs-expect)**2)/expect)
        chiValues[name] = chiScore
    
    featureByChi = sorted(chiValues.items(), key=lambda x:x[1], reverse=True)
    print(name1)
    print(featureByChi)
#%%

'''MULTICOLLINEARITY b/w dependent variables using chisquare function'''
cramerDict ={}
categoricalFeatures = list(categoryColumns.columns)
from scipy.stats import chi2_contingency
for i in range(len(categoricalFeatures)-1):
    name1 = categoricalFeatures[i]
    chiValues = {}
    for j in range(i+1,len(categoricalFeatures)):
        name = categoricalFeatures[j]
        df1 = creditData.groupby([name,name1])['class'].agg('count')
        df1 = df1.unstack()
        chi_val, p, dof, expected = chi2_contingency(df1)
        key = (name1,name)

        cram = (chi_val/1000000/min(len(df1.index)-1,len(df1.columns)-1))**0.5
        cramerDict[key] = cram

cramerList = sorted(cramerDict.items(), key = lambda x:x[1])
print(cramerList)
    
        
        
        
#%%

      
fileName = open('Features.csv','w')
fileName.write('Feature Name,Label Name,volume,volume percent,good count,good percent,bad count,bad percent\n')
for name in categoryColumns:
    fileName.write(name)
    if name != 'class':
        df1 = creditData.groupby([name,'class'])['class'].agg('count')
        df1 = df1.unstack().reset_index()
        for ix,label in enumerate(df1[name]):
            lab = label
            totalCount = df1['bad'][ix] + df1['good'][ix]
            labelPercent = totalCount / len(creditData[name]) * 100
            goodCount = df1['good'][ix]
            badCount = df1['bad'][ix]
            goodPercent = goodCount / totalCount * 100
            badPercent = badCount/totalCount *100
            
            fileName.write(',{0},{1},{2},{3},{4},{5},{6}\n'.format(lab,totalCount,labelPercent,goodCount,goodPercent,badCount,badPercent))

fileName.close()
                
                
             
#%%
'''Chi square using library and find p values'''
from scipy.stats import chi2_contingency

for name in categoryColumns:
    observed = []
    expected = []
    goodCountlist = []
    badCountlist = []
    if name != 'class':
        df1 = creditData.groupby([name,'class'])['class'].agg('count')
        df1 = df1.unstack()
# we create contingency table 
        table = df1
        print(name)
# Get chi-square value , p-value, degrees of freedom, expected frequencies using the function chi2_contingency
        chi_val, p, dof, expected = chi2_contingency(table)
# select significance value
        alpha = 0.05
# Determine whether to reject or keep your null hypothesis
        print('significance=%.3f, p=%.3f , %s' % (alpha, p, chi_val))
        if p <= alpha:
           print('Variables are associated (reject H0)')
        else:
            print('Variables are not associated(fail to reject H0)')
    
               
                
        

#%%
''' Mutual information method to filter categorical features'''
mutInfo = {}
import numpy as np
Columns = ['residence_since','num_dependents','existing_credits','installment_commitment']
del categoryColumns['class']
for name in categoryColumns:
   sum = 0
   df1 = creditData.groupby([name,'class'])['class'].agg('count')
   df1 = df1.unstack().reset_index()
   for ix,label in enumerate(df1[name]):
            lab = label
            totalCount = df1['bad'][ix] + df1['good'][ix]
            p_xybad = df1['bad'][ix]/1000000
            p_ygood = 699774/1000000
            p_ybad = 300226/1000000 
            p_xygood = df1['good'][ix]/1000000
            p_x = totalCount/1000000
            sum += p_xybad*(np.log2(p_xybad/(p_x*p_ybad))) + p_xygood*(np.log2(p_xygood/(p_x*p_ygood)))
   #print(name,sum)
   mutInfo[name] = sum
   
featureByMutual = sorted(mutInfo.items(),key=lambda x: x[1],reverse=True)
print(featureByMutual)
#%%
def search_score(featureByMutual, feature):
    for ix,val in enumerate(featureByMutual):
        if val[0] == feature:
            return val[1],ix+1

with open('Categorical Features Selection.csv','w') as file:
    file.write('Feature,chisquare,mutual info,rank by chi,rank by MI\n')
    for ix in range(len(featureByMutual)):
        rankByChi = ix+1
        feature = featureByChi[ix][0]
        mutualScore, rankByMutual= search_score(featureByMutual,feature)
        file.write('{0},{1},{2},{3},{4}\n'.format(feature,featureByChi[ix][1],mutualScore,
                                              rankByChi,rankByMutual))
























