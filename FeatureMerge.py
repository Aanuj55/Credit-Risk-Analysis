# -*- coding: utf-8 -*-
"""
Created on Mon May 11 12:43:46 2020

@author: AANUJ JAIN
"""
import pandas as pd
import numpy as np

    
def labelCountCumcount(col_data):
    ''' Calculate labels below certain percentile position for categorical data'''
    labelCount = col_data.value_counts()
    sortedLabel = list(labelCount.index)
    
    sortedCount = list(labelCount.values)
    cumCount = [sortedCount[0]]
    for i in range(1,len(sortedCount)):
        cumCount.append(sortedCount[i] + cumCount[i-1])
    cumCount = np.array(cumCount)
    labelPercentileDict = {}
    for p in range(0,101,5):
        value = (cumCount[-1]+1) * (p/100) -1
        label_below = len(cumCount[cumCount < value]) + 1
        labelPercentileDict[p]= label_below
        print("Number of labels below {0}th percentile is:{1:>10}".format(p,label_below))       
    
    return sortedLabel,labelPercentileDict.copy()
    

def merge(creditData,feature,sortedLabel,labelPercentileDict,startPer,endPer,newName):
    '''merge certain labels'''
    startIndex = labelPercentileDict[startPer]-1
    endIndex = labelPercentileDict[endPer]-1
    labelsToMerge = sortedLabel[startIndex:endIndex+1]
    newCol = []
    
    groupCount = creditData.groupby([feature,'class'])['class'].count()
    df = groupCount.unstack().reset_index()
    for label in creditData[feature]:
        if label in labelsToMerge:
            badCountIndex = list(df[feature]).index(label)
            eventRate = df['bad'][badCountIndex] / (df['bad'][badCountIndex]+df['good'][badCountIndex])
            if eventRate < 0.300226:
                newCol.append(newName+'_below')
            else:
                newCol.append(newName+'_above')
        else:
            newCol.append(label)
             
    return newCol


#%%

creditData = pd.read_csv('BNG(credit-g).csv')
categoricalData = creditData.select_dtypes('object')

del categoricalData['class']
#%%
sortedLabel,labelPercentileDict = labelCountCumcount(creditData['purpose'])
col_data = creditData['purpose']
categoricalData['purpose'] = merge(creditData,'purpose',sortedLabel,labelPercentileDict,95,100,'merge')



            
            
            













  
                        
                        
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
        
        