# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 13:23:03 2020

@author: AANUJ JAIN
"""
import datetime as dt
import numpy as np
#%%

file = open('BNG(credit-g).csv')

for line in file:
        col_names = line.strip().split(',')
        break
    
file.close()
print(col_names)

#%%
'''Extract column names from file'''
data = {}
for col in col_names:
    data[col] = []
file = open('BNG(credit-g).csv')
for line in file:
    row = line.strip().split(',')
    for j,item in enumerate(row):
        data[col_names[j]].append(item)

for key in data.keys():
    del data[key][0]
        
file.close()
#%%
'''Make a column of Dates'''
start = dt.datetime.strptime("2018-01-01", "%Y-%m-%d")
end = dt.datetime.strptime("2020-01-01", "%Y-%m-%d")
date_list = \
    [start + dt.timedelta(days=x) for x in range(0, (end-start).days)] 

data['Date'] = []
count = 1000000//730
rem = 1000000%730

for j in range(0,count):
    data['Date'] += date_list 
    
data['Date'] += date_list[0:rem]
data["Date"] = list(map(lambda x: dt.datetime.strftime(x,"%Y-%m-%d") ,data['Date']))
col_names.append('Date')

#%%


def col_type(col_data):
   count_str = count_num = count_date = 0
   check_row = list(np.random.randint(1,len(col_data),size=100)) 
   for val in check_row:
       try:
           v=float(col_data[val])
           count_num += 1
       except:
              try:
                  v =  dt.datetime.strptime(col_data[val] , "%Y-%m-%d")
                  count_date += 1
              except:
                    count_str += 1
                    
   if count_num > 97:
       return 'numeric'
   elif count_str > 97:
       return 'str'
   else:
       return 'dateTime'

              
#%%
def missingRecord(col_data):
    mis_set = ['',np.NaN,None,'nan']
    dum = col_data.copy()
    count = 0
    i =0
    for item in col_data:
        if item in mis_set:
            count += 1
            dum.remove(item)
        else:
            i += 1
    return (dum,count)


def notParsedRecord(col_data,dtype):
    count = 0
    dum = col_data[:]
    if dtype == 'numeric':
        try:
            new = list(map(float,col_data))
        except:
            for x in col_data:
                try:
                    v = float(x)
                except:
                    dum.remove(x)
                    count += 1
                    
    if dtype == 'date':
        try:
            new = list(map(lambda x: dt.datetime.strptime(x,'%Y-%m-%d'),col_data))
        except:
            for x in col_data:
                try:
                    v = dt.datetime.strptime(x,'%Y-%m-%d')
                except:
                    dum.remove(x)
                    count += 1
            
    return (dum, count)
                

#%%
def unique(col_data):
    return len(set(col_data))

def num_percentile(col_data,p):
    val_list = sorted(col_data)
    pos = p*(len(val_list)+1)/100
    if pos != int(pos):
        pos = int(pos)
        return (float(val_list[pos])+float(val_list[pos+1]))/2
    else:
        return(float(val_list[int(pos)]))
    
    
def cat_percentile(col_data,p):
    ''' 
       Calculate labels below certain percentile position for categorical data
    '''
    dict1 = {}
    for val in col_data:
        if val in dict1.keys():
            dict1[val] += 1
        else:
            dict1[val] = 1
    
    ct_label = list(dict1.values())
    ct_label.sort()
    
    for i in range(1,len(ct_label)):
        if i != 0:
           ct_label[i] = ct_label[i] + ct_label[i-1]
    
    ct_label = np.array(ct_label)
    value  = (max(ct_label)+1)*p/100
    label_above = len(ct_label[ct_label >= value]) -1
    label_below = len(ct_label) - label_above
    return label_below

def dat_percentile(col_data,p):
    col_data =  list(map(lambda x: dt.datetime.strptime(x,'%Y-%m-%d'),col_data))
    col_data.sort()
    pos = (len(col_data)+1) * (p/100)
    
    return dt.datetime.strftime(col_data[int(pos)], "%Y-%m-%d")


def max_of(col_data):
    return max(col_data)

def min_of(col_data):
    return min(col_data)

def top_N(col_data,dict1,ct_set):    
    ''' It return the list of top 5 labels with their probability
        otherwise return None
    '''
    top_list = []  
    for num in range(1,6):
        try:
            labels = dict1[ct_set[num-1]]
            prob = ct_set[num-1]/len(col_data)
            if len(labels) > 5:
                 chosen_label = list(np.random.choice(labels,5))
                 labels = chosen_label
            top_list.append((labels, round(prob*100)))
        except:
            top_list.append((None,None))
    return top_list

def get_dict(col_data):
    '''
       It provides necessary dictionary for frequency and
       related labels
    '''
    lab_cnt = {}
    for val in col_data:
        if val in lab_cnt.keys():
           lab_cnt[val] += 1
        else:
             lab_cnt[val] = 1
    
    ct_list = list(lab_cnt.values())
    ct_set = list(set(ct_list))
    ct_set.sort(reverse = True)
    
    dict1 = {}
    for count in ct_set:
        dict1[count] = []
        
    for lab,count in lab_cnt.items():
          dict1[count].append(lab)
    return (dict1.copy(),ct_set)
          
def last_N(col_data,dict1,ct_set):
     '''
        It return the list of last 2 labels with their probability
        otherwise return None
     '''
     last_list = []   
     for num in range(1,3):
        try:
           labels = dict1[ct_set[num-1]]
           prob = ct_set[num-1]/len(col_data)
           if len(labels) > 5:
               chosen_label = list(np.random.choice(labels,5))
               labels = chosen_label
           last_list.append((labels, round(prob*100)))
        except:
              last_list.append((None,None))
     return last_list

#%%
with open('BNG_stats.csv','w') as file:
    file.write('variableName,variableType,totalRecords,missingRecords,notParsecRecords')
    file.write(',uniqueLabels,perc_0.05,perc_0.25,perc_0.50,perc_0.75,perc_0.95,perc_0.99')
    file.write(',MIN,MAX,top_1_val,top_1_prob,top_2_val,top_2_prob,top_3_val,top_3_prob')
    file.write(',top_4_val,top_4_prob,top_5_val,top_5_prob,last_1_val,last_1_prob,last_2_val,last_2_prob\n')
    for name in col_names:
        data[name], mis_ct = missingRecord(data[name])
        d_type = col_type(data[name])
        print(d_type,name)
        
        if d_type == 'numeric':
            data[name],notParse = notParsedRecord(data[name],d_type)
            data[name] = list(map(float,data[name]))
            file.write('{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},-,-,-,-,-,-,-,-,-,-,-,-,-,-\n'.format(
                   name.capitalize(), d_type.upper(), str(len(data[name])), mis_ct, notParse,unique(data[name]),
                   num_percentile(data[name],5), num_percentile(data[name],25), num_percentile(data[name],50),
                   num_percentile(data[name],75), num_percentile(data[name],95), num_percentile(data[name],99),
                   min_of(data[name]), max_of(data[name])
                    )
                  )
            
        elif d_type == 'str':
            dict1 , ct_set = get_dict(data[name])
            top_list = top_N(data[name], dict1, ct_set)
            ct_set.sort()
            last_list = last_N(data[name], dict1, ct_set)
            top1_lab, top1_prob = top_list[0]
            top2_lab, top2_prob = top_list[1]
            top3_lab, top3_prob = top_list[2]
            top4_lab, top4_prob = top_list[3]
            top5_lab, top5_prob = top_list[4]
            last1_lab, last1_prob = last_list[0]
            last2_lab, last2_prob = last_list[1]
            notParse = 0
            d_type = 'object'
            file.write('{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},-,-,{12},{13},{14},{15},{16},{17},{18},{19},{20},{21},{22},{23},{24},{25}\n'.format(
                   name.capitalize(), d_type.upper(), str(len(data[name])), mis_ct, notParse,unique(data[name]),
                   cat_percentile(data[name],5),cat_percentile(data[name],25),cat_percentile(data[name],50),
                   cat_percentile(data[name],75),cat_percentile(data[name],95),
                   cat_percentile(data[name],99),top1_lab, top1_prob, top2_lab, top2_prob, top3_lab, top3_prob,
                   top4_lab, top4_prob,top5_lab, top5_prob, last1_lab, last1_prob, last2_lab, last2_prob)
                  )
        else:
            dict1 , ct_set = get_dict(data[name])
            top_list = top_N(data[name], dict1, ct_set)
            ct_set.sort()
            last_list = last_N(data[name], dict1, ct_set)
            top1_lab, top1_prob = top_list[0]
            top2_lab, top2_prob = top_list[1]
            top3_lab, top3_prob = top_list[2]
            top4_lab, top4_prob = top_list[3]
            top5_lab, top5_prob = top_list[4]
            last1_lab, last1_prob = last_list[0]
            last2_lab, last2_prob = last_list[1]
            data[name],notParse = notParsedRecord(data[name],d_type)
            d_type = 'datetime'
            file.write('{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},-,-,{12},{13},{14},{15},{16},{17},{18},{19},{20},{21},{22},{23},{24},{25}\n'.format(
                   name.capitalize(), d_type.upper(), str(len(data[name])), mis_ct, notParse,unique(data[name]),
                   dat_percentile(data[name],5),dat_percentile(data[name],25),dat_percentile(data[name],50),
                   dat_percentile(data[name],75),dat_percentile(data[name],95),
                   dat_percentile(data[name],95),top1_lab, top1_prob, top2_lab, top2_prob, top3_lab, top3_prob,
                   top4_lab, top4_prob,top5_lab, top5_prob, last1_lab, last1_prob, last2_lab, last2_prob)
                  )
           













        