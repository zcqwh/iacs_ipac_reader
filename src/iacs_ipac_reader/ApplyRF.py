#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 21:05:14 2022

@author: nana
"""
import numpy as np
import h5py



def p20(x):
    return np.percentile(x,20)
def p50(x):
    return np.percentile(x,50)
def p80(x):
    return np.percentile(x,80)

def mad(x):#definition of median absolute deviation
    return np.median(abs(x - np.median(x)))/0.67448975019608171

def getdata2(rtdc_path,userdef0):
    feature_name = ["Area","Area_Ratio"] #more features
    keys = ["area_um","area_ratio"]
    classes = [1,2]

    print(rtdc_path)
    NameList,List = [],[]
    
    rtdc_ds = h5py.File(rtdc_path, 'r')

    operations = [np.mean,np.median,np.std,mad,p20,p50,p80]
    operationnames = ["mean","median","std","mad","p20","p50","p80"]

    #get the numbers of events for each subpopulation
    #userdef0 = rtdc_ds["events"]["userdef0"][:]

    for cl in classes:
        ind_x = np.where(userdef0==cl)[0]
        perc = len(ind_x)/len(userdef0)
        NameList.append("events_perc_class_"+str(cl))
        List.append(perc)
        
        for k in range(len(keys)):
            values = rtdc_ds["events"][keys[k]][:][ind_x]
            #remove nan values and zeros
            ind = np.isnan(values)
            ind = np.where(ind==False)[0]
            values = values[ind]
            ind = np.where(values!=0)[0]
            values = values[ind]
    
            for o in range(len(operations)):
                NameList.append(feature_name[k]+"_"+operationnames[o]+"_class"+str(cl))
                if len(values)==0:
                    List.append(np.nan)
                else:
                    stat = operations[o](values)
                    List.append(stat)

    return [List,NameList]