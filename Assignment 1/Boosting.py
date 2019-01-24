# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 14:23:40 2017

@author: JTay
"""
import sklearn.model_selection as ms
from sklearn.ensemble import AdaBoostClassifier
from helpers import dtclf_pruned
import pandas as pd
from helpers import  basicResults,makeTimingCurve,iterationLC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

adult = pd.read_hdf('./data/datasets.hdf','adult')        
adultX = adult.drop('TARGET',1).copy().values
adultY = adult['TARGET'].copy().values

cont = pd.read_hdf('./data/datasets.hdf','contraceptive')     
contX = cont.drop('method',1).copy().values
contY = cont['method'].copy().values

alphas = [-1,-1e-3,-(1e-3)*10**-0.5, -1e-2, -(1e-2)*10**-0.5,-1e-1,-(1e-1)*10**-0.5, 0, (1e-1)*10**-0.5,1e-1,(1e-2)*10**-0.5,1e-2,(1e-3)*10**-0.5,1e-3]


adult_trgX, adult_tstX, adult_trgY, adult_tstY = ms.train_test_split(adultX, adultY, test_size=0.3, random_state=0,stratify=adultY)     
cont_trgX, cont_tstX, cont_trgY, cont_tstY = ms.train_test_split(contX, contY, test_size=0.3, random_state=0,stratify=contY)     

cont_base = dtclf_pruned(criterion='gini',class_weight='balanced',random_state=55)                
adult_base = dtclf_pruned(criterion='entropy',class_weight='balanced',random_state=55)
OF_base = dtclf_pruned(criterion='gini',class_weight='balanced',random_state=55)                
#paramsA= {'Boost__n_estimators':[1,2,5,10,20,30,40,50],'Boost__learning_rate':[(2**x)/100 for x in range(8)]+[1]}
paramsA= {'Boost__n_estimators':[1,2,5,10,20,30,45,60,80,100],
          'Boost__base_estimator__alpha':alphas}
#paramsM = {'Boost__n_estimators':[1,2,5,10,20,30,40,50,60,70,80,90,100],
#           'Boost__learning_rate':[(2**x)/100 for x in range(8)]+[1]}

paramsM = {'Boost__n_estimators':[1,2,5,10,20,30,45,60,80,100],
           'Boost__base_estimator__alpha':alphas}
                                   
         
cont_booster = AdaBoostClassifier(algorithm='SAMME',learning_rate=1,base_estimator=cont_base,random_state=55)
adult_booster = AdaBoostClassifier(algorithm='SAMME',learning_rate=1,base_estimator=adult_base,random_state=55)
OF_booster = AdaBoostClassifier(algorithm='SAMME',learning_rate=1,base_estimator=OF_base,random_state=55)

pipeM = Pipeline([('Scale',StandardScaler()),
                 ('Cull1',SelectFromModel(RandomForestClassifier(random_state=1),threshold='median')),
                 ('Cull2',SelectFromModel(RandomForestClassifier(random_state=2),threshold='median')),
                 ('Cull3',SelectFromModel(RandomForestClassifier(random_state=3),threshold='median')),
                 ('Cull4',SelectFromModel(RandomForestClassifier(random_state=4),threshold='median')),
                 ('Boost',cont_booster)])

pipeA = Pipeline([('Scale',StandardScaler()),                
                 ('Boost',adult_booster)])

#
cont_clf = basicResults(pipeM,cont_trgX,cont_trgY,cont_tstX,cont_tstY,paramsM,'Boost','cont')        
adult_clf = basicResults(pipeA,adult_trgX,adult_trgY,adult_tstX,adult_tstY,paramsA,'Boost','adult')        

#
#
#cont_final_params = {'n_estimators': 20, 'learning_rate': 0.02}
#adult_final_params = {'n_estimators': 10, 'learning_rate': 1}
#OF_params = {'learning_rate':1}

cont_final_params = cont_clf.best_params_
adult_final_params = adult_clf.best_params_
OF_params = {'Boost__base_estimator__alpha':-1, 'Boost__n_estimators':50}

##
pipeM.set_params(**cont_final_params)
pipeA.set_params(**adult_final_params)
makeTimingCurve(contX,contY,pipeM,'Boost','cont')
makeTimingCurve(adultX,adultY,pipeA,'Boost','adult')
#
pipeM.set_params(**cont_final_params)
iterationLC(pipeM,cont_trgX,cont_trgY,cont_tstX,cont_tstY,{'Boost__n_estimators':[1,2,5,10,20,30,40,50,60,70,80,90,100]},'Boost','cont')        
pipeA.set_params(**adult_final_params)
iterationLC(pipeA,adult_trgX,adult_trgY,adult_tstX,adult_tstY,{'Boost__n_estimators':[1,2,5,10,20,30,40,50]},'Boost','adult')                
pipeM.set_params(**OF_params)
iterationLC(pipeM,cont_trgX,cont_trgY,cont_tstX,cont_tstY,{'Boost__n_estimators':[1,2,5,10,20,30,40,50,60,70,80,90,100]},'Boost_OF','cont')                
pipeA.set_params(**OF_params)
iterationLC(pipeA,cont_trgX,cont_trgY,cont_tstX,cont_tstY,{'Boost__n_estimators':[1,2,5,10,20,30,40,50]},'Boost_OF','adult')                

             
