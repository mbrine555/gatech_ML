# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 14:23:40 2017

@author: JTay
"""

import numpy as np
from sklearn.neural_network import MLPClassifier
import sklearn.model_selection as ms
import pandas as pd
from helpers import  basicResults,makeTimingCurve,iterationLC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

# Load Data       
adult = pd.read_hdf('./data/datasets.hdf','adult')        
adultX = adult.drop('TARGET',1).copy().values
adultY = adult['TARGET'].copy().values

wine = pd.read_hdf('./data/datasets.hdf','wine')     
wineX = wine.drop('quality',1).copy().values
wineY = wine['quality'].copy().values

adult_trgX, adult_tstX, adult_trgY, adult_tstY = ms.train_test_split(adultX, adultY, test_size=0.3, random_state=0,stratify=adultY)     
wine_trgX, wine_tstX, wine_trgY, wine_tstY = ms.train_test_split(wineX, wineY, test_size=0.3, random_state=0,stratify=wineY)     

pipeA = Pipeline([('Scale',StandardScaler()),
                 ('MLP',MLPClassifier(max_iter=2000,early_stopping=True,random_state=55))])

pipeM = Pipeline([('Scale',StandardScaler()),
                 ('Cull1',SelectFromModel(RandomForestClassifier(random_state=1),threshold='median')),
                 ('Cull2',SelectFromModel(RandomForestClassifier(random_state=2),threshold='median')),
                 ('Cull3',SelectFromModel(RandomForestClassifier(random_state=3),threshold='median')),
                 ('Cull4',SelectFromModel(RandomForestClassifier(random_state=4),threshold='median')),
                 ('MLP',MLPClassifier(max_iter=2000,early_stopping=True,random_state=55))])

d = adultX.shape[1]
hiddens_adult = [(h,)*l for l in [1,2,3] for h in [d,d//2,d*2]]
alphas = [10**-x for x in np.arange(-1,5.01,1/2)]
alphasM = [10**-x for x in np.arange(-1,9.01,1/2)]
d = wineX.shape[1]
hiddens_wine = [(h,)*l for l in [1,2,3] for h in [d,d//2,d*2]]
params_adult = {'MLP__activation':['relu','logistic'],'MLP__alpha':alphas,'MLP__hidden_layer_sizes':hiddens_adult}
params_wine = {'MLP__activation':['relu','logistic'],'MLP__alpha':alphas,'MLP__hidden_layer_sizes':hiddens_wine}
#
wine_clf = basicResults(pipeM,wine_trgX,wine_trgY,wine_tstX,wine_tstY,params_wine,'ANN','wine')        
adult_clf = basicResults(pipeA,adult_trgX,adult_trgY,adult_tstX,adult_tstY,params_adult,'ANN','adult')        


#wine_final_params = {'MLP__hidden_layer_sizes': (500,), 'MLP__activation': 'logistic', 'MLP__alpha': 10.0}
#adult_final_params ={'MLP__hidden_layer_sizes': (28, 28, 28), 'MLP__activation': 'logistic', 'MLP__alpha': 0.0031622776601683794}

wine_final_params = wine_clf.best_params_
adult_final_params =adult_clf.best_params_
adult_OF_params =adult_final_params.copy()
adult_OF_params['MLP__alpha'] = 0
wine_OF_params =wine_final_params.copy()
wine_OF_params['MLP__alpha'] = 0

#raise

#
pipeM.set_params(**wine_final_params)  
pipeM.set_params(**{'MLP__early_stopping':False})                   
makeTimingCurve(wineX,wineY,pipeM,'ANN','wine')
pipeA.set_params(**adult_final_params)
pipeA.set_params(**{'MLP__early_stopping':False})                  
makeTimingCurve(adultX,adultY,pipeA,'ANN','adult')

pipeM.set_params(**wine_final_params)
pipeM.set_params(**{'MLP__early_stopping':False})               
iterationLC(pipeM,wine_trgX,wine_trgY,wine_tstX,wine_tstY,{'MLP__max_iter':[2**x for x in range(12)]+[2100,2200,2300,2400,2500,2600,2700,2800,2900,3000]},'ANN','wine')        
pipeA.set_params(**adult_final_params)
pipeA.set_params(**{'MLP__early_stopping':False})                  
iterationLC(pipeA,adult_trgX,adult_trgY,adult_tstX,adult_tstY,{'MLP__max_iter':[2**x for x in range(12)]+[2100,2200,2300,2400,2500,2600,2700,2800,2900,3000]},'ANN','adult')                

pipeM.set_params(**wine_OF_params)
pipeM.set_params(**{'MLP__early_stopping':False})                  
iterationLC(pipeM,wine_trgX,wine_trgY,wine_tstX,wine_tstY,{'MLP__max_iter':[2**x for x in range(12)]+[2100,2200,2300,2400,2500,2600,2700,2800,2900,3000]},'ANN_OF','wine')        
pipeA.set_params(**adult_OF_params)
pipeA.set_params(**{'MLP__early_stopping':False})               
iterationLC(pipeA,adult_trgX,adult_trgY,adult_tstX,adult_tstY,{'MLP__max_iter':[2**x for x in range(12)]+[2100,2200,2300,2400,2500,2600,2700,2800,2900,3000]},'ANN_OF','adult')                

