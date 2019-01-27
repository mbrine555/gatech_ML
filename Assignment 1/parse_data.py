# -*- coding: utf-8 -*-
"""
Credit to JohnTay https://github.com/JonathanTay/CS-7641-assignment-1
"""

import pandas as pd
import numpy as np

# Preprocess with adult dataset
adult = pd.read_csv('./data/adult.csv')
# Note that cap_gain > 0 => cap_loss = 0 and vice versa. Combine variables.
print(adult.ix[adult['capital-gain']>0]['capital-loss'].abs().max())
print(adult.ix[adult['capital-loss']>0]['capital-gain'].abs().max())
adult['cap_gain_loss'] = adult['capital-gain']-adult['capital-loss']
adult = adult.drop(['fnlwgt','education','capital-gain','capital-loss'],1)
adult['TARGET'] = pd.get_dummies(adult['TARGET'])
print(adult.groupby('occupation')['occupation'].count())
print(adult.groupby('native-country')['native-country'].count())
#http://scg.sdsu.edu/dataset-adult_r/
replacements = { 'Cambodia':' SE-Asia',
                'Canada':' British-Commonwealth',
                'China':' China',
                'Columbia':' South-America',
                'Cuba':' Other',
                'Dominican-Republic':' Latin-America',
                'Ecuador':' South-America',
                'El-Salvador':' South-America ',
                'England':' British-Commonwealth',
                'France':' Euro_1',
                'Germany':' Euro_1',
                'Greece':' Euro_2',
                'Guatemala':' Latin-America',
                'Haiti':' Latin-America',
                'Holand-Netherlands':' Euro_1',
                'Honduras':' Latin-America',
                'Hong':' China',
                'Hungary':' Euro_2',
                'India':' British-Commonwealth',
                'Iran':' Other',
                'Ireland':' British-Commonwealth',
                'Italy':' Euro_1',
                'Jamaica':' Latin-America',
                'Japan':' Other',
                'Laos':' SE-Asia',
                'Mexico':' Latin-America',
                'Nicaragua':' Latin-America',
                'Outlying-US(Guam-USVI-etc)':' Latin-America',
                'Peru':' South-America',
                'Philippines':' SE-Asia',
                'Poland':' Euro_2',
                'Portugal':' Euro_2',
                'Puerto-Rico':' Latin-America',
                'Scotland':' British-Commonwealth',
                'South':' Euro_2',
                'Taiwan':' China',
                'Thailand':' SE-Asia',
                'Trinadad&Tobago':' Latin-America',
                'United-States':' United-States',
                'Vietnam':' SE-Asia',
                'Yugoslavia':' Euro_2'}
adult['native-country'] = adult['native-country'].str.strip()
adult = adult.replace(to_replace={'native-country':replacements,
                                  'workclass':{' Without-pay': ' Never-worked'},
                                  'relationship':{' Husband': 'Spouse',' Wife':'Spouse'}})    
adult['native-country'] = adult['native-country'].str.strip()
print(adult.groupby('native-country')['native-country'].count())   
for col in ['workclass','marital-status','occupation','relationship','race','sex','native-country']:
    adult[col] = adult[col].str.strip()
    
adult = pd.get_dummies(adult)
adult = adult.rename(columns=lambda x: x.replace('-','_'))

adult.to_hdf('./data/datasets.hdf','adult',complib='blosc',complevel=9)

# Wine
# 'Quality' wine is defined as having a quality score > 6
wine = pd.read_csv('./data/winequality-white.csv', sep=';')
wine.columns = wine.columns.str.lower().str.replace(' ', '_')
wine['quality'] = (wine['quality'] > 6).astype('int')
wine.to_hdf('./data/datasets.hdf', 'wine', complib='blosc', complevel=9)
