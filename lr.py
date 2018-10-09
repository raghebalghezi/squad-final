#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 14:56:45 2018

@author: raghebal-ghezi
"""
import pandas as pd
import re
from sklearn import linear_model
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
def clean(x):
    return [float(i) for i in re.findall("\s\d.\d*",str(x))]


#load modified data
dataframe = pd.read_csv("./with_pos_overlap_score.csv",dtype={"word_overlap":list})

#extract relevent columns for features
train2 = dataframe[["is_impossible","cosine_sim","word_overlap","pos_tag_ovrlap","target"]]

#De-stringifying columns
train2["is_impossible"]= train2["is_impossible"].apply(lambda x:1 if x==True else 0)
train2["cosine_sim"]= train2["cosine_sim"].map(clean)
train2["word_overlap"]= train2["word_overlap"].apply(clean)
train2["pos_tag_ovrlap"]= train2["pos_tag_ovrlap"].apply(clean)
# Using a small portion of the data for easy computation
small_partion = train2.iloc[2000:40000]

#generating features for classification
t = pd.DataFrame()
for i,j in enumerate(small_partion.cosine_sim):
    for k,l in enumerate(j):
        if k < 11:
            t.loc[i, "column_cos_"+"%s"%k] = l

for i,j in enumerate(small_partion.word_overlap):
    for k,l in enumerate(j):
        if k < 11:
            t.loc[i, "column_wordOverlap_"+"%s"%k] = l

for i,j in enumerate(small_partion.pos_tag_ovrlap):
    for k,l in enumerate(j):
        if k < 11:
            t.loc[i, "column_POSOverlap_"+"%s"%k] = l
t["is_impossible"] = small_partion["is_impossible"]
t["target"] = small_partion["target"]

# clean null values
subset1 = t.iloc[:,:10].fillna(1)
subset2=t.iloc[:,11:].fillna(0)
train_final = pd.concat([subset1, subset2],axis=1, join_axes=[subset1.index])

#saving only feature vectors to disk
# train_final.to_csv("feature_set.csv")

#Classifiication

scaler = MinMaxScaler()
X = scaler.fit_transform(train_final.iloc[:,:-1])
train_x, test_x, train_y, test_y = train_test_split(X,train_final.iloc[:,-1], train_size=0.8, random_state = 5)
mul_lr = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg')
mul_lr.fit(train_x, train_y)

print("Multinomial Logistic regression Train Accuracy : ", metrics.accuracy_score(train_y, mul_lr.predict(train_x)))
print("Multinomial Logistic regression Test Accuracy : ", metrics.accuracy_score(test_y, mul_lr.predict(test_x)))
print(classification_report(test_y, mul_lr.predict(test_x)))
print(confusion_matrix(test_y, mul_lr.predict(test_x)))
