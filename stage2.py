#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 12:33:09 2018

@author: raghebal-ghezi
"""
import itertools
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


#pickled Pandas DataFrame that has the following cols:
# ['candid',  'const_tags',  'constituents',  'cosine_2',  
# 'cosine_3',  'left_span',  'question',  'right_span',  
# 'span_length',  'text',  'wh+tag',  'target',  'tfidf_sum',  
# 'tfidf_sum_2',  'tfidf_sum_3',  'glove_cos_2']
df = pd.read_pickle('serialized_df.pkl')


df_feature = pd.DataFrame() #to store the features
#try:
print("Generating features ...... (1 of 5)")
for i,j in enumerate(df.cosine_2): # prepare features for Contextual Overlap with window size 2
    for k,l in enumerate(j):
        if k < 31:
            df_feature.loc[i, "column_cos2_"+"%s"%k] = l
print("Generating features ...... (2 of 5)")
for i,j in enumerate(df.cosine_3): # prepare features for Contextual Overlap with window size 3
    for k,l in enumerate(j):
        if k < 31:
            df_feature.loc[i, "column_cos3_"+"%s"%k] = l
print("Generating features ...... (3 of 5)")            
for i,j in enumerate(df.tfidf_sum_2):# sum of tfidf values for Contextual Overlap with window size 2
    for k,l in enumerate(j):
        if k < 31:
            df_feature.loc[i, "column_tfidf2_"+"%s"%k] = l
print("Generating features ...... (4 of 5)")            
for i,j in enumerate(df.tfidf_sum_3): # sum of tfidf values for Contextual Overlap with window size 3
    for k,l in enumerate(j):
        if k < 31:
            df_feature.loc[i, "column_tfidf3_"+"%s"%k] = l
print("Generating features ...... (5 of 5)")
for i,j in enumerate(df.glove_cos_2): # distributional cos sim
    for k,l in enumerate(j):
        if k < 31:
            df_feature.loc[i, "column_gloveCos_"+"%s"%k] = l

# Appending the remaining features from original dataframe
df_feature['wh+tag'] = df['wh+tag'] 
df_feature['left_span'] = df['left_span']
df_feature['right_span'] = df['right_span']
df_feature['span_length'] = df['span_length']
df_feature['target'] = df.target

# restricting the number of target constituents to 30
train_final = df_feature[df_feature['target'] < 31]

# filling in missing values with zeros
train_final = train_final.fillna(0)

# enforcing the datatype to 'wh+tag' 
train_final['wh+tag'] = train_final['wh+tag'].map(lambda x:str(x).lower())
#fixing the indices of dataframe
train_final = train_final.reset_index(drop=True)

# encoding the 'wh+tag'  to numerical values
le = preprocessing.LabelEncoder()
train_final['wh+tag'] = le.fit_transform(train_final['wh+tag'])

# assigning X and y
X = train_final.drop('target',axis=1)
y = train_final['target']

# splitting and Shuffling
train_x, test_x, train_y, test_y = train_test_split(X,y, train_size=0.7, random_state = 5,shuffle=True)
# Using LR with Newton methods optimizer
mul_lr = linear_model.LogisticRegression(random_state=0, solver='newton-cg',multi_class='multinomial',n_jobs=-1)
mul_lr.fit(train_x, train_y)

# Printing Classification Report
print(classification_report(test_y, mul_lr.predict(test_x), labels=np.unique(mul_lr.predict(test_x))))


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize=2)

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('fig_1.pdf')


# Compute confusion matrix
cnf_matrix = confusion_matrix(test_y, mul_lr.predict(test_x), labels=[i for i in range(0,30)])
np.set_printoptions(precision=2)

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=np.unique(mul_lr.predict(test_x)),
                      title='Confusion matrix, with normalization', normalize=True)
