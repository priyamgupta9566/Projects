# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 12:24:11 2020

@author: Priyam Gupta
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset=pd.read_csv('Placement_Data.csv')
dataset=dataset.replace(0,np.NaN)
dataset.fillna(0,inplace=True)
dataset2 = dataset.iloc[:,:]
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 13:14].values
Z = dataset.iloc[:, 14:].values
dataset.head()
dataset.tail()
dataset.info()
dataset.shape

from sklearn import preprocessing
label_encoder=preprocessing.LabelEncoder()
dataset['status']=label_encoder.fit_transform(dataset['status'])

Y = pd.DataFrame(dataset2.status)

dataset.drop('sl_no',axis=1,inplace=True)
dataset.drop('status',axis=1,inplace=True)
dataset.drop('salary',axis=1,inplace=True)
column_name=dataset.columns.tolist()
desc_stat=dataset.describe()
skew_chk=dataset.skew()

corr=dataset.corr()
sns.heatmap(corr, 
            xticklabels = corr.columns.values,
            yticklabels = corr.columns.values,
            annot = True);

d=sns.pairplot(dataset,diag_kind='kde')
#d.savefig("Pair_plots.pdf")

dataset.hist(bins=500, figsize=(20,15))
#plt.savefig("attribute_histogram_plots_.pdf")
plt.show()

dataset.columns
list( dataset.columns )
X_features = list( dataset.columns )
placement_df = pd.get_dummies( dataset[X_features], drop_first = True )
column_name=placement_df.columns.tolist()

X = pd.DataFrame(placement_df)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.4, random_state =0)

import statsmodels.api as sm
logit = sm.Logit(y_train, sm.add_constant(X_train,prepend=True,has_constant='skip'))
lg = logit.fit() 
lg.summary()

def get_significant_vars(lm):
    var_p_vals_df = pd.DataFrame( lm.pvalues )
    var_p_vals_df['vars'] = var_p_vals_df.index
    var_p_vals_df.columns = ['pvals', 'vars']
    return list( var_p_vals_df[var_p_vals_df.pvals <= 0.05]['vars'] )

significant_vars = get_significant_vars( lg )


y_pred_df = pd.DataFrame( { "predicted_prob": lg.predict( sm.add_constant( X_test ) ) } )
Analysis=pd.concat([y_test,y_pred_df],axis=1)

'''Working on the Demarcation'''
y_pred_df['predicted'] = y_pred_df.predicted_prob.map( lambda x: 1 if x > 0.7 else 0)
Analysis=pd.concat([y_test,y_pred_df],axis=1)


from sklearn import metrics
def draw_cm( actual, predicted ):
    cm = metrics.confusion_matrix( actual, predicted, [1,0] )
    sns.heatmap(cm, annot=True,  fmt='.2f', xticklabels = ["Placed", "Not Placed"] , yticklabels = ["Placed", "Not Placed"] )
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

draw_cm( Analysis.status, Analysis.predicted )

'''Overall Accuracy & ROC Curve '''
print( 'Total Accuracy : ',np.round( metrics.accuracy_score( y_test, y_pred_df.predicted ), 2 ) )
print( 'Precision : ',np.round( metrics.precision_score( y_test, y_pred_df.predicted ), 2 ) )
print( 'Recall : ',np.round( metrics.recall_score( y_test, y_pred_df.predicted ), 2 ) )

auc_score = metrics.roc_auc_score( Analysis.status, Analysis.predicted_prob  )
round( float( auc_score ), 2 )

def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(6, 4))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    return fpr, tpr, thresholds
fpr, tpr, thresholds = draw_roc( Analysis.status, Analysis.predicted_prob )
    

from sklearn.ensemble import RandomForestClassifier
def randomforestclf(X_train, y_train,X_test, y_test):
    print("RandomForestClassifier")
    randomforest = RandomForestClassifier().fit(X_train, y_train)
    print(randomforest,'\n')
    
    # Predict target variables y for test data
    y_pred = randomforest.predict_proba(X_test)[:,1]

    # Get Cross Validation and Confusion matrix
    #get_eval(randomforest, X_train, y_train,y_test,y_pred)
    draw_roc (y_test,y_pred)
    return

# Random Forest
randomforestclf(X_train, y_train,X_test, y_test) 


randomforest = RandomForestClassifier(n_estimators=70).fit(X_train, y_train)

from sklearn.model_selection import cross_val_score
accuracies_RF = cross_val_score(estimator = randomforest, X = X_train, y = y_train, cv = 15)
accuracies_RF.mean()
accuracies_RF.std()
    