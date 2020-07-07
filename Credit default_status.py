# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 00:08:37 2019

@author: Priyam Gupta
"""

#PROJECT-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandasql as ps

dataset=pd.read_csv('German_Credit_Data.csv')
##Import all the required libararies and the data file.


dataset.head()
dataset.shape
dataset.tail()
dataset.info()
##some Description of dataset

column_name=dataset.columns.tolist()
desc_stat=dataset.describe()
skew_chk=dataset.skew()
corr=dataset.corr()
sns.heatmap(corr, 
            xticklabels = corr.columns.values,
            yticklabels = corr.columns.values,
            annot = True);
##Exploratory Data Analysis-EDA

            
d=sns.pairplot(dataset,diag_kind='kde')
#d.savefig("Pair_plots2.pdf")
            
dataset.hist(bins=500, figsize=(20,15))
#plt.savefig("attribute_histogram_plots2.pdf")
plt.show()
##Univariant Analysis


## Bivariate Analysis
#1. Status
'''
A11 : ... < 0 DM 
A12 : 0 <= ... < 200 DM 
A13 : ... >= 200 DM / salary assignments for at least 1 year 
A14 : no checking account 
'''

query = """SELECT status,Default_Status,
count(Default_Status) as Default_Count,
sum(Default_Status) as Total_Default_Count
 FROM dataset
group by status,Default_Status
order by status,Default_Status
"""
Status_Default=ps.sqldf(query, locals())
query_pct = """SELECT status,
sum(Total_Default_Count)*100/sum(Default_Count) as  Default_Pct
FROM Status_Default
group by status
order by status
"""
Status_Default_PCT=ps.sqldf(query_pct, locals())

#2. Duration month
query="""SELECT Durationinmonth,Default_Status,
count(Default_Status) as Default_count,
sum(Default_Status) as Total_Default_count
FROM dataset
group by Durationinmonth,Default_Status
order by Durationinmonth,Default_Status
"""
Durationinmonth_Default=ps.sqldf(query,locals())
query_pct="""SELECT Durationinmonth,
sum(Total_Default_count)*100/sum(Default_count) as Default_Pct
FROM Durationinmonth_Default
group by Durationinmonth
order by Durationinmonth
"""
Durationinmonth_Default_PCT=ps.sqldf(query_pct,locals())

#3. Credit History
'''
A30 : no credits taken/ all credits paid back duly
A31 : all credits at this bank paid back duly
A32 : existing credits paid back duly till now
A33 : delay in paying off in the past
A34 : critical account/other credits existing (not at this bank)
'''

query="""SELECT Credithistory,Default_Status,
count(Default_Status) as Default_count,
sum(Default_Status) as Total_Default_count
FROM dataset
group by Credithistory,Default_Status
order by Credithistory,Default_Status
"""
Credithistory_Default=ps.sqldf(query,locals())
query_pct="""SELECT Credithistory,
sum(Total_Default_count)*100/sum(Default_count) as Default_Pct
FROM Credithistory_Default
group by Credithistory
order by Credithistory
"""
Credithistory_Default_PCT=ps.sqldf(query_pct,locals())

#4. Purpose
'''
A40 : car (new)
A41 : car (used)
A42 : furniture/equipment
A43 : radio/television
A44 : domestic appliances
A45 : repairs
A46 : education
A47 : (vacation - does not exist?)
A48 : retraining
A49 : business
A410 : others
'''

query="""SELECT Purpose,Default_Status,
count(Default_Status) as Default_count,
sum(Default_Status) as Total_Default_count
FROM dataset
group by Purpose,Default_Status
order by Purpose,Default_Status
"""
Purpose_Default=ps.sqldf(query,locals())
query_pct="""SELECT Purpose,
sum(Total_Default_count)*100/sum(Default_count) as Default_Pct
FROM Purpose_Default
group by Purpose
order by Purpose
"""
Purpose_Default_PCT=ps.sqldf(query_pct,locals())

#5. Creditamount 
'''As Credit Amount is Discrete Data and non-categorical it will 
be insignificant to plot a discrete data over Categorical Data
'''

#6. Savingsaccount_bonds 
'''
A61 :          ... <  100 DM
A62 :   100 <= ... <  500 DM
A63 :   500 <= ... < 1000 DM
A64 :          .. >= 1000 DM
A65 :   unknown/ no savings account
'''

query="""SELECT Savingsaccount_bonds,Default_Status,
count(Default_Status) as Default_count,
sum(Default_Status) as Total_Default_count
FROM dataset
group by Savingsaccount_bonds,Default_Status
order by Savingsaccount_bonds,Default_Status
"""
Savingsaccount_bonds_Default=ps.sqldf(query,locals())
query_pct="""SELECT Savingsaccount_bonds,
sum(Total_Default_count)*100/sum(Default_count) as Default_Pct
FROM Savingsaccount_bonds_Default
group by Savingsaccount_bonds
order by Savingsaccount_bonds
"""
Savingsaccount_bonds_Default_PCT=ps.sqldf(query_pct,locals())

#7. Presentemploymentsince
'''
A71 : unemployed
A72 :       ... < 1 year
A73 : 1  <= ... < 4 years  
A74 : 4  <= ... < 7 years
A75 :       .. >= 7 years
'''

query="""SELECT Presentemploymentsince,Default_Status,
count(Default_Status) as Default_count,
sum(Default_Status) as Total_Default_count
FROM dataset
group by Presentemploymentsince,Default_Status
order by Presentemploymentsince,Default_Status
"""
Presentemploymentsince_Default=ps.sqldf(query,locals())
query_pct="""SELECT Presentemploymentsince,
sum(Total_Default_count)*100/sum(Default_count) as Default_Pct
FROM Presentemploymentsince_Default
group by Presentemploymentsince
order by Presentemploymentsince
"""
Presentemploymentsince_Default_PCT=ps.sqldf(query_pct,locals())

#8. Installmentrate
'''
As Installmentrate is Discrete Data and non-categorical it will 
be insignificant to plot a discrete data over Categorical Data
'''

#9.Personalstatusandsex
'''
A91 : male   : divorced/separated
A92 : female : divorced/separated/married
A93 : male   : single
A94 : male   : married/widowed
A95 : female : single
'''

query="""SELECT Personalstatusandsex,Default_Status,
count(Default_Status) as Default_count,
sum(Default_Status) as Total_Default_count
FROM dataset
group by Personalstatusandsex,Default_Status
order by Personalstatusandsex,Default_Status
"""
Personalstatusandsex_Default=ps.sqldf(query,locals())
query_pct="""SELECT Personalstatusandsex,
sum(Total_Default_count)*100/sum(Default_count) as Default_Pct
FROM Personalstatusandsex_Default
group by Personalstatusandsex
order by Personalstatusandsex
"""
Personalstatusandsex_Default_PCT=ps.sqldf(query_pct,locals())

#10. Guarantors
'''
Other debtors / guarantors
A101 : none
A102 : co-applicant
A103 : guarantor
'''
query="""SELECT Guarantors,Default_Status,
count(Default_Status) as Default_count,
sum(Default_Status) as Total_Default_count
FROM dataset
group by Guarantors,Default_Status
order by Guarantors,Default_Status
"""
Guarantors_Default=ps.sqldf(query,locals())
query_pct="""SELECT Guarantors,
sum(Total_Default_count)*100/sum(Default_count) as Default_Pct
FROM Guarantors_Default
group by Guarantors
order by Guarantors
"""
Guarantors_Default_PCT=ps.sqldf(query_pct,locals())

#11. Presentresidencesince
query="""SELECT Presentresidencesince,Default_Status,
count(Default_Status) as Default_count,
sum(Default_Status) as Total_Default_count
FROM dataset
group by Presentresidencesince,Default_Status
order by Presentresidencesince,Default_Status
"""
Presentresidencesince_Default=ps.sqldf(query,locals())
query_pct="""SELECT Presentresidencesince,
sum(Total_Default_count)*100/sum(Default_count) as Default_Pct
FROM Presentresidencesince_Default
group by Presentresidencesince
order by Presentresidencesince
"""
Presentresidencesince_Default_PCT=ps.sqldf(query_pct,locals())

#12. Property
'''
A121 : real estate
A122 : if not A121 : building society savings agreement/life insurance
A123 : if not A121/A122 : car or other, not in attribute 6
A124 : unknown / no property
'''

query="""SELECT Property,Default_Status,
count(Default_Status) as Default_count,
sum(Default_Status) as Total_Default_count
FROM dataset
group by Property,Default_Status
order by Property,Default_Status
"""
Property_Default=ps.sqldf(query,locals())
query_pct="""SELECT Property,
sum(Total_Default_count)*100/sum(Default_count) as Default_Pct
FROM Property_Default
group by Property
order by Property
"""
Property_Default_PCT=ps.sqldf(query_pct,locals())

#13. Age
'''
As Age is Discrete Data and non-categorical it will 
be insignificant to plot a discrete data over Categorical Data
'''

#14. Otherinstallmentplans
'''
A141 : bank
A142 : stores
A143 : none
'''
query = """SELECT Otherinstallmentplans,Default_Status,
count(Default_Status) as Default_Count ,
sum(Default_Status) as Total_Default_Count
FROM dataset
group by Otherinstallmentplans,Default_Status
order by Otherinstallmentplans,Default_Status
"""
Otherinstallmentplans_Default=ps.sqldf(query, locals())
query_pct = """SELECT Otherinstallmentplans,
sum(Total_Default_Count)*100/sum(Default_Count) as  Default_Pct
FROM Otherinstallmentplans_Default
group by Otherinstallmentplans
order by Otherinstallmentplans
"""
Otherinstallmentplans_PCT=ps.sqldf(query_pct, locals())


#15. Housing
'''
A151 : rent
A152 : own
A153 : for free
'''
query = """SELECT Housing,Default_Status,
count(Default_Status) as Default_Count ,
sum(Default_Status) as Total_Default_Count
FROM dataset
group by Housing,Default_Status
order by Housing,Default_Status
"""
Housing_Default=ps.sqldf(query, locals())
query_pct = """SELECT Housing,
sum(Total_Default_Count)*100/sum(Default_Count) as  Default_Pct
FROM Housing_Default
group by Housing
order by Housing
"""
Housing_PCT=ps.sqldf(query_pct, locals())

#16. No_of_existing_credit
query = """SELECT No_of_existing_credit,Default_Status,
count(Default_Status) as Default_Count ,
sum(Default_Status) as Total_Default_Count
FROM dataset
group by No_of_existing_credit,Default_Status
order by No_of_existing_credit,Default_Status
"""
No_of_existing_credit_Default=ps.sqldf(query, locals())
query_pct = """SELECT No_of_existing_credit,
sum(Total_Default_Count)*100/sum(Default_Count) as  Default_Pct
FROM No_of_existing_credit_Default
group by No_of_existing_credit
order by No_of_existing_credit
"""
No_of_existing_credit_PCT=ps.sqldf(query_pct,locals())

#17. Job
'''
A171 : unemployed/ unskilled  - non-resident
A172 : unskilled - resident
A173 : skilled employee / official
A174 : management/ self-employed/highly qualified employee/ officer
'''

query = """SELECT Job,Default_Status,
count(Default_Status) as Default_Count ,
sum(Default_Status) as Total_Default_Count
FROM dataset
group by Job,Default_Status
order by Job,Default_Status
"""
Job_Default=ps.sqldf(query, locals())
query_pct = """SELECT Job,
sum(Total_Default_Count)*100/sum(Default_Count) as  Default_Pct
FROM Job_Default
group by Job
order by Job
"""
Job_PCT=ps.sqldf(query_pct, locals())

#18. Numberofdependents
'''
1
2
'''
query = """SELECT Numberofdependents,Default_Status,
count(Default_Status) as Default_Count ,
sum(Default_Status) as Total_Default_Count
FROM dataset
group by Numberofdependents,Default_Status
order by Numberofdependents,Default_Status
"""
Numberofdependents_Default=ps.sqldf(query, locals())
query_pct = """SELECT Numberofdependents,
sum(Total_Default_Count)*100/sum(Default_Count) as  Default_Pct
FROM Numberofdependents_Default
group by Numberofdependents
order by Numberofdependents
"""
Numberofdependents_PCT=ps.sqldf(query_pct, locals())

#19. Telephone
'''
1
0
'''
query = """SELECT Telephone,Default_Status,
count(Default_Status) as Default_Count ,
sum(Default_Status) as Total_Default_Count
FROM dataset
group by Telephone,Default_Status
order by Telephone,Default_Status
"""
Telephone_Default=ps.sqldf(query, locals())
query_pct = """SELECT Telephone,
sum(Total_Default_Count)*100/sum(Default_Count) as  Default_Pct
FROM Telephone_Default
group by Telephone
order by Telephone
"""
Telephone_PCT=ps.sqldf(query_pct, locals())

#20. foreignworker
'''
A201- Yes
'''
query = """SELECT foreignworker,Default_Status,
count(Default_Status) as Default_Count ,
sum(Default_Status) as Total_Default_Count
FROM dataset
group by foreignworker,Default_Status
order by foreignworker,Default_Status
"""
foreignworker_Default=ps.sqldf(query, locals())
query_pct = """SELECT foreignworker,
sum(Total_Default_Count)*100/sum(Default_Count) as  Default_Pct
FROM foreignworker_Default
group by foreignworker
order by foreignworker
"""
foreignworker_PCT=ps.sqldf(query_pct, locals())


' Default Value Counts '
dataset.Default_Status.value_counts()

'Creating Dummy Features'
dataset.columns
list( dataset.columns )
X_features = list( dataset.columns )
X_features.remove( 'Default_Status' )
X_features

credit_df_complete = pd.get_dummies( dataset[X_features], drop_first = True )
column_name=credit_df_complete.columns.tolist()

'''Now there are 48 Columns'''
X = pd.DataFrame(credit_df_complete)
Y = pd.DataFrame(dataset.Default_Status)

'''Splitting Datasets into Train and Test '''
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state =0)

'''Developing Model Summary to select Significant Variables '''
import statsmodels.api as sm
logit = sm.Logit( y_train, sm.add_constant( X_train ) )
lg = logit.fit() 
lg.summary()

'''Making the Classifier Ready for Model Validation, not to be used now'''
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)

'''Finding the Significant Variable '''
def get_significant_vars( lm ):
    var_p_vals_df = pd.DataFrame( lm.pvalues )
    var_p_vals_df['vars'] = var_p_vals_df.index
    var_p_vals_df.columns = ['pvals', 'vars']
    return list( var_p_vals_df[var_p_vals_df.pvals <= 0.05]['vars'] )

significant_vars = get_significant_vars( lg )


''' Predict Test Data and Measure Accuracy of the Model'''
y_pred_df = pd.DataFrame( { "predicted_prob": lg.predict( sm.add_constant( X_test ) ) } )
Analysis=pd.concat([y_test,y_pred_df],axis=1)

'''Working on the Demarcation'''
y_pred_df['predicted'] = y_pred_df.predicted_prob.map( lambda x: 1 if x > 0.7 else 0)
Analysis=pd.concat([y_test,y_pred_df],axis=1)

'''Confusion Matrix'''
from sklearn import metrics
def draw_cm( actual, predicted ):
    cm = metrics.confusion_matrix( actual, predicted, [1,0] )
    sns.heatmap(cm, annot=True,  fmt='.2f', xticklabels = ["Default", "No Default"] , yticklabels = ["Default", "No Default"] )
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    
draw_cm( Analysis.Default_Status, Analysis.predicted )

'''Overall Accuracy & ROC Curve '''
print( 'Total Accuracy : ',np.round( metrics.accuracy_score( y_test, y_pred_df.predicted ), 2 ) )
print( 'Precision : ',np.round( metrics.precision_score( y_test, y_pred_df.predicted ), 2 ) )
print( 'Recall : ',np.round( metrics.recall_score( y_test, y_pred_df.predicted ), 2 ) )

auc_score = metrics.roc_auc_score( Analysis.Default_Status, Analysis.predicted_prob  )
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
fpr, tpr, thresholds = draw_roc( Analysis.Default_Status, Analysis.predicted_prob )

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)

'''Model Building with CART Analysis'''
from sklearn.tree import DecisionTreeClassifier 
# fit, train and cross validate Decision Tree with training and test data 
def dectreeclf(X_train, y_train,X_test, y_test):
    print("DecisionTreeClassifier")
    dec_tree = DecisionTreeClassifier(min_samples_split=10,min_samples_leaf=5).fit(X_train, y_train)
    print(dec_tree,'\n')
    
    # Predict target variables y for test data
    y_pred = dec_tree.predict_proba(X_test)[:,1]

    # Get Cross Validation and Confusion matrix
    #get_eval(dec_tree, X_train, y_train,y_test,y_pred)
    draw_roc(y_test,y_pred)
    return

# Decisiontree
dectreeclf(X_train, y_train,X_test, y_test) 

'''Model Building with Random Forest Analysis'''
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

'''Working on K-Fold Cross Validation'''

# Applying k-Fold Cross Validation

Log_Classifier=classifier.fit(X_train, y_train)
CART_Analysis = DecisionTreeClassifier(min_samples_split=10,min_samples_leaf=5).fit(X_train, y_train)
randomforest = RandomForestClassifier(n_estimators=70).fit(X_train, y_train)

from sklearn.model_selection import cross_val_score

accuracies_log = cross_val_score(estimator = Log_Classifier, X = X_train, y = y_train, cv = 15)
accuracies_log.mean()
accuracies_log.std()


accuracies_CART = cross_val_score(estimator = CART_Analysis, X = X_train, y = y_train, cv = 15)
accuracies_CART.mean()
accuracies_CART.std()

accuracies_RF = cross_val_score(estimator = randomforest, X = X_train, y = y_train, cv = 15)
accuracies_RF.mean()
accuracies_RF.std()

