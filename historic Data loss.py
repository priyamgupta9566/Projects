# -*- coding: utf-8 -*-
"""
Created on Tue May  5 19:51:52 2020

@author: Priyam Gupta
"""

# Conceptualise Linear Regression
##############################################################################################
# Data Preparation for Linear Regression -- Missing Data

#importing important libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandasql as ps

%matplotlib inline


#Importing the dataset
dataset=pd.read_csv('hld.csv')


# To See the Size of the Dataset from different aspects
dataset.head()
dataset.tail()
dataset.info()
dataset.shape

#Dropping Non Significant Variables - Policy Number
dataset.drop('Policy_Number',axis=1, inplace=True)

# Creating an array to collect the names of variables
column_name=dataset.columns.tolist()

#Complete Description - EDA

desc=dataset.describe()
dataset.skew()
dataset.sort_values('Age')
corr=dataset.corr()

sns.heatmap(corr, 
            xticklabels = corr.columns.values,
            yticklabels = corr.columns.values,
            annot = True);
            

#Start Observing the Distribution of Variables one by one (Univariate Analysis)

d=sns.pairplot(dataset,diag_kind='kde')
d.savefig("Pair_plots.pdf")
       

dataset.hist(bins=500, figsize=(20,15))
plt.savefig("attribute_histogram_plots_.pdf")
plt.show()

#Cap the Losses to 1200
q1 = """SELECT *,(case when losses>1200 then 1200 else losses end )as Losses_Calc FROM dataset """

capped_losses_dataset=ps.sqldf(q1, locals())

#Bivariate Analysis

#Age vs Losses

q1 = """SELECT Age,sum(Losses_Calc) as Losses FROM capped_losses_dataset group by Age Order by Age"""
AgevsLoss_dataset=ps.sqldf(q1, locals())

BVA=AgevsLoss_dataset.sort_values('Age')
plt.plot(BVA.Age,BVA.Losses,color='blue')
plt.title('Age vs Losses')
plt.xlabel('Age')
plt.ylabel('Losses')
plt.show()
plt.savefig("Age_Losses.pdf")

#Years_of_Driving_Experience vs Losses

q1 = """SELECT Years_of_Driving_Experience,sum(Losses_Calc) 
as Losses FROM capped_losses_dataset 
group by Years_of_Driving_Experience Order by Years_of_Driving_Experience"""
Years_of_ExpvsLoss_dataset=ps.sqldf(q1, locals())

plt.plot(Years_of_ExpvsLoss_dataset.Years_of_Driving_Experience,Years_of_ExpvsLoss_dataset.Losses,color='blue')
plt.title('Years_of_Driving_Experience vs Losses')
plt.xlabel('Years_of_Driving_Experience')
plt.ylabel('Losses')
plt.show()
plt.savefig("Years_of_Driving_Experience_Losses.pdf")



#Number_of_Vehicles vs Losses

q1 = """SELECT Number_of_Vehicles,sum(Losses_Calc) 
as Losses FROM capped_losses_dataset 
group by Number_of_Vehicles Order by Number_of_Vehicles"""
Number_of_VehiclesvsLoss_dataset=ps.sqldf(q1, locals())

plt.plot(Number_of_VehiclesvsLoss_dataset.Number_of_Vehicles,Number_of_VehiclesvsLoss_dataset.Losses,color='blue')
plt.title('Number_of_Vehicles vs Losses')
plt.xlabel('Number_of_Vehicles')
plt.ylabel('Losses')
plt.show()
#plt.savefig("Number_of_Vehicles_Losses.pdf")

#Gender vs Losses

q1 = """SELECT Gender,sum(Losses_Calc) 
as Losses FROM capped_losses_dataset 
group by Gender Order by Gender"""
GendervsLoss_dataset=ps.sqldf(q1, locals())

plt.plot(GendervsLoss_dataset.Gender,GendervsLoss_dataset.Losses,color='blue')
plt.title('Gender vs Losses')
plt.xlabel('Gender')
plt.ylabel('Losses')
plt.show()
#plt.savefig("Gender_Losses.pdf")

#Married vs Losses

q1 = """SELECT Married,sum(Losses_Calc) 
as Losses FROM capped_losses_dataset 
group by Married Order by Married"""
Married_dataset=ps.sqldf(q1, locals())

plt.plot(Married_dataset.Married,Married_dataset.Losses,color='blue')
plt.title('Married vs Losses')
plt.xlabel('Married')
plt.ylabel('Losses')
plt.show()
#plt.savefig("Married_Losses.pdf")


#Vehicle_Age vs Losses

q1 = """SELECT Vehicle_Age,sum(Losses_Calc) 
as Losses FROM capped_losses_dataset 
group by Vehicle_Age Order by Vehicle_Age"""
Vehicle_Age_dataset=ps.sqldf(q1, locals())

plt.plot(Vehicle_Age_dataset.Vehicle_Age,Vehicle_Age_dataset.Losses,color='blue')
plt.title('Vehicle_Age vs Losses')
plt.xlabel('Vehicle_Age')
plt.ylabel('Losses')
plt.show()
#plt.savefig("Vehicle_Age_Losses.pdf")

#Fuel_Type vs Losses

q1 = """SELECT Fuel_Type,sum(Losses_Calc) 
as Losses FROM capped_losses_dataset 
group by Fuel_Type Order by Fuel_Type"""
Fuel_Type_dataset=ps.sqldf(q1, locals())

plt.plot(Fuel_Type_dataset.Fuel_Type,Fuel_Type_dataset.Losses,color='blue')
plt.title('Fuel_Type vs Losses')
plt.xlabel('Fuel_Type')
plt.ylabel('Losses')
plt.show()
#plt.savefig("Fuel_Type_Losses.pdf")

#EDA Report Ends

#Model Data Preprocessing Steps

#Segregating independent variables and response variables
X = capped_losses_dataset.iloc[:, :-1].values
Y = capped_losses_dataset.iloc[:, 7].values



#Categorical Data - Character

from sklearn import preprocessing
myencoder=preprocessing.LabelEncoder()
X[:,3]=myencoder.fit_transform(X[:,3])
X[:,4]=myencoder.fit_transform(X[:,4])
X[:,6]=myencoder.fit_transform(X[:,6])
X = pd.DataFrame(X)
Y = pd.DataFrame(Y)

#Splitting the Datasets
from sklearn.model_selection import train_test_split
X_Train,X_Test,Y_Train,Y_Test=train_test_split(X,Y,test_size=0.3,random_state=0)

# Fitting Simple Linear Regression to the Training Dataset
# Have to fit Training but test should be used
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_Train,Y_Train)

#Predicting the Test Results , creating a vector of Predicted Sets
Y_Pred=regressor.predict(X_Test)

#Creating a single DF   
X_Train_df=pd.DataFrame(X_Train)
X_Test_df=pd.DataFrame(X_Test)
Y_Train_df=pd.DataFrame(Y_Train)
Y_Test_df=pd.DataFrame(Y_Test)
Y_Pred_df=pd.DataFrame(Y_Pred)
Analysis=pd.concat([X_Test_df,Y_Test_df,Y_Pred_df],axis=1)
###################################################################

#Building the optimal Model
#b0+b1X+B2X -- B0 should be 1
import statsmodels.formula.api as sm
#Bo is not considered so we will add 1 coulmn with the 
#values as 1 in our dataset
X=np.append(arr=np.ones((50,1)).astype(int),values=X,axis=1)

X_BE=X[:,[0,1,2,3,4,5]]
regressor_OLS=sm.OLS(endog=Y, exog=X_BE).fit()
regressor_OLS.summary()


X_BE=X[:,[0,1,3,4,5]]
regressor_OLS=sm.OLS(endog=Y, exog=X_BE).fit()
regressor_OLS.summary()


X_BE=X[:,[0,3,4,5]]
regressor_OLS=sm.OLS(endog=Y, exog=X_BE).fit()
regressor_OLS.summary()

X_BE=X[:,[0,3,5]]
regressor_OLS=sm.OLS(endog=Y, exog=X_BE).fit()
regressor_OLS.summary()


X_BE=X[:,[0,3]]
regressor_OLS=sm.OLS(endog=Y, exog=X_BE).fit()
regressor_OLS.summary()

#Automating the above part


import statsmodels.formula.api as sm
def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(Y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x
 
SL = 0.05
X_BE = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_BE, SL)

#Visualizing the Predicted Test Results
plt.scatter(X_Train,Y_Train,color='blue')
plt.plot(X_Train,regressor.predict(X_Train),color='red')
plt.title('Actual vs Predicted')
plt.xlabel('Years of Exp.')
plt.ylabel('Salary')
plt.show()

