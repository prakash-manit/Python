# library to be installed
# python --version
# latest version is 3.12.4
# python.exe -m pip install --upgrade pip
# pip install pandas
# pip install matplotlib
# pip install seaborn
# pip install missingno
# pip install scikit-learn
# pip install xgboost

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as ms
from sklearn import model_selection, metrics  #to include metrics for evaluation # this used to be cross_validation
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import matplotlib.pylab as plt

# Quiet warnings since this is a demo (it quiets future and deprecation warnings).
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

# Read the data and view the top portion to see what we are dealing with.
data=pd.read_csv('Telco-Customer-Churn.csv')
# print(data.head())
print (data.info)

# Convert the 'TotalCharges' column to numeric
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors = 'coerce')
# Get the rows where 'TotalCharges' is null
# print (data.loc[data['TotalCharges'].isna()==True])

# Above we see that the blank "TotalCharges" happen when customers have 0 months tenure so we will change those values to $0. Replace missing values in 'TotalCharges' column with 0
data[data['TotalCharges'].isna()==True] = 0
# print (data.loc[data['TotalCharges'].isna()==True])

# Get unique values in 'OnlineBackup' column
#print (data['OnlineBackup'].unique())

# See how many rows and columns.
# print(data.shape)

# More data cleanup: next we’ll convert the categorical values into numeric values.
data['gender'].replace(['Male','Female'],[0,1],inplace=True)
data['Partner'].replace(['Yes','No'],[1,0],inplace=True)
data['Dependents'].replace(['Yes','No'],[1,0],inplace=True)
data['PhoneService'].replace(['Yes','No'],[1,0],inplace=True)
data['MultipleLines'].replace(['No phone service','No', 'Yes'],[0,0,1],inplace=True)
data['InternetService'].replace(['No','DSL','Fiber optic'],[0,1,2],inplace=True)
data['OnlineSecurity'].replace(['No','Yes','No internet service'],[0,1,0],inplace=True)
data['OnlineBackup'].replace(['No','Yes','No internet service'],[0,1,0],inplace=True)
data['DeviceProtection'].replace(['No','Yes','No internet service'],[0,1,0],inplace=True)
data['TechSupport'].replace(['No','Yes','No internet service'],[0,1,0],inplace=True)
data['StreamingTV'].replace(['No','Yes','No internet service'],[0,1,0],inplace=True)
data['StreamingMovies'].replace(['No','Yes','No internet service'],[0,1,0],inplace=True)
data['Contract'].replace(['Month-to-month', 'One year', 'Two year'],[0,1,2],inplace=True)
data['PaperlessBilling'].replace(['Yes','No'],[1,0],inplace=True)
data['PaymentMethod'].replace(['Electronic check', 'Mailed check', 'Bank transfer (automatic)','Credit card (automatic)'],[0,1,2,3],inplace=True)
data['Churn'].replace(['Yes','No'],[1,0],inplace=True)
 
# print (data.info())

# Let's look at relationships between customer data and churn using correlation.
data = data.drop('customerID', axis = 1)
corr = data.corr()
sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values, annot = True, annot_kws={'size':12})
heat_map=plt.gcf()
heat_map.set_size_inches(20,15)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()

# Our goal is to avoid multicollinearity by dropping features that are closely correlated with each other. 
# For example here it is TotalCharges and MonthlyCharges. So we will drop TotalCharges.
data.pop('TotalCharges')

# Run info again to make sure TotalCharges has been dropped (popped off).
# print (data.info())

# Explore how many churn data points we have.
print(len(data['Churn']))
# Explore how many customers in this dataset have churned. Is this dataset 50% as the team suggests is the overall customer churn rate?  
print(data['Churn'].value_counts())

# This creates a bar graph of churn (Yes vs. No) so we can check how the data is balanced.
# data['Churn'].value_counts().plot(kind = 'bar', title = 'Bar Graph of Non-Churners vs Churners by Count (Churn is a 1)', color = 'blue', align = 'center')
# plt.show()
# The dataset does not have a huge imbalance which is good news! But also we clearly see it does not have the 50% as we would have thought. 

# Creates initial contingency table between Churn and gender. Male is 0, Female is 1.
# gender_churn_contingency = pd.crosstab(data["gender"], data["Churn"])
# print(gender_churn_contingency)
# Male and females churn at about the same rate, so not much to see here. Let's keep moving.

# # Explore the relationship between instances of Tech Support and Churn. 
# # Stacked Bar of Tech Support and Churn.
# tech_support_churn = pd.crosstab(data['TechSupport'], data['Churn'])
# tech_support_churn.plot(kind = 'bar', stacked = True)
# plt.ylabel('Count')
# plt.xlabel('Tech Support Count')
# plt.title('Churn Rate Relative to Uses of Tech Support (Churned is a 1)')
# plt.show()
# # We can see that non-churners use tech support more often than customers that end up churning.
# # So let's explore some ways to get people to use Tech Support more often so they cancel (churn) less. You can see notes for this at the bottom. 
# # Also, tech support in this data is just a Y/N. It would be useful in future to include how many tech support calls by customer 
# # so we could analyze how the number of tech support calls relates to churn.

# # Churn rate relative to tenure.
# # Stacked bar of tenure and churn.
# tenure_churn = pd.crosstab(data['tenure'], data['Churn'])
# tenure_churn.plot(kind = 'bar', stacked = True)
# plt.ylabel('Count')
# plt.xlabel('Tenure of Subscription')
# plt.title('Churn Rate Relative to Tenure of Subscription (Churned is a 1)')
# plt.show()
# # We can clearly see the longer a customer stays as a subscriber, the less they are likely to churn!

X_train, X_test, y_train, y_test = train_test_split(data.drop('Churn',axis=1), 
                                                    data['Churn'], test_size=0.30, 
                                                    random_state=101)
train=pd.concat([X_train,y_train],axis=1)

# Function to estimate the best value of n_estimators and fit the model with the given data.
def modelfit(alg, dtrain, predictors,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):    
    if useTrainCV:
        #to get the parameters of xgboost
        xgb_param = alg.get_xgb_params() 
        
        #to convert into a datastructure internally used by xgboost for training efficiency 
        # and speed
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        
        #xgb.cv is used to find the number of estimators required for the parameters 
        # which are set
        cvresult = xgb.cv(xgb_param, xgtrain, 
                          num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                        metrics='auc', early_stopping_rounds=early_stopping_rounds)        
        #setting the n_estimators parameter using set_params
        alg.set_params(n_estimators=cvresult.shape[0])        
        print(alg.get_xgb_params())
    
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain['Churn'],eval_metric='auc')    
    return alg

    
    # Function to get the accuracy of the model on the test data given the features considered.
def get_accuracy(alg,predictors):
    dtrain_predictions = alg.predict(X_test[predictors])
    dtrain_predprob = alg.predict_proba(X_test[predictors])[:,1]
    print ("\nModel Report")
    print ("Accuracy : %.4g" % metrics.accuracy_score(y_test.values, 
                                                      dtrain_predictions))
    print ("AUC Score (Train): %f" % metrics.roc_auc_score(y_test.values, 
                                                           dtrain_predprob))

# Function to get the feature importances based on the model fit.
def get_feature_importances(alg):
    #to get the feature importances based on xgboost we use fscore
    feat_imp = pd.Series(alg._Booster.get_fscore()).sort_values(ascending=False)
    print(feat_imp)
    
    #this shows the feature importances on a bar chart
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')

target = 'Churn'
IDcol = 'customerID'

# To return the XGBClassifier object based on the values of the features.
# XGBoost converts weak learners to strong learners through an ensemble method. 
# Unlike bagging, in the classical boosting the subset creation is not random and depends upon the performance of the previous models.

def XgbClass(learning_rate =0.1,n_estimators=1000,max_depth=5,min_child_weight=1,
             gamma=0,subsample=0.8,colsample_bytree=0.8):
    xgb1 = XGBClassifier(learning_rate=learning_rate,
                         n_estimators=n_estimators,
                         max_depth=max_depth,
                         min_child_weight=min_child_weight,
                         gamma=gamma,
                         subsample=subsample,
                         colsample_bytree=colsample_bytree)
    return xgb1

# Function to return the list of predictors.
# These are the initial parameters before tuning.
def drop_features(l):
    return [x for x in train.columns if x not in l]

# First Prediction: Use of initial parameters and without feature engineering
# from xgboost import XGBClassifier
# import xgboost as xgb

# predictors = drop_features([target, IDcol])
# xgb1=XgbClass()
# first_model=modelfit(xgb1, train, predictors)
# xgb1.fit(train[predictors],train['Churn'])




