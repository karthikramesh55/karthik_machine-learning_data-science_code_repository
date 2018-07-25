## 3. Reading in to Pandas ##

import pandas as pd

loans_2007 = pd.read_csv('loans_2007.csv', delimiter = ',')
loans_2007.head(1)

## First group of columns ##

loans_2007 = loans_2007.drop(['id','member_id', 'funded_amnt', 'funded_amnt_inv', 'grade', 'sub_grade', 'emp_title', 'issue_d'], axis=1)

## Second group of features ##

loans_2007 = loans_2007.drop(['zip_code','out_prncp', 'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp'], axis=1)

## Third group of features ##

loans_2007=loans_2007.drop(['total_rec_int','total_rec_late_fee','recoveries','collection_recovery_fee', 'last_pymnt_amnt', 'last_pymnt_d'], axis=1)

## Target column ##

print(loans_2007['loan_status'].value_counts())

## Binary classification ##

loans_2007 = loans_2007[(loans_2007['loan_status'] == 'Fully Paid') | (loans_2007['loan_status'] == 'Charged Off')]

status_replace = {"loan_status" : {"Fully Paid" : 1, "Charged Off" : 0}}
loans_2007 = loans_2007.replace(status_replace)

## Removing single value columns ##

orig_columns = loans_2007.columns
drop_columns = []
for col in orig_columns:
    col_series = loans_2007[col].dropna().unique()
    if len(col_series) == 1:
        drop_columns.append(col)
loans_2007 = loans_2007.drop(drop_columns, axis=1)
print(drop_columns)

##----------------------------------------------------------------------------------##

## Importing the above cleaned dataframe

loans = pd.read_csv('filtered_loans_2007.csv', delimiter = ',')
loans.head(2)

## Identifying the number of missing values in the dataframe

null_counts = loans.isnull().sum()
print(null_counts)

## Handling missing values ##

loans = loans.drop('pub_rec_bankruptcies', axis=1)

loans = loans.dropna(axis = 0)
print(loans.dtypes.value_counts())

## Text columns ##

object_columns_df = loans.select_dtypes(include=['object'])
object_columns_df.head(5)

## First 5 categorical columns ##

cols = ['home_ownership', 'verification_status', 'emp_length', 'term', 'addr_state']
for c in cols:
    print(loans[c].value_counts())

## The reason for the loan ##

print(loans["purpose"].value_counts())
print(loans["title"].value_counts())

## Categorical columns ##

mapping_dict = {
    "emp_length": {
        "10+ years": 10,
        "9 years": 9,
        "8 years": 8,
        "7 years": 7,
        "6 years": 6,
        "5 years": 5,
        "4 years": 4,
        "3 years": 3,
        "2 years": 2,
        "1 year": 1,
        "< 1 year": 0,
        "n/a": 0
    }
}



loans = loans.drop(['last_credit_pull_d','addr_state','title','earliest_cr_line'], axis = 1)

loans['int_rate'] = loans['int_rate'].str.rstrip('%').astype('float')
loans['revol_util'] = loans['revol_util'].str.rstrip('%').astype('float')
loans = loans.replace(mapping_dict)

## Dummy variables ##

import pandas as pd

dummy_df = pd.get_dummies(loans[['home_ownership','verification_status','purpose','term']])

loans = pd.concat([loans, dummy_df], axis = 1)

loans = loans.drop(['home_ownership','verification_status','purpose','term'],axis = 1)

##---------------------------------------------------------------------------------------------##

## Importing the above final cleaned version of the dataframe

loans = pd.read_csv('cleaned_loans_2007.csv', delimiter = ',')
loans.info()

## Picking an error metric ##

tn = (predictions == 0) & (loans['loan_status'] == 0)

tp = (predictions == 1) & (loans['loan_status'] == 1)

fn = (predictions == 0) & (loans['loan_status'] == 1)

fp = (predictions == 1) & (loans['loan_status'] == 0)


## Class imbalance ##

import numpy

# Predict that all loans will be paid off on time.
predictions = pd.Series(numpy.ones(loans.shape[0]))

fp_filter = (predictions == 1) & (loans["loan_status"] == 0)
fp = len(predictions[fp_filter])

# True positives.
tp_filter = (predictions == 1) & (loans["loan_status"] == 1)
tp = len(predictions[tp_filter])

# False negatives.
fn_filter = (predictions == 0) & (loans["loan_status"] == 1)
fn = len(predictions[fn_filter])

# True negatives
tn_filter = (predictions == 0) & (loans["loan_status"] == 0)
tn = len(predictions[tn_filter])

fpr = fp / (fp + tn)
print("The false positive rate or fallout is:", fpr)

tpr = tp / (tp + fn)
print("The true positive rate or fallout is:", tpr)


## Logistic Regression ##

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
cols = loans.columns
train_cols = cols.drop("loan_status")
features = loans[train_cols]
target = loans["loan_status"]
lr.fit(features, target)
predictions = lr.predict(features)

## Cross Validation ##

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

lr = LogisticRegression()
predictions = cross_val_predict(lr, features, target, cv = 3)
predictions = pd.Series(predictions)

# False positives.
fp_filter = (predictions == 1) & (loans["loan_status"] == 0)
fp = len(predictions[fp_filter])

# True positives.
tp_filter = (predictions == 1) & (loans["loan_status"] == 1)
tp = len(predictions[tp_filter])

# False negatives.
fn_filter = (predictions == 0) & (loans["loan_status"] == 1)
fn = len(predictions[fn_filter])

# True negatives
tn_filter = (predictions == 0) & (loans["loan_status"] == 0)
tn = len(predictions[tn_filter])
# Rates
tpr = tp  / (tp + fn)
fpr = fp  / (fp + tn)
print(tpr)
print(fpr)

## Penalizing the classifier ##

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

lr = LogisticRegression(class_weight = 'balanced')
predictions = cross_val_predict(lr, features, target)
predictions = pd.Series(predictions)

# False positives.
fp_filter = (predictions == 1) & (loans["loan_status"] == 0)
fp = len(predictions[fp_filter])

# True positives.
tp_filter = (predictions == 1) & (loans["loan_status"] == 1)
tp = len(predictions[tp_filter])

# False negatives.
fn_filter = (predictions == 0) & (loans["loan_status"] == 1)
fn = len(predictions[fn_filter])

# True negatives
tn_filter = (predictions == 0) & (loans["loan_status"] == 0)
tn = len(predictions[tn_filter])

# Rates
tpr = tp / (tp + fn)
fpr = fp / (fp + tn)

print(tpr)
print(fpr)


## Manual penalties ##

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

penalty = {0 : 10, 1 : 1}

lr = LogisticRegression(class_weight = penalty)
predictions = cross_val_predict(lr, features, target)
predictions = pd.Series(predictions)

# False positives.
fp_filter = (predictions == 1) & (loans["loan_status"] == 0)
fp = len(predictions[fp_filter])

# True positives.
tp_filter = (predictions == 1) & (loans["loan_status"] == 1)
tp = len(predictions[tp_filter])

# False negatives.
fn_filter = (predictions == 0) & (loans["loan_status"] == 1)
fn = len(predictions[fn_filter])

# True negatives
tn_filter = (predictions == 0) & (loans["loan_status"] == 0)
tn = len(predictions[tn_filter])

# Rates
tpr = tp / (tp + fn)
fpr = fp / (fp + tn)

print(tpr)
print(fpr)


## Random forests ##

from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_predict

rf = RandomForestClassifier(class_weight = 'balanced', random_state = 1)
predictions = cross_val_predict(rf, features, target, cv = 3)
predictions = pd.Series(predictions)

# False positives.
fp_filter = (predictions == 1) & (loans["loan_status"] == 0)
fp = len(predictions[fp_filter])

# True positives.
tp_filter = (predictions == 1) & (loans["loan_status"] == 1)
tp = len(predictions[tp_filter])

# False negatives.
fn_filter = (predictions == 0) & (loans["loan_status"] == 1)
fn = len(predictions[fn_filter])

# True negatives
tn_filter = (predictions == 0) & (loans["loan_status"] == 0)
tn = len(predictions[tn_filter])

# Rates
tpr = tp / (tp + fn)
fpr = fp / (fp + tn)

print(tpr)
print(fpr)

