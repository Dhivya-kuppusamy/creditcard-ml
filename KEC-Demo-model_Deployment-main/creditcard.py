import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import joblib
data = pd.read_csv("./fraudTest.csv")
data

data.isnull().sum()
# Example imputation for numeric columns
data['city_pop'].fillna(data['city_pop'].median(), inplace=True)
data['unix_time'].fillna(data['unix_time'].median(), inplace=True)
data['merch_lat'].fillna(data['merch_lat'].median(), inplace=True)
data['merch_long'].fillna(data['merch_long'].median(), inplace=True)
data['is_fraud'].fillna(0, inplace=True)  # Assuming is_fraud is a binary variable

    # Check for missing values in the entire dataset
missing_values = data.isnull().sum()




data['cc_num'],cc_name=pd.factorize(data['cc_num'])
data['category'],category_name=pd.factorize(data['category'])

data['trans_date_trans_time'],time_name=pd.factorize(data['trans_date_trans_time'])

data['amt'],amt_name=pd.factorize(data['amt'])

data['merchant'],merchant_name=pd.factorize(data['merchant'])

data['zip'],zip_name=pd.factorize(data['zip'])

data['lat'],lat_name=pd.factorize(data['lat'])

data['long'],long_name=pd.factorize(data['long'])
  
data['city_pop'],city_name=pd.factorize(data['city_pop'])



data['first'],first_name=pd.factorize(data['first'])

data['last'],last_name=pd.factorize(data['last'])

data['street'],street_name=pd.factorize(data['street'])

data['job'],job_name=pd.factorize(data['job'])

data['dob'],dob_name=pd.factorize(data['dob'])

data['trans_num'],trans_name=pd.factorize(data['trans_num'])

data['gender'],gender_name=pd.factorize(data['gender'])

data['city'],city_name=pd.factorize(data['city'])

data['state'],state_name=pd.factorize(data['state'])
data.drop(['unix_time', 'merch_lat', 'merch_long','Unnamed: 0'],axis=1, inplace=True)

X = data.drop('is_fraud', axis=1)  # Assuming 'is_fraud' is the target variable
y = data['is_fraud']
from sklearn import tree, metrics
dtree=tree.DecisionTreeClassifier(criterion='gini')#entrophy or gini
dtree.fit(X,y)
 # Save the model as a pickle file  
joblib.dump(dtree,"credit.joblib")
