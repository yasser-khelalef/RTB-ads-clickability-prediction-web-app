#this file is to build train and test our model
#importing required libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from numpy import savetxt
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

###########  read the dataset  #######################################################################
data_raw = pd.read_csv('train.csv')
data_raw2 = data_raw.drop(['timestamp' ,'bidid' ,'device_id' ,'user_id','support_id' ,'ad_id'], axis = 1) 

print(data_raw2['format'].value_counts())
print(data_raw2['support_type'].value_counts())
print(data_raw2['device_type'].value_counts())
print(data_raw2['device_os'].value_counts())
print(data_raw2['device_language'].value_counts())
print(data_raw2['device_model'].value_counts())
print(data_raw2['verticals_0'].value_counts())
print(data_raw2['verticals_1'].value_counts())
print(data_raw2['verticals_2'].value_counts())
print(data_raw2['vertical_3'].value_counts())
print(data_raw2['bid_price'].value_counts())
print(data_raw2['won_price'].value_counts())
print(data_raw2['bidfloor'].value_counts())

# We can see after this step that the format, device_type, and device_os features have only one class, Hence, we remove them from the dataset
# because they would have no influence
data_raw2 = data_raw.drop(['timestamp' ,'bidid' ,'device_id' ,'user_id','support_id' ,'ad_id', 'format','device_type','device_os'], axis = 1) 
#print(data_raw2.head)
######################################################################################################



###########  Convert catergorical inputs into numerical by creating dummy variables  #################
cat_vars=['device_model', 'support_type']
for var in cat_vars:
    cat_list='var'+'_'+var
    cat_list = pd.get_dummies(data_raw2[var], prefix=var)
    data1=data_raw2.join(cat_list)
    data_raw2=data1
cat_vars=['device_model', 'support_type']
data_vars=data_raw2.columns.values.tolist()
to_keep=[i for i in data_vars if i not in cat_vars]
data_final=data_raw2[to_keep]

device_language = data_final['device_language'].values.tolist()
for i in range(len(device_language)):
    if device_language[i] == 'en_EN' or device_language[i] == 'fr_FR':
        device_language[i] = 1 #big language
    elif device_language[i] == 'ar_AR' or device_language[i] == 'es_ES' or device_language[i] == 'ru_RU' or device_language[i] == 'de_DE' or device_language[i] =='zh_CN' or device_language[i] == 'hi_HI':
        device_language[i] = 2 #regional language
    else:
        device_language[i] = 3
        
device_language_new = pd.DataFrame(device_language, columns=['dev_lang'])
data_final = data_final.drop(['device_language'], axis = 1)
data_final['dev_lang'] = device_language
print(data_final.columns)
print(data_final.columns.values)

######################################################################################################


print(data_final['clicked'].value_counts())
# The data set is umbalanced
###########  Balancing the dataset using Up samole minority class method #############################
from sklearn.utils import resample
df_majority = data_final[data_final['clicked'] == 0]
df_minority = data_final[data_final['clicked'] == 1]
df_minority_upsampled = resample(df_minority,replace=True, n_samples=724386, random_state=123)
data_final_minup = pd.concat([df_majority, df_minority_upsampled])
print('The clicked values after using resample up minority class are : ')
print(data_final_minup.clicked.value_counts())

###########  Balancing the dataset using SMOTE  ######################################################
X = data_final.loc[:, data_final.columns != 'clicked']
y = data_final.loc[:, data_final.columns == 'clicked']

from imblearn.over_sampling import SMOTE
os = SMOTE(random_state=0)
cols2 = ['bidfloor' ,'bid_price' ,'won_price','device_model_iphone', 'device_model_ipod']
X = X[cols2]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
columns = X_train.columns
os_data_X,os_data_y=os.fit_sample(X_train, y_train)
os_data_X = pd.DataFrame(data=os_data_X,columns=columns )
os_data_y= pd.DataFrame(data=os_data_y,columns=['clicked'])
print('The clicked values after using SMOTE method are : ')
print(os_data_y.clicked.value_counts())
######################################################################################################

plt.figure(figsize=(15,15))
cor = data_raw2.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()


######################################################################################################
######### The heat map shows that there is no correlation between the features and the target variable


########### Building the model using Random Forest classifier ########################################
######## Using the unbalanced dataset directly
X = data_final.drop(['clicked'], axis = 1)
y = data_final['clicked']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state=0)
clf_4 = RandomForestClassifier()
clf_4.fit(X_train, y_train)
pred_y_4 = clf_4.predict(X_test)
accu = accuracy_score(y_test, pred_y_4)
print("The accuracy score for Random forest classification method using the unbalanced dataset is : ")
print(accu)
joblib_file = "UnbalancedData_RandForClass_Model.pkl"  
joblib.dump(clf_4, joblib_file)
savetxt('Test-RanForClass-unbalanced.csv', y_test, delimiter = ',')
savetxt('Results-RanForClass-unbalanced.csv', pred_y_4, delimiter = ',')


########### First using the up sample minority class data
X = data_final_minup.drop(['clicked'], axis = 1)
y = data_final_minup['clicked']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state=0)
clf_4 = RandomForestClassifier()
clf_4.fit(X_train, y_train)
pred_y_4 = clf_4.predict(X_test)
accu = accuracy_score(y_test, pred_y_4)
print("The accuracy score for Random forest classification method using upsample min class method is : ")
print(accu)
joblib_file = "UpSample_MinorityClass_RandForClass_Model.pkl"  
joblib.dump(clf_4, joblib_file)
savetxt('Test-RanForClass-upsaml.csv', y_test, delimiter = ',')
savetxt('Results-RanForClass-upsample.csv', pred_y_4, delimiter = ',')

########### Second using the SMOTE data

X = os_data_X
y = os_data_y
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state=0)
clf_4 = RandomForestClassifier()
clf_4.fit(X_train, y_train)
pred_y_4 = clf_4.predict(X_test)
accu = accuracy_score(y_test, pred_y_4)
print("The accuracy score for Random forest classification method using upsample min class method is : ")
print(accu)
joblib_file = "SMOTE_RandForClass_Model.pkl"  
joblib.dump(clf_4, joblib_file)
savetxt('Test-RanForClass-SMOTE', y_test, delimiter = ',')
savetxt('Results-RanForClass-SMOTE', pred_y_4, delimiter = ',')


