from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

data = pd.read_csv('datasets/hr/train.csv')
test_data = pd.read_csv('datasets/hr/test.csv')


data.department = data.department.replace(['Sales & Marketing', 'Operations', 'Technology', 'Analytics',
       'R&D', 'Procurement', 'Finance', 'HR', 'Legal'], [1,2,3,4,5,6,7,8,9])
data.education = data.education.replace(["Master's & above", "Bachelor's", 'Below Secondary'], [3,2,1])
data.gender = data.gender.replace(['f','m'], [1,2])

test_data.department = test_data.department.replace(['Sales & Marketing', 'Operations', 'Technology', 'Analytics',
       'R&D', 'Procurement', 'Finance', 'HR', 'Legal'], [1,2,3,4,5,6,7,8,9])
test_data.education = test_data.education.replace(["Master's & above", "Bachelor's", 'Below Secondary'], [1,2,3])
test_data.gender = test_data.gender.replace(['f','m'], [1,2])

bins = [0,20,30,40,50,60,70]
labels = [0,25,35,45,55,65]
data.age = pd.cut(data.age, bins=bins, labels=labels)
test_data.age = pd.cut(test_data.age, bins=bins, labels=labels)


excel=pd.DataFrame()
excel['employee_id'] = test_data.employee_id

data = data.drop(['region','employee_id','recruitment_channel'], axis=1)
test_data = test_data.drop(['region','employee_id','recruitment_channel'], axis=1)

data = data.fillna(0)
test_data = test_data.fillna(value=0)

X, y = data.iloc[:,:-1],data.iloc[:,-1]
##data_dmatrix = xgb.DMatrix(data=X,label=y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=123)

clf2 = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.7,
                max_depth = 8, alpha = 20, n_estimators = 10, verbose=False)

clf1 = RandomForestRegressor(n_estimators=50, criterion='mse', max_depth=None, min_samples_split=2, min_samples_leaf=2,
                             min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0,
                             min_impurity_split=None, bootstrap=False, oob_score=False, n_jobs=-1, random_state=None, verbose=15, warm_start=True)

clf3 = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=2,
                              min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0,
                              min_impurity_split=None, bootstrap=False, oob_score=False, n_jobs=None, random_state=None, verbose=0,
                              warm_start=False, class_weight=None)

########################################################
## NN model

NN_model = Sequential()

# The Input Layer :
NN_model.add(Dense(128, kernel_initializer='normal',input_dim = X_train.shape[1], activation='relu'))

# The Hidden Layers :
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))

# The Output Layer :
NN_model.add(Dense(1, kernel_initializer='normal',activation='linear'))

# Compile the network :
NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])


#NN_model.fit(X_train, y_train, epochs=3, batch_size=128, validation_split = 0.2)
##############################################################

clf=clf3
 
clf.fit(X_train,y_train)

preds = clf.predict(test_data)

excel['is_promoted'] = preds.astype('int')
excel.to_csv("Hr(clean)1.csv", index=False)

print(np.count_nonzero(preds == 1))

## My Rank		1011		Score	0.4627659574
