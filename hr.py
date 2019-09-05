import xgboost as xgb
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

data = pd.read_csv('datasets/hr/train.csv')
test_data = pd.read_csv('datasets/hr/test.csv')


data.department = data.department.replace(['Sales & Marketing', 'Operations', 'Technology', 'Analytics',
       'R&D', 'Procurement', 'Finance', 'HR', 'Legal'], [1,2,3,4,5,6,7,8,9])
data.education = data.education.replace(["Master's & above", "Bachelor's", 'Below Secondary'], [1,2,3])
data.gender = data.gender.replace(['f','m'], [1,2])
data.recruitment_channel = data.recruitment_channel.replace(['sourcing','other','referred'], [1,2,3])

test_data.department = test_data.department.replace(['Sales & Marketing', 'Operations', 'Technology', 'Analytics',
       'R&D', 'Procurement', 'Finance', 'HR', 'Legal'], [1,2,3,4,5,6,7,8,9])
test_data.education = test_data.education.replace(["Master's & above", "Bachelor's", 'Below Secondary'], [1,2,3])
test_data.gender = test_data.gender.replace(['f','m'], [1,2])
test_data.recruitment_channel = test_data.recruitment_channel.replace(['sourcing','other','referred'], [1,2,3])

excel=pd.DataFrame()
excel['employee_id'] = test_data.employee_id

data = data.drop(['region','employee_id'], axis=1)
test_data = test_data.drop(['region','employee_id'], axis=1)

X, y = data.iloc[:,:-1],data.iloc[:,-1]
data_dmatrix = xgb.DMatrix(data=X,label=y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=123)

xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.4, learning_rate = 0.8,
                max_depth = 8, alpha = 20, n_estimators = 20)

xg_reg.fit(X_train,y_train)

preds = xg_reg.predict(test_data)

excel['is_promoted'] = preds.astype('int')
excel.to_csv("Hr(clean)1.csv", index=False)
