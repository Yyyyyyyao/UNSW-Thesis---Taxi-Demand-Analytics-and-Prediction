import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import statistics


new_df = pd.read_csv('2017_01_final.csv')




# overall result
# RMSE: 29.440814
# Test acc: 13.35 acc

X, y = new_df.iloc[:, :-1], new_df.iloc[:, -1]

data_dmatrix = xgb.DMatrix(data=X,label=y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)

xg_reg = xgb.XGBRegressor(objective='reg:linear', colsample_bytree = 0.9, subsample= 0.96, min_child_weight = 6.19, learning_rate = 0.1186, max_depth=14, gamma=1, n_estimators = 463)

xg_reg.fit(X_train,y_train)
preds = xg_reg.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))

'''
df_monday = pd.read_csv('Monday_eval_dataset.csv')
X_mon_test, y_mon_test = df_monday.iloc[:, :-1], df_monday.iloc[:, -1]
mon_preds = xg_reg.predict(X_mon_test)
mon_mean = statistics.mean(abs(mon_preds - y_mon_test))
print(mon_preds)
print('Test acc: %.2f acc' % (mon_mean))
'''

testacc = statistics.mean(abs(y_test[:] - preds[:]))
print('Test acc: %.2f acc' % (testacc))


