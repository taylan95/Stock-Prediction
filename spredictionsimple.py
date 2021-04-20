#Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as web
import datetime as dt
import scipy
from scipy.stats import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,LSTM
import xgboost
from xgboost.sklearn import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn import metrics
from sklearn.metrics import r2_score

stockname = str(input("stock name: "))
site = str(input("site: "))
startdate = str(input("startdate: "))
enddate = str(input("enddate: "))
predictiondays = int(input("prediction days: "))

def get_stock(stockname,site,startdate,enddate):
    data = web.DataReader(stockname,data_source = site,start = startdate,end = enddate)
    return data.copy()

df = get_stock(stockname,site,startdate,enddate)

def generalgraph(df,stockname):
    plt.title("Close Prices")
    plt.ylabel(f"Stock Prices of {stockname}")
    df["Close"].plot(figsize = (25,15))
    plt.savefig('C:\\Users\\taylan.polat\\Desktop\\PredictionResults\\generalgraph.png',transparent=True,
                bbox_inches="tight",facecolor="w")
    
generalgraph(df,stockname)

df['SMA'] = df['Close'].rolling(window = predictiondays).mean()
df['STD'] = df['Close'].rolling(window = predictiondays).std()
df['Up'] = df['SMA'] + (df['STD'] * 2)
df['Down'] = df['SMA'] - (df['STD'] * 2)
df["Prediction"] = df[["Close"]].shift(-predictiondays)

def generalgraph2(df,stockname):
    variable_list = ["Close","SMA","Up","Down"]
    df[variable_list].plot(figsize = (25,16))
    plt.title("Bollinger Bands Strategy")
    plt.ylabel("Price")
    plt.savefig('C:\\Users\\taylan.polat\\Desktop\\PredictionResults\\generalgraph2.png',transparent=True,
                bbox_inches="tight",facecolor="w")
    
generalgraph2(df,stockname)

X = (np.array(df["Close"])[:-predictiondays]).reshape(-1,1)
y = np.array(df["Prediction"])[:-predictiondays]
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.25,random_state = 0)

#Models

lgbm = lgb.LGBMRegressor(num_leaves=31,
                        learning_rate=0.05,
                        n_estimators=20)
lgbm.fit(x_train, y_train,
        eval_set=[(x_test, y_test)],
        eval_metric='l1',
        early_stopping_rounds=5)

tree = DecisionTreeRegressor().fit(x_train,y_train)
lr = LinearRegression().fit(x_train,y_train)
xgb = XGBRegressor(n_estimators=1000, max_depth=7, eta=0.1, subsample=0.7, colsample_bytree=0.8).fit(x_train,y_train)
rfr = RandomForestRegressor(n_estimators = 100).fit(x_train,y_train)
gbr = GradientBoostingRegressor(n_estimators=600, 
    max_depth=5, 
    learning_rate=0.01, 
    min_samples_split=3).fit(x_train,y_train)
adaboost = AdaBoostRegressor(base_estimator=None, learning_rate=1.0, loss='linear',
         n_estimators=100, random_state=None).fit(x_train,y_train)
bag = BaggingRegressor(base_estimator=DecisionTreeRegressor(),
                       max_features = 1.0,
                       bootstrap_features = False,
                       random_state = 0).fit(x_train, y_train)

x_future = pd.DataFrame(df["Close"])[:-predictiondays]
x_future = x_future.tail(predictiondays)
x_future = np.array(x_future)

tree_predictions = tree.predict(x_future)
lr_predictions = lr.predict(x_future)
lgbm_predictions = lgbm.predict(x_future)
xgb_predictions = xgb.predict(x_future)
rfr_predictions = rfr.predict(x_future)
gbr_predictions = gbr.predict(x_future)
bag_predictions = bag.predict(x_future)
adaboost_predictions = adaboost.predict(x_future)

def predictiongraph(df,X,predictions,predname):
    valid = df[X.shape[0]:]
    valid["Predicted"] = predictions
    plt.figure(figsize = (25,15))
    plt.title(f"Model of {predname}")
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.plot(df["Close"])
    plt.plot(valid[["Close","Predicted"]])
    plt.legend(["Actual","Value","Predicted"])
    plt.savefig(f'C:\\Users\\taylan.polat\\Desktop\\PredictionResults\\prediction_{predname}.png',transparent=True,
                bbox_inches="tight",facecolor="w")
    
predictions1 = [tree_predictions,lr_predictions,xgb_predictions,rfr_predictions,
                gbr_predictions,bag_predictions,adaboost_predictions]
prednames = ["tree_predictions","lr_predictions","xgb_predictions","rfr_predictions","gbr_predictions","bag_predictions","adaboost_predictions"]

for pred,predname in zip(predictions1,prednames):
    predictiongraph(df,X,pred,predname)
    
def signal(predictions):
    valid = df[X.shape[0]:]
    valid["Predicted"] = predictions
    buying_signal = []
    selling_signal = []
    for x in range(len(valid["Predicted"])):
        if valid["Predicted"][x] < valid['Down'][x]:
            buying_signal.append(valid["Predicted"][x])
            selling_signal.append(valid["SMA"][x])
        elif valid["Predicted"][x] > valid['Up'][x]:
            selling_signal.append(valid["Predicted"][x])
            buying_signal.append(valid["SMA"][x])
        else:
            buying_signal.append(valid["SMA"][x])
            selling_signal.append(valid["SMA"][x])
    return(buying_signal,selling_signal)

def new_data_forecast(predictions,predname):
    new_data = df[X.shape[0]:]
    new_data["Predicted"] = predictions
    new_data["Buy"] = signal(predictions)[0]
    new_data["Sell"] = signal(predictions)[1]
    fig = plt.figure(figsize = (25,16))
    ax = fig.add_subplot(1,1,1)
    x_axis = new_data.index
    ax.fill_between(x_axis,new_data['Up'],new_data['Down'],color = 'lightgray')
    ax.plot(x_axis,new_data['SMA'],color = 'black',lw = 3,label = "Simple Moving Average",alpha = 0.5)
    ax.plot(x_axis,new_data['Predicted'],color = 'blue', lw = 3, label = f"Predicted Price of {predname}",alpha = 0.5)
    ax.scatter(x_axis, new_data['Buy'],color = 'green', label = "Buying",lw = 3,marker = "^",alpha = 0.7)
    ax.scatter(x_axis, new_data['Sell'],color = 'red', label = "Selling",lw = 3,marker = "o",alpha = 0.7)
    ax.set_title("Time of Stock Price")
    ax.set_ylabel("Stock Price")
    ax.set_xlabel("Date")
    plt.xticks(rotation = 45)
    ax.legend()
    plt.savefig(f'C:\\Users\\taylan.polat\\Desktop\\PredictionResults\\prediction_charts_of_{predname}.png',transparent=True,
                bbox_inches="tight",facecolor="w")
    
new_data_forecast(tree_predictions,"tree_predictions")
new_data_forecast(lr_predictions,"lr_predictions")
new_data_forecast(xgb_predictions,"xgb_predictions")
new_data_forecast(rfr_predictions,"rfr_predictions")
new_data_forecast(gbr_predictions,"gbr_predictions")
new_data_forecast(bag_predictions,"bag_predictions")
new_data_forecast(adaboost_predictions,"adaboost_predictions")

predictions1 = [tree_predictions,lr_predictions,xgb_predictions,rfr_predictions,gbr_predictions,
                bag_predictions,adaboost_predictions]

mae = []
mse = []
rmse = []
rsquare = []

for pred in predictions1:
    mae.append(metrics.mean_absolute_error(df["Close"][-predictiondays:],pred))
    mse.append(metrics.mean_squared_error(df["Close"][-predictiondays:],pred))
    rmse.append(np.sqrt(metrics.mean_squared_error(df["Close"][-predictiondays:],pred)))
    rsquare.append(r2_score(df["Close"][-predictiondays:],pred))
    
results = pd.DataFrame()
results["mae"] = mae
results["mse"] = mse
results["rmse"] = rmse
results["r2"] = rsquare
results.index = prednames

results.to_csv(r"C:\\Users\\taylan.polat\\Desktop\\PredictionResults\\modelresults.csv",index=True)

predictions_results = pd.DataFrame()
predictions_results["DTree"] = tree_predictions
predictions_results["LReg"] = lr_predictions
predictions_results["Xgboost"] = xgb_predictions
predictions_results["RandomForest"] = rfr_predictions
predictions_results["GBR"] = gbr_predictions
predictions_results["Bagging"] = bag_predictions
predictions_results["Adaboost"] = adaboost_predictions
predictions_results["Actual"] = df["Close"][-predictiondays:]
predictions_results.index = df.index[-predictiondays:]

predictions_results.to_csv(r"C:\\Users\\taylan.polat\\Desktop\\PredictionResults\\modelpredictionresults.csv",index=True)