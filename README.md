# Stock Prediction

In this study, a stock prediction model study was carried out by specifying the Stock name and date ranges on the site where the data is to be downloaded as Input.
If the number of days to be predicted is given as input to the predictiondays variable, it makes the number of days available as test observations.
The generalgraph.png file prepared by the "Py" file shows the distribution of the Close prices for the used Stock between the specified date.
The generalgraph2.png file prepared by the Py file shows the distribution of SMA, STD, Up, Down prices between the specified date for the Stock used.
Py file predicts Close prices until the specified forecast day using models such as LGBMRegressor, DecisionTreeRegressor, LinearRegression, XGBRegressor, 
RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor. 

* Prediction graphics produced for these models produce output as "prediction_charts_of_ [Modelname].png". 
* The location where the files will be produced can be updated by changing the fields specified in the Py file. 
* MSE, MAE, RMSE, R2 values produced by the models are created as "modelresults.csv" file.
* In addition, we can see the Prediction values produced by the models in the "modelpredictionresults.csv" file.

[linkedin]: https://www.linkedin.com/in/taylan-polat/

You can download requirements via "pip install -r requirements"