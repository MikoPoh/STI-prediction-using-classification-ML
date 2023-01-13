
# Prediction of STI using classification machine learning model

This repository calculate commonly used technical indicators to be used as features for 
classification machine learning model. Model will then predict 5 day EMA period of STI as 
a proxy of daily closing price.

## Libraries used

To run this project, the following libraries are needed

`yfinance` `numpy` `pandas` `matplotlib` `seaborn` `sklearn` `xgboost` 


## Introduction
Notebook STI prediction for optimization was first to predict closing price directly but
unable to due to nature of data. Thus, prediction change to 5 day EMA as a proxy for closing 
price. 

Technical indicators used:
- Exponential moving average (EMA)
- Moving average convergence/divergence (MACD)
- Stochastic
- Force index

Different time period was calculated and compared before loading into models. Difference 
between current value and previous day value for each indicator and time period is also 
calculated. Top 5 features from random forest's feature importance used for models comparing. 

Classification models compared:
- Logistic regression
- Gaussian
- Decision Tree
- K nearest neighbour
- Random forest
- SVC
- XG classifier

Classification models are compared based on the following:
- Accuracy
- Precision
- Log loss
- AUC score

Trained models are then ensemble into 1 final model using stacking classifier and voting
classifier.

Notebook STI prediction of 5 day EMA fine tune hyperparameters of only XG classifier, 
random forest, svc and logistic regression. Models are ensemble into 1 final model using 
stacking classifier and voting classifier.

Final model will be used to be predict 2022 data and be used for trading simulation.
## Result
Model comparison results
![model comparison](https://github.com/MikoPoh/STI-prediction-using-classification-ML/blob/main/Chart/models%20comparison.png?raw=true)

Trading simulation of 2022 predictions
![backtest](https://github.com/MikoPoh/STI-prediction-using-classification-ML/blob/main/Chart/backtest%20result.png?raw=true)

## Conclusion
5 days period EMA is easier to predict compared to closing price. Hyperparameter tuning does
not give very significant improvement due to nature of dataset. Prediction can be furthur 
improve by combining with another ML model such as LSTM which treat data as time series.
