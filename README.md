# Bike Sharing Prediction

Created on Fri Jun  4 15:47:11 2021

## Project details
   * @author: Mae Tafoya
   * Analysis type: Multiple linear regression

## Libraries to be used in this project

```import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
```


## Data provided by:
Hadi Fanaee-T

Laboratory of Artificial Intelligence and Decision Support (LIAAD), University of Porto

INESC Porto, Campus da FEUP

Rua Dr. Roberto Frias, 378

4200 - 465 Porto, Portugal
    
*Original Source: http://capitalbikeshare.com/system-data*

**Original Dataset shape: (173879, 17);
Analysis dataset shape: (17376, 47)**

## Defining the Problem

A bike sharing company needs to  to meet the bike rental demand on an hourly basis. We need to make predictions on when they need
to stock more bikes during a certain hour at a certain location.

## Exploring the data / Selecting our features to use in analysis

Remove unecessary columns
* Index - just a serial number, does not add value
* Date - bc each entry is '01-01-2011', no help to our analysis
* Casual and Registered - Dependent variable 'demand' is the sum of these two. No need for them.


This is a clean dataset, no null values found.

## Plot all features
To get an overall look at the data we will plot a histogram for each feature.

![continuous variables](https://github.com/mae-tafoya/bike-sharing/blob/master/Visualizations/histogram-ALL-Features.png)

* We can see that the predicted variable "demand" is NOT normally distributed.

Features that are fairly normally distributed: 
* atemp, hour, humidity, temp, and windspeed. (continuous/numerical variables)

Categorical columns:
* Season
* Year
* Month
* Hour
* Holiday
* Weekday
* Working Day
* Weather

## Plot Independent vs Dependent

**Visualize NUMERICAL features vs. predicted variable with scatter plot**

![numerical vs. demand](https://github.com/mae-tafoya/bike-sharing/blob/master/Visualizations/NumericVSDemand.png)

* Temp vs. Demand AND aTemp vs. Demand look identical
  * indicating possible high correlation between temp/atemp (if true, drop one of the two)
* Humidity vs Demand is very scattered - indicating no correlation to dependent variable (drop)
* Windspeed vs Demand shows pattern that indicates some correlation to dependent variable (drop)

**Visualize CATEGORICAL features vs. predicted variable**

![categorical vs. demand](https://github.com/mae-tafoya/bike-sharing/blob/master/Visualizations/CategoricalVSDemand.png)

* weekday, year, and working day will not help much in analysis as there is little fluctuation (drop)
* There is a lot of variation in Hours vs. Demand bar graph, so we will explore it in depth (timeseries data)

## Check for multicollinearity among the numerical features

Check also for any low corellation between demand and any other independent feature.
* Use correlation coefficient
* plot correlation matrix

![correlation matrix](https://github.com/mae-tafoya/bike-sharing/blob/master/Visualizations/Correlation.PNG)

* As per the matrix, it is confirmed that atemp/temp are highly correlated (remove atemp moving forward)
* There is little correlation in Windspeed vs. Demand (drop windspeed)

## Check for Autocorrelation with *.acorr*

First, we'll create a new df with the dependent variable and convert to float type. We'll check up to 12 lags, since the data is tracked 24 hours. 

![autocorrelation plot](https://github.com/mae-tafoya/bike-sharing/blob/master/Visualizations/Autocorrelation.png)

Looking at the acorr plot, we see high autocorrelation up to 5 lags. Anything after 5 lags, goes below .5 and the behavior becomes abnormal. With this
we will shift the data up to 3 lags. (Create new df for each lag and drop instances with Null values.)

Next, concatenate the lag dataframes (t_1, t_2, t_3) to our full dataset bikes_prep. We will use these 3 additional independent features (new columns) to predict Demand.

## Normalize Dependent Feature
We saw previously that Demand was not normally distributed. We can use log to bring the distribution closer to a bell curve.

![before log transformation](https://github.com/mae-tafoya/bike-sharing/blob/master/Visualizations/before-log-transformation.png)

![after log transformation](https://github.com/mae-tafoya/bike-sharing/blob/master/Visualizations/after-log-transformation.png)

## Create Dummy Variables for Categorical Features
This last feature engineering step brings the dataset shape to (17376, 47) - bikes_prep_lag

## Split the dataset to Train/Test
Since the demand feature is time dependent, we cannot randomly split the data. So we will have to 
split the data by using the first 70% of entries as training set and the remaining entries for testing.

## Train and fit our Regression Model

After fitting the model to our training data, we checked the R squared score to get a general idea of how well our regression line fits. 

**R2_train (training data) = 0.92**

**R2_test (test data) = 0.93**

With good output of R squared, we can go ahead and make our predictions and calculate the accuracy using RMSE and RMSLE.

**RMSE = 0.38**

**RMSLE = 0.356**

## Conclusion

After all our feature engineering our model is predicting with about 74% accuracy.

The features used in making these predictions are:
* Temp
* Humidity
* Lag units up to 3 for demand (t-1, t-2, t-3)
* Hot encoded variable columns for Season, Month, Hour, Weather 

We tried removing Humidity, Season, and Weather, but the accuracy went down slightly. 

**RMSE after removig more variables = 0.385**

**RMSLE after removing more variables was 0.36**
