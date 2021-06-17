"""
Created on Fri Jun  4 15:47:11 2021

Data provided by:
    Hadi Fanaee-T

    Laboratory of Artificial Intelligence and Decision Support (LIAAD), University of Porto
    INESC Porto, Campus da FEUP
    Rua Dr. Roberto Frias, 378
    4200 - 465 Porto, Portugal
    
    Original Source: http://capitalbikeshare.com/system-data

Project details
    @author: Mae Tafoya
    Analysis type: Multiple linear regression
    
"""
#---------------
# import libraries
#---------------
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math


#---------------------------------------------------------------
# Read the data
#---------------------------------------------------------------
bikes = pd.read_csv('hour.csv')

#---------------------------------------------------------------
# Explore the data / Feature Selection
#---------------------------------------------------------------

"""
Notes:
We do not neet the index copy, date columns.
We will also remove the casual and registered columns as the demand is the sum of the two.
Copy and remove the columns - use bikes_prep in analysis
"""
bikes_prep = bikes.copy()
bikes_prep = bikes_prep.drop(['index', 'date', 'casual', 'registered'], axis=1)

# check for missing values
bikes_prep.isnull().sum()
bikes_prep.hist(rwidth=0.9)
plt.tight_layout()

"""
Notes:
After plotting the histogram for each feature, we can see that the predicted variable "demand" is NOT normally distributed.
Features that are fairly normally distributed: atemp, hour, humidity, temp, and windspeed.
Looking at the df, we have categorical columns up to weather. Temp, atem, humidity and windspeed are numerical (continuous variables).
"""

#-------------------------------------------------------------
# Begin Visualize NUMERICAL features vs. predicted variable
#-------------------------------------------------------------

#Temp vs Demand
plt.subplot(2, 2, 1)
plt.title('Temperature Vs Demand')
plt.scatter(bikes_prep['temp'], bikes_prep['demand'], s=2, c='g') #plt.scatter(x, y, size, color)

#atemp vs Demand
plt.subplot(2, 2, 2)
plt.title('aTemp Vs Demand')
plt.scatter(bikes_prep['atemp'], bikes_prep['demand'], s=2, c='b')

#Humidity vs Demand
plt.subplot(2, 2, 3)
plt.title('Humidity Vs Demand')
plt.scatter(bikes_prep['humidity'], bikes_prep['demand'], s=2, c='m')

#Windspeed vs Demand
plt.subplot(2, 2, 4)
plt.title('Windspeed Vs Demand')
plt.scatter(bikes_prep['windspeed'], bikes_prep['demand'], s=2, c='c')

plt.tight_layout()

"""
Looking at the four scatter plots, we can see that temp vs demand and atemp vs demand look identical - indicating possible high correlation between temp/atemp
and humidity vs demand is very scattered - indicating no correlation to demand
and windspeed vs demand shows pattern that indicates some correlation to demand
"""
#-------------------------------------------------------------
# End NUM vs Demand plots
#-------------------------------------------------------------


#-------------------------------------------------------------
# Begin Visualize CATEGORICAL features vs. predicted variable
#-------------------------------------------------------------

# Season vs Demand - since season is categorized 1-4, we will:
# 1. create a 3x3 subplot
plt.subplot(3,3,1)
plt.title('Average Demand \n per Season')

# 2. create a list of unique season's values
cat_list = bikes_prep['season'].unique() # x axis for bar chart

# 3. get the avg (aka mean) demand per season using groupby
cat_average = bikes_prep.groupby('season').mean()['demand']

#4. plot in a bar chart
colors = ['g', 'r', 'm', 'b']
plt.bar(cat_list, cat_average, color=colors)

# Year vs Demand - repeat steps 1-4 from above
plt.subplot(3,3,2)
plt.title('Average Demand \n per Year')
cat_list = bikes_prep['year'].unique()
cat_average = bikes_prep.groupby('year').mean()['demand']
plt.bar(cat_list, cat_average, color=colors)

# Month vs Demand
plt.subplot(3,3,3)
plt.title('Average Demand \n per Month')
cat_list = bikes_prep['month'].unique()
cat_average = bikes_prep.groupby('month').mean()['demand']
plt.bar(cat_list, cat_average, color=colors)

# Hour vs Demand
plt.subplot(3,3,4)
plt.title('Average Demand \n per Hour')
cat_list = bikes_prep['hour'].unique()
cat_average = bikes_prep.groupby('hour').mean()['demand']
plt.bar(cat_list, cat_average, color=colors)

# Holiday vs Demand
plt.subplot(3,3,5)
plt.title('Average Demand \n per Holiday')
cat_list = bikes_prep['holiday'].unique()
cat_average = bikes_prep.groupby('holiday').mean()['demand']
plt.bar(cat_list, cat_average, color=colors)

# Weekday vs Demand
plt.subplot(3,3,6)
plt.title('Average Demand \n per Weekday')
cat_list = bikes_prep['weekday'].unique()
cat_average = bikes_prep.groupby('weekday').mean()['demand']
plt.bar(cat_list, cat_average, color=colors)

# Working Day vs Demand
plt.subplot(3,3,7)
plt.title('Average Demand \n per Working Day')
cat_list = bikes_prep['workingday'].unique()
cat_average = bikes_prep.groupby('workingday').mean()['demand']
plt.bar(cat_list, cat_average, color=colors)

# Weather vs Demand 
plt.subplot(3,3,8)
plt.title('Average Demand \n per Weather')
cat_list = bikes_prep['weather'].unique()
cat_average = bikes_prep.groupby('weather').mean()['demand']
plt.bar(cat_list, cat_average, color=colors)

plt.tight_layout(pad=.5)

"""
Notes: After looking at all the barcharts, we can see that weekday, year, and working day will not help much in analysis
so we can drop these features moving forward.

There is a lot of variation in the hours vs demand bar graph, so we will explore it in depth by itself as it is a timeseries data.
"""
#-------------------------------------------------------------
# End CATEGORICAL vs demand plots
#-------------------------------------------------------------

#-------------------------------------------------------------
# Check for multicollinearity ---> high correlation between two independent features (one of lin reg condition is there should be no multicollinearity )
# Check also for any low corellation between demand and any other independent feature
correlation = bikes_prep[['temp', 'atemp', 'humidity', 'windspeed', 'demand']].corr()

"""
Notes: We confirmed that atemp/temp are highly correlated so we will remove atemp moving forward.
Also, there is little correlation between demand and windspeed so we will remove this feature.
"""
bikes_prep = bikes_prep.drop(['weekday', 'year', 'workingday', 'atemp', 'windspeed'], axis=1)

#-------------------------------------------------------------
# Check autocorrelation (demand)
df1 = pd.to_numeric(bikes_prep['demand'], downcast = 'float')

plt.acorr(df1, maxlags=12)

"""
We can see autocorrelation in the dependent variable data 
Meaning the dependent feature entries (in each row) are dependent on each previous entry.
Later, we will shift the data to compare entries to one period back.
"""

#-------------------------------------------------------------
# Transform the 'demand' feature to the log distribution

df1 = bikes_prep['demand']
df2 = np.log(df1)

plt.figure()
plt.title('Before Log Transformation')
df1.hist(rwidth = 0.9, bins = 20)

plt.figure()
plt.title('After Log Transformation')
df2.hist(rwidth = 0.9, bins = 20)

"""
After using log function on the demand feature, we can see in the comparison that the distribution is 
still a bit skewed, but closer to a bell curve.
"""

bikes_prep['demand'] = np.log(bikes_prep['demand'])

#-------------------------------------------------------------
# Solve the autocorrelation issue by using shift to create lag

t_1 = bikes_prep['demand'].shift(+1).to_frame()
t_1.columns = ['t-1']

t_2 = bikes_prep['demand'].shift(+2).to_frame()
t_2.columns = ['t-2']

t_3 = bikes_prep['demand'].shift(+3).to_frame()
t_3.columns = ['t-3']

bikes_prep_lag = pd.concat([bikes_prep, t_1, t_2, t_3], axis=1)
bikes_prep_lag = bikes_prep_lag.dropna()

#-------------------------------------------------------------
# Create dummy variables for all categorical features
# Drop first category to avoid dummy trap

bikes_prep_lag['season'] = bikes_prep_lag['season'].astype('category')
bikes_prep_lag['holiday'] = bikes_prep_lag['holiday'].astype('category')
bikes_prep_lag['weather'] = bikes_prep_lag['weather'].astype('category')
bikes_prep_lag['month'] = bikes_prep_lag['month'].astype('category')
bikes_prep_lag['hour'] = bikes_prep_lag['hour'].astype('category')

bikes_prep_lag = pd.get_dummies(bikes_prep_lag, drop_first=True)

#---------------------------------------------------------------
# Process the data
#---------------------------------------------------------------

# Split into train/test sets
"""
Note:
Since the demand feature is time dependent, we cannot randomly split the data. So we will have to 
split into two groups so we do not lose the autocorrelation which we are using as a predictor.
"""
Y = bikes_prep_lag[['demand']]
X = bikes_prep_lag.drop(['demand'], axis=1)

# Create the size for 70% of the data (70% of all rows in bikes_prep_lag = 12,163.2)
tr_size = 0.7 * len(X) # with remainder, the result is a float
tr_size = int(tr_size) # change from float to int

X_train = X.values[0:tr_size] # take all rows from 0 up to 12,163
X_test = X.values[tr_size : len(X)] # all rows after 12,163

Y_train = Y.values[0:tr_size]
Y_test = Y.values[tr_size : len(Y)]

# Train the regression model
from sklearn.linear_model import LinearRegression

# create an instance of the model
standard_reg = LinearRegression()

#fit the model
standard_reg.fit(X_train, Y_train)

# get the Rsquared value
r2_train = standard_reg.score(X_train, Y_train)
r2_test = standard_reg.score(X_test, Y_test)

"""
Note:
Why is R-sq score important in regression? 
A higher score (closer to 1) means less error or the regression line is close to the points.
It also means that the variation in Y (dependent/predicted variable) is better explained
with the variations in X (independant/features variables).
"""

# create Y predictions
Y_predict = standard_reg.predict(X_test)

# Calculate the RMSE
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(Y_test, Y_predict))

# Calculate the RMSLE
Y_test_e = []
Y_predict_e = []

for i in range(0, len(Y_test)):
    Y_test_e.append(math.exp(Y_test[i]))
    Y_predict_e.append(math.exp(Y_predict[i]))
    
# sum of the logs and squares
log_sq_sum = 0.0    
    
for i in range(0, len(Y_test_e)):
    log_a = math.log(Y_test_e[i]+1)
    log_p = math.log(Y_predict_e[i]+1)
    log_diff = (log_a - log_p)**2
    log_sq_sum = log_sq_sum + log_diff
    
rmsle = math.sqrt(log_sq_sum/len(Y_test))

print("RMSLE score:")
print(rmsle)
    

#---------------------------------------------------------------
# Try dropping features to improve accuracy of model 
#---------------------------------------------------------------

bikes_prep_lag2 = bikes_prep_lag.drop(['humidity','season_2','season_3','season_4', 'holiday_1', 'weather_2', 'weather_3','weather_4'], axis=1)

Y_2 = bikes_prep_lag2[['demand']]
X_2 = bikes_prep_lag2.drop(['demand'], axis=1)

tr_size2 = 0.7 * len(X_2) # with remainder, the result is a float
tr_size2 = int(tr_size2) # change from float to int

X_train2 = X_2.values[0:tr_size2] # take all rows from 0 up to 12,163
X_test2 = X_2.values[tr_size2 : len(X_2)] # all rows after 12,163

Y_train2 = Y_2.values[0:tr_size2]
Y_test2 = Y_2.values[tr_size2 : len(Y_2)]


standard_reg2 = LinearRegression()

standard_reg2.fit(X_train2, Y_train2)

r2_train2 = standard_reg2.score(X_train2, Y_train2)
r2_test2 = standard_reg2.score(X_test2, Y_test2)

Y_predict2 = standard_reg2.predict(X_test2)

rmse_2 = math.sqrt(mean_squared_error(Y_test2, Y_predict2))

Y_test_e_2 = []
Y_predict_e_2 = []

for i in range(0, len(Y_test2)):
    Y_test_e_2.append(math.exp(Y_test2[i]))
    Y_predict_e_2.append(math.exp(Y_predict2[i]))
    
log_sq_sum_2 = 0.0    
    
for i in range(0, len(Y_test_e_2)):
    log_a2 = math.log(Y_test_e_2[i]+1)
    log_p2 = math.log(Y_predict_e_2[i]+1)
    log_diff2 = (log_a2 - log_p2)**2
    log_sq_sum_2 = log_sq_sum_2 + log_diff2
    
rmsle_2 = math.sqrt(log_sq_sum_2/len(Y_test2))

print("RMSLE_2 score:")
print(rmsle_2)
