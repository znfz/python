import pandas as pd
df=pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/automobileEDA.csv')

#1. Create an equation for 'peak-rpm' vs. 'price'

x=df[['peak-rpm']]
y=df['price']
from sklearn.linear_model import LinearRegression
lm=LinearRegression()
lm.fit(x,y)
m=lm.coef_
y=lm.intercept_
print(m,y)

yhat=lm.predict(x)
yhat[30]

import seaborn as sns
sns.regplot(x='peak-rpm',y='price',data=df)

df[['peak-rpm','price']].corr()

#2. Create a linear-regression model for 'highway-mpg' vs. price

x=df[['highway-mpg']]
y=df['price']
from sklearn.linear_model import LinearRegression
lm=LinearRegression()
lm.fit(x,y)
m=lm.coef_
y=lm.intercept_
print(m,y)

yhat=lm.predict(x)
yhat[30]

import seaborn as sns
sns.regplot(x='highway-mpg',y='price',data=df)

df[['highway-mpg','price']].corr()

#3. Create a multiple linear model using 'horsepower','curb-weight','engine-size','highway-mpg'

z=df[['horsepower','curb-weight','engine-size','highway-mpg']]
y=df['price']
from sklearn.linear_model import LinearRegression
lm=LinearRegression()
lm.fit(z,y)
m=lm.coef_
c=lm.intercept_
print(m,c)

#4. Find residual regression between 'highway-mpg' vs. 'price'

x=df[['highway-mpg']]
y=df['price']

import seaborn as sns
sns.residplot(x='highway-mpg',y='price',data=df)
