#1. Develop a linear model between 'peak-rpm' vs. 'price'
import pandas as pd
df=pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/automobileEDA.csv')
x=df[['peak-rpm']]
x1=df['peak-rpm']
y=df['price']

from sklearn.linear_model import LinearRegression
lm=LinearRegression()
lm.fit(x,y)
yhat=lm.predict(x)
import matplotlib.pyplot as plt
plt.plot(x1,yhat,color='red')
plt.scatter(x1,y)
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
print('Gradients:',lm.coef_)
print('Intercept:',lm.intercept_)
print('R2:',lm.score(x,y))
print('MSE:',mean_squared_error(y,yhat))

#2. Develop a polynomial model between 'peak-rpm' vs. 'price' 
import pandas as pd
df=pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/automobileEDA.csv')
x=df[['peak-rpm']]
x1=df['peak-rpm']
y=df['price']

from sklearn.preprocessing import PolynomialFeatures
pm=PolynomialFeatures(degree=2)
x_poly=pm.fit_transform(x)
from sklearn.linear_model import LinearRegression
lm=LinearRegression()
lm.fit(x_poly,y)

yhat=lm.predict(x_poly)

import matplotlib.pyplot as plt
plt.plot(x1,yhat,color='red')
plt.scatter(x1,y,)

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
print('R2',r2_score(y,yhat))
print('MSE:',mean_squared_error(y,yhat))
print('Gradient:',lm.coef_)
print('Intercept:',lm.intercept_)

#3. Develop a Multiple Linear Regression for price
import pandas as pd
df=pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/automobileEDA.csv')
z=df[['horsepower','curb-weight','engine-size','highway-mpg']]
y=df['price']

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
lm=LinearRegression()
lm.fit(z,y)

yhat=lm.predict(z)
print('Gradient:',lm.coef_)
print('Intercept',lm.intercept_)
print('R2:',r2_score(y,yhat))
print('MSE:',mean_squared_error(y,yhat))

import matplotlib.pyplot as plt
plt.plot(y)
plt.plot(yhat)
