import pandas as pd
df=pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/automobileEDA.csv')

#1. Create a linear regression model for engine-size vs price

x=df[['engine-size']]
y=df['price']
import matplotlib.pyplot as plt
plt.scatter(x,y)

from sklearn.linear_model import LinearRegression
lm1=LinearRegression()
lm1.fit(x,y)

print(lm.coef_)
print(lm.intercept_)

yhat=lm.predict(x)
