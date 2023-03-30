#1. Create a MLR model 

import pandas as pd
import numpy as np
df=pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/automobileEDA.csv')
x=df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg','normalized-losses','symboling']]
y=df['price']

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
lm=LinearRegression()
lm.fit(x,y)
yhat=np.round(lm.predict(x),1)
print('Predicted values are',yhat[:3])
print('Actual values are',y[:3].tolist())
print('R2:',np.round(lm.score(x,y),4))
print('The gradients are',np.round(lm.coef_,1))
print('The intercept is',np.round(lm.intercept_,1))
import matplotlib.pyplot as plt
x_range=range(len(x))
plt.plot(x_range,y,color='red')
plt.scatter(x_range,yhat)
plt.xlabel('#')
plt.ylabel('Price')

#2.Do a cross_val 3 times
from sklearn.model_selection import cross_val_score
cvs=np.round(cross_val_score(lm,x,y,cv=3),3)
print('R2:',cvs)
from sklearn.model_selection import cross_val_predict
yhat=np.round(cross_val_predict(lm,x,y,cv=3),3)
print('Predicted',yhat[:3])
print('Actual',(y[:3].tolist()))

#3. Create a MLR with 40% test sample

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,random_state=0)
from sklearn.linear_model import LinearRegression
lm=LinearRegression()
lm.fit(x_train,y_train)
yhat=np.round(lm.predict(x_test),1)

print('Predicted values are',yhat[:3])
print('Actual values are',y_test[:3].tolist())
print('R2:',np.round(lm.score(x_test,y_test),3))
print('The gradients are',np.round(lm.coef_,3))
print('The intercepts are',np.round(lm.intercept_,1))
import matplotlib.pyplot as plt
x_num=range(len(x_test))
plt.plot(x_num,y_test,color='red')
plt.scatter(x_num,y_test)
plt.xlabel('#')
plt.ylabel('Price')

#4. Create a PR with 40% test sample with degree=2

import pandas as pd
df=pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/automobileEDA.csv')
x=df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]
y=df['price']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,random_state=0)

from sklearn.preprocessing import PolynomialFeatures
pm=PolynomialFeatures(degree=2)
x_trainp=pm.fit_transform(x_train)
x_testp=pm.fit_transform(x_test)

from sklearn.linear_model import LinearRegression
lm=LinearRegression()
lm.fit(x_trainp,y_train)
yhat=np.round(lm.predict(x_testp),1)
R2=np.round(lm.score(x_testp,y_test),3)
print('Predicted values are',yhat[:3].tolist())
print('Actual values are',y_test[:3].tolist())
print('The gradients are',lm.coef_)
print('The intercept is',np.round((lm.intercept_),2))
print('R2:',R2)

x_num=range(len(x_test))
import matplotlib.pyplot as plt
plt.scatter(x_num,y_test)
plt.plot(x_num,yhat,color='red')
plt.xlabel('#')
plt.ylabel('Price')

#6. Create PM of degree=5

import pandas as pd
import numpy as np
df=pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/automobileEDA.csv')
x=df[['horsepower']]
y=df['price']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,random_state=0)

from sklearn.preprocessing import PolynomialFeatures
pm=PolynomialFeatures(degree=5)
x_trainp=pm.fit_transform(x_train)
x_testp=pm.fit_transform(x_test)

from sklearn.linear_model import LinearRegression
lm=LinearRegression()
lm.fit(x_trainp,y_train)
yhat=lm.predict(x_testp)

import matplotlib.pyplot as plt
plt.scatter(x_test,y_test)
plt.scatter(x_test,yhat)
plt.xlabel('Price')
plt.ylabel('Horsepower')
int=np.round(lm.intercept_,1)
print('Gradients:',lm.coef_)
print('Intercept',int)
R2=np.round(lm.score(x_testp,y_test),3)
print('R2:',R2)

#7. Develop a Polynomial model

import pandas as pd
import numpy as np
df=pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/automobileEDA.csv')
x=df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg','normalized-losses','symboling']]
y=df['price']

from sklearn.preprocessing import PolynomialFeatures
pm=PolynomialFeatures(degree=2)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,random_state=0)

x_trainp=pm.fit_transform(x_train)
x_testp=pm.fit_transform(x_test)

from sklearn.linear_model import Ridge

rm=Ridge(alpha=10000000000)
rm.fit(x_trainp,y_train)
yhat=np.round(rm.predict(x_testp),1)
#print('Predicted:',yhat[:3].tolist())
#print('Actual:',y_test[:3].tolist())

R2_te=rm.score(x_testp,y_test)
R2_tr=rm.score(x_trainp,y_train)
print('R2 for testing:',np.round(R2_te,3))
print('R2 for training:',np.round(R2_tr,3))

x_num=range(len(x_test))
import matplotlib.pyplot as plt
plt.scatter(x_num,y_test)
plt.plot(x_num,yhat,color='red')
plt.xlabel('#')
plt.ylabel('Price')

#All these models should be used in consideration when developing predictive models.
