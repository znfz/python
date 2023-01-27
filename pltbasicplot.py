import pandas as pd
lists=["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
        "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
        "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
        "peak-rpm","city-mpg","highway-mpg","price"]

df=pd.read_csv('data.data',header=None)
df.columns=lists

#1. print first 3 rows
#df.head(3)

#2. save dataset
#df.to_excel('res.xlsx',index=False)

#3. find out types
#df.dtypes

#4.statistics for two columns
#df[['length','width']].describe()

#5. replace ?
import numpy as np
df.replace('?',np.nan, inplace=True)

#6. convert stroke & horsepower to float
df[['stroke','horsepower']]=df[['stroke','horsepower']].astype(float)

#7. calculate mean of horsepower/replace missing values
hp_ave=df['horsepower'].mean()
df['horsepower'].replace(np.nan,hp_ave)

#8 calculate mean of peak-rpm/replace missing values
df['peak-rpm']=df['peak-rpm'].astype(float)
pr_ave=df['peak-rpm'].mean(axis=0)
df['peak-rpm'].replace(np.nan,pr_ave)

df[['bore','stroke','price','peak-rpm']]=df[['bore','stroke','price','peak-rpm']].astype(float)
df['normalized-losses']=df['normalized-losses'].astype(float)

#9 convert mpg to L/100km w/ new column
df['city-L/100km']=235.215/df['city-mpg']
df['highway-L/100km']=235.215/df['highway-mpg']

#10. Normalize length, width, height
df['length']=df['length']/df['length'].max()
df['width']=df['width']/df['width'].max()
df['height']=df['height']/df['height'].max()

#11. 

df['fuel-type']

dummy=pd.get_dummies(df['fuel-type'])
dummy.rename({'fuel-type':'fuel-dummy'})

res=pd.concat([df,dummy])
res.columns
