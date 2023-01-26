import pandas as pd
headers=["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
        "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
        "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
        "peak-rpm","city-mpg","highway-mpg","price"]

df=pd.read_csv('data.data',header=None)
df.columns=headers

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
res=df.replace('?',np.nan)

#6. convert stroke & horsepower to float
res[['stroke','horsepower']]=res[['stroke','horsepower']].astype(float)

#7. calculate mean of horsepower/replace missing values
hp_ave=res['horsepower'].mean()
res['horsepower'].replace(np.nan,hp_ave)

#8 calculate mean of peak-rpm/replace missing values
res['peak-rpm']=res['peak-rpm'].astype(float)
pr_ave=res['peak-rpm'].mean(axis=0)
res['peak-rpm'].replace(np.nan,pr_ave)

res.dtypes
res[['bore','stroke','price','peak-rpm']]=res[['bore','stroke','price','peak-rpm']].astype(float)
res['normalized-losses']=res[['normalized-losses']].astype(int)
