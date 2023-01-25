#file name is data.csv
#assume file has no headers and no columns
import pandas as pd

df=pd.read_csv('data.csv', Header=None)

#create a list as headers. csv file has 4 columns
headers=['a','b','c','d']
df.columns=headers

#sort df by b
df.sort_values['b']

#print first 3 and last 3 rows
df.head(3)
df.tail(3)

               
