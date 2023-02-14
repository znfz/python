import pandas as pd
df=pd.read_csv('https://raw.githubusercontent.com/fivethirtyeight/data/master/college-majors/women-stem.csv')
df.head()
#1. Categorize major category
df.value_counts('Major_category').astype('int').to_frame()
#2. Plot histogram 
import matplotlib.pyplot as plt
x=df['Major_category']
plt.hist(x)
plt.title('Major Count')
plt.ylabel('Count')
