import pandas as pd
df=pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/automobileEDA.csv')
#1. Find relationship between bore, stroke, compression-ratio & horsepower
df[['bore','stroke','compression-ratio','horsepower']].corr

#2. Create a scatterplot of engine-size vs price
import matplotlib.pyplot as plt
x=df['engine-size']
y=df['price']
plt.xlabel('Engine Size')
plt.ylabel('Price')
plt.scatter(x,y)

#3. What is the correlation between engine-size & price?
df[['engine-size','price']].corr()

#4. Is peak-rpm a a good predictor of price?
x=df['peak-rpm']
y=df['price']
plt.scatter(x,y)
plt.xlabel('Peak-RPM'); plt.ylabel('Price')
df[['peak-rpm','price']].corr()

#5. Find correlation between stroke & price

print(df[['stroke','price']].corr())
x=df['stroke']
y=df['price']
plt.scatter(x,y)
plt.xlabel('Stroke')
plt.ylabel('Price')

#6. Plot relationship between stroke vs. price using seaborn
import seaborn as sns
sns.regplot(x='stroke',y='price',data=df)

#7. Relationship between body-style vs price
sns.boxplot(x='body-style',y='price',data=df)
#'body-style' is not a good indicator of price

#8. Relationship between engine-location vs price
sns.boxplot(x='engine-location',y='price',data=df)
#Engine in the rear is more expensive

#9. Relationship between drive-wheels vs price
sns.boxplot(x='drive-wheels',y='price',data=df)
#rwd is most expensive, followed by fwd, followed by 4wd

#10. Convert the categorical variable drive-wheels to a count
res=df['drive-wheels'].value_counts().to_frame()
res

#11. Rename drive-wheels to Value-Counts
res.rename(columns={'drive-wheels':'Value-Counts'},inplace=True)
res.index.name='Drive-Wheels'
res

#12. engine-location is a catergorical variable. Let's change it to value-count
res=df['engine-location'].value_counts().to_frame()
res.rename(columns={'engine-location':'Count'},inplace=True)
res.index.name='Location'
res

