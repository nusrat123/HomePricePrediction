import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('home_price.csv')

##print(df)

x = df[['area']]
y = df['price']

##print(y)


plt.scatter(df['area'],df['price'],marker='+',color='red')
plt.xlabel('Area in square ft')
plt.ylabel('Price in taka')
plt.title('Home Price Graph')
##plt.show()

##plt.plot(x,y)
##plt.show()

xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size=.30, random_state=1)
##print(xtrain)

reg = LinearRegression()
reg.fit(xtrain,ytrain)

##print(reg.predict(xtest))

plt.scatter(df['area'],df['price'],marker='+',color='red')
plt.xlabel('Area in square ft')
plt.ylabel('Price in taka')
plt.title('Home Price Graph')

plt.plot(df.area, reg.predict(df[['area']]))
plt.show()

print(reg.predict([[3500]]))

