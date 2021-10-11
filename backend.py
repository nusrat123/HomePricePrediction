from flask import Flask, redirect, url_for, request, render_template
import flask
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

plt.scatter(df['area'],df['price'],marker='+',color='red')
plt.xlabel('Area in square ft')
plt.ylabel('Price in taka')
plt.title('Home Price Graph')


app = Flask(__name__)



@app.route('/',defaults = {'name':" "})
@app.route('/<name>')
def mainpage(name):
    return render_template("index.html",name = name)


@app.route('/about')
def aboutpage():
    return render_template("about.html")

@app.route('/submit',methods = ['GET'])
def sub():
    print(request.method)
    if request.method == 'GET':
        a=request.args['area']
        print(1)
        ##print(a)

        plt.plot(df.area, reg.predict(df[['area']]))
        ##plt.show()
        ##print(reg.predict([[a]]))
        b=reg.predict([[a]])
        print(b)

        return redirect(url_for('mainpage', name= b))
    else:
        return redirect(url_for('mainpage'))


if __name__ == '__main__':
    app.run(debug = True)
