from flask import Flask, render_template, request
import numpy as np
import pickle
import pandas
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, label_binarize
import os

# Machine learning

from sklearn.model_selection import train_test_split
from sklearn import model_selection, tree, preprocessing, metrics, linear_model
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier

filename = 'train.csv'


model = pickle.load(open('titanic.pkl', 'rb'))


app = Flask(__name__)

@app.route("/")
def hello_world():
     return render_template('index.html')
@app.route("/form")
def formpage():
     return render_template('form.html')
@app.route("/predict",methods=['GET', 'POST'])#methods attribute a must for the form to work
def predict1():
    
    
    if request.form['b']==1:
     data2=1
     data12=0
     data13=0
    elif request.form['b']==2:
     data2=0
     data12=1
     data13=0
    else: 
     data2=0
     data12=0
     data13=1
    data2 = request.form['b']
    data12 = request.form['b']
    data13 = request.form['b']
    if request.form['d']==0:
         data3=1
         data4=0
    else:
         data4=1
         data3=0
    data4 = request.form['d']
    data3 = request.form['d']
    data5 = request.form['f']
    data6 = request.form['g']
    data8 = request.form['i']
    data9 = request.form['j']
    data10= request.form['k']
    data11= request.form['l']
    
    
    arr = np.array([[data5,data6,data8,data9,data10,data11,data4,data3,data2,data12,data13]])
   
                  
    pred = model.predict(arr)
    return render_template('predict.html',data=pred)
     

if __name__=='__main__':
 app.run( debug=True)