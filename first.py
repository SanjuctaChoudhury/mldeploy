import numpy as np
import pickle
import pandas 
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, label_binarize


from sklearn.model_selection import train_test_split
from sklearn import model_selection, tree, preprocessing, metrics, linear_model
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier

train = pandas.read_csv('train.csv')
test = pandas.read_csv('test.csv')

df_bin = pandas.DataFrame() 
df_con = pandas.DataFrame() 
df_bin['Survived'] = train['Survived']
df_con['Survived'] = train['Survived']
df_bin['Pclass'] = train['Pclass']
df_con['Pclass'] = train['Pclass']
df_bin['Sex'] = train['Sex']
df_bin['Sex'] = np.where(df_bin['Sex'] == 'female', 1, 0) # change sex to 0 for male and 1 for female
df_con['Sex'] = train['Sex']
df_bin['SibSp'] = train['SibSp']
df_con['SibSp'] = train['SibSp']
df_bin['Parch'] = train['Parch']
df_con['Parch'] = train['Parch']
df_con['Fare'] = train['Fare'] 
df_bin['Fare'] = pandas.cut(train['Fare'], bins=5) # discretised
# Add Embarked to sub dataframes
df_bin['Embarked'] = train['Embarked']
df_con['Embarked'] = train['Embarked']
df_con = df_con.dropna(subset=['Embarked'])
df_bin = df_bin.dropna(subset=['Embarked'])
df_embarked_one_hot = pandas.get_dummies(df_con['Embarked'], 
                                     prefix='embarked')

df_sex_one_hot = pandas.get_dummies(df_con['Sex'], 
                                prefix='sex')

df_plcass_one_hot = pandas.get_dummies(df_con['Pclass'], 
                                   prefix='pclass')
df_con_enc = pandas.concat([df_con, 
                        df_embarked_one_hot, 
                        df_sex_one_hot, 
                        df_plcass_one_hot], axis=1)

# Drop the original categorical columns (because now they've been one hot encoded)
df_con_enc = df_con_enc.drop(['Pclass', 'Sex', 'Embarked'], axis=1)
selected_df = df_con_enc
# Split the dataframe into data and labels
X_train = selected_df.drop('Survived', axis=1) # data
y_train = selected_df.Survived # labels

    # Decision Tree Classifier
dtclassifier=DecisionTreeClassifier()
dtclassifier.fit(X_train, y_train)
print(X_train.columns)
                                                            
                                                               
pickle.dump(dtclassifier, open('titanic.pkl', 'wb'))

