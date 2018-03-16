import numpy as np
import pandas as pd
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.model_selection import train_test_split

#chargement du fichier 
df_titanic = pd.read_csv('train.csv')
df_titanic.sample(10)

for col in df_titanic.loc[:,['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']].columns:
    sns.violinplot(x=col, y='Survived', data=df_titanic,  inner=None)
    plt.show()

#fonction préparation des données

#df_test = pd.read_csv('train.csv')

def prepare_data(df, verbose):
    df['Age'] = df['Age'].fillna(df['Age'].median()) #remplace les nan dans Age et Fare par 
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())# la valeur médiane
    
    def age_cat(age):
        if age<=3:
            return 0
        elif age<= 12 and age>3:
            return 1
        elif age<= 18 and age>12:
            return 2
        elif age<= 40 and age>18:
            return 3
        elif age<= 60 and age>40:
            return 4
        elif age>60:
            return 5

    def col_cat(data):
        data['cat_age'] = data['Age'].map(age_cat)
        return data
    
    df=col_cat(df)
    
    def has_cabin(cabin):
        if type(cabin) == type(1.3):
            return 0
        return 1
 
    df['Has_Cabin'] = df['Cabin'].map(has_cabin)
    if verbose:
        sns.violinplot(x='Has_Cabin', y='Survived', data=df,  inner=None)
        plt.show()
        df = df.loc[:,['Survived', 'Pclass', 'Age', 'Sex', 'SibSp', 'Parch', 'Fare', 'Has_Cabin', 'Embarked', 'cat_age']]
    else:
        df = df.loc[:,['Pclass', 'Age', 'Sex', 'SibSp', 'Parch', 'Fare', 'Has_Cabin', 'Embarked', 'cat_age']]
    
    df = pd.get_dummies(df)
    df['First_Class_Women'] = np.where((df['Sex_female'] == 1) & (df['Pclass'] == 1), 1, 0)
    
    if verbose:
        sns.violinplot(x='First_Class_Women', y='Survived', data=df,  inner=None)
        plt.show()
    
    return df
print(df_titanic.columns)
df_titanic = prepare_data(df_titanic, True)

list_col = ['Pclass', 'Age', 'Sex_female', 'Has_Cabin', 'Fare', 'First_Class_Women', 'Embarked_C', 'Embarked_Q', 'Embarked_S', 'Parch', 'SibSp']
X = df_titanic[list_col]
Y = df_titanic['Survived']


X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.2)

if False:
    for c in np.arange(0.1, 10, 0.1):
        #linear logistic regression
        logi_reg = lm.LogisticRegression(C=c)
        logi_reg.fit(X_train, np.ravel(Y_train))
        print('linear:', c, ': ' , logi_reg.score(X_valid, np.ravel(Y_valid)))
else:
    c = 0.036
    logi_reg = lm.LogisticRegression(C=c)
    logi_reg.fit(X_train, np.ravel(Y_train))    
    print('linear: ', logi_reg.score(X_valid, np.ravel(Y_valid)))
#tree
clf = tree.DecisionTreeClassifier()
clf.fit(X_train, Y_train)
print('tree: ', clf.score(X_valid, Y_valid))

df_test = pd.read_csv('test.csv')
passenger_id_tmp = df_test['PassengerId']

df_test = prepare_data(df_test, False)
df_test

X_test = df_test.loc[:, list_col]
result = logi_reg.predict(X_test)
result

data = pd.DataFrame({'PassengerId':passenger_id_tmp, 'Survived': result})
data.set_index('PassengerId').to_csv('my_test.csv')