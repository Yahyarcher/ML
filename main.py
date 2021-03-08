import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import seaborn as sns
from math import *
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import AdaBoostClassifier 
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import VotingClassifier

#infos sur la prediction 
def pred(model_object,predictors,compare):
    """1.model_object = modele
       2.predictors = données à predire
       3.compare = y_train"""
    predicted = model_object.predict(predictors)
    # Determiner les faux positifs et les vrais positifs 
    cm = pd.crosstab(compare,predicted)
    TN = cm.iloc[0,0]
    FN = cm.iloc[1,0]
    TP = cm.iloc[1,1]
    FP = cm.iloc[0,1]
    print("<<------- Prediction ------->> ")
    print(cm)
    print()
    ##verifier la precision
    print('<<------- Classification  ------->>')
    print('Précision : ', round(((TP+TN)*100)/(TP+TN+FP+FN),2))
    print()
   
    print(classification_report(compare,predicted))
 


#---------------------------main--------------------------------------#

#Extraction des données
f = open('mars_train.csv')
Volcans = pd.read_csv(f)
file = open('mars_unknown.csv')
pre =pd.read_csv(file)

Xe = pre
X = Volcans.drop(["1"], axis=1)
y = Volcans["1"]
#ACP  
pca = PCA(n_components=50)
Xp = pca.fit_transform(X)    
#unknown    
Xe = pca.fit_transform(Xe)
#train data
X_train, X_test, y_train, y_test = train_test_split(Xp,y,test_size=0.2,random_state=0)
#format standard    
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
#unknown
Xe = scaler.fit_transform(Xe)

bags = BaggingClassifier(SVC(), random_state=10, n_jobs=6).fit(X_train, y_train)
ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200)
#------------------best score------------------------#
vote = VotingClassifier(estimators=[('bagging(svc)', bags), ('ada(tree)', ada)], voting='soft').fit(X_train, y_train)
pred(vote,X_test,y_test)
#unknown
ye = vote.predict(Xe)
print(ye)
#Generation de Y final
df =  pd.DataFrame(ye)
df.to_csv('prediction.csv', sep=',', header=None, index=None)