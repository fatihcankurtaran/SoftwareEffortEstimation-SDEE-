from sklearn.neighbors import KNeighborsRegressor
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import mglearn
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR
from sklearn.feature_selection import SelectFromModel



training1 = pd.read_csv("dataset1/trainingdata.csv",delimiter =",",index_col=False)
testing1 = pd.read_csv("dataset1/testingdata.csv",delimiter=",",index_col=False)
training2 = pd.read_csv("dataset2/trainingdata.csv",delimiter =",",index_col=False)
testing2 = pd.read_csv("dataset2/testingdata.csv",delimiter=",",index_col=False)
training3 = pd.read_csv("dataset3/trainingdata.csv",delimiter =",",index_col=False)
testing3 = pd.read_csv("dataset3/testingdata.csv",delimiter=",",index_col=False)
training4 = pd.read_csv("dataset4/trainingdata.csv",delimiter =",",index_col=False)
testing4 = pd.read_csv("dataset4/testingdata.csv",delimiter=",",index_col=False)
training5 = pd.read_csv("dataset5/trainingdata.csv",delimiter =",",index_col=False)
testing5 = pd.read_csv("dataset5/testingdata.csv",delimiter=",",index_col=False)

k,l = np.shape(training1)
p = pd.concat([training1,testing1])
my = pd.get_dummies(p,columns=['Development_platform','Language_Type','Resource_Level'])
training1 = my.iloc[:k]
testing1= my.iloc[k:]

k,l = np.shape(training2)
p = pd.concat([training2,testing2])
my = pd.get_dummies(p,columns=['Development_platform','Language_Type','Resource_Level'])
training2 = my.iloc[:k]
testing2= my.iloc[k:]

k,l = np.shape(training3)
p = pd.concat([training3,testing3])
my = pd.get_dummies(p,columns=['Development_platform','Language_Type','Resource_Level'])
training3 = my.iloc[:k]
testing3= my.iloc[k:]

k,l = np.shape(training4)
p = pd.concat([training4,testing4])
my = pd.get_dummies(p,columns=['Development_platform','Language_Type','Resource_Level'])
training4 = my.iloc[:k]
testing4= my.iloc[k:]

k,l = np.shape(training5)
p = pd.concat([training5,testing5])
my = pd.get_dummies(p,columns=['Development_platform','Language_Type','Resource_Level'])
training5 = my.iloc[:k]
testing5= my.iloc[k:]


#print(np.shape(t))
#print(np.shape(j))
#print(my.iloc[1])

#training1 = pd.get_dummies(training1,columns=['Development_platform','Language_Type','Resource_Level'])
#testing1 = pd.get_dummies(testing1)
#training2 = pd.get_dummies(training2)
#testing2 = pd.get_dummies(testing2)
#training3 = pd.get_dummies(training3)
#testing3 = pd.get_dummies(testing3)
#training4 = pd.get_dummies(training4)
#testing4 = pd.get_dummies(testing4)
#training5 = pd.get_dummies(training5)
#testing5 = pd.get_dummies(testing5)




X_train1 = np.concatenate((training1.iloc[:,0:1],training1.iloc[:,2:]),axis=1)
X_train2 = np.concatenate((training2.iloc[:,0:1],training2.iloc[:,2:]),axis=1)
X_train3 = np.concatenate((training3.iloc[:,0:1],training3.iloc[:,2:]),axis=1)
X_train4 = np.concatenate((training4.iloc[:,0:1],training4.iloc[:,2:]),axis=1)
X_train5 = np.concatenate((training5.iloc[:,0:1],training5.iloc[:,2:]),axis=1)
y_train1 =np.concatenate((testing1.iloc[:,0:1],testing1.iloc[:,2:]),axis=1)
y_train2 =np.concatenate((testing2.iloc[:,0:1],testing2.iloc[:,2:]),axis=1)
y_train3 =np.concatenate((testing3.iloc[:,0:1],testing3.iloc[:,2:]),axis=1)
y_train4 =np.concatenate((testing4.iloc[:,0:1],testing4.iloc[:,2:]),axis=1)
y_train5 =np.concatenate((testing5.iloc[:,0:1],testing5.iloc[:,2:]),axis=1)

X_test1 = training1.loc[:,'Effort']
X_test2=  training2.loc[:,'Effort']
X_test3 = training3.loc[:,'Effort']
X_test4 = training4.loc[:,'Effort']
X_test5 = training5.loc[:,'Effort']

y_test1 = testing1.loc[:,'Effort']
y_test2 = testing2.loc[:,'Effort']
y_test3 = testing3.loc[:,'Effort']
y_test4 = testing4.loc[:,'Effort']
y_test5= testing5.loc[:,'Effort']

t_train = []
t_test = []
X_train = []
X_test = []
y_train =[]
y_test = []

X_train.append(X_train1)
X_train.append(X_train2)
X_train.append(X_train3)
X_train.append(X_train4)
X_train.append(X_train5)

X_test.append(X_test1)
X_test.append(X_test2)
X_test.append(X_test3)
X_test.append(X_test4)
X_test.append(X_test5)

y_train.append(y_train1)
y_train.append(y_train2)
y_train.append(y_train3)
y_train.append(y_train4)
y_train.append(y_train5)

y_test.append(y_test1)
y_test.append(y_test2)
y_test.append(y_test3)
y_test.append(y_test4)
y_test.append(y_test5)
for j in range(0,5):
    t_train.append(np.concatenate([X_train[j],y_train[j]],axis=0))
    t_test.append(np.concatenate([X_test[j],y_test[j]],axis=0))
print("----------")
print(t_train[0])
print(np.shape(t_train[0]))

#print(X_train2)
knn2 = KNeighborsRegressor(n_neighbors=2)
knn1 = KNeighborsRegressor(n_neighbors=2)
knn2.fit(X_train[3],X_test[3])
print(knn2.score(X_train[3],X_test[3]))
predict = cross_val_predict(knn1,t_train[0],t_test[0],cv=10)
print("this is predicted cross vall")
print(predict)
print(mean_absolute_error(t_test[0],predict))


loc = LogisticRegression()
loc.fit(X_train[3],X_test[3])
print(loc.score(X_train[3],X_test[3]))
#print(loc.score(y_train[3],y_test[3]))
print(np.shape(y_train[3]))
print(np.shape(X_train[3]))

for i in range(5):
    print(np.shape(X_train[i]))
    print(np.shape(X_test[i]))
    print(np.shape(y_train[i]))
    print(np.shape(y_test[i]))
"""
# ------------------------------
data1 = open('dataset5/testingData.csv' ,'rt')
data1a = open('dataset5/trainingData.csv' ,'rt')

testdata1 = np.loadtxt(data1, delimiter=",")
trainingdata1 = np.loadtxt(data1a, delimiter=",")

X_train1 = trainingdata1[0:,0:4]
X_test1 = testdata1[0:,0:4]
y_train1 = trainingdata1[0:,4:]
y_test1 = testdata1[0:,4:]
print(X_train1.shape);print(X_test1.shape);print(y_train1.shape);print(y_test1.shape)
knn1 = KNeighborsRegressor(n_neighbors=2)
knn1.fit(X_train1,y_train1)
y_pred1 = knn1.predict(X_test1)

print("Train set score : {:.2f}".format(knn1.score(X_train1,y_train1)))
print("Test set score: {:.2f}".format(knn1.score(X_test1, y_test1)))



data2 = open('dataset2/testingData.csv' ,'rt')
data2a = open('dataset2/trainingData.csv' ,'rt')
testdata2 = np.loadtxt(data2, delimiter=",")
trainingdata2 = np.loadtxt(data2a, delimiter=",")
X_train2 = trainingdata2[0:,0:4]
X_test2 = testdata2[0:,0:4]
y_train2 = trainingdata2[0:,4:]
y_test2 = testdata2[0:,4:]
print(X_train2.shape);print(X_test2.shape);print(y_train2.shape);print(y_test2.shape)
knn2 = KNeighborsRegressor(n_neighbors=2)
knn2.fit(X_train2,y_train2)
y_pred1 = knn2.predict(X_test2)
print("Train set score dataset 2 : {:.2f}".format(knn2.score(X_train2,y_train2)))
print("Test set score dataset 2 : {:.2f}".format(knn2.score(X_test2, y_test2)))

#-------------------------------------------------------


"""

for k in range (1,8):
    for n in range(0,5):

        knn = KNeighborsRegressor(n_neighbors=k)
        knn2 = KNeighborsRegressor(n_neighbors=k)
        knn.fit(X_train[n], X_test[n])
        predict_test = knn.predict(y_train[n])
        predict = cross_val_predict(knn2, t_train[n], t_test[n], cv=10)
        print("KNN-Train MAE for "+k.__str__() +" neighbors " +(n+1).__str__() + "_set: {:.2f}".format(mean_absolute_error(y_test[n],predict_test)))
        print("KNN-Hole MAE for  "+k.__str__() +" neighbors " +(n+1).__str__() + "_set: {:.2f}".format(mean_absolute_error(t_test[n], predict)))
        #print("KNN-Train score for " + k.__str__() + " neighbors " + (n + 1).__str__() + "_set: {:.2f}".format(knn.score(X_train[n],X_test[n])))
        #print("KNN-Test score for  " + k.__str__() + " neighbors " + (n + 1).__str__() + "_set: {:.2f}".format(knn.score(y_train[n], y_test[n])))


for n in range(0,5):
    lr = LinearRegression()
    lr2 =LinearRegression()
    lr.fit(X_train[n], X_test[n])
    test_p = lr.predict(y_train[n])
    hole_p = cross_val_predict(lr2, t_train[n], t_test[n], cv=10)

    print("Linear Reg-Test MAE for " +(n+1).__str__()  + "_set: {:.2f}".format(mean_absolute_error(y_test[n], test_p)))
    print("Linear Reg-CV  MAE for " +(n+1).__str__() + "_set: {:.2f}".format(mean_absolute_error(t_test[n], hole_p)))
    print("Linear Reg-Test ME for " + (n + 1).__str__() + "_set: ",(np.mean([y_test[n]- test_p])))
    print("Linear Reg-CV  ME for " + (n + 1).__str__() + "_set: ",(np.mean([t_test[n]-hole_p])))
    #print("Linear Reg-Train score for " + (n + 1).__str__() + "_set: {:.2f}".format(lr.score(X_train[n], X_test[n])))
    #print("Linear Reg-Test  score for " + (n + 1).__str__() + "_set: {:.2f}".format(lr.score(y_train[n], y_test[n])))
for k in [0.001,1,10,100,1000,10000]:
    for n in range(0,5):
        lasso = Lasso(alpha = k,max_iter=1000000).fit(X_train[n],X_test[n])
        ridge = Ridge(alpha = k).fit(X_train[n],X_test[n])
        lasso_test_p = lasso.predict(y_train[n])
        lasso_hole_p = cross_val_predict(lasso,t_train[n],t_test[n],cv=10)
        ridge_test_p = ridge.predict(y_train[n])
        ridge_hole_p = cross_val_predict(ridge,t_train[n],t_test[n],cv=10)
        print("Ridge Test-MAE for alpha: "+k.__str__() +" and " + (n+1).__str__() + "_set: {:.2f}".format(mean_absolute_error(y_test[n], ridge_test_p)))
        print("Ridge CV-MAE for alpha:  "+k.__str__() +" and "  + (n+1).__str__() + "_set: {:.2f}".format(mean_absolute_error(t_test[n], ridge_hole_p)))
        print("Lasso Test-MAE for alpha: " + k.__str__() + " and " + (n+1).__str__() + "_set: {:.2f}".format(mean_absolute_error(y_test[n], lasso_test_p)))
        print("Lasso CV-MAE for alpha:  " + k.__str__() + " and " + (n+1).__str__() + "_set: {:.2f}".format(mean_absolute_error(t_test[n], lasso_hole_p)))
        #print("Ridge Regression Train set score for alpha:" + k.__str__() + " and " + n.__str__() + "_set: {:.2f}".format(ridge.score(X_train[n], X_test[n])))
        #print("Ridge Regression Test  set score for alpha:" + k.__str__() + " and " + n.__str__() + "_set: {:.2f}".format(  ridge.score(y_train[n], y_test[n])))
        #print("Lasso Regression Train set score for alpha:" + k.__str__() + " and " + n.__str__() + "_set: {:.2f}".format(lasso.score(X_train[n], X_test[n])))
        #print("Lasso Regression Test  set score for alpha:" + k.__str__() + " and " + n.__str__() + "_set: {:.2f}".format(lasso.score(y_train[n], y_test[n])))
        print("Ridge Test-ME for alpha: " + k.__str__() + " and " + (n + 1).__str__() + "_set: ",(np.mean([y_test[n]-ridge_test_p])))
        print("Ridge CV-ME for alpha:  " + k.__str__() + " and " + (n + 1).__str__() + "_set: ",(np.mean([t_test[n]-ridge_hole_p])))
        print("Lasso Test-ME for alpha: " + k.__str__() + " and " + (n + 1).__str__() + "_set: ",(np.mean([y_test[n]-lasso_test_p])))
        print("Lasso CV-ME for alpha:  " + k.__str__() + " and " + (n + 1).__str__() + "_set: ",(np.mean([t_test[n]-lasso_hole_p])))
for p in range (1, 6):
    for n in range(0,5):
        tree = DecisionTreeRegressor(max_depth=p)
        tree.fit(X_train[n],X_test[n])
        test_p= tree.predict(y_train[n])
        hole_p= cross_val_predict(tree,t_train[n],t_test[n],cv=10)
        print("DecisionTreeR-Test MAE depth: "+p.__str__()  + " and "+(n+1).__str__() + "_set: {:.2f}".format(mean_absolute_error(y_test[n], test_p)))
        print("DecisionTreeR-CV  MAE depth: "+p.__str__()  + " and "+(n+1).__str__() + "_set: {:.2f}".format(mean_absolute_error(t_test[n], hole_p)))
        #print("DecisionTree R Train set score for depth" + p.__str__() + "and" + n.__str__() + "_set: {:.2f}".format(tree.score(X_train[n], X_test[n])))
        #print("DecisionTree R Test  set score for depth" + p.__str__() + "and" + n.__str__() + "_set: {:.2f}".format(tree.score(y_train[n], y_test[n])))
        print("DecisionTreeR-TEST  ME depth: " + p.__str__() + " and " + (n + 1).__str__() + "_set: ",(np.mean([y_test[n] - test_p])))
        print("DecisionTreeR-CV ME depth: " + p.__str__() + " and " + (n + 1).__str__() + "_set: ",(np.mean([t_test[n]-hole_p])))


for n in range(0,5):
    svr = SVR(kernel= 'linear',max_iter=-1)
    svr.fit(X_train[n],X_test[n])
    hole_p = cross_val_predict(svr,t_train[n],t_test[n],cv=10)
    select = SelectFromModel(SVR(kernel='linear',max_iter=-1))
    select.fit(X_train[n],X_test[n])
    print("Selected features: from the set"+(n+1).__str__()+"",select.get_support())

    print("svr: Train score for set="+(n+1).__str__()+": {:.2f}".format(svr.score(X_train[n],X_test[n])))
    print("svr: Test  score for set=" +(n+1).__str__()+": {:.2f}".format(svr.score(y_train[n],y_test[n])))

    print("SVR and CV score for set="+(n+1).__str__()+"",np.mean(cross_val_score(svr,t_train[n],t_test[n],cv=10),axis=0))
    predict = svr.predict(y_train[n])
    print("SVR-TEST MAE for set:" +(n+1).__str__()+" {:.2f}".format(mean_absolute_error(y_test[n],predict)))
    print("SVR-CV MAE for   set:"+(n+1).__str__()+" {:.2f}".format(mean_absolute_error(t_test[n],hole_p)))

    print("SVR-CV ME for set: " + (n + 1).__str__() + ": ", np.mean([t_test[n]-hole_p]))
    print("SVR-TEST ME for set: "+(n+1).__str__()+": ",(np.mean([y_test[n]-predict])))
