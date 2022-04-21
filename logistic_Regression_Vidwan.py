import pandas as pd
import sklearn
data=pd.read_csv("datafile.csv")
print(data.columns)
data=data[['Year','DayofWeek','DepTime','UniqueCarrier','Origin','Dest','DepDelay']]
data.loc[data['DepDelay']<=15,'DepDelay']=0
data.loc[data['DepDelay']>15,'DepDelay']=1
print(data.head())

print(data.info())
data=data.dropna()
print(data.info())

from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
data['UniqueCarrier_N']=labelencoder.fit_transform(data['UniqueCarrier'])
# data['TailNum_N']=labelencoder.fit_transform(data['TailNum'])
data['Origin_N']=labelencoder.fit_transform(data['Origin'])
data['Dest_N']=labelencoder.fit_transform(data['Dest'])


print(data.info())
#
# # print(data['UniqueCarrier_N'].unique())
# # print(data['TailNum_N'].unique())
# # print(data['Origin_N'].unique())
#
#
X=data[['Year','DayofWeek','DepTime','UniqueCarrier_N','Origin_N','Dest_N']]
Y=data['DepDelay']

print(X.head())
print(Y.head())

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.20)
#
#
# #nochange
from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression(max_iter=1000)
results=lr.fit(X_train,y_train)
y_pred=lr.predict(X_test)
coef=logreg.coef_[0]
print(coef)
print(results.summary())
from sklearn import metrics
cnf_matrix=metrics.confusion_matrix(y_test,y_pred)
print(cnf_matrix)
acc=metrics.accuracy_score(y_test,y_pred)


