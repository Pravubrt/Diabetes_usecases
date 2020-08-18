import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,f1_score,classification_report

diabetes= pd.read_csv('diabetes.csv')
print(diabetes.info())

zero_not_accepted=['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
for column in zero_not_accepted:
    diabetes[column]= diabetes[column].replace(0,np.nan)
    mean=int(diabetes[column].mean(skipna=True))
    diabetes[column]= diabetes[column].replace(np.nan,mean)
#print(diabetes.head())
X= diabetes.iloc[:,0:8]
y=diabetes.iloc[:,8]
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=0)


sc_X= StandardScaler()
X_train= sc_X.fit_transform(X_train)
X_test= sc_X.fit_transform(X_test)

import math
print(math.sqrt(len(y_test)))
classifier= KNeighborsClassifier(n_neighbors=33,p=2)
classifier.fit(X_train,y_train)
y_pred= classifier.predict(X_test)

cm= confusion_matrix(y_test,y_pred)
print('cm',cm)
error_rate=[]
import matplotlib.pyplot as plt
for i in range(1,40):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i=knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))
plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue',linestyle='dashed',marker='o',markerfacecolor='red',markersize=10)
plt.title('Error Rate Vs K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.show()

print(classification_report(y_test,y_pred))
print(accuracy_score(y_test,y_pred))