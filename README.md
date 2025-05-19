# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.       
2.Read the data frame using pandas.         
3.Get the information regarding the null values present in the dataframe.      
4.Split the data into training and testing sets.          
5.convert the text data into a numerical representation using CountVectorizer.     
6.Use a Support Vector Machine (SVM) to train a model on the training data and make predictions on the testing data.       
7.Finally, evaluate the accuracy of the model.         

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by:UDHAYA PRAKASH V     
RegisterNumber:212224240177
*/
```
```
import pandas as pd
data = pd.read_csv("D:/introduction to ML/jupyter notebooks/spam.csv",encoding = 'windows-1252')
from sklearn.model_selection import train_test_split
data
data.shape
x = data['v2'].values
y = data['v1'].values
x.shape
y.shape
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.35,random_state = 48)
x_train
x_train.shape
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
x_train = cv.fit_transform(x_train)
x_test = cv.transform(x_test)
from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train,y_train)
y_pred = svc.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
acc = accuracy_score(y_test,y_pred)
acc
con = confusion_matrix(y_test,y_pred)
print(con)
cl = classification_report(y_test,y_pred)
print(cl)

```

## Output:
![image](https://github.com/23004513/Implementation-of-SVM-For-Spam-Mail-Detection/assets/138973069/3bd71a5c-def8-4e9c-8d3d-1668fc7985d4)

![image](https://github.com/23004513/Implementation-of-SVM-For-Spam-Mail-Detection/assets/138973069/8d765463-acba-4aca-aad7-8cb1a0d0fcb9)

![image](https://github.com/23004513/Implementation-of-SVM-For-Spam-Mail-Detection/assets/138973069/cb8f773a-81b4-4f89-8af8-2b57f4c40b3f)

![image](https://github.com/23004513/Implementation-of-SVM-For-Spam-Mail-Detection/assets/138973069/79d59541-d0c3-4259-8a13-aa7f7d24600a)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
