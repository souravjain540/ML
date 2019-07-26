import numpy as np
#1-D ARRAY
arr=np.array([1,2,3])
print(arr)
#2-D ARRAY
a=np.array([[1,2,3],[4,5,6]])                          m
print(a)
#DIMENSIONS OF ARRAY
print(arr.ndim)
print(a.ndim)
#SHAPE OF ARRAY
print(arr.shape)
print(a.shape)

#changing shape
a.shape=(6,)
print(a)

#question 
import numpy as np
arr=np.array([
        [1,2,3],
        [4,5,6],
        [7,8,9]
        ])
print(arr)
print("dimension of array is",arr.ndim)
print('size of array is',arr.shape)
N=int(input('Enter value of N '))
arr1=arr*N
print(arr1)
arr.shape=(9,)
print('New dimensions is ',arr.ndim)
print(sum(arr))
print(max(arr))
print(min(arr))
#changing it to 3d
arr.shape=(3,3,1)
print(arr)


#arange(start=0,end,diff) , function,where end value is excluded
arr=np.arange(1,51,.25)
print(arr)

#linspace(start,stop,num=50,endpoint=True,retstep=False,dtype=None,axis=0)

#to generate 50 elements from 1 to 10 where 10 is excluded
arr=np.linspace(1,10,endpoint=False)
print(arr)
#to generate 9 elements from 1 to 10 where 10 is included
arr=np.linspace(1,10,9)
print(arr)

#ones(shape,dtype=None)    ret a new array of shape given filled with one 

arr=np.ones([5,5,5,5,5])
print('array is',arr)
print('array size is',arr.size)
print('array dimensions is',arr.ndim)
print('array shape is',arr.shape)

#zeros same as ones rather filled with zeroes

arr=np.zeros([5,5,5,5,5])
print('array is',arr)
print('array size is',arr.size)
print('array dimensions is',arr.ndim)
print('array shape is',arr.shape)


#MATRIX OPERATIONS

a=np.array([
        [1,2,3],
        [4,5,6],
        [7,8,9]
        ])

b=np.ones([3,3])
print((a+b))
print((a-b))
print((a*b))#element by element matrix multiplication
print((a.dot(b)))#normal matrix multiplcation


#PANDAS

#pandas series
import pandas as pd
data=pd.Series([11,14,55,64,32,12])
print(data)
#index and values
print(data.index)
print(data.values)

#Addition of two Series
import pandas as pd
students=['Girish Attri','Neha Dhamija','Ravi Verma','Ankit Jain','Nikhil']
term1_marks=[43,35,65,45,49]
term2_marks=[42,49,43,40,42]
s1=pd.Series(term1_marks, students)
s2=pd.Series(term2_marks, students)
total=s1+s2
print(total)
#accessing values
term1_marks=[43,35,65,45,49]
students=['Girish Attri','Neha Dhamija','Ravi Verma','Ankit Jain','Nikhil']
s=pd.Series(term1_marks, students)
print(s['Girish Attri'])

#Filtering

students=['Girish Attri','Neha Dhamija','Ravi Verma','Ankit Jain','Nikhil']
term1_marks=[43,35,65,45,49]

s=pd.Series(term1_marks, students)

pass_students=s[s>50]
fail_students=s[s<50]
print(pass_students, end="\n\n")
print(fail_students, end="\n\n")
#series from dictionary
marks={
    "Girish Attri" : 84,
    "Ravi verma" : 74,
    "Ankit Jain" : 84,
    "Nikhil Rastogi" : 89,
    "Neha Dhamija" : 84,
    "Shrishti Verma" : 90
}

s=pd.Series(marks)

print(s)
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

marks={
    "Girish Attri" : 84,
    "Ravi verma" : 74,
    "Pankaj Sharma" : None,
    "Ankit Jain" : 84,
    "Nikhil Rastogi" : 89,
    "Sanjay Gupta" : None,
    "Neha Dhamija" : 84,
    "Shrishti Verma" : 90
}

s=pd.Series(marks)

print(s, end="\n\n")

#Filtering NaN
print(s.dropna(), end="\n\n")

#Filling NaN
print(s.fillna(0), end="\n\n")

#Filling Nan with astype
print(s.fillna(0).astype(int), end="\n\n")



marks={
    "Girish Attri" : 84,
    "Ravi verma" : 74,
    "Pankaj Sharma" : None,
    "Ankit Jain" : 84,
    "Nikhil Rastogi" : 89,
    "Sanjay Gupta" : None,
    "Neha Dhamija" : 84,
    "Shrishti Verma" : 90
}

s=pd.Series(marks)

print(s, end="\n\n")

#Filtering NaN
print(s.dropna(), end="\n\n")

#Filling NaN
print(s.fillna(0), end="\n\n")

#Filling Nan with astype
print(s.fillna(0).astype(int), end="\n\n")
marks={
    "Girish Attri" : 84,
    "Ravi verma" : 74,
    "Pankaj Sharma" : None,
    "Ankit Jain" : 84,
    "Nikhil Rastogi" : 89,
    "Sanjay Gupta" : None,
    "Neha Dhamija" : 84,
    "Shrishti Verma" : 90
}

s=pd.Series(marks)

print(s, end="\n\n")

#Filtering NaN
print(s.dropna(), end="\n\n")

#Filling NaN
print(s.fillna(0), end="\n\n")

#Filling Nan with astype
print(s.fillna(0).astype(int), end="\n\n")
marks={
    "Girish Attri" : 84,
    "Ravi verma" : 74,
    "Pankaj Sharma" : None,
    "Ankit Jain" : 84,
    "Nikhil Rastogi" : 89,
    "Sanjay Gupta" : None,
    "Neha Dhamija" : 84,
    "Shrishti Verma" : 90
}

s=pd.Series(marks)

print(s, end="\n\n")

#Filtering NaN
print(s.dropna(), end="\n\n")

#Filling NaN
print(s.fillna(0), end="\n\n")

#Filling Nan with astype
print(s.fillna(0).astype(int), end="\n\n")marks={
    "Girish Attri" : 84,
    "Ravi verma" : 74,
    "Pankaj Sharma" : None,
    "Ankit Jain" : 84,
    "Nikhil Rastogi" : 89,
    "Sanjay Gupta" : None,
    "Neha Dhamija" : 84,
    "Shrishti Verma" : 90
}

s=pd.Series(marks)

print(s, end="\n\n")

#Filtering NaN
print(s.dropna(), end="\n\n")

#Filling NaN
print(s.fillna(0), end="\n\n")

#Filling Nan with astype
print(s.fillna(0).astype(int), end="\n\n")marks={
    "Girish Attri" : 84,
    "Ravi verma" : 74,
    "Pankaj Sharma" : None,
    "Ankit Jain" : 84,
    "Nikhil Rastogi" : 89,
    "Sanjay Gupta" : None,
    "Neha Dhamija" : 84,
    "Shrishti Verma" : 90
}

s=pd.Series(marks)

print(s, end="\n\n")

#Filtering NaN
print(s.dropna(), end="\n\n")

#Filling NaN
print(s.fillna(0), end="\n\n")

#Filling Nan with astype
print(s.fillna(0).astype(int), end="\n\n")marks={
    "Girish Attri" : 84,
    "Ravi verma" : 74,
    "Pankaj Sharma" : None,
    "Ankit Jain" : 84,
    "Nikhil Rastogi" : 89,
    "Sanjay Gupta" : None,
    "Neha Dhamija" : 84,
    "Shrishti Verma" : 90
}

s=pd.Series(marks)

print(s, end="\n\n")

#Filtering NaN
print(s.dropna(), end="\n\n")

#Filling NaN
print(s.fillna(0), end="\n\n")

#Filling Nan with astype
print(s.fillna(0).astype(int), end="\n\n")marks={
    "Girish Attri" : 84,
    "Ravi verma" : 74,
    "Pankaj Sharma" : None,
    "Ankit Jain" : 84,
    "Nikhil Rastogi" : 89,
    "Sanjay Gupta" : None,
    "Neha Dhamija" : 84,
    "Shrishti Verma" : 90
}

s=pd.Series(marks)

print(s, end="\n\n")

#Filtering NaN
print(s.dropna(), end="\n\n")

#Filling NaN
print(s.fillna(0), end="\n\n")

#Filling Nan with astype
print(s.fillna(0).astype(int), end="\n\n")

#Creating DataFrames using Pandas Series

years = [2014, 2015, 2016, 2017]
shop1 = pd.Series([2409.14, 2941.01, 3496.83, 3119.55], index=years)
shop2 = pd.Series([1203.45, 3441.62, 3007.83, 3619.53], index=years)
shop3 = pd.Series([3412.12, 3491.16, 3457.19, 1963.10], index=years)

shops_df = pd.concat( [shop1, shop2, shop3] )           #Column Wise Concatenation
print(shops_df, end="\n\n")

shops_df = pd.concat( [shop1,shop2,shop3], axis=1)
print(shops_df, end="\n\n")

#Column Names

years = [2014, 2015, 2016, 2017]
shop1 = pd.Series([2409.14, 2941.01, 3496.83, 3119.55], index=years)
shop2 = pd.Series([1203.45, 3441.62, 3007.83, 3619.53], index=years)
shop3 = pd.Series([3412.12, 3491.16, 3457.19, 1963.10], index=years)

shops_df = pd.concat( [shop1,shop2,shop3], axis=1)

col_names = ['Shop 1', 'Shop 2', 'Shop 3']
shops_df.columns = col_names

print(shops_df, end="\n\n")

#Dataframes from Dictionaries'
cities = {"name": ["London", "Berlin", "Madrid", "Rome", 
                   "Paris", "Vienna", "Bucharest", "Hamburg", 
                   "Budapest", "Warsaw", "Barcelona", 
                   "Munich", "Milan"],
          
          "country": ["England", "Germany", "Spain", "Italy",
                      "France", "Austria", "Romania", 
                      "Germany", "Hungary", "Poland", "Spain",
                      "Germany", "Italy"],
          
          "population": [8615246, 3562166, 3165235, 2874038,
                         2273305, 1805681, 1803425, 1760433,
                         1754000, 1740119, 1602386, 1493900,
                         1350680],
          }
city_frame = pd.DataFrame(cities)
city_frame


print(city_frame["population"])       #as dictionary like way

#or

print(city_frame.population)          #as attribute

area = [1572, 891.85, 605.77, 1285, 
        105.4, 414.6, 228, 755, 
        525.2, 517, 101.9, 310.4, 
        181.8]
# area could have been designed as a list, a Series, an array or a scalar   
city_frame["area"] = area
print(city_frame)

# Matplotlib

#Matplotlib is a plotting library for Python. It is used along with NumPy to provide an environment that is an effective open source alternative for MatLab
#Importing pyplot
from matplotlib import pyplot as plt

#Plotting to our canvas
plt.plot([1,2,3],[4,5,1])

#Showing what we plotted
plt.show()


x = ['2013','2014','2015','2016','2017']
y = [434,543,346,634,564]

plt.plot(x,y)

plt.title('ABC Pvt Ltd.')
plt.ylabel('Sales')
plt.xlabel('Year')

plt.show()

# **Histograms**

import matplotlib.pyplot as plt

population_ages = [22,55,62,45,21,22,34,42,42,4,99,102,110,120,121,122,130,111,115,112,80,75,65,54,44,43,42,48]

bins = [0,10,20,30,40,50,60,70,80,90,100,110,120,130]

plt.hist(population_ages, bins, histtype='bar', rwidth=0.8)

plt.xlabel('Age Groups')
plt.ylabel('Count')
plt.title('Population Chart')
plt.show()

#Pie Chart
import matplotlib.pyplot as plt

voting = [54354,65443,34634,23423,1233]
candidates = ['A','B','C','D','E']
cols = ['c','m','r','b','g']

plt.pie(voting,
        labels=candidates,
        colors=cols,
        startangle=90,
        shadow= True,
        explode=(0,0,0,0,0.1),
        autopct='%d%%'
       )

plt.title('Candidate wise Voting %')
plt.show()

#Scikit Learn LinearRegression Classifier
#---------------------------------------

import numpy as np
from sklearn.linear_model import LinearRegression

x = np.array([1,2,3,4,5])
x.shape=(-1,1)
y = np.array([3,4,5,6,7])

#Create Classifier Object
clf = LinearRegression()

#Train the Classifier
clf.fit(x,y)

#Evaluating the Accuracy of the Model
score=clf.score(x,y)
print("Accuracy of The Model : ", score)

#Predictions
predict_x=np.array([6,8,9])
predict_x.shape=(-1,1)
predict_y=clf.predict(predict_x)
print("Prediction Data : ", predict_y)



LabelEncoder
-------------

Encode labels with value between 0 and n_classes-1.

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit([1, 2, 2, 6])
LabelEncoder()
le.classes_
array([1, 2, 6])
le.transform([1, 1, 2, 6]) 
array([0, 0, 1, 2]...)
le.inverse_transform([0, 0, 1, 2])
array([1, 1, 2, 6])






Handling Null/NaN data in a dataset.


Filling Missing Values with Mean data.


s1=pd.Series([34,65,34,None,43,54,None,23,54,65,34,None])
#Mean
mn=s1.mean()
#Fill
s1=s1.fillna(mn)

print(s1)



Imputer
--------

For various reasons, many real world datasets contain missing values, often encoded as blanks, NaNs or other placeholders. Such datasets however are incompatible with scikit-learn estimators which assume that all values in an array are numerical, and that all have and hold meaning. A basic strategy to use incomplete datasets is to discard entire rows and/or columns containing missing values. However, this comes at the price of losing data which may be valuable (even though incomplete). A better strategy is to impute the missing values, i.e., to infer them from the known part of the data.


sklearn.impute.SimpleImputer
-----------------------------

The SimpleImputer class provides basic strategies for imputing missing values. Missing values can be imputed with a provided constant value, or using the statistics (mean, median or most frequent) of each column in which the missing values are located.


#Mean

from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np

s1=np.array([34,65,34,None,43,54,None,23,54,65,34,None])
s1.shape=(-1,1)

imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp.fit(s1)		


#Filling Data
s1=imp.transform(s1)

print(s1)


#Most Frequent

from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np

s1=np.array([34,65,34,None,43,54,None,34,54,65,34,None])
s1.shape=(-1,1)

imp = SimpleImputer(strategy='most_frequent')
imp.fit(s1)		


#Filling Data
s1=imp.transform(s1)

print(s1)



    
import numpy as np
from sklearn.linear_model import LinearRegression

#Converting Text Data to Numbers
def Mapper(data):
    unique=list(set(data))
    unique.sort()
    count=1
    map_data={}
    for value in unique:
        map_data[value]=count
        count+=1
    return map_data

def ConvertTextToNumber(data,maps):
    for i in range(len(data)):
        data[i]=maps[data[i]]



xdata=['A','B','C','D','A','C','B','D','C','E']
ydata=[1.1,3.0,4.2,7,1.2,4.3,3.1,7.7,4.5,10.2]

maps=Mapper(xdata)
ConvertTextToNumber(xdata,maps)

x = np.array(xdata)
x.shape=(-1,1)
y = np.array(ydata)



#Create Classifier Object
clf = LinearRegression()

#Train the Classifier
clf.fit(x,y)

#Evaluating the Accuracy of the Model
score=clf.score(x,y)
print("Accuracy of The Model : ", score)

#Predictions
predictdata=['A','E']
ConvertTextToNumber(predictdata,maps)
predict_x=np.array(predictdata)
predict_x.shape=(-1,1)
predict_y=clf.predict(predict_x)
print("Prediction Data : ", predict_y)





#PRE PROCESSING PART

import numpy as np
from sklearn import preprocessing , neighbors , svm , model_selection
import pandas as pd

df= pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data')
df.replace('?',-99999,inplace=True)
col_names=['id','clump_thickness','cell_size','cell shape','adhesion','epithelial','nuclei','brandchlomatin','nucleoli','mitoses','class']
df.columns=col_names
print(df.head())
df.drop(['id'],1,inplace=True)

X=np.array(df.drop(['class'],1))
y=np.array(df['class'])

X_train,X_test,y_train,y_test = model_selection.train_test_split(X,y,test_size=0.2)




clf=neighbors.KNeighborsClassifier()
clf.fit(X_train,y_train)
confidence=clf.score(X_test,y_test)
print("Confidence is  :-  ", int(confidence*100),'%')

example_measures=np.array([4,.2,1,1,1,2,3,2,1])
example_measures.shape=(1,-1)
prediction=clf.predict(example_measures)
print('Cancer state : ','Benign' if prediction==2 else 'Malignant')


from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split , GridSearchCV

pipeline=Pipeline([('clf',svm.SVC(kernel='linear',C=1,gamma=1))])
params={'clf_C': (0.1,0.5,1,2,5,10,20)
    }
clf=svm.SVC()
clf.fit(X_train,y_train)
confidence=clf.score(X_test,y_test)
print("Confidence is  :-  ", int(confidence*100),'%')

example_measures=np.array([4,.2,1,1,1,2,3,2,1])
example_measures.shape=(1,-1)
prediction=clf.predict(example_measures)
print('Cancer state : ','Benign' if prediction==2 else 'Malignant')


clf=svm.SVC()
clf.fit(X_train,y_train)
confidence=clf.score(X_test,y_test)
print("Confidence is  :-  ", int(confidence*100),'%')

example_measures=np.array([4,.2,1,1,1,2,3,2,1])
example_measures.shape=(1,-1)
prediction=clf.predict(example_measures)
print('Cancer state : ','Benign' if prediction==2 else 'Malignant')
