import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

df=pd.read_csv('Data.csv', encoding = "ISO-8859-1")
train = df[df['Date'] < '20150101']
test = df[df['Date'] > '20141231']

# Removing punctuations
train1=train.iloc[:,2:27]
train1.replace("[^a-zA-Z]"," ",regex=True, inplace=True)

# Renaming column names for ease of access
list1= [i for i in range(25)]
new_Index=[str(i) for i in list1]
train1.columns= new_Index

# Convertng headlines to lower case
for index in new_Index:
    train1[index]=train1[index].str.lower()

train2 = []
for row in range(0,len(train1.index)):
    train2.append(' '.join(str(x) for x in train1.iloc[row,0:25]))
    
## implement BAG OF WORDS
countvector=CountVectorizer(ngram_range=(2,2))
x_train=countvector.fit_transform(train2)
y_train=train['Label']

# implement RandomForest Classifier
randomclassifier=RandomForestClassifier(n_estimators=200,criterion='entropy')
randomclassifier.fit(x_train,y_train)

# Removing punctuations
test1=test.iloc[:,2:27]
test1.replace("[^a-zA-Z]"," ",regex=True, inplace=True)

# Renaming column names for ease of access
list1= [i for i in range(25)]
new_Index=[str(i) for i in list1]
test1.columns= new_Index

# Convertng headlines to lower case
for index in new_Index:
    test1[index]=test1[index].str.lower()
    
test2= []
for row in range(0,len(test1.index)):
    test2.append(' '.join(str(x) for x in test1.iloc[row,2:27]))
    
x_test = countvector.transform(test2)
y_test = test['Label']
## Predict for the Test Dataset
y_pred = randomclassifier.predict(x_test)

matrix=confusion_matrix(y_test,y_pred)
score=accuracy_score(y_test,y_pred)
report=classification_report(y_test,y_pred)
print(report)