import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, classification_report, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn import utils
from sklearn.metrics import ConfusionMatrixDisplay

# load dataset
data = pd.read_csv('StudentsPerformance.csv')
print(data.head())
print(data.shape)

# create new column 'average_score'
data['average_score'] = np.int_(data[['math score', 'reading score', 'writing score']].mean(axis=1))
print(data.head())

# define function to convert average_score to letter grade
def letter_grade(average_score):
    if average_score >= 90:
        return 'A'
    elif average_score < 90 and average_score >= 80:
        return 'B'
    elif average_score < 80 and average_score >= 70:
        return 'C'
    elif average_score < 70 and average_score >= 60:
        return 'D'
    else:
        return 'E'

# apply letter_grade function to create new column 'grades'
data['grades'] = data.apply(lambda x: letter_grade(x['average_score']), axis = 1 )

# check for missing values
print(data.isnull().sum())

# explore categorical variables
print(data['gender'].value_counts())
print(data['race/ethnicity'].value_counts())
print(data['parental level of education'].value_counts())
print(data['lunch'].value_counts())
print(data['test preparation course'].value_counts())

# encode categorical variables using LabelEncoder
le = LabelEncoder()
for x in data:
    if data[x].dtypes=='object':
        data[x] = le.fit_transform(data[x])
print(utils.multiclass.type_of_target(data[x].astype('int')))
print(data.head())

# drop unnecessary columns and split dataset into train and test sets
data = data.drop(columns=['average_score'])
x = data.drop(columns=['grades'])
y = data['grades']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=13)
print(x_train.shape)
print(x_test.shape)

# Naive Bayes
model = GaussianNB()
model.fit(x_train, y_train)
y_predict = model.predict(x_test)
print(accuracy_score(y_test,y_predict))
print(classification_report(y_test, y_predict, target_names=le.classes_))
cm = confusion_matrix(y_test, y_predict)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=None)
disp.plot()
plt.show()
