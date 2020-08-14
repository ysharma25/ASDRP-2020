import pandas as pd
from sklearn.metrics import accuracy_score
from pandas.plotting import scatter_matrix
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
import sklearn

doc = pd.read_csv("/Users/ember/OneDrive/Desktop/county data across the US.csv")
#X and Y Variables
y_var = doc.pop('FAKE Risk Class')
x_var = doc.copy()


x_train, x_test, y_train, y_test = train_test_split(x_var,y_var, test_size=0.3)

#This replaces values with the value in the mean
for i in doc.columns:
    doc[i] = doc[i].fillna(doc[i].mean())

#THIS CHECKS FOR NaN and INFINITE VALUES V
print(x_train.notnull().values.all())
print(np.isfinite(x_train).all())
print(x_train.isnull().values.all())
print(np.isfinite(x_train).all())



model = RandomForestClassifier(n_estimators=100,
                               random_state=50,
                               max_features = 'sqrt',
                               max_depth = 200)
#Fit the model onto the data
model.fit(x_train, y_train)

#This will tell you which factors affect the number of cases the most
feature_importance = pd.DataFrame({'Factor' : x_train.columns, 'Effect' : model.feature_importances_})
feature_importance.sort_values('Effect', ascending=True, inplace=True)

print(feature_importance)

predictions = model.predict(x_test)
print(predictions)
print(accuracy_score(y_test, predictions))

