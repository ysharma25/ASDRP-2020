import pandas as pd
from sklearn.metrics import accuracy_score
from pandas.plotting import scatter_matrix
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
import sklearn

doc = pd.read_csv("/Users/ember/OneDrive/Desktop/county data-across the US.csv") # Add filepath of csv file
#X and Y Variables
y_var = doc.pop('Risk/Case Class')
x_var = doc.copy()

x_train, x_test, y_train, y_test = train_test_split(x_var,y_var, test_size=0.3)

model = RandomForestClassifier(bootstrap=False, ccp_alpha=0.0, class_weight=None,
                               criterion='gini', max_depth=100, max_features='auto',
                               max_leaf_nodes=50, max_samples=1,
                               min_impurity_decrease=0.0, min_impurity_split=None,
                               min_samples_leaf=1, min_samples_split=2,
                               min_weight_fraction_leaf=0.0, n_estimators=1000,
                               n_jobs=-1, oob_score=False, random_state=50, verbose=0,
                               warm_start=False)


#Fit the model onto the data
model.fit(x_train, y_train)

#This will tell you which factors affect the number of cases the most
feature_importance = pd.DataFrame({'Factor' : x_train.columns, 'Feature Importance' : model.feature_importances_})
feature_importance.sort_values('Effect', ascending=True, inplace=True)

print(feature_importance)
predictions = model.predict(x_test)

#This will print the model's predictions and accuracy score
print(predictions)
print("Accuracy Score:" + str(accuracy_score(y_test, predictions)))

sklearn.metrics.plot_confusion_matrix(model, x_test, y_test)

beds = 6
employment = 40
income = 50000
sum_temp = 70
wint_temp = 40
hispanic = 10
afam = 5
asian = 50
poverty = 15
pets = 20
facilites = 4
airports = 0
insurance = 30
household = 3.74
elderly = 45
youth = 30

#Predict the risk level
inputs = np.array([beds, employment, income, sum_temp, wint_temp, hispanic, afam, asian,
                   poverty, pets, facilites, airports, insurance, household, elderly, youth]).reshape(1,-1)
print('Predicted Risk level: ' + model.predict(inputs))
