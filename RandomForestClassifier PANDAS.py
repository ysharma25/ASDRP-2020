import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

doc = pd.read_csv("/Users/ember/OneDrive/Desktop/County data without spaces.csv")
a_names = np.array(doc.pop('Classes Example'))

train, test, train_labels, test_labels = train_test_split(doc, a_names, stratify = a_names, test_size = 0.3, random_state = 50)

train = train.fillna(train.mean())
test = test.fillna(test.mean())
tr_attributes = list(train.columns)

'''train.head(5)

le = LabelEncoder()
train['County Name'] = le.fit_transform(train['County Name'].astype('str'))
test['County Name'] = le.fit_transform(train['County Name'].astype('str'))

train.head(5)'''

print(train)
print(test)
print(train_labels)
print(tr_attributes)

model = RandomForestClassifier(n_estimators=100,
                               random_state=50,
                               max_features = 'sqrt')

print(model.fit(train, train_labels))
