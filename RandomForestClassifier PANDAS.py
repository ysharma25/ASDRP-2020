import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

doc = pd.read_csv("/Users/yojita/Documents/EXAMPLE county data without spaces NOT REAL DATA")
a_names = np.array(doc.pop('Classes Example'))

train, test, train_labels, test_labels = train_test_split(doc, a_names, stratify = a_names, test_size = 0.3, random_state = 50)

train = train.fillna(train.mean())
test = test.fillna(test.mean())
tr_attributes = list(train.columns)

'''print(train)
print(test)
print(tr_attributes)'''

model = RandomForestClassifier(n_estimators=100,
                               random_state=50,
                               max_features = 'sqrt',
                               n_jobs = -1, verbose = 1)
model.fit(train, train_labels)
