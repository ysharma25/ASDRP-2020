import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
doc = pd.read_csv("/Users/ember/OneDrive/Desktop/EXAMPLE county data without spaces NOT COMPLETELY ACCURATE DATA.csv")
a_names = np.array(doc.pop('Classes Example'))

train, test, train_labels, test_labels = train_test_split(doc, a_names, stratify = a_names, test_size = 0.3, random_state = 60)

train = train.fillna(train.mean())
test = test.fillna(test.mean())
tr_attributes = list(train.columns)

print(len(train))
print(len(test))
print(tr_attributes)
                   
