#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here
from sklearn.cross_validation import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(features,labels, random_state=42, test_size=0.3)

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score,recall_score
clf=DecisionTreeClassifier(random_state=0)
clf.fit(xtrain,ytrain)
ypred=clf.predict(xtest)
#print recall_score(ytest,ypred)
pred=[0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
true = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]
print precision_score(true,pred)
print recall_score(true,pred)