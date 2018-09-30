#!/usr/bin/python 

""" 
    Skeleton code for k-means clustering mini-project.
"""




import pickle
import numpy
import matplotlib.pyplot as plt
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit






### load in the dict of dicts containing all the data on each person in the dataset
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
### there's an outlier--remove it! 
data_dict.pop("TOTAL", 0)

salary = []
ex_stok = []
for users in data_dict:
    val = data_dict[users]["salary"]
    if val == 'NaN':
        continue
    salary.append(float(val))
    val = data_dict[users]["exercised_stock_options"]
    if val == 'NaN':
        continue
    ex_stok.append(float(val))
    
salary = [min(salary),20000.0,max(salary)]
ex_stok = [min(ex_stok),1000000.0,max(ex_stok)]

print salary
print ex_stok

salary = numpy.array([[e] for e in salary])
ex_stok = numpy.array([[e] for e in ex_stok])

from sklearn.preprocessing import MinMaxScaler 
scaler_salary = MinMaxScaler()
scaler_stok = MinMaxScaler()

rescaled_salary = scaler_salary.fit_transform(salary)
rescaled_stock = scaler_salary.fit_transform(ex_stok)

print rescaled_salary
print rescaled_stock

