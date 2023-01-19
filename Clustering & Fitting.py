# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 12:45:20 2023

@author: SamuelOkachi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import scipy.optimize as opt
from scipy.optimize import curve_fit
import seaborn as sns
import err_ranges as err

def solution(filename,countries,columns,indicator):
    df = pd.read_csv(filename,skiprows=4)
    df = df[df['Indicator Name'] == indicator]
    df = df[columns]
    df.set_index('Country Name', inplace = True)
    #df = df.loc[countries]
    df = df.dropna()
    return df,df.transpose()


filename = 'API_19_DS2_en_csv_v2_4773766.csv'
countries = ['Germany','Argentina','Brazil','Nigeria','Afghanistan']
columns = ['Country Name', '1990','2019']
indicators = ['CO2 emissions (metric tons per capita)']

year_co2,cnty_co2= solution(filename,countries,columns,indicators[0])

#Scatter Plot 
scatter = year_co2.plot('1990', '2019',kind='scatter')
print(scatter)

#c02 emission
x = year_co2.values
print(x)

#Plot showing K-MEANS
plt.figure()
cs = []
for i in range(1, 10):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=100, n_init=10, random_state=0)
    kmeans.fit(x)
    cs.append(kmeans.inertia_)
plt.plot(range(1, 10), cs)
plt.title('Elbow clustering')
plt.xlabel('Number of clusters')
plt.ylabel('CS')
plt.show()

#K-means showing Number of iteration as Numpy arrays
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=100, n_init=10, random_state=0)
y_kmeans = kmeans.fit_predict(x)
print(y_kmeans)


#Using y-means to get a scatter plot and identify our centroids
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s = 50, c = 'blue',label = 'label 0')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s = 50, c = 'orange',label = 'label 1')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s = 50, c = 'green',label = 'label 2')
#plt.scatter(x[y_kmeans == 3, 0], x[y_kmeans == 3, 1], s = 50, c = 'blue',label = 'Iabel 3')
#plt.scatter(x[y_kmeans == 4, 0], x[y_kmeans == 4, 1], s = 50, c = 'black',label = 'Iabel 4')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s = 10, c = 'red', label = 'Centroids')
plt.title('Clusters and Centroids')
plt.legend()
plt.show()


#MODEL FITING
#define the objective function
def linear(x, a, b):
        s = a + b*x
        return s
# create a few points with normal distributed random errors
xarr = np.linspace(0.0, 5.0, 15)
yarr = linear(xarr, 1.0, 0.2)
ymeasure = yarr + np.random.normal(0.0, 0.5, len(yarr))

lin_list = []
for x, ym in zip(xarr, ymeasure):
    lin_list.append([x, ym])

df_lin = pd.DataFrame(lin_list, columns=["1990", "2019"])
print(df_lin)


param, covar = opt.curve_fit(linear, year_co2["1990"], year_co2["2019"])
plt.figure()
plt.plot(year_co2["1990"], year_co2["2019"], "go", label="co2 emission")
plt.plot(year_co2["1990"], linear(year_co2["1990"], *param), label="fit")
plt.plot(year_co2["2019"], linear(year_co2["2019"], 1.0, 0.2), label="a = 1.0+0.02*x")
plt.xlabel("1990")
plt.ylabel("2019")
plt.legend()
plt.show()
