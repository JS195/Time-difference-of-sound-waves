from Functions import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas as pd
import sklearn as sk
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import random
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy import optimize
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import silhouette_score
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler
import umap.umap_ as umap
import time

# Values found from https://www.engineeringtoolbox.com/sound-speed-solids-d_713.html
#Steel
cp1 = 1500
cs2 = 3150
cp2 = 5800
steel = deltat(cp1, cs2, cp2)
#plotter(steel)

#Brass: 
cp1 = 1500
cs2 = 2110
cp2 = 4700
brass = deltat(cp1, cs2, cp2)

#Concrete
cp1 = 1500
cs2 = 3200
cp2 = 3700
concrete = deltat(cp1, cs2, cp2)

# Lead (rolled)
cp1 = 1500
cs2 = 690
cp2 = 1960
lead = deltat(cp1, cs2, cp2)

# Aluminium 
cp1 = 1500
cs2 = 3040
cp2 = 6420
aluminium = deltat(cp1, cs2, cp2)

# Copper (rolled)
cp1 = 1500
cs2 = 2270
cp2 = 5010
copper = deltat(cp1, cs2, cp2)

# Magnesium (rolled)
cp1 = 15001
cs2 = 3050
cp2 = 5570
magnesium = deltat(cp1, cs2, cp2)

SVC_accuracy_scores = []
for E in range(250):
    noisy_df = generate_noisy_data(steel, brass, concrete, lead, aluminium, copper, magnesium, E)
    accuracy_scores = calculate_SVC_accuracy(noisy_df)
    SVC_accuracy_scores.extend(accuracy_scores)

knn_accuracy_scores = []
for E in range(250):
    noisy_df = generate_noisy_data(steel, brass, concrete, lead, aluminium, copper, magnesium, E)
    accuracy_scores = calculate_KNN_accuracy(noisy_df)
    knn_accuracy_scores.extend(accuracy_scores)

#grad_accuracy_scores = []
#for E in range(250):
#    noisy_df = generate_noisy_data(steel, brass, concrete, lead, aluminium, copper, magnesium, E)
#    accuracy_scores = calculate_gradient_accuracy(noisy_df)
#    grad_accuracy_scores.extend(accuracy_scores)

rf_accuracy_scores = []
for E in range(250):
    noisy_df = generate_noisy_data(steel, brass, concrete, lead, aluminium, copper, magnesium, E)
    accuracy_scores = randomForest(noisy_df)
    rf_accuracy_scores.extend(accuracy_scores)

axis1 = np.linspace(0.0000001, 0.000025, 250)

plt.plot(axis1, SVC_accuracy_scores)
plt.xlabel('E value')
plt.ylabel('Accuracy')
plt.title('Accuracy of a SVM for increasing values of E')
plt.show()

plt.plot(axis1, knn_accuracy_scores)
plt.xlabel('E value')
plt.ylabel('Accuracy')
plt.title('Accuracy of a KNN for increasing values of E')

#plt.plot(axis1, grad_accuracy_scores)
#plt.xlabel('E value')
#plt.ylabel('Accuracy')
#plt.title('Accuracy of GB for increasing values of E')

plt.plot(axis1, rf_accuracy_scores)
plt.xlabel('E value')
plt.ylabel('Accuracy')
plt.title('Accuracy of an RFC for increasing values of E')

sil_scores =[]
for E in range (1000):
    noisysteel = multiplicative_noise(steel,E)
    noisybrass = multiplicative_noise(brass,E)
    noisyconcrete = multiplicative_noise(concrete,E)
    noisylead = multiplicative_noise(lead,E)
    df = (pd.DataFrame(np.array(noisysteel + noisybrass + noisyconcrete + noisylead))).dropna(axis=1)
    df['label'] = ''
    X = df.drop('label', axis=1)

    kmeans = KMeans(n_clusters=4)
    kmeans.fit(X)

    y_pred = kmeans.predict(X)
    y_pred = y_pred.astype(str)

    sil_score = silhouette_score(X, y_pred)
    sil_scores.append(sil_score)

axis1 = np.linspace(0.00001, (1000*0.00001), num=1000)
plt.plot(axis1, sil_scores)
plt.xlabel('E value (proportional to error)')
plt.ylabel('Silhouette score')
plt.title('The silhouette scores from running kmeans for different values of E')
plt.show()