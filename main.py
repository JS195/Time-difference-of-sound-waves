import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def deltat(cp1, cs2, cp2):
    d = 100000
    theta_p1 = np.linspace(0, np.pi/2, num=100)
    g2 = cp2/cp1
    g1 = cs2/cp1
    delta_t1 = np.array([(d/cp1)*(np.cos(theta) + 1/np.sqrt(1-g1**2*np.sin(theta)**2)*(g1*np.sin(theta)**2-1/g1)) for theta in theta_p1]) 
    delta_t2 = np.array([(d/cp1)*(np.cos(theta) + 1/np.sqrt(1-g2**2*np.sin(theta)**2)*(g2*np.sin(theta)**2-1/g2)) for theta in theta_p1])
    delta_t3 = delta_t2-delta_t1
    return(delta_t3)

def plotter(delta_t3):
    theta_p1 = np.linspace(0, np.pi/2, num=100)
    plt.plot(theta_p1, delta_t3,'r')
    plt.xlabel('theta_p1 (rad)')
    plt.ylabel('Time (s)')
    plt.title('Difference in time between longitudinal and shear waves reaching the reciever for a variety of angles')
    plt.show()

# Values found from https://www.engineeringtoolbox.com/sound-speed-solids-d_713.html
#Steel
steel_cp1 = 1500
steel_cs2 = 3150
steel_cp2 = 5800
steel = deltat(steel_cp1, steel_cs2, steel_cp2)
#plotter(steel)

#Brass: 
brass_cp1 = 1500
brass_cs2 = 2110
brass_cp2 = 4700
brass = deltat(brass_cp1, brass_cs2, brass_cp2)
#plotter(brass)

#Concrete
concrete_cp1 = 1500
concrete_cs2 = 3200
concrete_cp2 = 3700
concrete = deltat(concrete_cp1, concrete_cs2, concrete_cp2)
#plotter(concrete)

# Lead (rolled)
lead_cp1 = 1500
lead_cs2 = 690
lead_cp2 = 1960
lead = deltat(lead_cp1, lead_cs2, lead_cp2)
#plotter(lead)

def noise(delta_t1):
    E = 1
    datasets = []
    for i in range(5):
        noise = np.random.normal(0, 1, len(delta_t1)) * E
        delta_t1_noisy = delta_t1 + noise
        datasets.append(delta_t1_noisy)
    return datasets

def plot_datasets(datasets):
    for i, dataset in enumerate(datasets):
        plt.plot(dataset, label=f'dataset {i + 1}')
    plt.legend()
    plt.show()

noisysteel = noise(steel)
noisybrass = noise(brass)
noisyconcrete = noise(concrete)
noisylead = noise(lead)

#plot_datasets(noisysteel)
#plot_datasets(noisybrass)
#plot_datasets(noisyconcrete)
#plot_datasets(noisylead)

df = (pd.DataFrame(np.array(noisysteel+noisybrass+noisyconcrete+noisylead))).dropna(axis=1)

scaled_df = StandardScaler().fit_transform(df)

def Kmeans_method(filename,num_components):
    new_cluster = np.array(filename)
    kmeans = KMeans(n_clusters= num_components, random_state=0)
    label = kmeans.fit_predict(new_cluster)
    centroids = kmeans.cluster_centers_
    for i in range(0,4):
        plt.scatter(new_cluster[label == i , 0] ,new_cluster[label == i , 1], label = i, s = 10)
        plt.scatter(centroids[:,0] , centroids[:,1] , s = 30, color = 'k', marker="x")
        plt.legend()
    return label

Kmeans_method(scaled_df,4)