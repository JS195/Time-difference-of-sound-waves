import matplotlib.pyplot as plt
import numpy as np
import umap.umap_ as umap
import pandas as pd

def deltat(cp1, cs2, cp2):
    d = 0.1
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

def multiplicative_noise(delta_t1,E):
    E = E*0.0000001
    datasets = []
    for i in range(100):
        noise = np.random.normal(0, 1, len(delta_t1))*E
        delta_t1_noisy = delta_t1+noise
        datasets.append(delta_t1_noisy)
    return datasets

def umap_embedding(df, n_components=2, random_state=42):
    mapper = umap.UMAP(n_components=n_components, random_state=random_state)
    embedding = mapper.fit_transform(df.values)
    return pd.DataFrame(embedding, columns=[f"UMAP_{i+1}" for i in range(n_components)], index=df.index)