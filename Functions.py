import matplotlib.pyplot as plt
import numpy as np
import umap.umap_ as umap
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

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

def generate_noisy_data(s, b, co, l, a, c, m, E):
    noisysteel = multiplicative_noise(s, E)
    noisybrass = multiplicative_noise(b, E)
    noisyconcrete = multiplicative_noise(co, E)
    noisylead = multiplicative_noise(l, E)
    noisyaluminium = multiplicative_noise(a, E)
    noisycopper = multiplicative_noise(c, E)
    noisymagnesium = multiplicative_noise(m, E)

    df = (pd.DataFrame(np.array(noisysteel + noisybrass + noisyconcrete + noisylead + noisyaluminium + noisycopper + noisymagnesium))).dropna(axis=1)
    df['label'] = ''
    df.loc[:len(noisysteel)-1, 'label'] = '0'
    df.loc[len(noisysteel):len(noisysteel) + len(noisybrass) - 1, 'label'] = '1'
    df.loc[len(noisysteel) + len(noisybrass):len(noisysteel) + len(noisybrass) + len(noisyconcrete) - 1, 'label'] = '2'
    df.loc[len(noisysteel) + len(noisybrass) + len(noisyconcrete):len(noisysteel) + len(noisybrass) + len(noisyconcrete) + len(noisylead) - 1, 'label'] = '3'
    df.loc[len(noisysteel) + len(noisybrass) + len(noisyconcrete) + len(noisylead):len(noisysteel) + len(noisybrass) + len(noisyconcrete) + len(noisylead) + len(noisyaluminium) - 1, 'label'] = '4'
    df.loc[len(noisysteel) + len(noisybrass) + len(noisyconcrete) + len(noisylead) + len(noisyaluminium):len(noisysteel) + len(noisybrass) + len(noisyconcrete) + len(noisylead) + len(noisyaluminium) + len(noisycopper) - 1, 'label'] = '5'
    df.loc[len(noisysteel) + len(noisybrass) + len(noisyconcrete) + len(noisylead) + len(noisyaluminium) + len(noisycopper):, 'label'] = '6'
    df['label'] = df['label'].astype(int)

    return df

def calculate_SVC_accuracy(df):
    SVC_accuracy_scores = []
    scaler = MinMaxScaler()

    X_train, X_test, y_train, y_test = train_test_split(df.drop('label', axis=1), df['label'], test_size=0.2, random_state=0)
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    clf = SVC(kernel='rbf')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    SVC_accuracy_scores.append(accuracy)

    return SVC_accuracy_scores
