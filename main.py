from Functions import *
import time
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

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



def calculate_SVC_accuracy_scores(steel, brass, concrete, lead, aluminium, copper, magnesium, E_range):
    SVC_accuracy_scores = []
    scaler = MinMaxScaler()

    for E in E_range:
        noisysteel = multiplicative_noise(steel, E)
        noisybrass = multiplicative_noise(brass, E)
        noisyconcrete = multiplicative_noise(concrete, E)
        noisylead = multiplicative_noise(lead, E)
        noisyaluminium = multiplicative_noise(aluminium, E)
        noisycopper = multiplicative_noise(copper, E)
        noisymagnesium = multiplicative_noise(magnesium, E)
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



axis1 = np.linspace(0.0000001, 0.000025, 250)

plt.plot(axis1, SVC_accuracy_scores)
plt.xlabel('E value')
plt.ylabel('Accuracy')
plt.title('Accuracy of a SVM for increasing values of E')
plt.show()