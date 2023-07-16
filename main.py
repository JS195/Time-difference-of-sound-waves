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

SVC_accuracy_scores = []
for E in range(250):
    noisy_df = generate_noisy_data(steel, brass, concrete, lead, aluminium, copper, magnesium, E)
    accuracy_scores = calculate_SVC_accuracy(noisy_df)
    SVC_accuracy_scores.extend(accuracy_scores)

axis1 = np.linspace(0.0000001, 0.000025, 250)

plt.plot(axis1, SVC_accuracy_scores)
plt.xlabel('E value')
plt.ylabel('Accuracy')
plt.title('Accuracy of a SVM for increasing values of E')
plt.show()