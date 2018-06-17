
# Cartes Auto Adaptatives / Self Organizing Map: SOM




# Importation des librairies
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importation des données
dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values   #[toute les lignes, toute les colonnes sauf la dernière] en tableau
y = dataset.iloc[:, -1].values   #[toute les lignes, la dernière colonne] en tableau


# Changement d'échelle / Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)


# Entrainement du SOM
from minisom import MiniSom
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)   #implémentation du SOM à partir de la librairie
som.random_weights_init(X)   #initialisation des poids aléatoirement
som.train_random(data = X, num_iteration = 100)


# Visualisation des résultats (des MID (mean internal node distance / distance moyenne interne neuronale))
from pylab import bone, pcolor, colorbar, plot, show
bone()   #initialise le graph
pcolor(som.distance_map().T)   #affiche les distance MID et les colorise en fonction de celle-ci
colorbar()   #affiche l'échelle
markers = ['o', 's']   #o=circle, s=square   
colors = ['r', 'g']   #r=red, g=green
for i, x in enumerate(X):   #pour chaque(i) valeur(x) de X 
    w = som.winner(x)   #neurone gagnant
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()


# Finding the frauds
mappings = som.win_map(X)
frauds = np.concatenate((mappings[(8,1)], mappings[(6,8)]), axis = 0)
frauds = sc.inverse_transform(frauds)