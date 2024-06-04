# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 21:28:08 2022

@author: https://scikit-learn.org/stable/modules/tree.html#tips-on-practical-use

"""

# criterion log_loss requires-Python >=3.8
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
import hickle as hkl
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt 


x,y_t,x_norm,x_n_s,y_t_s = hkl.load('./badanie_metaparamterow\haberman.hkl')
# dane w orientacji kolumnowej [[],[],[]]

# Zmiana wartości cechy wyjściowej z 1,2 na 0,1
y_t_s -= 1
# Przetransponowanie cech wejściowych na orientacje wierszową
x=x_n_s.T
# Przetransponowanie cechy wyjściowej na orientacje wierszową
y_t = np.squeeze(y_t_s)

# dla pierwszego zestawu 2 metaparametrów
criterion_vec = ["gini", "entropy", "log_loss"] 
criterion_num_vec = np.array(range(len(criterion_vec))) # [0 1 2] wektor kryteriów
# Wypisanie wszystkich kryteriów 
for ind_criterion in range(len(criterion_vec)):
    print(criterion_vec[ind_criterion])

max_depth_vec = np.array(range(1,21,1)) # [ 1 2 3 ... 20] wektor badanych głebokości

data = x
target = y_t

# Ustalenie ilości części (foldów) na którą zostaną podzielone dane
CVN = 10
skfold = StratifiedKFold(n_splits=CVN)
# Stworzenie pustego wektora na dane wyjsciowe dla kazdej kombinacji metaparemtrow
PK_cr_md_vec = np.zeros([len(criterion_vec),len(max_depth_vec)])

for criterion_ind in range(len(criterion_vec)): 
    for max_depth_ind in range(len(max_depth_vec)):                           
        PK_vec = np.zeros(CVN)
        for i, (train, test) in enumerate(skfold.split(data, target), start=0):
            x_train, x_test = data[train], data[test]
            y_train, y_test = target[train], target[test]
            # print(x_train)
            # Inicjalizacja klasyfikatora drzewa decyzyjnego z odpowiednimi parametrami.
            decision_tree = DecisionTreeClassifier(random_state=0, max_depth=max_depth_vec[max_depth_ind],  criterion=criterion_vec[criterion_ind])
            decision_tree = decision_tree.fit(x_train, y_train)
            result = decision_tree.predict(x_test)

            if max_depth_vec[max_depth_ind] == 1:
               plot_tree(decision_tree)

            # Liczba próbek w zbiorze testowym
            n_test_samples = test.size
            # Obliczenie dokładności predykcji dla tego podziału.
            PK_vec[i] = np.sum(result == y_test) / n_test_samples
          
        # Obliczenie średniej dokładności dla danej kombinacji kryterium i głębokości drzewa.
        PK_cr_md_vec[criterion_ind,max_depth_ind] = np.mean(PK_vec)
        print("criterion: {} | max_depth: {} | PK: {}".format(criterion_vec[criterion_ind], max_depth_vec[max_depth_ind], PK_cr_md_vec[criterion_ind,max_depth_ind]))

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(criterion_num_vec, max_depth_vec)
surf = ax.plot_surface(X, Y, PK_cr_md_vec.T, cmap='viridis')

ax.set_xlabel('criterion')
ax.set_xticks(np.array(range(len(criterion_vec))))
ax.set_xticklabels(criterion_vec)
ax.set_ylabel('max_depth')
ax.set_zlabel('PK')

ax.view_init(30, 200)  
plt.show()
plt.savefig("Fig.1_dtree_CV_experiment.png",bbox_inches='tight')          