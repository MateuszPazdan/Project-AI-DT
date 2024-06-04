import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
import hickle as hkl
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt


# Wczytanie danych z pliku hickle
x, y_t, x_norm, x_n_s, y_t_s = hkl.load('./badanie_metaparamterow/haberman.hkl')

# Zmiana wartości cechy wyjściowej z 1,2 na 0,1
y_t_s -= 1

# Przetransponowanie cech wejściowych na orientacje wierszową
x = x_n_s.T

# Przetransponowanie cechy wyjściowej na orientacje wierszową
y_t = np.squeeze(y_t_s)

# Definicja metaparametrów do badania
min_impurity_decrease_vec = np.linspace(0, 1, 10)
ccp_alpha_vec = np.linspace(0, 1, 10)

data = x
target = y_t

print('x', data)
print('y',target)

# Ustalenie ilości części (foldów) na którą zostaną podzielone dane
CVN = 10
skfold = StratifiedKFold(n_splits=CVN)

# Stworzenie pustego wektora na dane wyjściowe dla każdej kombinacji metaparametrów
PK_cr_md_vec = np.zeros([len(min_impurity_decrease_vec), len(ccp_alpha_vec)])

for min_impurity_decrease_ind in range(len(min_impurity_decrease_vec)):
    for ccp_alpha_ind in range(len(ccp_alpha_vec)):
        PK_vec = np.zeros(CVN)
        for i, (train, test) in enumerate(skfold.split(data, target), start=0):
            x_train, x_test = data[train], data[test]
            y_train, y_test = target[train], target[test]
            

            # Inicjalizacja klasyfikatora drzewa decyzyjnego z odpowiednimi parametrami.
            decision_tree = DecisionTreeClassifier(
                random_state=0,
                max_depth=1,
                criterion='entropy',
                splitter='random',
                min_samples_split=2,
                min_samples_leaf=1,
                min_weight_fraction_leaf=0.03,
                max_features=3,
                max_leaf_nodes=10,
                min_impurity_decrease=min_impurity_decrease_vec[min_impurity_decrease_ind],
                ccp_alpha=ccp_alpha_vec[ccp_alpha_ind]
            )
            decision_tree = decision_tree.fit(x_train, y_train)
            result = decision_tree.predict(x_test)

            if min_impurity_decrease_vec[min_impurity_decrease_ind] == 0:
                plot_tree(decision_tree)

            # Liczba próbek w zbiorze testowym
            n_test_samples = test.size
            # Obliczenie dokładności predykcji dla tego podziału.
            PK_vec[i] = np.sum(result == y_test) / n_test_samples

        # Obliczenie średniej dokładności dla danej kombinacji kryterium i głębokości drzewa.
        PK_cr_md_vec[min_impurity_decrease_ind, ccp_alpha_ind] = np.mean(PK_vec)
        print("min_impurity_decrease: {} | ccp_alpha: {} | PK: {}".format(
            min_impurity_decrease_vec[min_impurity_decrease_ind],
            ccp_alpha_vec[ccp_alpha_ind],
            PK_cr_md_vec[min_impurity_decrease_ind, ccp_alpha_ind]
        ))

# Wizualizacja wyników
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(min_impurity_decrease_vec, ccp_alpha_vec)
surf = ax.plot_surface(X, Y, PK_cr_md_vec.T, cmap='viridis')

ax.set_xlabel('min_impurity_decrease')
ax.set_ylabel('ccp_alpha')
ax.set_zlabel('PK')

ax.view_init(30, 200)
plt.show()
plt.savefig("Fig.5_dtree_CV_experiment_min_impurity_decrease_ccp_alpha.png", bbox_inches='tight')
