import hickle as hkl
import numpy as np
import matplotlib.pyplot as plt

filename = './przygotowanie hkl\haberman.txt'
data = np.loadtxt(filename, delimiter=',', dtype=str)
# Wyciagamy wszystkie cechy z 1,2,3 kolumny i transonujemy aby uzyskac orientacje kolumnową
x = data[:,0:-1].astype(float).T
# wyciagamy wszystkie cechy wynikowe z ostatniej kolumny  
y_t = data[:,-1].astype(float)
# przeksztalcamy aby uzyskac ostnia kolumne w jednym wierszu
y_t = y_t.reshape(1,y_t.shape[0])                                                                   # dla kodowanie klas naturalno-liczbowego


# min i max dla każdej z cech przed normalizacją
# print(np.transpose([np.array(range(x.shape[0])), x.min(axis=1), x.max(axis=1)]))
# normalizacja do przedziału <-1,1> wg zależnoci
# x_norm = (x_norm_max-x_norm_min)*(x-x_min)/(x_max-x_min) + x_norm_min
# w której x_norm_max oraz x_norm_min są docelowymi wartociami rozpiętoci cechy

# wyciagamy najmniejsza wartosc kazdej cechy [30. 58. 0.]
x_min = x.min(axis=1)
print(x_min)
# wyciagamy najwieksza wartosc kazdej cechy [83. 69. 52.]
x_max = x.max(axis=1)
print(x_max)
# ustalenie min, max po znormalizowaniu 
x_norm_max = 1
x_norm_min = -1

# stworzenie tablicy zerowej o takim samym kształcie jak x do przechowywania znormalizowanych danych.
x_norm = np.zeros(x.shape)

print(x.shape[0])

# sprawdzenie rozpiętoci cech przed normalizacją
s = np.transpose([np.array(range(x.shape[0])), x.min(axis=1), x.max(axis=1)])
# print("\n---------------------\n", "Przed normalizacja\n ---------------------\n", s)
print(s)

# przejscie po kazdym wierszu i znormalizowanie danych do przedzialu <-1, 1>
for i in range(x.shape[0]):
    x_norm[i,:] = (x_norm_max-x_norm_min)/(x_max[i]-x_min[i])* \
        (x[i,:]-x_min[i]) + x_norm_min

# sprawdzenie rozpiętoci cech po normalizacji
s = np.transpose([np.array(range(x.shape[0])), x_norm.min(axis=1), x_norm.max(axis=1)])
# print("\n---------------------\n", "Po normalizacji\n ---------------------\n", s)

# sortowanie y_t
y_t_s_ind = np.argsort(y_t)
x_n_s = np.zeros(x.shape)
y_t_s = np.zeros(y_t.shape)
for i in range(x.shape[1]):
    y_t_s[0,i] = y_t[0,y_t_s_ind[0,i]]
    x_n_s[:,i] = x_norm[:,y_t_s_ind[0,i]]
# Do pliku dane są zapisane w orientacji kolumnowej kolumna - proba
plt.plot(y_t_s[0])
plt.show()
hkl.dump([x,y_t,x_norm,x_n_s,y_t_s],'haberman.hkl')
