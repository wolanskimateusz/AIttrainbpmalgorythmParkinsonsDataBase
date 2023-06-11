import numpy as np
import hickle as hkl
import matplotlib.pyplot as plt

#
#   PRZYGOTOWANIE DANYCH
#

filename = "parkinsons.txt"
data = np.loadtxt(filename, delimiter=",", dtype=str)


x_old = data[:, 1:-1].astype(float)  # Wczytania danych poza pierszą kolumną
x = np.delete(x_old, 16, axis=1)  # usunięcie kolumny 17 z wynikami
x = x.T  # transpozycja


y_t = data[:, -7].astype(float)  # wczytanie kolumny z wynikami
y_t = y_t.reshape(1, y_t.shape[0])  # nadanie odpowiednich wymiarów macierzy

print(y_t)


x_min = x.min(axis=1)
x_max = x.max(axis=1)
x_norm_max = 1
x_norm_min = 0
x_norm = np.zeros(x.shape)
for i in range(x.shape[0]):
    x_norm[i, :] = (x_norm_max - x_norm_min) / (x_max[i] - x_min[i]) * (
        x[i, :] - x_min[i]
    ) + x_norm_min
# znalezienie minimum i maximum w kolumnie, następnie dla każdej wartości obliczenie jej odpowiednika z przedziału <0,1>


print(
    np.transpose(
        [np.array(range(1, x.shape[0] + 1)), x_norm.min(axis=1), x_norm.max(axis=1)]
    )
)

print(type(y_t))

plt.plot(y_t[0])
plt.show()

hkl.dump([x, y_t, x_norm], "parkinsons.hkl")


# if epoch % 10 == 0:
#    print(self.SSE)
