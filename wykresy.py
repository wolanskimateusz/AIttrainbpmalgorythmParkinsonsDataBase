from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np


#
#   Wykresy K1 K2
#

fig1, ax1 = plt.subplots(subplot_kw={"projection": "3d"})


filename = "PKlist.txt"
data1 = np.loadtxt(filename, delimiter=",", dtype=float)

ax1.set_title("Wykres ilości neuronów i błędu PK")

x1 = data1[:, 0].astype(int)
ax1.set_xlabel("K1")
y1 = data1[:, 1].astype(int)
ax1.set_ylabel("K2")
z1 = data1[:, 2].astype(float)
ax1.set_zlabel("PK")

surf1 = ax1.plot_trisurf(x1, y1, z1, cmap=cm.coolwarm, linewidth=0, antialiased=False)


ax1.zaxis.set_major_locator(LinearLocator(10))
ax1.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))

fig1.colorbar(surf1, shrink=0.5, aspect=5)

#
#   Wykresy ksi_inc i ksi_dec
#

# mc = 0.8

fig2, ax2 = plt.subplots(subplot_kw={"projection": "3d"})


filename = "ksilist0-8mc.txt"
data2 = np.loadtxt(filename, delimiter=",", dtype=float)

ax2.set_title("Wykres zmian ksi dla mc = 0,8")

x2 = data2[:, 0].astype(float)
ax2.set_xlabel("ksi_inc")
y2 = data2[:, 1].astype(float)
ax2.set_ylabel("ksi_dec")
z2 = data2[:, 2].astype(float)
ax2.set_zlabel("PK")

surf2 = ax2.plot_trisurf(x2, y2, z2, cmap=cm.coolwarm, linewidth=0, antialiased=False)


ax2.zaxis.set_major_locator(LinearLocator(10))
ax2.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))

fig2.colorbar(surf2, shrink=0.5, aspect=5)


# mc = 0.85

fig3, ax3 = plt.subplots(subplot_kw={"projection": "3d"})


filename = "ksilist0-85mc.txt"
data3 = np.loadtxt(filename, delimiter=",", dtype=float)

ax3.set_title("Wykres zmian ksi dla mc = 0,85")

x3 = data3[:, 0].astype(float)
ax3.set_xlabel("ksi_inc")
y3 = data3[:, 1].astype(float)
ax3.set_ylabel("ksi_dec")
z3 = data3[:, 2].astype(float)
ax3.set_zlabel("PK")

surf3 = ax3.plot_trisurf(x3, y3, z3, cmap=cm.coolwarm, linewidth=0, antialiased=False)


ax3.zaxis.set_major_locator(LinearLocator(10))
ax3.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))

fig3.colorbar(surf3, shrink=0.5, aspect=5)


# mc = 0.9

fig4, ax4 = plt.subplots(subplot_kw={"projection": "3d"})


filename = "ksilist0-9mc.txt"
data4 = np.loadtxt(filename, delimiter=",", dtype=float)

ax4.set_title("Wykres zmian ksi dla mc = 0,9")

x4 = data4[:, 0].astype(float)
ax4.set_xlabel("ksi_inc")
y4 = data4[:, 1].astype(float)
ax4.set_ylabel("ksi_dec")
z4 = data4[:, 2].astype(float)
ax4.set_zlabel("PK")

surf4 = ax4.plot_trisurf(x4, y4, z4, cmap=cm.coolwarm, linewidth=0, antialiased=False)


ax4.zaxis.set_major_locator(LinearLocator(10))
ax4.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))

fig4.colorbar(surf4, shrink=0.5, aspect=5)


# mc = 0.95

fig5, ax5 = plt.subplots(subplot_kw={"projection": "3d"})


filename = "ksilist0-95mc.txt"
data5 = np.loadtxt(filename, delimiter=",", dtype=float)

ax5.set_title("Wykres zmian ksi dla mc = 0,95")

x5 = data5[:, 0].astype(float)
ax5.set_xlabel("ksi_inc")
y5 = data5[:, 1].astype(float)
ax3.set_ylabel("ksi_dec")
z5 = data5[:, 2].astype(float)
ax5.set_zlabel("PK")

surf5 = ax5.plot_trisurf(x5, y5, z5, cmap=cm.coolwarm, linewidth=0, antialiased=False)


ax5.zaxis.set_major_locator(LinearLocator(10))
ax5.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))

fig5.colorbar(surf5, shrink=0.5, aspect=5)


# mc = 0.99

fig6, ax6 = plt.subplots(subplot_kw={"projection": "3d"})


filename = "ksilist0-99mc.txt"
data6 = np.loadtxt(filename, delimiter=",", dtype=float)

ax6.set_title("Wykres zmian ksi dla mc = 0,99")

x6 = data6[:, 0].astype(float)
ax6.set_xlabel("ksi_inc")
y6 = data6[:, 1].astype(float)
ax6.set_ylabel("ksi_dec")
z6 = data6[:, 2].astype(float)
ax6.set_zlabel("PK")

surf6 = ax6.plot_trisurf(x6, y6, z6, cmap=cm.coolwarm, linewidth=0, antialiased=False)


ax6.zaxis.set_major_locator(LinearLocator(10))
ax6.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))

fig6.colorbar(surf6, shrink=0.5, aspect=5)


plt.show()
