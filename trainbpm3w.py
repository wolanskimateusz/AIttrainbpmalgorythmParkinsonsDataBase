import hickle as hkl
import numpy as np
import nnet as net
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold


class mlp_ma_3w:
    def __init__(
        self,
        x,
        y_t,
        K1,
        K2,
        K3,
        lr,
        err_goal,
        disp_freq,
        mc,
        ksi_inc,
        ksi_dec,
        er,
        max_epoch,
    ):
        self.x = x
        self.L = self.x.shape[0]
        self.y_t = y_t
        self.K1 = K1
        self.K2 = K2
        self.K3 = K3
        self.lr = lr
        self.err_goal = err_goal
        self.disp_freq = disp_freq
        self.mc = mc
        self.ksi_inc = ksi_inc
        self.ksi_dec = ksi_dec
        self.er = er
        self.max_epoch = max_epoch
        self.K4 = y_t.shape[0]
        self.SSE_vec = []
        self.PK_vec = []
        self.w1, self.b1 = net.nwtan(self.K1, self.L)
        self.w2, self.b2 = net.nwtan(self.K2, self.K1)
        self.w3, self.b3 = net.nwtan(self.K3, self.K2)
        self.w4, self.b4 = net.rands(self.K4, self.K3)
        hkl.dump(
            [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3, self.w4, self.b4],
            "wagi4w.hkl",
        )
        (
            self.w1,
            self.b1,
            self.w2,
            self.b2,
            self.w3,
            self.b3,
            self.w4,
            self.b4,
        ) = hkl.load("wagi4w.hkl")
        (
            self.w1_t_1,
            self.b1_t_1,
            self.w2_t_1,
            self.b2_t_1,
            self.w3_t_1,
            self.b3_t_1,
            self.w4_t_1,
            self.b4_t_1,
        ) = (self.w1, self.b1, self.w2, self.b2, self.w3, self.b3, self.w4, self.b4)
        self.SSE = 0
        self.lr_vec = list()

    def predict(self, x):
        self.y1 = net.tansig(np.dot(self.w1, x), self.b1)
        self.y2 = net.tansig(np.dot(self.w2, self.y1), self.b2)
        self.y3 = net.tansig(np.dot(self.w3, self.y2), self.b3)
        self.y4 = net.purelin(np.dot(self.w4, self.y3), self.b4)
        return self.y4

    def train(self, x_train, y_train):
        for epoch in range(1, self.max_epoch + 1):
            self.y4 = self.predict(x_train)
            self.e = y_train - self.y4
            self.SSE_t_1 = self.SSE
            self.SSE = net.sumsqr(self.e)
            self.PK = (
                1 - sum((abs(self.e) >= 0.5).astype(int)[0]) / self.e.shape[1]
            ) * 100  # obliczanie procentowego błędu uczenia
            self.PK_vec.append(self.PK)

            if self.SSE < self.err_goal:  # porównywanie błędu z progiem
                break

            if epoch % 10 == 0:
                print(self.SSE)

            if np.isnan(self.SSE):  # zmiania współczynników uczenia
                break
            else:
                if self.SSE > self.er * self.SSE_t_1:
                    self.lr *= self.ksi_dec
                elif self.SSE < self.SSE_t_1:
                    self.lr *= self.ksi_inc
            self.lr_vec.append(self.lr)
            self.d4 = net.deltalin(self.y4, self.e)
            self.d3 = net.deltatan(self.y3, self.d4, self.w4)
            self.d2 = net.deltatan(self.y2, self.d3, self.w3)
            self.d1 = net.deltatan(self.y1, self.d2, self.w2)
            self.dw1, self.db1 = net.learnbp(self.x, self.d1, self.lr)
            self.dw2, self.db2 = net.learnbp(self.y1, self.d2, self.lr)
            self.dw3, self.db3 = net.learnbp(self.y2, self.d3, self.lr)
            self.dw4, self.db4 = net.learnbp(self.y3, self.d4, self.lr)

            (
                self.w1_temp,
                self.b1_temp,
                self.w2_temp,
                self.b2_temp,
                self.w3_temp,
                self.b3_temp,
                self.w4_temp,
                self.b4_temp,
            ) = (
                self.w1.copy(),
                self.b1.copy(),
                self.w2.copy(),
                self.b2.copy(),
                self.w3.copy(),
                self.b3.copy(),
                self.w4.copy(),
                self.b4.copy(),
            )
            # zmiana wag
            self.w1 += self.dw1 + self.mc * (self.w1 - self.w1_t_1)
            self.b1 += self.db1 + self.mc * (self.b1 - self.b1_t_1)
            self.w2 += self.dw2 + self.mc * (self.w2 - self.w2_t_1)
            self.b2 += self.db2 + self.mc * (self.b2 - self.b2_t_1)
            self.w3 += self.dw3 + self.mc * (self.w3 - self.w3_t_1)
            self.b3 += self.db3 + self.mc * (self.b3 - self.b3_t_1)
            self.w4 += self.dw4 + self.mc * (self.w4 - self.w4_t_1)
            self.b4 += self.db4 + self.mc * (self.b4 - self.b4_t_1)

            # zapisywanie wcześniejszych wag potrzebne przy obliczaniu kolejnych
            (
                self.w1_t_1,
                self.b1_t_1,
                self.w2_t_1,
                self.b2_t_1,
                self.w3_t_1,
                self.b3_t_1,
                self.w4_t_1,
                self.b4_t_1,
            ) = (
                self.w1_temp,
                self.b1_temp,
                self.w2_temp,
                self.b2_temp,
                self.w3_temp,
                self.b3_temp,
                self.w4_temp,
                self.b4_temp,
            )
            self.SSE_vec.append(self.SSE)


x, y_t, x_norm = hkl.load("parkinsons.hkl")

max_epoch = 1000
err_goal = 0.25
disp_freq = 10
lr = 0.00001
mc = 0.95
ksi_inc = 1.05
ksi_dec = 0.8
er = 1.04
K1 = 20
K2 = 22
K3 = 8
data = x_norm.T
target = y_t
CVN = 10
skfold = StratifiedKFold(n_splits=CVN)
PK_vec = np.zeros(CVN)


for i, (train, test) in enumerate(skfold.split(data, np.squeeze(target)), start=0):
    x_train, x_test = data[train], data[test]
    y_train, y_test = np.squeeze(target)[train], np.squeeze(target)[test]

    mlpnet = mlp_ma_3w(
        x_train.T,
        y_train,
        K1,  # K1
        K2,  # K2
        K3,
        lr,
        err_goal,
        disp_freq,
        mc,
        ksi_inc,
        ksi_dec,
        er,
        max_epoch,
    )
    mlpnet.train(x_train.T, y_train.T)
    result = mlpnet.predict(x_test.T)
    n_test_samples = test.size
    PK_vec[i] = (
        1 - sum((abs(result - y_test) >= 0.5).astype(int)[0]) / n_test_samples
    ) * 100
    print("Test #{:<2}: PK_vec {} test_size {} ".format(i, PK_vec[i], n_test_samples))

PK = np.mean(PK_vec)
print("PK {}".format(PK))
