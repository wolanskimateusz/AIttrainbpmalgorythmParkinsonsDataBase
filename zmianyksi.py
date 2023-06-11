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
        self.lr = lr
        self.err_goal = err_goal
        self.disp_freq = disp_freq
        self.mc = mc
        self.ksi_inc = ksi_inc
        self.ksi_dec = ksi_dec
        self.er = er
        self.max_epoch = max_epoch
        self.K3 = y_t.shape[0]
        self.SSE_vec = []
        self.PK_vec = []
        self.w1, self.b1 = net.nwtan(self.K1, self.L)
        self.w2, self.b2 = net.nwtan(self.K2, self.K1)
        self.w3, self.b3 = net.rands(self.K3, self.K2)
        hkl.dump([self.w1, self.b1, self.w2, self.b2, self.w3, self.b3], "wagi3w.hkl")
        self.w1, self.b1, self.w2, self.b2, self.w3, self.b3 = hkl.load("wagi3w.hkl")
        self.w1_t_1, self.b1_t_1, self.w2_t_1, self.b2_t_1, self.w3_t_1, self.b3_t_1 = (
            self.w1,
            self.b1,
            self.w2,
            self.b2,
            self.w3,
            self.b3,
        )
        self.SSE = 0
        self.lr_vec = list()

    def predict(self, x):
        self.y1 = net.tansig(np.dot(self.w1, x), self.b1)
        self.y2 = net.tansig(np.dot(self.w2, self.y1), self.b2)
        self.y3 = net.purelin(np.dot(self.w3, self.y2), self.b3)
        return self.y3

    def train(self, x_train, y_train):
        for epoch in range(1, self.max_epoch + 1):
            self.y3 = self.predict(x_train)
            self.e = y_train - self.y3
            self.SSE_t_1 = self.SSE
            self.SSE = net.sumsqr(self.e)
            self.PK = (
                1 - sum((abs(self.e) >= 0.5).astype(int)[0]) / self.e.shape[1]
            ) * 100
            self.PK_vec.append(self.PK)

            if self.SSE < self.err_goal:
                break

            if epoch % 10 == 0:
                print(self.SSE)

            if np.isnan(self.SSE):
                break
            else:
                if self.SSE > self.er * self.SSE_t_1:
                    self.lr *= self.ksi_dec
                elif self.SSE < self.SSE_t_1:
                    self.lr *= self.ksi_inc
            self.lr_vec.append(self.lr)
            self.d3 = net.deltalin(self.y3, self.e)
            self.d2 = net.deltatan(self.y2, self.d3, self.w3)
            self.d1 = net.deltatan(self.y1, self.d2, self.w2)
            self.dw1, self.db1 = net.learnbp(self.x, self.d1, self.lr)
            self.dw2, self.db2 = net.learnbp(self.y1, self.d2, self.lr)
            self.dw3, self.db3 = net.learnbp(self.y2, self.d3, self.lr)

            (
                self.w1_temp,
                self.b1_temp,
                self.w2_temp,
                self.b2_temp,
                self.w3_temp,
                self.b3_temp,
            ) = (
                self.w1.copy(),
                self.b1.copy(),
                self.w2.copy(),
                self.b2.copy(),
                self.w3.copy(),
                self.b3.copy(),
            )

            self.w1 += self.dw1 + self.mc * (self.w1 - self.w1_t_1)
            self.b1 += self.db1 + self.mc * (self.b1 - self.b1_t_1)
            self.w2 += self.dw2 + self.mc * (self.w2 - self.w2_t_1)
            self.b2 += self.db2 + self.mc * (self.b2 - self.b2_t_1)
            self.w3 += self.dw3 + self.mc * (self.w3 - self.w3_t_1)
            self.b3 += self.db3 + self.mc * (self.b3 - self.b3_t_1)

            (
                self.w1_t_1,
                self.b1_t_1,
                self.w2_t_1,
                self.b2_t_1,
                self.w3_t_1,
                self.b3_t_1,
            ) = (
                self.w1_temp,
                self.b1_temp,
                self.w2_temp,
                self.b2_temp,
                self.w3_temp,
                self.b3_temp,
            )
            self.SSE_vec.append(self.SSE)


x, y_t, x_norm = hkl.load("parkinsons.hkl")

max_epoch = 1000
err_goal = 0.25
disp_freq = 10
lr = 0.00001
mc = 0.99  # zmiany
ksi_inc = 1.05  # zmiany
ksi_dec = 0.7  # zmiany
er = 1.04
K1 = 20
K2 = 22
data = x_norm.T
target = y_t
CVN = 10
skfold = StratifiedKFold(n_splits=CVN)
PK_vec = np.zeros(CVN)
PK_list = []
ksi_inc_list = [0.8, 0.9, 1, 1.05, 1.1, 1.15, 1.2]
ksi_dec_list = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]


for ki in ksi_inc_list:
    for kj in ksi_dec_list:
        for i, (train, test) in enumerate(
            skfold.split(data, np.squeeze(target)), start=0
        ):
            x_train, x_test = data[train], data[test]
            y_train, y_test = np.squeeze(target)[train], np.squeeze(target)[test]

            mlpnet = mlp_ma_3w(
                x_train.T,
                y_train,
                K1,  # K1
                K2,  # K2
                lr,
                err_goal,
                disp_freq,
                mc,
                ki,  # ksi_inc
                kj,  # ksi_dec
                er,
                max_epoch,
            )
            mlpnet.train(x_train.T, y_train.T)
            result = mlpnet.predict(x_test.T)
            n_test_samples = test.size
            PK_vec[i] = (
                1 - sum((abs(result - y_test) >= 0.5).astype(int)[0]) / n_test_samples
            ) * 100
            print(
                "Test #{:<2}: PK_vec {} test_size {} K1:{} K2:{}".format(
                    i, PK_vec[i], n_test_samples, ki, kj
                )
            )
        PK_by_neurons = (ki, kj, np.mean(PK_vec))
        PK_list.append(PK_by_neurons)

# PK = np.mean(PK_vec)
# print("PK {}".format(PK))

for i in PK_list:
    print("ksi_inc:", i[0], " ksi_dec:", i[1], " PK:", i[2])
    np.savetxt("ksilist0-99mc.txt", PK_list, fmt="%2.6f", delimiter=",")
