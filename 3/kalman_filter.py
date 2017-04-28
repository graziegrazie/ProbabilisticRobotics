import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines  as line

class LinearKalmanFilter:
    def __init__(self):
        self.name = "";
        
        self.matA = np.matrix(np.identity(2))
        self.matB = np.matrix(np.identity(2))
        self.matC = np.matrix(np.identity(2))

        self.noiseQ = np.matrix(np.identity(2))
        self.noiseR = np.matrix(np.identity(2) * 2)

        self.prev_mu    = np.empty((2, 1))
        self.prev_sigma = np.identity(2)

    def predict(self, u):
        tmp_mu    = self.matA * self.prev_mu + self.matB * u
        tmp_sigma = self.matA * self.prev_sigma * self.matA.T + self.noiseQ

        return tmp_mu, tmp_sigma

    def update(self, z, mu, sigma):
        inv = np.linalg.inv(self.matC * sigma * self.matC.T + self.noiseR)
        gain = sigma * self.matC.T * inv
    
        updated_mu    = mu + gain * (z.T - self.matC * mu)
        updated_sigma = (np.matrix(np.identity(2)) - gain * self.matC) * sigma

        return updated_mu, updated_sigma

    def exec_lkf(self, u, z):
        tmp_mu, tmp_sigma = self.predict(u)
        updated_mu, updated_sigma = self.update(z, tmp_mu, tmp_sigma)

        self.prev_mu    = updated_mu
        self.prev_sigma = updated_sigma

if __name__ == '__main__':
    lkf = LinearKalmanFilter()
    
    u = np.matrix(np.array([[2],[2]]))

    x = np.empty((2, 1))
    X = np.matrix(x.T)
    Z = np.matrix(x.T)

    for i in xrange(5):
        x = lkf.matA * x + lkf.matB * u + np.random.multivariate_normal([0, 0], lkf.noiseQ, 1).T
        X = np.append(X, x.T, axis=0)
        
        z = lkf.matC * x + np.random.multivariate_normal([0, 0], lkf.noiseR, 1).T
        Z = np.append(Z, z.T, axis=0)

    MU = np.matrix(np.array([[0, 0]]), float)

    for i in xrange(1, 6):
        lkf.exec_lkf(u, Z[i])
        MU = np.append(MU, lkf.prev_mu.T, axis=0)

    plt.plot(X[:, 0],  X[:, 1],  marker="o", color="red"  , label="state")
    plt.plot(Z[:, 0],  Z[:, 1],  marker="x", color="green", label="measure")
    plt.plot(MU[:, 0], MU[:, 1], marker="*", color="blue" , label="update")
    plt.show()
