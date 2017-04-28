import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines  as line

class LinearKalmanFilter:
    def __init__(self):
        self.name = "";
        
        self.matA = np.matrix(np.identity(2))
        self.matB = np.matrix(np.identity(2))
        self.matC = np.matrix(np.random.rand(2, 2))

        self.noiseQ = np.matrix(np.identity(2))
        self.noiseR = np.matrix(np.identity(2) * 2)

        self.prev_mu = np.empty((2, 1))
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
    z = np.matrix(np.array([[2 * j for i in xrange(2)] for j in xrange(0, 6)]))

    x = np.empty((2, 1))
    X = np.array([x])
    Y = np.array([x])

    mu_list = np.empty((0, 2), float)

    for i in xrange(1, 6):
        lkf.exec_lkf(u, z[i])
        mu_list = np.append(mu_list, lkf.prev_mu.T, axis=0)

    plt.plot(z[:, 0],       z[:, 1],       marker="o", color="red" , label="predict")
    plt.plot(mu_list[:, 0], mu_list[:, 1], marker="x", color="blue", label="update")
    plt.show()
