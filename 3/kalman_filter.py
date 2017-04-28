import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines  as line

class LinearKalmanFilter:
    def __init__(self):
        self.name = "";
        
        self.matA = np.identity(2)
        self.matB = np.identity(2)
        self.matC = np.random.rand(2, 2)

        self.prev_mu = np.empty((2, 1))
        self.curr_mu = np.empty((2, 1))

        self.noiseQ = np.matrix(np.identity(2))
        self.noiseR = np.matrix(np.identity(2) * 2)

        self.prev_sigma = np.identity(2)
        self.curr_sigma = np.identity(2)

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

    mu_list = np.array([[0, 0]])

    for i in xrange(1, 6):
        print(X[i-1])
        X = np.append(X, lkf.matA * X[i-1] + lkf.matB * u + np.random.multivariate_normal([0, 0], lkf.noiseQ, 1).T)
        Y = np.append(Y, lkf.matC * X[i-1] + np.random.multivariate_normal([0, 0], lkf.noiseR, 1).T)
        
        lkf.exec_lkf(u, z[i])
        mu_list = np.append(mu_list, np.array([[lkf.prev_mu[0]], [lkf.prev_mu[1]]]))

    mu_list = np.delete(mu_list, 0)
    X       = np.delete(X, 0)

    z = z.getA()
    z = np.array([z[i][0] for i in range(len(z))])
    
    plt.plot(z, z,    marker="o", color="red")
    plt.plot(mu_list, marker="x", color="blue")
    plt.plot(X,       marker="|", color="green")
    plt.plot(Y,       marker=".", color="orange")
    plt.show()
