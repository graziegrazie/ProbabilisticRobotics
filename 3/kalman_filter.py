import numpy as np
import matplotlib.pyplot as plt

class linear_kalman_filter:
    def __init__(self):
        self.A = np.matrix([[  1,   0],
                            [  0,   1]])
        self.B = np.matrix([[0.1,   0],
                            [  0, 0.1]])
        self.C = np.matrix([[  1,   0],
                            [  0,   1]])

        self.Q = np.matrix([[1, 0],
                            [0, 1]])
        self.R = np.matrix([[2, 0],
                            [0, 2]])

    def predict(self, prev_mu, prev_sigma, u):
        mu_    = self.A * prev_mu    + self.B * u
        sigma_ = self.A * prev_sigma * self.A.T + self.R
        
        return mu_, sigma_

    def update(self, mu_, sigma_, z):
        inv = np.linalg.inv(self.C * sigma_ * self.C.T + self.Q)
        
        K     = sigma_ * self.C.T * inv
        mu    = mu_ + K * (z - self.C * mu_)
        sigma = (np.identity(2) - K * self.C) * sigma_

        return mu, sigma

if __name__ == '__main__':
    lkf  = linear_kalman_filter()

    mu0    = np.matrix([0, 0]).T
    sigma0 = np.matrix([[0, 0],
                        [0, 0]])

    u0 = np.matrix([10, 10])
    U  = np.matrix(np.array(u0), float)

    X  = np.array([[0, 0]])
    Z  = np.array([[0, 0]])
    MU = np.array([[0, 0]])

    # following block store points of parabola curve in X & Z
    for t in np.arange(0.1, 2, 0.1):
        u = np.array([[10, 10 - 10 * t]])
        U = np.append(U, u, axis=0)

        x = np.array([[10*t, 10*t - 0.5 * 10 * t**2]])
        X = np.append(X, x, axis=0)
                      
        z = x + np.random.multivariate_normal([0, 0], np.identity(2), 1)
        Z = np.append(Z, z, axis=0)

    prev_mu    = mu0
    prev_sigma = sigma0

    # following block
    for t in xrange(0, 20):
        mu_, sigma_ = lkf.predict(prev_mu, prev_sigma, U[t].T)
        mu,  sigma  = lkf.update(mu_,      sigma_,     np.matrix(Z[t]).T)

        prev_mu    = mu
        prev_sigma = sigma

        MU = np.append(MU, np.array([[mu.item(0), mu.item(1)]]), axis=0)

    plt.plot( X[:, 0],  X[:, 1], "+-", color="red",   label="ideal curve")
    plt.plot( Z[:, 0],  Z[:, 1], "o-", color="green", label="measured curve")
    plt.plot(MU[:, 0], MU[:, 1], "x-", color="blue",  label="LKFed curve")
    plt.legend()
    plt.title("Fig. Various Parabola Curve")
    plt.xlabel("Px[m]")
    plt.ylabel("Py[m]")
    plt.show()
