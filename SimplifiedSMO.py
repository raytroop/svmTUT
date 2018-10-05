"""
This is an implementatin of Sequential minimal optimization, SMO
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.base import BaseEstimator
from utils import plot_svc_decision_boundary
# to make this notebook's output stable across runs
np.random.seed(42)


class MySMO(BaseEstimator):
    def __init__(self, C=2, max_passes=10000, tol=.001):
        self.C = C
        self.max_passes = max_passes
        self.tol = tol

    def fit(self, X, y):
        print(X.shape, y.shape)
        print(np.unique(y))

        def calc_f(xk, alphas, X, y, b):
            return np.dot(alphas * y, X.dot(xk)) + b

        def calc_E(xk, yk, alphas, X, y, b):
            return calc_f(xk, alphas, X, y, b) - yk

        def calc_L(yi, yj, ai, aj, C):
            if yi != yj:
                return max(0, aj - ai)
            else:
                return max(0, ai + aj - C)

        def calc_H(ai, aj, yi, yj, C):
            if yi != yj:
                return min(C, aj - ai + C)
            else:
                return min(C, ai + aj)

        def update_aj(aj, yj, Ei, Ej, eta, H, L):
            """Eq. 12 and 15"""
            aj = aj - yj * (Ei - Ej) / eta
            if aj > H:
                return H
            elif aj < L:
                return L
            else:
                return aj

        def update_ai(ai, yi, yj, aj_old, aj):
            """Eq. 16"""
            return ai + yi * yj * (aj_old - aj)

        def calc_b(b1, b2, ai, aj, C):
            if 0 < ai < C:
                return b1
            elif 0 < aj < C:
                return b2
            else:
                return (b1 + b2) / 2

        m = X.shape[0]
        alphas = np.zeros(m)
        b = 0
        passes = 0
        y = y * 2 - 1  # -1 if t==0, +1 if t==1
        while passes < self.max_passes:
            #     print(passes, end=',')
            num_changed_alphas = 0
            for i in range(m):
                ai = alphas[i]
                xi = X[i]
                yi = y[i]
                Ei = calc_E(xi, yi, alphas, X, y, b)
                if (yi * Ei < -self.tol and ai < self.C) or (yi * Ei > self.tol and ai > 0):
                    j = np.random.choice([_ for _ in range(m) if _ != i])
                    aj = alphas[j]
                    xj = X[j]
                    yj = y[j]
                    Ej = calc_E(xj, yj, alphas, X, y, b)

                    ai_old = ai
                    aj_old = aj

                    L = calc_L(ai, aj, yi, yj, self.C)
                    H = calc_H(ai, aj, yi, yj, self.C)
                    if L == H:
                        continue

                    eta = 2 * xi.dot(xj) - xi.dot(xi) - xj.dot(xj)
                    if eta >= 0:
                        continue

                    aj = update_aj(aj, yj, Ei, Ej, eta, H, L)
                    alphas[j] = aj
                    if np.abs(aj - aj_old) < 1e-5:
                        continue

                    ai = update_ai(ai, yi, yj, aj_old, aj)
                    alphas[i] = ai

                    # Eq. 17
                    b1 = b - Ei - yi * (ai - ai_old) * xi.dot(xi) - \
                        yj * (aj - aj_old) * xi.dot(xj)
                    # Eq. 18
                    b2 = b - Ej - yi * (ai - ai_old) * xi.dot(xj) - \
                        yj * (aj - aj_old) * xj.dot(xj)
                    b = calc_b(b1, b2, ai, aj, self.C)
                    num_changed_alphas += 1
            if num_changed_alphas == 0:
                passes += 1
            else:
                passes = 0

        self.intercept_ = np.array([b])
        w = np.sum(alphas[:, None] * y[:, None] * X, axis=0).reshape(-1, 1)
        self.coef_ = np.array([w])
        # support vector definiton
        # support_vectors_idx = (y[:, None] * (X.dot(w) + b) < 1).ravel()
        support_vectors_idx = (alphas > 0).ravel()
        self.support_vectors_ = X[support_vectors_idx]
        return self

    def decision_function(self, X):
        return X.dot(self.coef_[0]) + self.intercept_[0]

    def predict(self, X):
        return (self.decision_function(X) >= 0).astype(np.float64)


# toy dataset
X = np.array([
    [0.5, 0.5],
    [1, 1],
    [1.5, 2],
    [2, 1]
])
y = np.array([0, 0, 1, 1])

C = 1e8
svm_clf = MySMO(C=C)
svm_clf.fit(X, y)

yr = y.ravel()
plt.figure(figsize=(12, 3.2))
plt.subplot(121)
plt.plot(X[:, 0][yr == 1], X[:, 1][yr == 1], "g^", label="p")
plt.plot(X[:, 0][yr == 0], X[:, 1][yr == 0], "bs", label="n")
plot_svc_decision_boundary(svm_clf, 0, 3)
plt.xlabel("x0", fontsize=14)
plt.ylabel("x1", fontsize=14)
plt.title("MySMO", fontsize=14)

svm_clf2 = SVC(kernel="linear", C=C)
svm_clf2.fit(X, y)
plt.subplot(122)
plt.plot(X[:, 0][yr == 1], X[:, 1][yr == 1], "g^")
plt.plot(X[:, 0][yr == 0], X[:, 1][yr == 0], "bs")
plot_svc_decision_boundary(svm_clf2, 0, 3)
plt.xlabel("x0", fontsize=14)
plt.title("SVC", fontsize=14)
plt.show()

### Simplified SMO can NOT handle large datasets and ensure that the algorithm converges #########
### full SMO algorithm needed

# # Training set
# iris = datasets.load_iris()
# X = iris["data"][:, (2, 3)]  # petal length, petal width
# y = (iris["target"] == 2).astype(np.float64).reshape(-1, 1)  # Iris-Virginica
# yr = y.ravel()

# plt.figure(figsize=(12, 3.2))
# plt.subplot(121)
# svm_clf = MySMO()
# svm_clf.fit(X, yr)
# plt.plot(X[:, 0][yr == 1], X[:, 1][yr == 1], "g^", label="Iris-Virginica")
# plt.plot(X[:, 0][yr == 0], X[:, 1][yr == 0], "bs", label="Not Iris-Virginica")
# plot_svc_decision_boundary(svm_clf, 4, 6)
# plt.xlabel("Petal length", fontsize=14)
# plt.ylabel("Petal width", fontsize=14)
# plt.title("MyLinearSVC", fontsize=14)
# plt.axis([4, 6, 0.8, 2.8])
# plt.show()
