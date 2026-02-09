import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# Simple linearly separable data
np.random.seed(7)
class_1 = np.random.randn(20, 2) + [2, 2]
class_2 = np.random.randn(20, 2) + [-2, -2]

X = np.vstack((class_1, class_2))
y = np.hstack((np.ones(20), -np.ones(20)))

# Train a linear SVM
clf = SVC(kernel="rbf", C=1.0)
clf.fit(X, y)

# Create grid to evaluate decision function
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                     np.linspace(y_min, y_max, 500))
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(8, 6))
# Hyperplane (level 0) and margins (±1)
contours = plt.contour(xx, yy, Z, levels=[-1, 0, 1], colors=["C0", "k", "C1"],
                       linestyles=["--", "-", "--"], linewidths=2)
fmt = {c: label for c, label in zip(contours.levels, ["Margin (-1)", "Hyperplane", "Margin (+1)"])}
plt.clabel(contours, fmt=fmt, inline=True, fontsize=10)

# Data points with support vectors highlighted
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.bwr, edgecolors="k", s=70)
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
            s=120, facecolors='none', edgecolors='yellow', linewidths=2,
            label="Support vectors")

plt.title("Linear SVM: Hyperplane and Margins")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
 