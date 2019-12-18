# Chapter 3: Classification
# https://learning.oreilly.com/library/view/hands-on-machine-learning/9781492032632/ch03.html

#%%
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.base import clone

# %%
# Open SSL error. How do?
mnist = fetch_openml("mnist_784", version=1)

X, y = mnist["data"], mnist["target"]

# %%
some_digit = X[0]
some_digit_image = some_digit.reshape(28, 28)

plt.imshow(some_digit_image, cmap="binary")
plt.axis("off")
plt.show()

# %%
y = y.astype(np.uint8)
# Data is pre-shuffled per sklearn datasets
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# %%
# Beginning with a binary classifier to simplify the problem.
# 5 and not 5
y_train_5 = y_train == 5
y_test_5 = y_test == 5

# %%
# random_state allows us to have reproducible results
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)

# %%
sgd_clf.predict([some_digit])

# %%
# Re-implementing cross validation

# StratifiedKFold performs stratified sampling to produce folds
# that contain a representative ratio of each class.
skfolds = StratifiedKFold(n_splits=3, random_state=42)

for train_index, test_index in skfolds.split(X_train, y_train_5):
  # Create a clone of the classifier, train that clone on training
  # folds, and finally make predictions on the test fold
  clone_clf = clone(sgd_clf)
  X_train_folds = X_train[train_index]
  y_train_folds = y_train_5[train_index]
  X_test_fold = X_train[test_index]
  y_test_fold = y_train_5[test_index]

  clone_clf.fit(X_train_folds, y_train_folds)
  y_pred = clone_clf.predict(X_test_fold)
  n_correct = sum(y_pred == y_test_fold)
  print(n_correct / len(y_pred))

# %%
cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")