# Chapter 3: Classification
# https://learning.oreilly.com/library/view/hands-on-machine-learning/9781492032632/ch03.html

#%%
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.base import clone, BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
    roc_curve,
    roc_auc_score,
)

# %%
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

# %% [markdown]
# # Binary Classification
# Simplifying the classification problem by making it 5 or not 5, instead of
# predicting all of the numbers in the dataset.

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
# array([0.95035, 0.96035, 0.9604 ])

# Don't get too excited about the 95% accuracy

# %%
# Classifier that guesses that an image is not a 5
class BadClassifier(BaseEstimator):
    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)


cross_val_score(BadClassifier(), X_train, y_train_5, cv=3, scoring="accuracy")
# array([0.91125, 0.90855, 0.90915])
# Only about 10% of the images are 5s, so guessing not a
# 5 is right 90% of the time.

# Accuracy is not the generally preferred metric for this reason.


# %%
# Perform K-fold cross-validation but return predictions made on each
# test fold instead of evaluation scores.
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)

# This ensures that we never touch the test set (to avoid bias) but still
# get predictions on data that the model hasn't seen during training.

# %%
confusion_matrix(y_train_5, y_train_pred)
# array([[53892,   687],
#        [ 1891,  3530]], dtype=int64)
#
# [[true negative, false positive],
#  [false negative, true positive]]

# %%
# Perfect predictions look like the following:
confusion_matrix(y_train_5, y_train_5)


# %%
# Precision = Accuracy of positive predictions, TP / (TP + FP)
print("precision:", precision_score(y_train_5, y_train_pred))

# Recall = Ratio of positive instances correctly identified, TP / (TP + FN)
print("recall:", recall_score(y_train_5, y_train_pred))

# Depending on the project, you may wish to have high precision and low recall, or
# vise-versa. F1 score is bad in these situations, since it favors
# models with similar precision and recall.

# F1 Score = Harmonic mean of precision and recall (more weight to low values)
print("f1:", f1_score(y_train_5, y_train_pred))


# %%
# We can tinker with the model's threshold (default 0) to have it favor
# precision or recall.

y_scores = cross_val_predict(
    sgd_clf, X_train, y_train_5, cv=3, method="decision_function"
)

precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")


plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()


# %%

threshold_90_precision = thresholds[np.argmax(precisions >= 0.90)]
y_train_pred_90 = y_scores >= threshold_90_precision

print("precision:", precision_score(y_train_5, y_train_pred_90))
print("recall:", recall_score(y_train_5, y_train_pred_90))


#%% [markdown]
# # ROC Curve
# Similar to precision/recall curve, but plots true positive rate (recall)
# against false positive rate (TPR vs. FPR).
#
# The higher the TPR (recall), the more false positives (FPR).
#
# ## AUC
# "Area under the curve". A perfect classifier has a ROC AUC of 1.


# %%
fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)


def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], "k--")


# The higher the TPR (recall), the more false positives (FPR).
# A good classifier stays towards the top-left corner.
plot_roc_curve(fpr, tpr)
plt.show()


# %%
roc_auc_score(y_train_5, y_scores)


# %%
# Let's compare to a random forest classifier
#
# Note that the random forest classifier doesn't have a decision_function() method,
# but sues predict_proba() instead.
#
# The predict_proba() method returns an array containing a row per instance and
# a column per class, each containing the probability that the given instance
# belongs to the given class (e.g., 70% chance that the image represents a 5).

forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(
    forest_clf, X_train, y_train_5, cv=3, method="predict_proba"
)

y_scores_forest = y_probas_forest[:, 1]  # score = proba of positive class
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5, y_scores_forest)

plt.plot(fpr, tpr, "b:", label="SGD")
plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
plt.legend(loc="lower right")
plt.show()


# %% [markdown]
# # Multiclass Classification
