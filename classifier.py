import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
# from sklearn.neural_network import MLPClassifier

h = .02  # step size in the mesh

names = ["Nearest Neighbors",
        "Linear SVM",
        "Decision Tree",
        "Naive Bayes",
        "Neural Network"
]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    DecisionTreeClassifier(max_depth=5),
    GaussianNB(),
    # MLPClassifier(activation='relu', algorithm='l-bfgs', alpha=1e-05,
    #    batch_size='auto', beta_1=0.9, beta_2=0.999, early_stopping=False,
    #    epsilon=1e-08, hidden_layer_sizes=(5, 2), learning_rate='constant',
    #    learning_rate_init=0.001, max_iter=200, momentum=0.9,
    #    nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
    #    tol=0.0001, validation_fraction=0.1, verbose=False,
    #    warm_start=False)
]
columns = ['id','clump_thickness','unif_cell_shape','marg_adhesion','single_epith_cell','bare_nuclei','bland_chromatin','normal_nucleoli','mitoses','class']
figure = plt.figure(figsize=(27, 9))

df = pd.read_csv('breast-cancer-wisconsin.data.csv')
df.replace('?', -99999, inplace=True)
columns_to_drop = [
    'id',
]
df.drop(columns_to_drop,1,inplace=True)
X = np.array(df[['clump_thickness','unif_cell_shape']])
print X.shape
y = np.array(df['class'])

X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

i = 1

x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# just plot the dataset first
cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])
ax = plt.subplot(1, len(classifiers) + 1, i)
# Plot the training points
ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
# and testing points
ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6)
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_xticks(())
ax.set_yticks(())
ax.set_title("Full Dataset")

i += 1

# iterate over classifiers
for name, clf in zip(names, classifiers):
    ax = plt.subplot(1, len(classifiers) + 1, i)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    # print clf
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    if hasattr(clf, "decision_function"):
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    print np.unique(Z)
    ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

    # Plot also the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
    # and testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
               alpha=0.6)

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(name)
    ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
            size=15, horizontalalignment='right')
    i += 1
figure.subplots_adjust(left=.02, right=.98)
plt.show()
