from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target,
                                                    stratify=cancer.target, random_state=66)

training_accuracy = []
test_accuracy = []

# try n_neighbours from 1 to 10
neighbours_settings = range(1, 11)

for n_neighbours in neighbours_settings:
    # build the model
    clf = KNeighborsClassifier(n_neighbors=n_neighbours)
    clf.fit(X_train, y_train)
    # record training accuracy
    training_accuracy(clf.score(X_train, y_train))
    # record testing accuracy
    test_accuracy(clf.score(X_test, y_test))

plt.plot(neighbours_settings, training_accuracy, label="training accuracy")
plt.plot(neighbours_settings, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbours")
plt.legend()
plt.show(block=True)