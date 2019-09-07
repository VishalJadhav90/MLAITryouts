import numpy as np

x = np.array([[-1, -1], [-2, -1], [-3, 0], [1, 2], [1, 1], [2, 2]])
y = np.array([-1, -1, -1, 1, 1, 1])

from sklearn.naive_bayes import GaussianNB
cls = GaussianNB()
cls.fit(x, y)
predictions = cls.predict([[-0.2, -1]])

from sklearn.metrics import accuracy_score
print(accuracy_score(predictions, [-1]))
