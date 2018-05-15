from sklearn.naive_bayes import GaussianNB
from sklearn import datasets, metrics

iris = datasets.load_iris()
classifier = GaussianNB()
classifier.fit(iris.data, iris.target)
score = metrics.accuracy_score(iris.target, classifier.predict(iris.data))
print("Accuracy: %f" % score)

newfeatures=[[4.9,3.1,1.5,0.2],[5,3,1.5,0.2],[6,2.5,1.5,0.2],[4.2,1.3,1.7,0],[5.15,2.9,3.5,1.2]]
predictions=classifier.predict(newfeatures)
for p in predictions:
	print("Prediction: %i" % p);