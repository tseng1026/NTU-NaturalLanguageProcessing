import numpy as np
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

def Classify(data, task, type = "SVM"):
	dataTrain, dataTest, lablTrain, lablTest = train_test_split(data, task, test_size=0.25)

	clfy = None
	if type == "KNN":
		clfr = KNeighborsClassifier(n_jobs=4)
		params = {'n_neighbors': [3,5,7,9], 'weights':['uniform', 'distance']}
		clfr = GridSearchCV(clfr, params, cv=3, n_jobs=4)
	
	elif type == "DT":
		clfr = DecisionTreeClassifier(max_depth=800, min_samples_split=5)
		params = {'criterion':['gini','entropy']}
		clfr = GridSearchCV(clfr, params, cv=3, n_jobs=4)

	elif type == "RF":
		clfr = RandomForestClassifier(max_depth=800, min_samples_split=5)
		params = {'n_estimators': [n for n in range(50,200,50)], 'criterion':['gini','entropy'], }
		clfr = GridSearchCV(clfr, params, cv=3, n_jobs=4)

	elif type == "LR":
		clfr = LogisticRegression(multi_class='auto', solver='newton-cg',)
		clfr = GridSearchCV(clfr, {"C":np.logspace(-3,3,7), "penalty":["l2"]}, cv=3, n_jobs=4)
	
	elif type == "SVM":
		clfr = SVC()
		clfr = GridSearchCV(clfr, {'C':[0.001, 0.01, 0.1, 1, 10]}, cv=3, n_jobs=4)

	else:
		print("Wrong Classifier Type!")
		return -1, -1
	
	clfr.fit(dataTrain, lablTrain)
	clfr = clfr.best_estimator_
	
	accuracy = accuracy_score(lablTest, clfr.predict(dataTest))
	score    =       f1_score(np.array(lablTest), np.array(clfr.predict(dataTest)), average='macro')

	return accuracy, score
