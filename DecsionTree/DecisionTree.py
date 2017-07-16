from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn import linear_model

#[Height, Weight, Shoesize]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

# Decision Tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X,Y)
prediction = clf.predict([[181, 80, 43]])
print (prediction)

# Naive bayes
nb = GaussianNB()
nb = nb.fit(X,Y)
prediction = nb.predict([[181, 80, 43]])
print (prediction)

# Logistic Regression
clf = linear_model.SGDClassifier()
clf.fit(X, Y)
prediction = clf.predict([[181, 80, 43]])
print (prediction)

# Support Vector Machine
clf = svm.SVC()
clf.fit(X, Y) 
prediction = clf.predict([[181, 80, 43]])
print (prediction)