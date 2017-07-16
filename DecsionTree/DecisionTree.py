from sklearn import tree

#[Height, Weight, Shoesize]
X = [[181, 80, 44],[177,78,42]]
Y = ['male', 'female']

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X,Y)

prediction = clf.predict([[181, 80, 43]])

print (prediction)
