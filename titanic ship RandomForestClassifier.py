from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


X = pd.read_csv('http://bit.ly/kaggletrain')
X.head()
X.shape
y = X.pop('Survived')
X.describe()
# notice that I have missing data of the kaggletrain, So fill missed age with the average of the average
X['Age'].fillna(X.Age.mean(), inplace=True)
X.describe()  # now it is 891 data


numeric_variable = list(X.dtypes[X.dtypes != 'object'].index)
X[numeric_variable].head()

X_train, X_test, y_train, y_test = train_test_split(
    X[numeric_variable], y, random_state=0)


model = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=42)
model.fit(X_train, y_train)
model.score(X_train, y_train)  # 1
model.score(X_test, y_test)  # .73


def describe_catagorial(X):
    from IPython.display import display, HTML
    display(HTML(X[X.columns[X.dtypes == 'object']].describe().to_html()))


describe_catagorial(X)

X.drop(['Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)
X.head()


def clean_cabin(X):
    try:
        return X[0]
    except TypeError:
        return 'None'


X["Cabin"] = X.Cabin.apply(clean_cabin)
X.head()

X.Cabin
describe_catagorial(X)  # now I have 891 caib not 204 after filling no cabin with None

catagorial_variable = ["Sex", "Cabin", "Embarked"]

for Variable in catagorial_variable:
    # fill missing data with the word missing:
    X[Variable].fillna('Missing', inplace=True)
    # create array of dummies
    dummies = pd.get_dummies(X[Variable], prefix=Variable)
    # update X to include dummise and drop orignal variable
    X = pd.concat([X, dummies], axis=1)
    X.drop([Variable], axis=1, inplace=True)

# now print X
X.head()
X.shape

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0)


model_cat = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=42)
model_cat.fit(X_train, y_train)
model_cat.score(X_train, y_train)  # .98
model_cat.score(X_test, y_test)  # .83


# Variable importance mesures
model_cat.feature_importances_
# now we want to plot it to ease the determination:
feature_imortances = pd.Series(model_cat.feature_importances_, index=X.columns)
feature_imortances
feature_imortances.sort_values()  # no effect :(
feature_imortances.plot(kind='barh', figsize=(7, 6))


training_accuracy = []
test_accuracy = []
n_estimators_option = [30, 50, 100, 200, 500, 1000, 2000]
for trees in n_estimators_option:
    model = RandomForestClassifier(trees, oob_score=True, n_jobs=1, random_state=42)
    model.fit(X_train, y_train)
    training_accuracy.append(model.score(X_train, y_train))
    test_accuracy.append(model.score(X_test, y_test))

plt.plot(n_estimators_option, training_accuracy, label='Accuracy of the training set')
plt.plot(n_estimators_option, test_accuracy, label='Accuracy of the test set')


training_accuracy = []
test_accuracy = []
max_feature_option = ["auto", None, "sqrt", "log2", .9, .2]
for max_features in max_feature_option:
    model = RandomForestClassifier(n_estimators=500, oob_score=True,
                                   n_jobs=-1, random_state=42, max_features=max_features)
    model.fit(X_train, y_train)
    training_accuracy.append(model.score(X_train, y_train))
    test_accuracy.append(model.score(X_test, y_test))
training_accuracy

test_accuracy

pos = np.arange(6)+0.5  # +.5 desn't effect
names = max_feature_option
plt.barh(pos, training_accuracy, align="center", color="magenta")  # allign center dosen't effect
plt.xlabel("accuracy score", color="Red")
plt.ylabel("max feature option", color="Red")
plt.title("score vs max feature koption", color="blue")
plt.tick_params(axis="x", colors="k")
plt.tick_params(axis="y", colors="k")
plt.yticks(pos, names)
plt.show()


training_accuracy = []
test_accuracy = []
min_samples_leaf_option = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
for min_samples in min_samples_leaf_option:
    model = RandomForestClassifier(n_estimators=500, oob_score=True,
                                   n_jobs=-1, random_state=42, max_features=.2, min_samples_leaf=min_samples)
    model.fit(X_train, y_train)
    training_accuracy.append(model.score(X_train, y_train))
    test_accuracy.append(model.score(X_test, y_test))

test_accuracy
training_accuracy


pos = np.arange(10)+0.5  # +.5 desn't effect
names = min_samples_leaf_option
plt.barh(pos, training_accuracy, align="center", color="magenta")  # allign center dosen't effect
plt.xlabel("accuracy score", color="Red")
plt.ylabel("min_samples_leaf_option", color="Red")
plt.title("score vs min_samples_leaf_optionn", color="blue")
plt.tick_params(axis="x", colors="k")
plt.tick_params(axis="y", colors="k")
plt.yticks(pos, names)
plt.show()


# Final Model
X_test = pd.read_csv('D:/python/scikit-learn/CSVs/test.csv')
X_test.describe()
X_test['Age'].fillna(X_test.Age.mean(), inplace=True)
X_test['Fare'].fillna(X_test.Fare.mean(), inplace=True)
X_test.describe()  # now it is 891 data
X_test.dtypes
describe_catagorial(X_test)

X_test.drop(['Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)
X_test.head()
X_test.dtypes


def describe_catagorial(X_test):
    from IPython.display import display, HTML
    display(HTML(X_test[X_test.columns[X_test.dtypes == 'object']].describe().to_html()))


describe_catagorial(X_test)


def clean_cabin(X_test):
    try:
        return X_test[0]
    except TypeError:
        return 'None'


X_test["Cabin"] = X_test.Cabin.apply(clean_cabin)
describe_catagorial(X_test)

X_test.Cabin
describe_catagorial(X_test)  # now I have 891 caib not 204 after filling no cabin with None


catagorial_variable_X_test = ["Sex", "Cabin", "Embarked"]

for Variable in catagorial_variable_X_test:
    # fill missing data with the word missing:
    X_test[Variable].fillna('Missing', inplace=True)
    # create array of dummies
    dummies = pd.get_dummies(X_test[Variable], prefix=Variable)
    # update X to include dummise and drop orignal variable
    X_test = pd.concat([X_test, dummies], axis=1)
    X_test.drop([Variable], axis=1, inplace=True)

# now print X
X_test.head()
X_test.shape
X.drop(['Cabin_T', 'Embarked_Missing'], axis=1, inplace=True)
X.columns
X.dtypes
X_test.dtypes


model = RandomForestClassifier(n_estimators=1000, oob_score=True, n_jobs=-1,
                               random_state=42, max_features=.2, min_samples_leaf=2)
model.fit(X, y)
model.score(X, y)


answer = model.predict(X_test)  # highe number refer to the class
answer

df = pd.DataFrame(answer, columns=["solution"])
df
X_test_raw = pd.read_csv('D:/python/scikit-learn/CSVs/test.csv')

result = pd.concat([X_test_raw, df], axis=1, sort=False)
result

result.columns
result.drop(['Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket',
             'Fare', 'Cabin', 'Embarked'], axis=1, inplace=True)
result.head()
result.set_index('PassengerId', inplace=True)
result.to_csv("titanic RandomForestClassifier answer.csv")
