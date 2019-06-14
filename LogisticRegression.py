import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import scale, normalize
from sklearn.preprocessing import Imputer
from sklearn import metrics
import matplotlib.pyplot as plt
import random

train_data = pd.read_csv('C:/Users/Yogesh Kumar Ahuja/Desktop/twitter Spammer Detection/kaggle_train.csv',encoding = "ISO-8859-1")

bot_data = pd.read_csv("C:/Users/Yogesh Kumar Ahuja/Desktop/twitter Spammer Detection/bots_data.csv")
nonbot_data = pd.read_csv("C:/Users/Yogesh Kumar Ahuja/Desktop/twitter Spammer Detection/nonbots_data.csv")
test_data = pd.read_csv("C:/Users/Yogesh Kumar Ahuja/Desktop/twitter Spammer Detection/test.csv",encoding = "ISO-8859-1")


#cleaning the data
train_attr = train_data[['followers_count', 'friends_count', 'listedcount', 'favourites_count', 'statuses_count', 'verified']]
train_label = train_data[['bot']]

bot_attr = bot_data[['followers_count', 'friends_count', 'listedcount', 'favourites_count', 'statuses_count', 'verified']]
bot_label = bot_data[['bot']]

nonbot_attr = nonbot_data[['followers_count', 'friends_count', 'listedcount', 'favourites_count', 'statuses_count', 'verified']]
nonbot_label = nonbot_data[['bot']]

test_attr = test_data[['followers_count', 'friends_count', 'listed_count', 'favourites_count', 'statuses_count', 'verified']]
test_label = test_data[['bot']]

#Training the classifier

logreg = LogisticRegression().fit(train_attr, train_label.as_matrix())

#Training on test data

actual = np.array(test_label)
predicted = logreg.predict(test_attr)
pred = np.array(predicted)

accuracy = accuracy_score(actual, pred) * 100
precision = precision_score(actual, pred) * 100
recall = recall_score(actual, pred) * 100
f1 = f1_score(actual, pred)
auc = roc_auc_score(actual, pred)
print('Accuracy is {:.4f}%\n\
Precision is {:.4f}%\n\
Recall is {:.4f}%\n\
F1 Score is {:.4f}\n\
Area Under Curve is {:.4f}'.format(accuracy, precision, recall, f1, auc))

fpr, tpr, thresholds = metrics.roc_curve(actual, pred)
auc = metrics.auc(fpr, tpr)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b',
label='AUC = %0.2f'% auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

#Training on bot data
actual = np.array(bot_label)
predicted = logreg.predict(bot_attr)
pred = np.array(predicted)

accuracy = accuracy_score(actual, pred) * 100
precision = precision_score(actual, pred) * 100
recall = recall_score(actual, pred) * 100
f1 = f1_score(actual, pred)
print('Accuracy is {:.4f}%\n\
Precision is {:.4f}%\n\
Recall is {:.4f}%\n\
F1 Score is {:.4f}'.format(accuracy, precision, recall, f1))

#Training on non-bot data
actual = np.array(nonbot_label)
predicted = logreg.predict(nonbot_attr)
pred = np.array(predicted)

accuracy = accuracy_score(actual, pred) * 100
precision = precision_score(actual, pred) * 100
recall = recall_score(actual, pred) * 100
f1 = f1_score(actual, pred)
print('Accuracy is {:.4f}%\n\
Precision is {:.4f}%\n\
Recall is {:.4f}%\n\
F1 Score is {:.4f}'.format(accuracy, precision, recall, f1))

#Cross-Validation on training set Train-Test split

train_X, test_X, train_Y, test_Y = train_test_split(train_attr, train_label, test_size=0.4, random_state=0)

X = train_X.as_matrix()
Y = train_Y.as_matrix()

logreg = LogisticRegression().fit(X, Y)

actual = np.array(test_Y)
predicted = logreg.predict(test_X)
pred = np.array(predicted)

accuracy = accuracy_score(actual, pred) * 100
precision = precision_score(actual, pred) * 100
recall = recall_score(actual, pred) * 100
f1 = f1_score(actual, pred)
auc = roc_auc_score(actual, pred)
print('Accuracy is {:.4f}%\n\
Precision is {:.4f}%\n\
Recall is {:.4f}%\n\
F1 Score is {:.4f}\n\
Area Under Curve is {:.4f}'.format(accuracy, precision, recall, f1, auc))

fpr, tpr, thresholds = metrics.roc_curve(actual, pred)
auc = metrics.auc(fpr, tpr)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b',
label='AUC = %0.2f'% auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

#K-Folds
kf = KFold(n_splits=4)

X = train_attr.as_matrix()
Y = train_label.as_matrix()

i = 0
for train_indices, test_indices in kf.split(X):
    i += 1
    train_X = X[train_indices, :]
    train_Y = Y[train_indices]
    test_X = X[test_indices, :]
    test_Y = Y[test_indices]
    logreg = LogisticRegression().fit(train_X, train_Y)
    pred = logreg.predict(test_X)

    accuracy = accuracy_score(test_Y, pred) * 100
    precision = precision_score(test_Y, pred) * 100
    recall = recall_score(test_Y, pred) * 100

    print
    'For split {}'.format(i)
    print
    '    Accuracy is {:.4f}%\n\
    Precision is {:.4f}%\n\
    Recall is {:.4f}%\n'.format(accuracy, precision, recall)

#Testing the Kaggle test data
X = train_attr.as_matrix()
Y = train_label.as_matrix()

logreg = LogisticRegression().fit(train_X, train_Y)


test_data = pd.read_csv('C:/Users/Yogesh Kumar Ahuja/Desktop/twitter Spammer Detection/kaggle_test.csv',encoding = "ISO-8859-1")
test_attr = test_data[
  ['followers_count', 'friends_count', 'listed_count', 'favorites_count', 'statuses_count', 'verified']]
test_attr.replace('None', np.NaN, inplace=True)
test_attr['followers_count'] = test_attr['followers_count'].astype(np.float64)
test_attr['friends_count'] = test_attr['friends_count'].astype(np.float64)
test_attr['listed_count'] = test_attr['listed_count'].astype(np.float64)
test_attr['favorites_count'] = test_attr['favorites_count'].astype(np.float64)
test_attr['statuses_count'] = test_attr['statuses_count'].astype(np.float64)
test_attr['verified'] = np.where(test_attr['verified'] == 'TRUE', True, False)

imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
test_attr = imp.fit_transform(test_attr)

pred = logreg.predict(test_attr)

train_data = pd.read_csv('C:/Users/Yogesh Kumar Ahuja/Desktop/twitter Spammer Detection/kaggle_train.csv',encoding = "ISO-8859-1")
train_label = train_data[['bot']]
test_data = pd.read_csv("C:/Users/Yogesh Kumar Ahuja/Desktop/twitter Spammer Detection/test.csv",encoding = "ISO-8859-1")
test_label = test_data[['bot']]

train_X, test_X, train_Y, test_Y = train_test_split(train_data, train_label, test_size=0.4, random_state=0)

X = train_X[['followers_count', 'friends_count', 'listedcount', 'favourites_count', 'statuses_count', 'verified']]
Y = train_Y

X = normalize(X)
Y = normalize(Y)

logreg = LogisticRegression().fit(X, Y)

Z = test_X[['followers_count', 'friends_count', 'listedcount', 'favourites_count', 'statuses_count', 'verified']]
Z = normalize(Z)
actual = np.array(test_Y)
predicted = logreg.predict(Z)
pred = np.array(predicted)

sc = test_X['screen_name'].as_matrix()
i = 0
for name in sc:
    if 'bot' in name or 'Bot' in name or 'bOt' in name or 'boT' in name or 'BOT' in name or 'BOt' in name or 'BoT' in name or 'bOT' in name:
        pred[i] = 1
    i += 1

print(pred)