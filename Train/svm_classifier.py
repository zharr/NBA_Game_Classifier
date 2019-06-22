from sklearn import svm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mlxtend.plotting import plot_decision_regions
from sklearn.decomposition import PCA


# import data
df = pd.read_csv('./Processed_Data/preprocessed_games_17_18.csv')
df_all = df[['FG%_A', 'FT%_A', #'PTS_A',
             'FG%_H', 'FT%_H', #'PTS_H,'
              'Away_Win', 'Home_Win']]

df_all = df_all.head(650)
val = df_all.iloc[631:650]
df_train = df_all.iloc[0:630]

# shuffle training data
#df_train = df_train.sample(frac=1).reset_index(drop=True)
print("Validation Set 1")
y_train = df_train[['Away_Win', 'Home_Win']].values
y_train_classes = np.argmax(y_train,axis=1)

x_train = df_train.drop(['Away_Win', 'Home_Win'], axis=1)

# Normalizing the data (subtracting mean and divide by std)
mean = x_train['FG%_A'].mean()
std = x_train['FG%_A'].std()
x_train.loc[:, 'FG%_A'] = (x_train['FG%_A'] - mean) / std
mean = x_train['FT%_A'].mean()
std = x_train['FT%_A'].std()
x_train.loc[:, 'FT%_A'] = (x_train['FT%_A'] - mean) / std
mean = x_train['FG%_H'].mean()
std = x_train['FG%_H'].std()
x_train.loc[:, 'FG%_H'] = (x_train['FG%_H'] - mean) / std
mean = x_train['FT%_H'].mean()
std = x_train['FT%_H'].std()
x_train.loc[:, 'FT%_H'] = (x_train['FT%_H'] - mean) / std
x_train_vals = x_train.values

pca = PCA(n_components=2).fit(x_train_vals)
pca_2d = pca.transform(x_train_vals)
print(pd.DataFrame(pca.components_,columns=x_train.columns,index = ['PC-1','PC-2']))
# training
clf = svm.SVC(kernel='rbf',gamma='scale')
clf.fit(x_train, y_train_classes)
#clf.fit(pca_2d,y_train_classes)

# Plot Decision Region using mlxtend's awesome plotting function
#plot_decision_regions(X=pca_2d,
#                      y=y_train_classes,
#                      clf=clf,
#                      legend=2)

# Update plot object with X/Y axis labels and Figure Title
#plt.title('SVM Decision Region Boundary', size=16)
#plt.show()

# testing
y_test = val[['Away_Win', 'Home_Win']].values
y_test_classes = np.argmax(y_test,axis=1)

x_test = val.drop(['Away_Win', 'Home_Win'], axis=1)

# Normalizing the data (subtracting mean and divide by std)
mean = x_test['FG%_A'].mean()
std = x_test['FG%_A'].std()
x_test.loc[:, 'FG%_A'] = (x_test['FG%_A'] - mean) / std
mean = x_test['FT%_A'].mean()
std = x_test['FT%_A'].std()
x_test.loc[:, 'FT%_A'] = (x_test['FT%_A'] - mean) / std
mean = x_test['FG%_H'].mean()
std = x_test['FG%_H'].std()
x_test.loc[:, 'FG%_H'] = (x_test['FG%_H'] - mean) / std
mean = x_test['FT%_H'].mean()
std = x_test['FT%_H'].std()
x_test.loc[:, 'FT%_H'] = (x_test['FT%_H'] - mean) / std
x_test = x_test.values


prediction = clf.predict(x_test)
total = len(y_test)
correct = 0
for i in range(total):
    if prediction[i] == y_test_classes[i]:
        correct += 1

acc = (correct/total)*100
print("Accuracy for Val 1: "+str(acc)+"%")


# plotting


# now do cross validation on generated splits
for i in range(4):
    print("Validation Set: "+str(i+2))
    df_train_cross = df_all.sample(frac=1).reset_index(drop=True)

    msk = np.random.rand(len(df_train_cross)) < 0.95384615
    df_train = df_train_cross[msk]
    val = df_train_cross[~msk]

    y_train = df_train[['Away_Win', 'Home_Win']].values
    y_train_classes = np.argmax(y_train, axis=1)

    x_train = df_train.drop(['Away_Win', 'Home_Win'], axis=1)

    # Normalizing the data (subtracting mean and divide by std)
    mean = x_train['FG%_A'].mean()
    std = x_train['FG%_A'].std()
    x_train.loc[:, 'FG%_A'] = (x_train['FG%_A'] - mean) / std
    mean = x_train['FT%_A'].mean()
    std = x_train['FT%_A'].std()
    x_train.loc[:, 'FT%_A'] = (x_train['FT%_A'] - mean) / std
    mean = x_train['FG%_H'].mean()
    std = x_train['FG%_H'].std()
    x_train.loc[:, 'FG%_H'] = (x_train['FG%_H'] - mean) / std
    mean = x_train['FT%_H'].mean()
    std = x_train['FT%_H'].std()
    x_train.loc[:, 'FT%_H'] = (x_train['FT%_H'] - mean) / std
    x_train = x_train.values

    # training
    clf = svm.SVC(gamma='scale')
    clf.fit(x_train, y_train_classes)

    # testing
    y_test = val[['Away_Win', 'Home_Win']].values
    y_test_classes = np.argmax(y_test, axis=1)

    x_test = val.drop(['Away_Win', 'Home_Win'], axis=1)

    # Normalizing the data (subtracting mean and divide by std)
    mean = x_test['FG%_A'].mean()
    std = x_test['FG%_A'].std()
    x_test.loc[:, 'FG%_A'] = (x_test['FG%_A'] - mean) / std
    mean = x_test['FT%_A'].mean()
    std = x_test['FT%_A'].std()
    x_test.loc[:, 'FT%_A'] = (x_test['FT%_A'] - mean) / std
    mean = x_test['FG%_H'].mean()
    std = x_test['FG%_H'].std()
    x_test.loc[:, 'FG%_H'] = (x_test['FG%_H'] - mean) / std
    mean = x_test['FT%_H'].mean()
    std = x_test['FT%_H'].std()
    x_test.loc[:, 'FT%_H'] = (x_test['FT%_H'] - mean) / std
    x_test = x_test.values

    prediction = clf.predict(x_test)
    total = len(y_test)
    correct = 0
    for i in range(total):
        if prediction[i] == y_test_classes[i]:
            correct += 1

    acc = (correct / total) * 100
    print("Accuracy: " + str(acc) + "%")



