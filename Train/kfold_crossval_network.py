import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneOut,KFold
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv('./Processed_Data/preprocessed_games_17_18.csv')
df_all = df[['FG%_A', 'FT%_A', #'PTS_A',
             'FG%_H', 'FT%_H', #'PTS_H,'
              'Away_Win', 'Home_Win']]
#df_all = df[['FG%_A','3P%_A','FT%_A','ORB_A','DRB_A','AST_A','STL_A','BLK_A','TOV_A',
#             'FG%_H','3P%_H','FT%_H','ORB_H','DRB_H','AST_H','STL_H','BLK_H','TOV_H',
#             'Away_Win', 'Home_Win']]

df_all = df_all.head(650)
val = df_all.iloc[631:650]
df_train = df_all.iloc[0:630]

# now start tensorflow portion
in_size = 4 #4
inputs = tf.placeholder(tf.float32, shape=(None, in_size), name='inputs')
label = tf.placeholder(tf.float32, shape=(None, 2), name='label')

# First layer
hid1_size = 64 # 64
w1 = tf.Variable(tf.random_normal([hid1_size, in_size], name='w1'))
b1 = tf.Variable(tf.constant(0.1, shape=(hid1_size, 1)), name='b1')
y1 = tf.nn.dropout(tf.nn.sigmoid(tf.add(tf.matmul(w1, tf.transpose(inputs)), b1)), keep_prob=0.5)

# Second layer
#hid2_size = 256
#w2 = tf.Variable(tf.random_normal([hid2_size, hid1_size],name='w2'))#, stddev=0.01), name='w2')
#b2 = tf.Variable(tf.constant(0.1, shape=(hid2_size, 1)), name='b2')
#y2 = tf.nn.dropout(tf.nn.sigmoid(tf.add(tf.matmul(w2, y1), b2)), keep_prob=0.5)

# Output layer
wo = tf.Variable(tf.random_normal([2, hid1_size], name='wo'))
bo = tf.Variable(tf.random_normal([2, 1]), name='bo')
yo = tf.transpose(tf.add(tf.matmul(wo, y1), bo))

# Prediction
pred = tf.nn.softmax(yo)
pred_label = tf.argmax(pred, 1, name='pred_label')
correct_label = tf.argmax(label, 1)
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(label, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# Loss function and optimizer
lr = tf.placeholder(tf.float32, shape=(), name='learning_rate')
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=yo, labels=label))
optimizer = tf.train.AdamOptimizer(lr).minimize(loss)

# training parameters
epochs = 250
learning_rate = 0.0001
batch_size = 1
x = []
y = []
y_max = 0
train_away = 0
valid_away = 0
val_accuracies = []
# Start a new tensorflow session and initialize variables
print("Validation Set: 1 (Games ordered by date)")
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    for epoch in range(epochs):
        avg_cost = 0.0

        # shuffle training data
        df_train = df_train.sample(frac=1).reset_index(drop=True)

        y_train = df_train[['Away_Win', 'Home_Win']].values
        x_train = df_train.drop(['Away_Win', 'Home_Win'], axis=1)
        labels_train = y_train

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

        if False:
            mean = x_train['3P%_A'].mean()
            std = x_train['3P%_A'].std()
            x_train.loc[:, '3P%_A'] = (x_train['3P%_A'] - mean) / std
            mean = x_train['ORB_A'].mean()
            std = x_train['ORB_A'].std()
            x_train.loc[:, 'ORB_A'] = (x_train['ORB_A'] - mean) / std
            mean = x_train['DRB_A'].mean()
            std = x_train['DRB_A'].std()
            x_train.loc[:, 'DRB_A'] = (x_train['DRB_A'] - mean) / std
            mean = x_train['AST_A'].mean()
            std = x_train['AST_A'].std()
            x_train.loc[:, 'AST_A'] = (x_train['AST_A'] - mean) / std
            mean = x_train['STL_A'].mean()
            std = x_train['STL_A'].std()
            x_train.loc[:, 'STL_A'] = (x_train['STL_A'] - mean) / std
            mean = x_train['BLK_A'].mean()
            std = x_train['BLK_A'].std()
            x_train.loc[:, 'BLK_A'] = (x_train['BLK_A'] - mean) / std
            mean = x_train['TOV_A'].mean()
            std = x_train['TOV_A'].std()
            x_train.loc[:, 'TOV_A'] = (x_train['TOV_A'] - mean) / std
            mean = x_train['3P%_H'].mean()
            std = x_train['3P%_H'].std()
            x_train.loc[:, '3P%_H'] = (x_train['3P%_H'] - mean) / std
            mean = x_train['ORB_H'].mean()
            std = x_train['ORB_H'].std()
            x_train.loc[:, 'ORB_H'] = (x_train['ORB_H'] - mean) / std
            mean = x_train['DRB_H'].mean()
            std = x_train['DRB_H'].std()
            x_train.loc[:, 'DRB_H'] = (x_train['DRB_H'] - mean) / std
            mean = x_train['AST_H'].mean()
            std = x_train['AST_H'].std()
            x_train.loc[:, 'AST_H'] = (x_train['AST_H'] - mean) / std
            mean = x_train['STL_H'].mean()
            std = x_train['STL_H'].std()
            x_train.loc[:, 'STL_H'] = (x_train['STL_H'] - mean) / std
            mean = x_train['BLK_H'].mean()
            std = x_train['BLK_H'].std()
            x_train.loc[:, 'BLK_H'] = (x_train['BLK_H'] - mean) / std
            mean = x_train['TOV_H'].mean()
            std = x_train['TOV_H'].std()
            x_train.loc[:, 'TOV_H'] = (x_train['TOV_H'] - mean) / std

        x_train = x_train.values

        num_runs = int(np.ceil(x_train.shape[0] / batch_size))

        # For each epoch, we go through all the samples we have
        for i in range(num_runs):
            # using cosine simulated annealing for the learning rate
            if i != (num_runs - 1):
                start = i * batch_size
                end = start + batch_size
                batch_x = x_train[start:end, :]
                batch_y = labels_train[start:end, :]
            else:
                start = i * batch_size
                batch_x = x_train[start:, :]
                batch_y = labels_train[start:, :]
            cor, l,  _, c = sess.run([correct_label, pred_label, optimizer, loss], feed_dict={lr:learning_rate,
                                                                                              inputs: batch_x,
                                                                                              label: batch_y})
            avg_cost += c

        avg_cost /= x_train.shape[0]
        x.append(epoch)
        y.append(avg_cost)

        # Print the cost in this epoch to the console
        if epoch % 10 == 0:
            print("Epoch: {:3d}     Train Cost: {:.4f}".format(epoch, avg_cost))

    acc_train = accuracy.eval(feed_dict={inputs: x_train, label: labels_train})
    print("Train accuracy: {:3.2f}%".format(acc_train * 100.0))
    saver.save(sess, './tmp/model.ckpt')

    labels_val = val[['Away_Win', 'Home_Win']].values
    x_val = val.drop(['Away_Win', 'Home_Win'], axis=1)

    mean = x_val['FG%_A'].mean()
    std = x_val['FG%_A'].std()
    x_val.loc[:, 'FG%_A'] = (x_val['FG%_A'] - mean) / std
    mean = x_val['FT%_A'].mean()
    std = x_val['FT%_A'].std()
    x_val.loc[:, 'FT%_A'] = (x_val['FT%_A'] - mean) / std
    mean = x_val['FG%_H'].mean()
    std = x_val['FG%_H'].std()
    x_val.loc[:, 'FG%_H'] = (x_val['FG%_H'] - mean) / std
    mean = x_val['FT%_H'].mean()
    std = x_val['FT%_H'].std()
    x_val.loc[:, 'FT%_H'] = (x_val['FT%_H'] - mean) / std

    if False:
        mean = x_val['3P%_A'].mean()
        std = x_val['3P%_A'].std()
        x_val.loc[:, '3P%_A'] = (x_val['3P%_A'] - mean) / std
        mean = x_val['ORB_A'].mean()
        std = x_val['ORB_A'].std()
        x_val.loc[:, 'ORB_A'] = (x_val['ORB_A'] - mean) / std
        mean = x_val['DRB_A'].mean()
        std = x_val['DRB_A'].std()
        x_val.loc[:, 'DRB_A'] = (x_val['DRB_A'] - mean) / std
        mean = x_val['AST_A'].mean()
        std = x_val['AST_A'].std()
        x_val.loc[:, 'AST_A'] = (x_val['AST_A'] - mean) / std
        mean = x_val['STL_A'].mean()
        std = x_val['STL_A'].std()
        x_val.loc[:, 'STL_A'] = (x_val['STL_A'] - mean) / std
        mean = x_val['BLK_A'].mean()
        std = x_val['BLK_A'].std()
        x_val.loc[:, 'BLK_A'] = (x_val['BLK_A'] - mean) / std
        mean = x_val['TOV_A'].mean()
        std = x_val['TOV_A'].std()
        x_val.loc[:, 'TOV_A'] = (x_val['TOV_A'] - mean) / std
        mean = x_val['3P%_H'].mean()
        std = x_val['3P%_H'].std()
        x_val.loc[:, '3P%_H'] = (x_val['3P%_H'] - mean) / std
        mean = x_val['ORB_H'].mean()
        std = x_val['ORB_H'].std()
        x_val.loc[:, 'ORB_H'] = (x_val['ORB_H'] - mean) / std
        mean = x_val['DRB_H'].mean()
        std = x_val['DRB_H'].std()
        x_val.loc[:, 'DRB_H'] = (x_val['DRB_H'] - mean) / std
        mean = x_val['AST_H'].mean()
        std = x_val['AST_H'].std()
        x_val.loc[:, 'AST_H'] = (x_val['AST_H'] - mean) / std
        mean = x_val['STL_H'].mean()
        std = x_val['STL_H'].std()
        x_val.loc[:, 'STL_H'] = (x_val['STL_H'] - mean) / std
        mean = x_val['BLK_H'].mean()
        std = x_val['BLK_H'].std()
        x_val.loc[:, 'BLK_H'] = (x_val['BLK_H'] - mean) / std
        mean = x_val['TOV_H'].mean()
        std = x_val['TOV_H'].std()
        x_val.loc[:, 'TOV_H'] = (x_val['TOV_H'] - mean) / std

    x_val = x_val.values

    acc = 0

    for i in range(x_val.shape[0]):
        winner = sess.run(pred_label, feed_dict={inputs:x_val[i,None]})
        if winner == 0:
            valid_away += 1
        if labels_val[i][winner] == 1:
            acc += 1
    print(valid_away)
    v_acc = ((acc/x_val.shape[0])*100.0)
    val_accuracies.append(v_acc)
    print("Validation accuracy:  {:3.2f}%".format(v_acc))
    sess.close()
    plt.plot(x, y)
    plt.axis([0, epochs, 0, max(y)])
    plt.xlabel("Epoch")
    plt.ylabel("Cost")
    plt.title("Training Cost vs. Epoch")
    plt.show()


if False:
# now do cross validation on generated splits
    for i in range(4):
        print("Validation Set: "+str(i+2))
        df_train_cross = df_all.sample(frac=1).reset_index(drop=True)

        msk = np.random.rand(len(df_train_cross)) < 0.95384615
        df_train = df_train_cross[msk]
        val = df_train_cross[~msk]

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(epochs):
                avg_cost = 0.0

                df_train = df_train.sample(frac=1).reset_index(drop=True)

                y_train = df_train[['Away_Win', 'Home_Win']].values
                x_train = df_train.drop(['Away_Win', 'Home_Win'], axis=1)
                labels_train = y_train

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

                num_runs = int(np.ceil(x_train.shape[0] / batch_size))

                # For each epoch, we go through all the samples we have
                for i in range(num_runs):
                    # using cosine simulated annealing for the learning rate
                    if i != (num_runs - 1):
                        start = i * batch_size
                        end = start + batch_size
                        batch_x = x_train[start:end, :]
                        batch_y = labels_train[start:end, :]
                    else:
                        start = i * batch_size
                        batch_x = x_train[start:, :]
                        batch_y = labels_train[start:, :]
                    cor, l, _, c = sess.run([correct_label, pred_label, optimizer, loss], feed_dict={lr: learning_rate,
                                                                                                     inputs: batch_x,
                                                                                                     label: batch_y})
                    avg_cost += c

                avg_cost /= x_train.shape[0]
                x.append(epoch)
                y.append(avg_cost)

                # Print the cost in this epoch to the console
                if epoch % 10 == 0:
                    print("Epoch: {:3d}     Train Cost: {:.4f}".format(epoch, avg_cost))

            acc_train = accuracy.eval(feed_dict={inputs: x_train, label: labels_train})
            print("Train accuracy: {:3.2f}%".format(acc_train * 100.0))

            labels_val = val[['Away_Win', 'Home_Win']].values
            x_val = val.drop(['Away_Win', 'Home_Win'], axis=1)

            mean = x_val['FG%_A'].mean()
            std = x_val['FG%_A'].std()
            x_val.loc[:, 'FG%_A'] = (x_val['FG%_A'] - mean) / std
            mean = x_val['FT%_A'].mean()
            std = x_val['FT%_A'].std()
            x_val.loc[:, 'FT%_A'] = (x_val['FT%_A'] - mean) / std
            mean = x_val['FG%_H'].mean()
            std = x_val['FG%_H'].std()
            x_val.loc[:, 'FG%_H'] = (x_val['FG%_H'] - mean) / std
            mean = x_val['FT%_H'].mean()
            std = x_val['FT%_H'].std()
            x_val.loc[:, 'FT%_H'] = (x_val['FT%_H'] - mean) / std
            x_val = x_val.values

            acc = 0

            for i in range(x_val.shape[0]):
                winner = sess.run(pred_label, feed_dict={inputs: x_val[i, None]})
                if winner == 0:
                    valid_away += 1
                if labels_val[i][winner] == 1:
                    acc += 1


            v_acc = ((acc / x_val.shape[0]) * 100.0)
            val_accuracies.append(v_acc)
            print("Validation accuracy:  {:3.2f}%".format(v_acc))
            sess.close()

    print(np.mean(val_accuracies))


# Functions to implement
#   graph loss/epoch
#   for app save current team averages in a DB and update everyday


########### Adam Optimizer
    # activation: sigmoid
    # lr: 0.0001
    # layer size: 64
    # epochs: 100
    # 78% test

    # activation: sigmoid
    # lr: 0.0001
    # layer size: 32
    # epochs: 100
    # 63.48% test

    # activation: sigmoid
    # lr: 0.0001
    # layer size: 128
    # epochs: 100
    # 59.76% test

    # activation: sigmoid
    # lr: 0.0001
    # layer size: 256
    # epochs: 100
    # 53.89% test

    # activation: sigmoid
    # lr: 0.0001
    # layer size: 32,64
    # epochs: 100
    # 73.01% test

    # activation: sigmoid
    # lr: 0.0001
    # layer size: 64,128
    # epochs: 300
    # 70.68% test

    # activation: sigmoid
    # lr: 0.0001
    # layer size: 128,256
    # epochs: 300
    # 56.37% test

###########
    # 4 SHOOTING ONLY
    # 78%

    # All stats
    # 68%
