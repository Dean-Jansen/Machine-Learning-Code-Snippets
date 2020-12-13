import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
import tensorflow as tf

# prepare data
iris = datasets.load_iris()
x_vals = np.array([x[0:4] for x in iris.data])
y_vals = np.array(iris.target)

y_vals = np.eye(len(set(y_vals)))[y_vals]

x_vals = (x_vals - x_vals.min(0)) / x_vals.ptp(0)

np.random.seed(59)
train_indices = np.random.choice(len(x_vals), round(len(x_vals) * 0.8), replace=False)
test_indices =np.array(list(set(range(len(x_vals))) - set(train_indices)))

x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]

feature_number = len(x_vals_train[0])

k = 5

x_data_train = tf.placeholder(shape=[None, feature_number], dtype=tf.float32)
y_data_train = tf.placeholder(shape=[None, len(y_vals[0])], dtype=tf.float32)
x_data_test = tf.placeholder(shape=[None, feature_number], dtype=tf.float32)

# manhattan distance
distance = tf.reduce_sum(tf.abs(tf.subtract(x_data_train, tf.expand_dims(x_data_test, 1))), axis=2)

# nearest k points
_, top_k_indices = tf.nn.top_k(tf.negative(distance), k=k)
top_k_label = tf.gather(y_data_train, top_k_indices)

sum_up_predictions = tf.reduce_sum(top_k_label, axis=1)
prediction = tf.argmax(sum_up_predictions, axis=1)


sess = tf.Session()
prediction_outcome = sess.run(prediction, feed_dict={x_data_train: x_vals_train,
                               x_data_test: x_vals_test,
                               y_data_train: y_vals_train})

# evaluation
accuracy = 0
for pred, actual in zip(prediction_outcome, y_vals_test):
    if pred == np.argmax(actual):
        accuracy += 1

print(accuracy / len(prediction_outcome))