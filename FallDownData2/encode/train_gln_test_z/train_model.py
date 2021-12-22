import os
import tensorflow as tf
from tensorflow import keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
import numpy as np
import json
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix   #copy

train_files = ['../ggt/ggt.npz', '../lp/lp.npz', '../nn/nn.npz']
test_files = ['../zj/zj.npz']

length_8_files = ['../ggt/ggt.npz', '../zj/zj.npz']

train_X = np.concatenate([np.load(i)['X'][:,:,1:] if i in length_8_files else np.load(i)['X'] for i in train_files], axis=0)
train_y = np.concatenate([np.load(i)['y'] for i in train_files], axis=0)

test_X = np.concatenate([np.load(i)['X'][:,:,1:] if i in length_8_files else np.load(i)['X'] for i in test_files], axis=0)
test_y = np.concatenate([np.load(i)['y'] for i in test_files], axis=0)

train_X[np.isnan(train_X)] = 0.
test_X[np.isnan(test_X)] = 0.
train_y = np.array(list(map(lambda x : 1 if x == 5 else 0, train_y)), dtype=np.int32)
test_y = np.array(list(map(lambda x : 1 if x == 5 else 0, test_y)), dtype=np.int32)

results = {}
results['acc'] = []
results['auc'] = []
results['confMat'] = []  #add
fold = 0

index=np.arange(train_X.shape[0])
np.random.shuffle(index)
train_y=train_y[index]
train_X=train_X[index]

#2
model = tf.keras.Sequential([
keras.layers.Masking(mask_value=0., input_shape=(500, 7)), 
keras.layers.Bidirectional(keras.layers.LSTM(256, return_sequences=True), input_shape=(500, 7)),
keras.layers.Bidirectional(keras.layers.LSTM(128)),
keras.layers.Dense(units=64, activation='relu'),
keras.layers.Dense(units=1, activation='sigmoid')
])
#2

model.compile(optimizer='adam', loss=keras.losses.BinaryCrossentropy(from_logits=False), metrics=['accuracy','AUC'])
TensorBoard_callback = keras.callbacks.TensorBoard('logs_without_timestamp/' + str(fold) +"/" , histogram_freq=1)
EarlyStopping_callback = keras.callbacks.EarlyStopping(patience=5)
checkpoint_callback = keras.callbacks.ModelCheckpoint('checkpoint_without_timestamp/fold_' + str(fold) +"/" , save_best_only=True, save_weights_only=True)

model.fit(x=train_X, y=train_y, batch_size=64, epochs=30, validation_split=0.2, callbacks=[TensorBoard_callback, EarlyStopping_callback, checkpoint_callback], verbose=2)

#1
y_pred = model.predict(test_X)
roc = roc_auc_score(test_y, y_pred)
y_pred[y_pred >= 0.5 ] =1
y_pred[y_pred < 0.5] = 0
acc = accuracy_score(test_y, y_pred)
confMat = confusion_matrix(test_y, y_pred).astype(str)
#acc = accuracy_score(np.argmax(test_y, axis=-1), np.argmax(y_pred, axis=-1))
print("In the fold {0}: acc {1}, roc {2}".format(fold, acc, roc))
results['acc'].append(acc)
results['auc'].append(roc)
confMat = [list(A)for A in confMat]
results['confMat'].append(confMat)
fold +=1
#1

with open('auc_acc_without_timestamp.txt','a') as f:
    f.write(json.dumps(results))