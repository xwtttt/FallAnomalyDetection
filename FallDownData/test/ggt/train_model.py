import os
import tensorflow as tf
from tensorflow import keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
import numpy as np
import json
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score

data = np.load('ggt.npz')       #mark final.npz
label = data['y']
Bigdata = data['X']
Bigdata[np.isnan(Bigdata)] = 0.
results = {}
results['acc'] = []
results['auc'] = []
fold = 0
label[label < 5] = 0
label[label==5] = 1
y = label
index=np.arange(440)
np.random.shuffle(index)
y=y[index]
Bigdata=Bigdata[index]
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=22)
for train, test in kfold.split(np.zeros(len(y)), y):
    print("run fold:{0}".format(fold))
    train_x, train_y = Bigdata[train], y[train]
    test_x, test_y = Bigdata[test], y[test]
    #train_y, test_y = tf.one_hot(train_y, depth=6).numpy(), tf.one_hot(test_y, depth=6).numpy()
    model = tf.keras.Sequential([
    keras.layers.Masking(mask_value=0., input_shape=(500, 8)), 
    keras.layers.LSTM(units=256,return_sequences=True),
    keras.layers.LSTM(units=128,return_sequences=False),
    keras.layers.Dense(units=64, activation='relu'),
    keras.layers.Dense(units=1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss=keras.losses.BinaryCrossentropy(from_logits=False), metrics=['accuracy','AUC'])
    TensorBoard_callback = keras.callbacks.TensorBoard('logs/' + str(fold) +"/" , histogram_freq=1)
    EarlyStopping_callback = keras.callbacks.EarlyStopping(patience=5)
    checkpoint_callback = keras.callbacks.ModelCheckpoint('checkpoint/fold_' + str(fold) +"/" , save_best_only=True, save_weights_only=True)
    
    model.fit(x=train_x, y=train_y, batch_size=64, epochs=30, validation_split=0.2, callbacks=[TensorBoard_callback, EarlyStopping_callback, checkpoint_callback], verbose=2)

    y_pred = model.predict(test_x)
    y_pred[y_pred >= 0.5 ] =1
    y_pred[y_pred < 0.5] = 0
    acc = accuracy_score(test_y, y_pred)
    #acc = accuracy_score(np.argmax(test_y, axis=-1), np.argmax(y_pred, axis=-1))
    roc = roc_auc_score(test_y, y_pred)
    print("In the fold {0}: acc {1}, roc {2}".format(fold, acc, roc))
    results['acc'].append(acc)
    results['auc'].append(roc)
    fold +=1

with open('auc_acc.txt','a+') as f:
    f.write(json.dumps(results))