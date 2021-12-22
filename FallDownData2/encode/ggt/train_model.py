import os
import tensorflow as tf
from tensorflow import keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
import numpy as np
import json
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix   #copy

data = np.load('ggt.npz')       #mark final.npz
label = data['y']
Bigdata = data['X']
Bigdata[np.isnan(Bigdata)] = 0.
#Bigdata = Bigdata[:,:,1:]
results = {}
results['acc'] = []
results['auc'] = []
results['confMat'] = []  #add
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
    #2
    model = tf.keras.Sequential([
    keras.layers.Masking(mask_value=0., input_shape=(500, 8)), 
    keras.layers.Bidirectional(keras.layers.LSTM(256, return_sequences=True), input_shape=(500, 8)),
    keras.layers.Bidirectional(keras.layers.LSTM(128)),
    keras.layers.Dense(units=64, activation='relu'),
    keras.layers.Dense(units=1, activation='sigmoid')
    ])
    #2

    model.compile(optimizer='adam', loss=keras.losses.BinaryCrossentropy(from_logits=False), metrics=['accuracy','AUC'])
    TensorBoard_callback = keras.callbacks.TensorBoard('logs_without_timestamp/' + str(fold) +"/" , histogram_freq=1)
    EarlyStopping_callback = keras.callbacks.EarlyStopping(patience=5)
    checkpoint_callback = keras.callbacks.ModelCheckpoint('checkpoint_without_timestamp/fold_' + str(fold) +"/" , save_best_only=True, save_weights_only=True)
    
    model.fit(x=train_x, y=train_y, batch_size=64, epochs=1, validation_split=0.2, callbacks=[TensorBoard_callback, EarlyStopping_callback, checkpoint_callback], verbose=2)

    #1
    y_pred = model.predict(test_x)
    roc = roc_auc_score(test_y, y_pred)
    y_pred[y_pred >= 0.5 ] =1
    y_pred[y_pred < 0.5] = 0
    acc = accuracy_score(test_y, y_pred)
    confMat = confusion_matrix(test_y, y_pred).astype(str).astype(int)
    #acc = accuracy_score(np.argmax(test_y, axis=-1), np.argmax(y_pred, axis=-1))
    print("In the fold {0}: acc {1}, roc {2}".format(fold, acc, roc))
    results['acc'].append(acc)
    results['auc'].append(roc)
    confMat = [list(A)for A in confMat]
    results['confMat'].append(confMat)
    fold +=1
    #1

with open('auc_acc_confMat.txt','a') as f:
    f.write(json.dumps(results))