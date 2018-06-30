# Version: 1.0 - 2018/06/29
# Contact: walid.daboubi@gmail.com

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense
from keras import regularizers
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint, TensorBoard
import numpy as np
import tensorflow as tf
import sys
from sklearn.metrics import (confusion_matrix, precision_recall_curve, auc,
                             roc_curve, recall_score, classification_report, f1_score,
                             precision_recall_fscore_support)



df = pd.read_csv("url_enriched_data.csv")


# Dimension of the dataset
print df.shape

# Missing values
print df.isnull().values.any()

frauds = df[df.label == 1]
normal = df[df.label == 0]

print   frauds.shape

#print frauds.Amount.describe()
#print normal.Amount.describe()

data = df.drop(['domain'], axis=1)
print data.shape
X_train, X_test = train_test_split(data, test_size=0.2)

# Take only the normal cases
X_train = X_train[X_train.label == 0]
X_train = X_train.drop(['label'], axis=1)

y_test = X_test['label']
X_test = X_test.drop(['label'], axis=1)
X_train = X_train.values
X_test = X_test.values
print X_train.shape
print X_test.shape



input_dim = X_train.shape[1]
encoding_dim = input_dim

input_layer = Input(shape=(input_dim, ))
encoder = Dense(encoding_dim, activation="tanh",
                activity_regularizer=regularizers.l1(10e-5))(input_layer)
encoder = Dense(int(encoding_dim), activation="relu")(encoder)

encoder = Dense(int(encoding_dim-2), activation="relu")(encoder)
code = Dense(int(encoding_dim-4), activation='tanh')(encoder)
decoder = Dense(int(encoding_dim-2), activation='tanh')(code)

decoder = Dense(int(encoding_dim), activation='tanh')(encoder)
decoder = Dense(input_dim, activation='relu')(decoder)
autoencoder = Model(inputs=input_layer, outputs=decoder)


nb_epoch = 100
batch_size = 60


autoencoder.compile(optimizer='adam',
                    loss='mean_squared_error',
                    metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath="model.h5",
                               verbose=0,
                               save_best_only=True)
tensorboard = TensorBoard(log_dir='./logs',
                          histogram_freq=0,
                          write_graph=True,
                          write_images=True)

history = autoencoder.fit(X_train, X_train,
                    epochs=nb_epoch,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_data=(X_test, X_test),
                    verbose=1,
                    callbacks=[checkpointer, tensorboard]).history




autoencoder = load_model('model.h5')
predictions = autoencoder.predict(X_test)
mse = np.mean(np.power(X_test - predictions, 2), axis=1)
error_df = pd.DataFrame({'reconstruction_error': mse,'true_class': y_test})
fraud_error_df = error_df[error_df['true_class'] == 1]


threshold=0
f1=0
recall=0
accuracy=0
while (recall < 0.5 or accuracy < 0.6):
    print '**************************'
    print threshold
    threshold+=.0005
    y_pred = [1 if e > threshold else 0 for e in error_df.reconstruction_error.values]
    conf_matrix = confusion_matrix(error_df.true_class, y_pred)
    tn, fp, fn, tp = conf_matrix.ravel()
    precision = 1.*tp/(tp+fp)
    recall = 1.*tp/(tp+fn)
    f1=(2*recall*precision)/(recall+precision)
    print 'TP:'+str(tp)
    print 'FP:'+str(fp)
    print 'TN:'+str(tn)
    print 'FN:'+str(fn)
    accuracy=1.*(tp+tn)/(tp+tn+fp+fn)
    print 'Accuracy:'+str(accuracy)
    print 'Precision:'+str(precision)
    print 'Recall:'+str(recall)
    print 'F1:'+str(f1)


groups = error_df.groupby('true_class')
fig, ax = plt.subplots()

for name, group in groups:
    ax.plot(group.index, group.reconstruction_error, marker='o', ms=2, linestyle='',
            label= "Malicious URL" if name == 1 else "Normal URL", color= 'green' if name == 1 else 'orange')
ax.hlines(threshold, ax.get_xlim()[0], ax.get_xlim()[1], colors="red", zorder=100, label='Threshold')
ax.legend()
plt.title("Reconstruction error for different classes")
plt.ylabel("Reconstruction error")
plt.xlabel("Data point index")
plt.show();
