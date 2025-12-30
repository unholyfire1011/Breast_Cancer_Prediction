import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.datasets
from sklearn.model_selection import train_test_split
import tensorflow as tf
tf.random.set_seed(3)
from tensorflow import keras


bcd = sklearn.datasets.load_breast_cancer()
data_frame = pd.DataFrame(bcd.data, columns=bcd.feature_names)
data_frame['label'] = bcd.target
data_frame.groupby('label').mean()
X = data_frame.drop(columns='label', axis=1)
Y = data_frame['label']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(30, )),
    keras.layers.Dense(20, activation='relu'),
    keras.layers.Dense(2 , activation='sigmoid')
])

model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

history = model.fit(X_train, Y_train, validation_split=0.1, epochs=100)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper right')
plt.show()

l,a = model.evaluate(X_test, Y_test)
print(l,a)

y_pred = model.predict(X_test)
y_pred_labels = [np.argmax(i) for i in y_pred]
print(y_pred_labels)
input_data  = (17.99,10.38,122.8,1001,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189
)
np_input_data = np.asarray(input_data)
input_data_reshaped = np_input_data.reshape(1,-1)
input_data_reshaped = scaler.transform(input_data_reshaped)
prediction = model.predict(input_data_reshaped)
prediction_label = np.argmax(prediction)
if prediction_label == 0:
    print('The breast cancer is Malignant(Cancerous)')
else:
    print('The breast cancer is Benign(Non-cancerous)')
