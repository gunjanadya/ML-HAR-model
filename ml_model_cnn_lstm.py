import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Dense, LSTM, Dropout, Flatten
from keras.utils import plot_model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import seaborn as sns

data = pd.read_csv('processed_data.csv')

data['attr_time'] = pd.to_datetime(data['attr_time'])

# split dataset 80/20
num_rows = data.shape[0]
divide = int(num_rows*0.8)

X_train = data.iloc[:divide,:-1]
Y_train = data.iloc[:divide,-1]
X_test = data.iloc[divide:,:-1]
Y_test = data.iloc[divide:,-1]

# Exclude 'attr_time' from the features for scaling
X_train = X_train.drop(columns=['attr_time'])
X_test = X_test.drop(columns=['attr_time'])

# Normalize X data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.fillna(X_train.mean()))
X_test = scaler.fit_transform(X_test.fillna(X_test.mean()))

# Convert Y labels to one-hot encoding
encoder = LabelEncoder()
Y_train_encoded = encoder.fit_transform(Y_train)
Y_train_onehot = to_categorical(Y_train_encoded)

# Reshape X_train to have the correct input shape (samples, time_steps, features)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Shape of the input data
n_timesteps, n_features = X_train.shape[1], X_train.shape[2]
n_outputs = Y_train_onehot.shape[1]

# Building the model
model = Sequential()

# Convolutional layers
model.add(Conv1D(filters=64, kernel_size=4, activation='relu', padding='same', input_shape=(n_timesteps, n_features)))
model.add(Conv1D(filters=64, kernel_size=4, activation='relu', padding='same'))
model.add(Dropout(0.5))
model.add(MaxPooling1D(pool_size=4))

# LSTM layer
model.add(LSTM(100, activation='relu', input_shape=(n_timesteps, n_features)))
model.add(Dropout(0.5))

# Dense layers
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(n_outputs, activation='softmax'))


# Compile the model with gradient clipping
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Plot the model architecture
plot_model(model, show_shapes=True, show_layer_names=True)

# Train the model
train_epochs = 50
batch_size = 32
validation_split = 0.2

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(
    X_train, Y_train_onehot,
    epochs=train_epochs, batch_size=batch_size,
    verbose=True, validation_split=validation_split,
    shuffle=True, callbacks=[early_stopping]
)

# Plot training history
plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
