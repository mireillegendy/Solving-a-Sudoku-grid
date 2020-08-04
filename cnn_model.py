import tensorflow as tf
import keras
import keras.backend as Kb
import keras.layers as kl
mnist = tf.keras.datasets.mnist
batch_size = 128
num_classes = 10
epochs = 10
img_rows, img_cols = 28, 28
(x_train, y_train), (x_test, y_test) = mnist.load_data()
(x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = mnist.load_data()
if Kb.image_data_format == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, 28, 28)
    x_test = x_test.reshape(x_test.shape[0], 1, 28, 28, )
    input_shape = (1, 28, 28)
else:
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    input_shape = (28, 28, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3, 3), input_shape = (28, 28, 1)))
model.add(tf.keras.layers.Activation('relu'))
tf.keras.layers.BatchNormalization(axis=-1)
model.add(tf.keras.layers.Conv2D(32,(3, 3)))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
kl.BatchNormalization(axis=-1)
model.add(tf.keras.layers.Conv2D(64, (3, 3)))
model.add(tf.keras.layers.Activation('relu'))
kl.BatchNormalization(axis=-1)
model.add(tf.keras.layers.Conv2D(64, (3, 3)))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(tf.keras.layers.Flatten())
tf.keras.layers.BatchNormalization()
model.add(tf.keras.layers.Dense(512))
model.add(tf.keras.layers.Activation('relu'))
tf.keras.layers.BatchNormalization()
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(10))
model.add(tf.keras.layers.Activation('softmax'))
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=128, epochs = 10, verbose=2, validation_data = (x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
model.save('CNN_model')
CNN_model = tf.keras.models.load_model('CNN_model')
predictions = CNN_model.predict([x_test])