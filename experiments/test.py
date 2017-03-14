from keras.models import Model
from keras.layers import Dense, Input, Flatten
from gym_learning_to_learn.datasets import mnist

x = Input((28, 28))
h = Flatten()(x)
h = Dense(512, activation='tanh')(h)
h = Dense(512, activation='tanh')(h)
y = Dense(10, activation='softmax')(h)
m = Model(x, y)
m.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
(x_train, y_train), (x_val, y_val), (x_test, y_test) = mnist.load_data()

f = m.fit(x_train, y_train, validation_data=(x_val, y_val), nb_epoch=1, verbose=0)
print(f.history)
yp = m.predict(x_train, verbose=0)
print(yp)
ytrain = m.evaluate(x_train, y_train, verbose=0)
yval = m.evaluate(x_val, y_val, verbose=0)
print(ytrain)
print(yval)
