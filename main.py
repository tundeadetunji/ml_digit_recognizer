import tensorflow as t
import numpy as n
import matplotlib.pyplot as plotter


mnist = t.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# #get to know data we're working with:
# #data for image 1
# print(x_train[0])
# # preview image 1
# plotter.imshow(x_train[0], cmap=plotter.cm.binary)
# plotter.show()

x_train = t.keras.utils.normalize(x_train, axis=1)
x_test = t.keras.utils.normalize(x_test, axis=1)

model = t.keras.models.Sequential()
model.add(t.keras.layers.Flatten())
model.add(t.keras.layers.Dense(128, activation=t.nn.relu))
model.add(t.keras.layers.Dense(128, activation=t.nn.relu))
model.add(t.keras.layers.Dense(10, activation=t.nn.softmax))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=3)

# evaluation: how good/bad is loss/accuracy
# val_loss, val_acc = model.evaluate(x_test, y_test)
# print('loss: ' + str(val_loss) + '\naccuracy: ' + str(val_acc))

#save for later
model.save('recognizer.model')

#load from save
new_model = t.keras.models.load_model('recognizer.model')

# predict...
prediction = new_model.predict([x_test])

# preview raw data from prediction
# print(prediction)

# what's the first character's prediction be?
print(n.argmax(prediction[0]))  # 7, prediction of x_test[0]
# really? 7? let's show the image, to confirm
plotter.imshow(x_test[0])
plotter.show()
