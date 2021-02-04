
from keras.models import Sequential, load_model
from keras.layers import Conv2D, Dropout, BatchNormalization, MaxPooling2D,Dense, Activation, Flatten
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
import os
import glob
import cv2
import numpy as np
from keras.preprocessing.image import load_img
list_classes = os.listdir("../data/train")
image_shape = (150,150,3)
num_classes = 6
def load_data(data_path):
    X = []
    Y = []
    list_classes = os.listdir(data_path)
    for class_name in list_classes:
        list_img_paths = glob.glob(os.path.join(data_path, class_name) + "/*.jpg")
        for img_path in list_img_paths:
            img = cv2.imread(img_path)
            img = cv2.resize(img, (image_shape[0], image_shape[1]))
            img_arr = np.array((img - 127.5) / 127.5)
            X.append(img_arr)
            Y.append(to_categorical(list_classes.index(class_name),len(list_classes)))
    return np.array(X), np.array(Y)

def classification_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(filters=8, kernel_size=(3,3), padding='same', activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=16, kernel_size=(3,3),padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=32, kernel_size=(3,3), padding= 'same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=64, kernel_size=(3,3), padding= 'same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(3,3), padding='same'))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), padding='same'))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model


def train(epochs= 500, batch_size = 16):
    model = classification_model(input_shape=image_shape, num_classes=num_classes)
    optimizer = Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics = ['accuracy'])

    trainX, trainy = load_data('../data/train')
    valX, valy = load_data('../data/val')

    current_checkpoint_subdir = os.listdir('checkpoint')
    new_checkpoint_subdir = os.path.join("checkpoint", str(len(current_checkpoint_subdir) + 1))
    os.makedirs(new_checkpoint_subdir, exist_ok=False)

    current_log_subdir = os.listdir("logs")
    new_log_subdir = os.path.join("logs", str(len(current_log_subdir) + 1))
    os.makedirs(new_log_subdir, exist_ok=False)

    tensorboard = TensorBoard(log_dir=new_log_subdir)
    early_stopper = EarlyStopping(monitor='val_accuracy', mode='max', patience=10)
    checkpointer = ModelCheckpoint(filepath=os.path.join(new_checkpoint_subdir, "{epoch:03d}-{val_accuracy:.3f}.hdf5"),
                                   monitor='val_accuracy', mode='max', verbose=1,
                                   save_best_only=True)

    model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, validation_data=(valX, valy), callbacks=[tensorboard, early_stopper,checkpointer])
    model.save(os.path.join(new_checkpoint_subdir, "model.h5"))

def test():
    testX, testy = load_data("../data/test")
    model = load_model("checkpoint/1/model.h5")
    result = model.evaluate(testX, testy)
    print("loss: ", result[0])
    print("accuracy: ", result[1])
if __name__ == "__main__":
    train(epochs = 30)
    #test()



