import keras 
from keras import layers
from keras.models import Sequential
import glob
import os
import numpy as np
import cv2
from keras.preprocessing.image import ImageDataGenerator
#from keras import backend as K
#print(K.tensorflow_backend._get_available_gpus())



color2idx = {
    "red": 0,
    "yellow": 1,
    "green": 2
}
idx2color = {
    0: "red",
    1: "yellow",
    2: "green"
}

def read_data():
    impaths = glob.glob(os.path.join("images", "*.png"))#[-1700:]
    impaths = impaths[-1700:]
    np.random.shuffle(impaths)
    num_ims = len(impaths)
    images = np.zeros((num_ims, 600, 800, 3), dtype=np.float32)
    classes = np.zeros((num_ims, 3), dtype=np.uint8)
    for idx, impath in enumerate(impaths):
        im = cv2.imread(impath)
        im = im.astype(np.float32) / 128 - 1
        images[idx] = im
        cls_ = None
        for color, cidx in color2idx.items():
            if color in impath:
                cls_ = cidx
        if cls_ is None:
            raise AttributeError
        classes[idx, cls_] = 1
    #images = images.astype(np.float32) / 128 - 1
    # Shuffle together
    val_size =  len(images) // 5 # 20% validationset
    img_train, img_val = np.split(images, [-val_size])
    cls_train, cls_val = np.split(classes, [-val_size])
    #idx_train = idx[:-val_size]
    #idx_val = idx[-val_size:]
    #img_train, img_val = 
    #img_train, img_val = images[idx_train], images[idx_val]
    #cls_train, cls_val = classes[idx_train], classes[idx_val]

    print("Loaded dataset")
    print("X_train shape: {}, Y_train shape: {},".format(img_train.shape, cls_train.shape))
    print("X_train shape: {}, Y_train shape: {},".format(img_val.shape, cls_val.shape))
    return img_train, cls_train, img_val, cls_val


def get_model():
    model = Sequential([
        layers.Conv2D(16, (3, 3), strides=2, padding="same",  input_shape=(600, 800,3)),
        layers.LeakyReLU(),
        layers.BatchNormalization(),
        layers.Conv2D(16, (3, 3), strides=2, padding="same",), # 400 x 300
        layers.LeakyReLU(),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), strides=2, padding="same",), # 200 x 150
        layers.LeakyReLU(),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), strides=2, padding="same",), # 100 x 75
        layers.LeakyReLU(),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), strides=2, padding="same",), # 50 x 37
        layers.LeakyReLU(),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), strides=2, padding="same",), # 25 x 18
        layers.LeakyReLU(),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), strides=2, padding="same",), # 12 x 9
        layers.LeakyReLU(),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), strides=2, padding="same",), # 6 x 4
        layers.LeakyReLU(),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), strides=2, padding="same",), # 3 x 2
        layers.LeakyReLU(),
        layers.BatchNormalization(),
        layers.Flatten(),
        layers.Dropout(0.2),
        layers.Dense(128),
        layers.LeakyReLU(),
        layers.Dense(64),
        layers.LeakyReLU(),
        layers.Dense(3, activation="softmax")
    ])
    return model

from keras.callbacks import LearningRateScheduler
epoch_num = 0
def lr_scheduler(x):
    print(x)
    global epoch_num
    epoch_num += 1
    if epoch_num >= 5 and  epoch_num <= 12:
        print("Setting down")
        return 0.0001
    if epoch_num >= 12:
        return 0.00001
    print("The same!")
    return 0.0001

if __name__ == "__main__":
    X_train, Y_train, X_val, Y_val = read_data()
    train_datagen = ImageDataGenerator(
        width_shift_range=0.2,
        height_shift_range=.2,
        #shear_range=.2,
        #zoom_range=.2,
        #horizontal_flip=True,
        #ill_mode="nearest"
    ).flow(X_train, Y_train, batch_size=32, shuffle=True)
    validation_datagen = ImageDataGenerator(
    ).flow(X_val, Y_val, batch_size=32)
    model = get_model()
    adam = keras.optimizers.Adam(0.001)
    print(model.summary())
    cb_lr = LearningRateScheduler(lr_scheduler)
    model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=["accuracy"])
    model.load_weights("good_model_weights.h5")
    #model.fit_generator(train_datagen, steps_per_epoch = len(X_train) // 32, epochs=20, validation_data=validation_datagen,validation_steps=len(X_val)//32, callbacks=[cb_lr])
    model.fit(X_train, Y_train, validation_data=(X_val, Y_val), shuffle=True, batch_size=32, epochs=20, callbacks=[cb_lr])
    print(Y_train)
    print(model.evaluate(X_val, Y_val))
    model.save("model_weights_v3.h5")
