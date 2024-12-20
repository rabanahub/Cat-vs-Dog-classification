from keras.models import load_model
from keras.preprocessing import image
#from keras.utils import load_img, img_to_array
import numpy as np
from os import listdir
from os.path import isfile, join

import os
os.chdir("D:\\felixPythan\\Rohit Sir\\dog_cat_classifier")

# dimensions of our images
img_width, img_height = 150, 150

# load the model we saved
model = load_model('model.h5')
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

mypath = "Predict/"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
print(onlyfiles)
# predicting images
dog_counter = 0
cat_counter  = 0
for file in onlyfiles:
    img = image.load_img("C:\\felixPythan\\Rohit Sir\\dog_cat_classifier\\predict\\cat_110.jpg")
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    classes = model.predict(images, batch_size=2)
    classes = classes[0][0]

    if classes == 0:
        print(file + ": " + 'cat')
        cat_counter += 1
    else:
        print(file + ": " + 'dog')
        dog_counter += 1
print("Total Dogs :",dog_counter)
print("Total Cats :",cat_counter)