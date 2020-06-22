import pickle
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
from keras import backend as K
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.imagenet_utils import decode_predictions, preprocess_input
import foolbox


# Helper functions
import helper
from attack import PixelAttacker

matplotlib.style.use('ggplot')
np.random.seed(100)

#加载数据

with open('data/imagenet_classes.pkl', 'rb') as f:
    class_names = pickle.load(f)
word_to_class = {w:i for i,w in enumerate(class_names)}
# Image URLs given by ImageNet
urls = [
    'http://farm4.static.flickr.com/3154/2585091536_78f528fdee.jpg',
]

# Where each file will be saved to
filenames = [
    'data/balloon.jpg'
]

# The labels of each corresponding image
labels = [
    word_to_class['balloon']
]
# Download all image files
# for url, filename in zip(urls, filenames):
#     print('Downloading', filename)
#     helper.download_from_url(url, filename)

originals = [load_img(filename, target_size=(224, 224)) for filename in filenames] # Load an image in PIL format
images = np.array([img_to_array(original) for original in originals]) # Convert the PIL image to a numpy array

helper.plot_image(images[0])


#加载模型
model = keras.applications.ResNet50()

#processed_images = preprocess_input(images.copy()) # Prepare the image for the model
processed_images=images
predictions = model.predict(processed_images) # Get the predicted probabilities for each class
label = decode_predictions(predictions) # Convert the probabilities to class labels

print(label)

#扰动图片
pixel = np.array([112, 112, 255, 255, 0]) # pixel = x,y,r,g,b
image_perturbed = helper.perturb_image(pixel, images)
helper.plot_image(image_perturbed[0])

#preprocessed_perturbed = preprocess_input(image_perturbed.copy())
preprocessed_perturbed =image_perturbed
predictions = model.predict(preprocessed_perturbed)
label = decode_predictions(predictions)

print(label)
'''
#攻击
# Should output /device:GPU:0
K.tensorflow_backend._get_available_gpus()
models = [model]

test = processed_images, np.array([labels])

attacker = PixelAttacker(models, test, class_names, dimensions=(224, 224))
result = attacker.attack(0, model, maxiter=3, verbose=True,plot=False,pixel_count=5)
#helper.plot_image(result[0])
helper.plot_image(result[0], result[4], class_names, result[5])
print(decode_predictions(np.expand_dims(result[9],axis=0)))
print("真实类别：",result[4])
'''
########################################
model = keras.applications.DenseNet201()
with tf.keras.backend.get_session().as_default():
    fmodel=foolbox.models.KerasModel(model,bounds=(0,255),predicts='logits')
    attack=foolbox.attacks.FGSM(fmodel)
    #adversarials=attack(processed_images,np.array([417]),unpack=False)[0]
    image, label = foolbox.utils.imagenet_example()
    adversarials = attack(np.expand_dims(image,axis=0), np.expand_dims(label,axis=0), unpack=False)[0]
    print(np.argmax(fmodel.forward_one(image)), label)
    print(adversarials.perturbed)
    helper.plot_image(image)
    helper.plot_image(adversarials.perturbed-image)


