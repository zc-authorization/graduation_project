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
from keras.applications.resnet50 import ResNet50
# Custom Networks
from differential_evolution import differential_evolution
from networks.resnet import ResNet
from networks.lenet import LeNet
from keras.datasets import cifar10
from keras.models import Model, load_model


def attack(images,labels,class_names,img_id, model, target=None, pixel_count=1,
           maxiter=75, popsize=400, verbose=False, plot=False, dimensions=(32, 32)):
    # Change the target class based on whether this is a targeted attack or not
    targeted_attack = target is not None
    target_class = target if targeted_attack else labels[img_id, 0]

    ack = PixelAttacker([model],(images,labels),class_names)

    # Define bounds for a flat vector of x,y,r,g,b values
    # For more pixels, repeat this layout
    dim_x, dim_y = dimensions
    bounds = [(0, dim_x), (0, dim_y), (0, 256), (0, 256), (0, 256)] * pixel_count

    # Population multiplier, in terms of the size of the perturbation vector x
    popmul = max(1, popsize // len(bounds))

    # Format the predict/callback functions for the differential evolution algorithm
    def predict_fn(xs):
        return ack.predict_classes(xs, images[img_id], target_class, model, target is None)

    def callback_fn(x, convergence):
        return ack.attack_success(x, images[img_id], target_class, model, targeted_attack, verbose)

    # Call Scipy's Implementation of Differential Evolution
    attack_result = differential_evolution(
        predict_fn, bounds, maxiter=maxiter, popsize=popmul,
        recombination=1, atol=-1, callback=callback_fn, polish=False)

    # Calculate some useful statistics to return from this function
    attack_image = helper.perturb_image(attack_result.x, images[img_id])[0]
    prior_probs = model.predict(np.array([images[img_id]]))[0]
    predicted_probs = model.predict(np.array([attack_image]))[0]
    predicted_class = np.argmax(predicted_probs)
    actual_class = labels[img_id, 0]
    success = predicted_class != actual_class
    cdiff = prior_probs[actual_class] - predicted_probs[actual_class]

    # Show the best attempt at a solution (successful or not)
    if plot:
        helper.plot_image(attack_image, actual_class, class_names, predicted_class)

    return [attack_image, model.name, pixel_count, img_id, actual_class, predicted_class, success, cdiff, prior_probs,
            predicted_probs, attack_result.x]


class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
#(x_train, y_train), (x_test, y_test) = cifar10.load_data()
images, labels=foolbox.utils.samples(dataset='cifar10',index=0,batchsize=10)
images = np.array([img_to_array(original) for original in images]) # Convert the PIL image to a numpy array
#print(np.shape(y_test))
print(np.shape(labels.reshape(10,1)))
labels=labels.reshape(10,1)
#自定义网络
# lenet = LeNet()
resnet = ResNet()

# model_filename = 'networks/models/resnet.h5'
# resnet=load_model(model_filename)

models = [resnet]

network_stats, correct_imgs = helper.evaluate_models(models,images, labels )
correct_imgs = pd.DataFrame(correct_imgs, columns=['name', 'img', 'label', 'confidence', 'pred'])
network_stats = pd.DataFrame(network_stats, columns=['name', 'accuracy', 'param_count'])

print(network_stats)

pixel = np.array([16, 20,  0, 255, 255])
model = resnet

image_id=1
true_class = labels[image_id, 0]
prior_confidence = model.predict_one(images[image_id])[true_class]
confidence = helper.predict_classes(pixel, images[image_id], true_class, model)[0]

print('Confidence in true class', class_names[true_class], 'is', confidence)
print('Prior confidence was', prior_confidence)
helper.plot_image(helper.perturb_image(pixel, images[image_id])[0])




list=attack(images,labels,class_names,image_id,model,pixel_count=3,verbose=True,plot=True)
print(list[10])
