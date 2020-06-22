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
from getClearPixels import clean_pixels
from networks.resnet import ResNet
from networks.lenet import LeNet
from keras.datasets import cifar10
from keras.models import Model, load_model

######Tensor("input_1_1:0", shape=(?, 32, 32, 3), dtype=float32)
from single_attack import single_pixel_attack

keras.backend.set_learning_phase(0)
# (x_train, y_train), (x_test, y_test) = cifar10.load_data()
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
images_id=2
images, labels=foolbox.utils.samples(dataset='cifar10',index=0,batchsize=10)
images = np.array([img_to_array(original) for original in images]) # Convert the PIL image to a numpy array

# 绘画干净图片
helper.plot_image(images[images_id])
# print(np.shape(images))
# print(np.shape(labels))
#
# print(images[images_id][27,22])
# print(images[images_id][21][24][1])
# print(images[images_id][22][7][2])


#自定义网络
# lenet = LeNet()
# resnet = ResNet()
# resnet.train()
model_filename = 'networks/models/resnet.h5'
resnet=load_model(model_filename)

models = [resnet]

# network_stats, correct_imgs = helper.evaluate_models(models,x_test, y_test )
# correct_imgs = pd.DataFrame(correct_imgs, columns=['name', 'img', 'label', 'confidence', 'pred'])
# network_stats = pd.DataFrame(network_stats, columns=['name', 'accuracy', 'param_count'])
#
# print(network_stats)

#预测干净图片
print("干净图片的类别：",class_names[np.argmax(resnet.predict(np.asarray([images[images_id]],dtype=np.float32)))])

# with tf.keras.backend.get_session().as_default():
fmodel = foolbox.models.KerasModel(resnet,bounds=(0, 255))
# apply attack on source image
attack_ = foolbox.v1.attacks.FGSM(fmodel)
# image=np.asarray(images[1],dtype=np.float32)
# image=np.transpose(image,(2,0,1))
# label=np.array(labels[1])
#对抗样本
adversarial = attack_(images[images_id], labels[images_id])
#预测对抗样本
print("对抗样本的类别",class_names[np.argmax(resnet.predict(np.asarray([adversarial])))])
#绘画对抗样本
helper.plot_image(adversarial)

##########将原始样本与对抗样本化成一张图
plt.figure()
plt.subplot(1,2,1)
plt.xticks([])
plt.yticks([])
plt.imshow(images[images_id].astype(np.uint8))
plt.subplot(1,2,2)
plt.imshow(adversarial.astype(np.uint8))
plt.xticks([])
plt.yticks([])
plt.show()
##########



#绘画扰动
helper.plot_image(adversarial-images[images_id])

#过滤敏感点
model=resnet
labels=labels.reshape(10,1)
list=single_pixel_attack(images,labels,class_names,images_id,model,pixel_count=10,verbose=True,plot=True)
print("扰动点坐标",list[10])

#获取敏感点在干净图片上的值
clean_rgb=clean_pixels(list[10],images[images_id])
#过滤敏感点 用干净值替代对抗样本对应位置 生成防御图
defense_image = helper.perturb_image(np.array(clean_rgb), adversarial)[0]
#绘画防御图
helper.plot_image(defense_image)
#绘画敏感点
helper.plot_image(adversarial-defense_image)

##########将扰动样本与敏感点化成一张图
plt.figure()
plt.subplot(1,2,1)
plt.xticks([])
plt.yticks([])
adv_images=adversarial-images[images_id]
plt.imshow(adv_images.astype(np.uint8))
plt.subplot(1,2,2)
sen_points=adversarial-defense_image
plt.imshow(sen_points.astype(np.uint8))
plt.xticks([])
plt.yticks([])
plt.show()
##########


#预测防御效果
print("防御后的类别",class_names[np.argmax(resnet.predict(np.asarray([defense_image])))])