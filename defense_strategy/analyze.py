import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.preprocessing.image import load_img, img_to_array
import foolbox
import helper
from getClearPixels import clean_pixels
from keras.models import Model, load_model
from single_attack import single_pixel_attack

keras.backend.set_learning_phase(0)

batchsize=110
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
images, labels=foolbox.utils.samples(dataset='cifar10',index=0,batchsize=batchsize)
images = np.array([img_to_array(original) for original in images]) # Convert the PIL image to a numpy array

model_filename = 'networks/models/resnet.h5'
resnet=load_model(model_filename)
models = [resnet]

#样本个数
n=100
#成功的个数
m=0
fmodel = foolbox.models.KerasModel(resnet, bounds=(0, 255))
# 攻击
attack_ = foolbox.v1.attacks.RandomProjectedGradientDescent(fmodel)
for images_id in range(1,2):
    print("images_id",images_id)
    # 绘画干净图片
    helper.plot_image(images[images_id])
    # 预测干净图片的类别
    clean_image_label=np.argmax(resnet.predict(np.asarray([images[images_id]], dtype=np.float32)))
    print("干净图片的类别：",clean_image_label)

    adversarial = attack_(images[images_id], labels[images_id])
    # try:
    #     # 生成对抗样本
    #     adversarial = attack_(images[images_id], labels[images_id])
    # except:
    #     continue

    # 预测对抗样本的类别
    adversarial_label=np.argmax(resnet.predict(np.asarray([adversarial])))
    print("对抗样本的类比：",adversarial_label)
    if clean_image_label==adversarial_label:
        continue
    # 绘画对抗样本
    helper.plot_image(adversarial)

    # attack_2 = foolbox.v1.attacks.BIM(fmodel)
    # adversarial_2 = attack_2(images[images_id], labels[images_id])
    # # 预测对抗样本的类别
    # adversarial_label_2 = np.argmax(resnet.predict(np.asarray([adversarial_2])))
    # print("对抗样本的类比2：", adversarial_label_2)
    # # 绘画对抗样本
    # helper.plot_image(adversarial_2)


    # 绘画扰动
    helper.plot_image(adversarial - images[images_id])
    # 过滤敏感点
    model = resnet
    labels = labels.reshape(batchsize, 1)
    list = single_pixel_attack(images, labels, class_names, images_id, model, pixel_count=100, verbose=True, plot=True)
    ###print("扰动点坐标", list[10])
    # 获取敏感点在干净图片上的值
    clean_rgb = clean_pixels(list[10], images[images_id])
    # 过滤敏感点 用干净值替代对抗样本对应位置 生成防御图
    defense_image = helper.perturb_image(np.array(clean_rgb), adversarial)[0]
    # 绘画防御图
    ###helper.plot_image(defense_image)
    # 绘画敏感点
    ###helper.plot_image(adversarial - defense_image)
    # 预测防御后的类别
    defense_label=np.argmax(resnet.predict(np.asarray([defense_image])))
    print("防御后的类别：",defense_label)
    if defense_label==clean_image_label:
        m=m+1

print("样本总数：",n," 防御成功的样本数：",m," 防御成功率：",m/n)
