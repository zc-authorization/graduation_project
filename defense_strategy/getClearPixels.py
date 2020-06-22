import foolbox
import numpy as np
from keras.preprocessing.image import load_img, img_to_array

import helper


def clean_pixels(xs,image):
    xyrgb=[]
    # if(xs.ndim<2):
    #     xs=np.array([xs])
    xs=xs.astype(int)
    pixels=np.split(xs,len(xs) // 5)
    for pixel in pixels:
        x,y,*rgb=pixel
        xyrgb.append(x)
        xyrgb.append(y)
        xyrgb.append(image[x,y][0])
        xyrgb.append(image[x, y][1])
        xyrgb.append(image[x, y][2])
    return xyrgb


if __name__ == '__main__':
    images_id=1
    images, labels=foolbox.utils.samples(dataset='cifar10',index=0,batchsize=10)
    images = np.array([img_to_array(original) for original in images]) # Convert the PIL image to a numpy array
    helper.plot_image(images[images_id])

    xs=[26.95498939,21.91616303,114.05668558,244.69620732,249.57373676,0,0,0,0,0,20,20,20,20,20]
    xy=clean_pixels(np.array(xs),images[images_id])
    print(xy)