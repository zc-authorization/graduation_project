import foolbox
import numpy as np
import torchvision.models as models

# instantiate model (supports PyTorch, Keras, TensorFlow (Graph and Eager), JAX, MXNet and many more)
model = models.resnet18(pretrained=True).eval()
preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
fmodel = foolbox.models.PyTorchModel(model, bounds=(0, 1), num_classes=1000, preprocessing=preprocessing)

# get a batch of images and labels and print the accuracy
images, labels = foolbox.utils.samples(dataset='imagenet', batchsize=16, data_format='channels_first', bounds=(0, 1))
print(np.mean(fmodel.forward(images).argmax(axis=-1) == labels))
# -> 0.9375

# apply the attack
attack = foolbox.attacks.FGSM(fmodel)
adversarials = attack(images, labels)
# if the i'th image is misclassfied without a perturbation, then adversarials[i] will be the same as images[i]
# if the attack fails to find an adversarial for the i'th image, then adversarials[i] will all be np.nan

# Foolbox guarantees that all returned adversarials are in fact in adversarials
print(np.mean(fmodel.forward(adversarials).argmax(axis=-1) == labels))
# -> 0.0


# In rare cases, it can happen that attacks return adversarials that are so close to the decision boundary,
# that they actually might end up on the other (correct) side if you pass them through the model again like
# above to get the adversarial class. This is because models are not numerically deterministic (on GPU, some
# operations such as `sum` are non-deterministic by default) and indepedent between samples (an input might
# be classified differently depending on the other inputs in the same batch).

# You can always get the actual adversarial class that was observed for that sample by Foolbox by
# passing `unpack=False` to get the actual `Adversarial` objects:
attack = foolbox.attacks.FGSM(fmodel, distance=foolbox.distances.Linf)
adversarials = attack(images, labels, unpack=False)

adversarial_classes = np.asarray([a.adversarial_class for a in adversarials])
print(labels)
print(adversarial_classes)
print(np.mean(adversarial_classes == labels))  # will always be 0.0

# The `Adversarial` objects also provide a `distance` attribute. Note that the distances
# can be 0 (misclassified without perturbation) and inf (attack failed).
distances = np.asarray([a.distance.value for a in adversarials])
print("{:.1e}, {:.1e}, {:.1e}".format(distances.min(), np.median(distances), distances.max()))
print("{} of {} attacks failed".format(sum(adv.distance.value == np.inf for adv in adversarials), len(adversarials)))
print("{} of {} inputs misclassified without perturbation".format(sum(adv.distance.value == 0 for adv in adversarials), len(adversarials)))


import matplotlib.pyplot as plt

image = images[15]

adversarial = attack(images[15:16], labels[15:16],unpack=False)[0]

print("扰动前: ",labels[15],"-->扰动后: ",adversarial.adversarial_class)

print(fmodel.forward(np.expand_dims(image,axis=0)).argmax(axis=-1))
print(fmodel.forward(np.expand_dims(adversarial.perturbed,axis=0)).argmax(axis=-1))


# CHW to HWC
image = image.transpose(1, 2, 0)
adversarial = adversarial.perturbed.transpose(1, 2, 0)

plt.figure()

plt.subplot(1, 3, 1)
plt.title('Original')
plt.imshow(image)
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Adversarial')
plt.imshow(adversarial)
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Difference')
difference = adversarial - image
plt.imshow(difference / abs(difference).max() * 0.2 + 0.5)
plt.axis('off')

plt.show()



##########################################
attack=foolbox.attacks.SinglePixelAttack(fmodel)
adversarials=attack(images[15:16], labels[15:16],max_pixels=4)
print(np.mean(fmodel.forward(adversarials).argmax(axis=-1) == labels))

