**NOTE: For users interested in multi-GPU, we recommend looking at the newer [cifar10_estimator](https://github.com/tensorflow/models/tree/master/tutorials/image/cifar10_estimator) example instead.**

---

CIFAR-10 is a common benchmark in machine learning for image recognition.

http://www.cs.toronto.edu/~kriz/cifar.html

Code in this directory demonstrates how to use TensorFlow to train and evaluate a convolutional neural network (CNN) on both CPU and GPU. We also demonstrate how to train a CNN over multiple GPUs.

Detailed instructions on how to get started available at:


DETAILS ABOUT THIS CNN ARCHITECTURE >>>>>>
cifar10.py = the main of the arch with the choquet pooling 
cifar10_train.py = train dataset
cifar10_eval.py = test dataset

http://tensorflow.org/tutorials/deep_cnn/

cifar10_eval.py output:
![](https://uploaddeimagens.com.br/imagens/cifar10_eval-output1-png)

cifar10_train.py output:
![](https://uploaddeimagens.com.br/imagens/cifar10_train-output-png)

