# NNCodesAndProjects
Codes for studying and projects from Neural Networks

Implementation in python of many popular neural networks, their improvements based on research and a few projects.

Each project will contain own readme explaining how it works.

### Prerequisites

All dependencies can be installed by running

```
pip install -r requirements.txt
```

### Generative Adversarial Networks - CIFAR10

Own implementation of GANs for generating CIFAR10 images.
```
python GAN/gan.py
```
NOTE - cifar_input.py reads CIFAR10 data, downloaded to a separate folder donwloaded from: https://www.cs.toronto.edu/~kriz/cifar.html

Results:
Measuring result of GANs is usually subjective. I decided to measure the performance of this network by training new Convolutional NN only on generated images (output from GAN) and testing what how well will such CNN perform on real CIFAR10 dataset. Result is around 65% of accuracy on test set (remember that training and validation set are not touched).

Resources:

Generative Adversarial Nets - https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf

Improved Techniques for Training GANs - https://arxiv.org/pdf/1606.03498.pdf

GAN hacks - https://github.com/soumith/ganhacks

### CIFAR 10 - classification

Convolutional NN is used to classify CIFAR10 dataset. Deep, residual network with additional improvements - Adam optimizer, using exponential moving averages, learnt variance and bias when normalizing batch.
Code reaches accuracy of 93.5%

```
python CIFAR10/convolutional.py
```

This trains and tests the network. Trained parameters are saved using saver and can be reused.
Plenty of state-of-art improvements to standard CNN are implemented in this project.
I found those to be most influential - Residual networks, wide vs deep networks, replacing pooling with convolutions with stride 2.

Resources:

Deep Residual Learning for Image Recognition (Microsoft Research) - https://arxiv.org/abs/1512.03385

Striving for simplicity - https://arxiv.org/pdf/1412.6806.pdf

### Text generation - LSTM
Implemetation of recurent neural network - long short term memory network.
Trained on a very famous Polish book - Pan Tadeusz. Tries to predict next letter of the text based on last read letters (how many exactly is a hyper-parameter). After such training (after each epoch) new text is generated, which starts with only beginning of one sentence and the network tries to recreate the text.
The result are very exciting - text has a lot features of Polish language and would be recognized by any Polish speaker as an attempt to write some actual text.

```
python LSTM/lstm2.py
```

Resources:

http://colah.github.io/posts/2015-08-Understanding-LSTMs/
