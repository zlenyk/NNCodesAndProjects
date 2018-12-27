# NNCodesAndProjects
Codes for studying and projects from Neural Networks

Implementation in python of many popular neural networks, their improvements based on research and a few projects.

Each project will contain own readme explaining how it works.

### Prerequisites

All dependencies can be installed by running

```
pip install -r requirements.txt
```

### CIFAR 10

Convolutional NN is used to classify CIFAR10 dataset. Deep, residual network with additional improvements - Adam optimizer, using exponential moving averages, learnt variance and bias when normalizing batch.
Code reaches accuracy of 93.5%

```
python convolutional.py
```

This trains and tests the network. Trained parameters are saved using saver and can be reused.
