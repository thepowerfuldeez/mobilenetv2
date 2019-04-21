This is my implementation of MobileNetV2 (as well as V1) from paper 
[MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)
and paper [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)
With comparison with ResNet and MobilenetV1 (will be soon)


# Depthwise Separable Convolution
This is a form of factorized convolutions
which factorize a standard convolution into a depthwise
convolution and a 1×1 convolution called a pointwise convolution. 
So the difference from standard convolution is instead of
filtering and combining inputs into a new set of outputs
in one step the one splits this into two layers, a separate layer 
for filtering and a separate layer for combining. 
And for every input channel used different convolution filter.

Depthwise convolution is extremely efficient relative to
standard convolution. However it only filters input channels, it does not combine them to create new features. So
an additional layer that computes a linear combination of
the output of depthwise convolution via 1 × 1 convolution
is needed in order to generate these new features

Comparable models: ResNet50 (23M params), ResNet18 (11M params), MobilenetV1 (2.2M, 1.8M, 0.8M, 0.2M, 0.05M params)
Validation accuracy of different models on different datasets
| 				    | Fashion MNIST   | CIFAR10     | CIFAR100   |
| ----------------- | :--------------:| :----------:| :--------: |
| ResNet18      	|		          |	  79.36%	    |   
| MobileNet(w=1)    |                 |   73.03%    |
| MobileNet(w=0.75) |                 |   69.26%    |
| MobileNet(w=0.5) 	|                 |   67.54%    |
| MobileNet(w=0.25) |                 |   62.69%    |
| MobileNet(w=0.032)|                 |   39.82%    |