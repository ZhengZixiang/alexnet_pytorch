# alexnet_pytorch
AlexNet is implementated with Pytorch. And it use the dataset CIFAR-10 to realize ten classfication task instead of ImageNet dataset in [paper](http://xanadu.cs.sjsu.edu/~drtylin/classes/cs267_old/ImageNet%20DNN%20NIPS2012(2).pdf).

#### Local Response Normalization
Pytorch has implemented Local Response Normalization (LRN) API in `torch.nn.LocalResponseNorm`, but I do not use this interface and write the LRN class in my way.

#### Biases Initailization
In paper, it initialized the convolution layer's biases in below way:
> We initialized the neuron biases in the second, fourth, and fifth convolutional layers, as well as in the fully-connected hidden layers, with the constant 1. This initialization accelerates the early stages of learning by providing the ReLUs with positive inputs. We initialized the neuron biases in the remaining layers with the constant 0.

I do not follow this setting, since I had fine-tuninged the biases and found it has the extremely slight effect to accuracy whether setting 0 or 1 for convolution biases in practice.
