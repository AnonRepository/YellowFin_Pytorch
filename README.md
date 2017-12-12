# YellowFin

YellowFin is an auto-tuning optimizer based on momentum SGD **which requires no manual specification of learning rate and momentum**. It measures the objective landscape on-the-fly and tunes momentum as well as learning rate using local quadratic approximation.

The implementation here can be **a drop-in replacement for any optimizer in PyTorch**. It supports ```step``` and ```zero_grad``` functions like any PyTorch optimizer after ```from yellowfin import YFOptimizer```. **We also provide interface to manually set the learning rate schedule at every iteration for finer control (see Detailed Guideline Section)**.

For more usage details, please refer to the inline documentation of ```tuner_utils/yellowfin.py```. Example usage can be found here for [ResNext on CIFAR10](https://github.com/AnonRepository/YellowFin_Pytorch/blob/master/pytorch-cifar/main.py#L106) and [Tied LSTM on PTB](https://github.com/AnonRepository/YellowFin_Pytorch/blob/master/word_language_model/main.py#L278).

**YellowFin is under active development. Many members of the community have kindly submitted issues and pull requests. We are incorporating fixes and smoothing things out. As a result the repository code is in flux. Please make sure you use the latest version and submit any issues you might have!**


## Setup instructions for experiments
Please clone the master branch and follow the instructions to run YellowFin on [ResNext](https://arxiv.org/abs/1611.05431) for CIFAR10 and [tied LSTM](https://arxiv.org/pdf/1611.01462.pdf) on Penn Treebank for language modeling. The models are adapted from [ResNext repo](https://github.com/kuangliu/pytorch-cifar) and [PyTorch example tied LSTM repo](https://github.com/pytorch/examples/tree/master/word_language_model) respectively. Thanks to the researchers for developing the models. **For more experiments on more convolutional and recurrent neural networks, please refer to our [Tensorflow implementation](https://github.com/AnonRepository/YellowFin) of YellowFin**.

Note YellowFin is tested with PyTorch v0.2.0 for compatibility. It is tested under Python 2.7.

### Run CIFAR10 ResNext experiments
The experiments on 110 layer ResNet with CIFAR10 and 164 layer ResNet with CIFAR100 can be launched using
```
cd pytorch-cifar
python main.py --logdir=path_to_logs --opt_method=YF
```

### Run Penn Treebank tied LSTM experiments
The experiments on multiple-layer LSTM on Penn Treebank can be launched using
```
cd word_language_model
python main.py --emsize 650 --nhid 650 --dropout 0.5 --epochs 40 --tied --opt_method=YF --logdir=path_to_logs --cuda
```

For more experiments, please refer to our [YellowFin Tensorflow Repo](https://github.com/AnonRepository/YellowFin).

## Detailed guidelines
* **Basic use**: ```optimizer = YFOptimizer(parameter_list)``` uses the uniform setting (i.e. without tuning) for all the PyTorch and Tensorflow experiments in our paper. 

* **Interface for manual finer control**: If you want to more finely control the learning rate (say using a manually set constant learning rate), or you want to use the typical lr-dropping technique after a ceritain number of epochs, please use ```set_lr_factor()``` in the YFOptimizer class. E.g. if you want to use a manually set constant learning rate, you can run ```set_lr_factor(desired_lr / self._lr)``` before ```self.step()``` at each iteration. Or e.g., if you want to always multiply a factor 2.0 to the learning rate originally tuned by YellowFin, you may use ```optimizer.set_lr_factor(2.0)``` right after ```optimizer = YFOptimizer(parameter_list)``` and before training with YellowFin. More details can be found [here](https://github.com/AnonRepository/YellowFin_Pytorch/blob/master/pytorch-cifar/main.py#L163). **(The argument ```lr``` and ```mu``` during ```YFOptimizer``` initialization are dummy, only for backward compatibility)**

* **Gradient clipping**: The default setting uses adaptive gradient clipping to prevent gradient explosion, thresholding norm of gradient to the square root of our estimated maximal curvature. There are three cases regarding gradient clipping. We recommend first turning off gradient clipping, and only turning it on when necessary. 

  * If you want to manually set threshold to clip the gradient, please first use ```adapt_clip=False``` to turn off the auto-clipping feature. Then, you can consider either using the ```clip_thresh=thresh_on_the_gradient_norm``` argument when initializing the YFOptimizer to clip acoording to your set threshold inside YFOptimizer, or clipping the gradient outside of YFOptimizer before ```step()``` is called.
  
  * If you want to totally turn off gradient clipping in YFOptimizer, please use ```clip_thresh=None, adapt_clip=False``` when initializing the YFOptimizer.

* **Normalization**: When using log probability style losses, please make sure the loss is properly normalized. In some RNN/LSTM cases, the cross_entropy need to be averaged by the number of samples in a minibatch. Sometimes, it also needs to be averaged over the number of classes and the sequence length of each sample in some PyTorch loss functions. E.g. in nn.MultiLabelSoftMarginLoss, ```size_average=True``` needs to be set.

## Implementation for other platforms
For Tensorflow users, we implemented [YellowFin Tensorflow Repo](https://github.com/AnonRepository/YellowFin).
