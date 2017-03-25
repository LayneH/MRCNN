# MRCNN
Implementation of [Manifold Regularized Convolutional Neural Networks][MRCNN].

---
## Dependency

* [Numpy][np]
* [Scipy][scipy]
* [Tensorflow][tf] >= 1.0
* [Pretrained VGG Net][model]
* [CIFAR-100 Dataset][cifar]

---
## Usage
Enter `python MRCNN.py` in bash to run the program with default setting.

Use `--gpu_id` to specify the GPU devices to use. By default, all GPUs available will be used.

Use `--verbose` to set the level of verbosity. By default, no loggin info will be printed.

Use `--vgg_path` to specify the path of pretrained VGG Net. By default, the model is located under `vgg/` directory

Use `--result_path` to specify file to save the result. By default, the file is "result.csv".

Use `--help` to aquire more information.

## Citation
If you'd like to compare our framework with your work, please cite our [paper][MRCNN] as following:

[MRCNN]:http://pdf...
[model]:http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat
[np]:https://github.com/numpy/numpy/blob/master/INSTALL.rst.txt
[scipy]:https://github.com/scipy/scipy/blob/master/INSTALL.rst.txt
[tf]:http://tensorflow.org
[cifar]:https://www.cs.toronto.edu/~kriz/cifar.html
