# Get CIFAR100
mkdir -p datasets
wget -P datasets https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
tar -xzvf cifar-100-python.tar.gz
rm cifar-100-python.tar.gz

# Get pretrained VGG Net
mkdir -p vgg
wget -P vgg http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat
