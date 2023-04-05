## Homework 1: 构建两层神经网络分类器

### 董巧乐 22110980004

### Environment

First prepare a python environment and install required packages:

```
pip install numpy
pip install pickle
```


### Dataset

Then download [MNIST](http://yann.lecun.com/exdb/mnist/) and unzip them use:
```
gunzip *.gz
```

### Training

Training a model utilize train.py :
```
python train.py --path2data [path to MNIST dataset]
```

### Parameter Search

Search the parameter by:
```
python parameter_search.py
```

It will print the best hyper-parameter according to accuracy of validation set.

### Test

Test the model on test set and get the final accuracy.

```
python test.py --path2pkl [saved model path] --hidden_dims [hidden dimension of model]
```
