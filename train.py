import os
import numpy as np
import pickle
import struct
import argparse


def log(obj, filename='ckpts/log.txt'):
    """save log"""
    print(obj)
    with open(filename, 'a') as f:
        print(obj, file=f)


class DataSet:
    def __init__(self, path, batch_size):
        self.path = path
        self.batch_size = batch_size
        image, labels = self.load_data(self.path, split='train')

        self.train_image, self.train_label = image[:, :55000], labels[:, :55000]
        self.val_image, self.val_label = image[:, 55000:], labels[:, 55000:]
        self.test_image, self.test_label = self.load_data(self.path, split='t10k')
        print('MNIST: {} training images, {} validation images, {} test images.'.format(self.train_image.shape[1],
                                                                                        self.val_image.shape[1],
                                                                                        self.test_image.shape[1]))

    def load_data(self, path, split='train'):
        """load MNIST from `path` according to train/test split"""
        labels_path = os.path.join(path,'{}-labels-idx1-ubyte'.format(split))
        images_path = os.path.join(path,'{}-images-idx3-ubyte'.format(split))
        with open(labels_path, 'rb') as lbpath:
            magic, n = struct.unpack('>II', lbpath.read(8))
            labels = np.fromfile(lbpath, dtype=np.uint8)

        with open(images_path, 'rb') as imgpath:
            magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
            images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
        labels = self.get_one_hot_label(labels, 10)
        return images.T, labels

    def get_one_hot_label(self, y, c):
        """get one hot labels"""
        y = np.eye(c)[y.reshape(-1)].T
        return y

    def normalize(self, x):
        """normalize x to [-1, 1]"""
        x = x.astype('float32')
        return (x / 255.0 - 0.5) / 0.5

    def shuffle_data(self, x, y):
        """shuffle the data randomly and drop_last=False"""
        batch_size = self.batch_size
        m = x.shape[1]
        mini_batches = []

        permutation = list(np.random.permutation(m))
        shuffled_x = x[:, permutation]
        shuffled_y = y[:, permutation].reshape((y.shape[0], m))

        num_batches = int(np.floor(m / batch_size))
        for k in range(0, num_batches):
            mini_batch_X = shuffled_x[:, k * batch_size:(k + 1) * batch_size]
            mini_batch_Y = shuffled_y[:, k * batch_size:(k + 1) * batch_size]

            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        if m % batch_size != 0:
            mini_batch_X = shuffled_x[:, batch_size * num_batches:]
            mini_batch_Y = shuffled_y[:, batch_size * num_batches:]

            mini_batch = (self.normalize(mini_batch_X), mini_batch_Y)
            mini_batches.append(mini_batch)

        return mini_batches


class MLP:
    def __init__(self, hidden_dims=100):
        """two layer MLP with ReLu activation function"""
        self.hidden_dims = hidden_dims
        self.W1 = np.random.normal(0.0, np.sqrt(2 / 28 * 28), (hidden_dims, 28 * 28))
        self.b1 = np.zeros((hidden_dims, 1))
        self.W2 = np.random.normal(0.0, np.sqrt(2 / hidden_dims), (10, hidden_dims))
        self.b2 = np.zeros((10, 1))

    def save_model(self, filename='latest.pkl'):
        """save model to filename"""
        parameters = {'W1': self.W1,
                      'b1': self.b1,
                      'W2': self.W2,
                      'b2': self.b2}
        with open(filename, 'wb') as file:
            pickle.dump(parameters, file)
        return

    def load_model(self, filename='latest.pkl'):
        """load model from filename"""
        with open(filename, 'rb') as file:
            parameters = pickle.load(file)
            self.W1 = parameters['W1']
            self.b1 = parameters['b1']
            self.W2 = parameters['W2']
            self.b2 = parameters['b2']
        return

    def forward(self, x):
        f1 = np.dot(self.W1, x) + self.b1
        # relu
        a1 = np.maximum(0, f1)
        f2 = np.dot(self.W2, a1) + self.b2
        # softmax
        exp_f2 = np.exp(f2 - np.max(f2, axis=0, keepdims=True))  # avoid big numbers after np.exp
        prob = exp_f2 / (1e-8 + np.sum(exp_f2, axis=0, keepdims=True))
        return prob, f1, a1, f2

    def one_step_optimization(self, x, y, weight_decay=0.0, lr=1e-4):
        """calculate gradient and optimize parameters"""
        m = x.shape[1]
        prob, f1, a1, f2 = self.forward(x)

        # loss
        logprobs = np.multiply(-np.log(prob + 1e-8), y)
        loss = 1. / m * np.sum(logprobs) + (np.sum(np.square(self.W1)) + np.sum(np.square(self.W2))) * weight_decay / 2.0

        # gradient
        df2 = 1. / m * (prob - y)
        dW2 = np.dot(df2, a1.T) + weight_decay * self.W2
        db2 = np.sum(df2, axis=1, keepdims=True)

        da1 = np.dot(self.W2.T, df2)
        df1 = np.multiply(da1, np.int64(a1 > 0))
        dW1 = np.dot(df1, x.T) + weight_decay * self.W1
        db1 = np.sum(df1, axis=1, keepdims=True)

        # sgd
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        return loss

    def test(self, x, y, weight_decay=0.0):
        m = x.shape[1]
        prob, f1, a1, f2 = self.forward(x)
        # loss
        logprobs = np.multiply(-np.log(prob + 1e-8), y)
        loss = 1. / m * np.sum(logprobs) + (
                    np.sum(np.square(self.W1)) + np.sum(np.square(self.W2))) * weight_decay / 2.0

        pred_y = np.argmax(prob, axis=0)
        label = np.argmax(y, axis=0)
        accuracy = np.mean(label == pred_y)
        return accuracy, loss


def train(dataset, model, init_lr=0.1, lr_decay=0.5, decay_epochs=50, weight_decay=0.0, num_epochs=200):
    """
    training function
    """

    best_valid_accuracy = 0

    lr = init_lr

    filename = 'ckpts/init_lr={}_hidden_dims={}_weight_decay={}.txt'.format(init_lr, model.hidden_dims, weight_decay)

    for i in range(num_epochs):
        minibatches = dataset.shuffle_data(dataset.train_image, dataset.train_label)

        losses = []
        for batch in minibatches:
            (x, y) = batch
            loss = model.one_step_optimization(x, y, weight_decay, lr)
            losses.append(loss)

        accuracy, _ = model.test(dataset.train_image, dataset.train_label, weight_decay)
        log("epoch: {}, training set accuracy: {:.2%}, loss: {:.4f}".format(i + 1, accuracy, np.mean(losses)), filename)

        accuracy, loss = model.test(dataset.val_image, dataset.val_label, weight_decay)
        log("validation set accuracy: {:.2%}, loss: {:.4f}".format(accuracy, loss), filename)

        if accuracy > best_valid_accuracy:
            best_valid_accuracy = accuracy
            model.save_model(filename=filename.replace('txt', 'pkl'))
            log('current best model on validation set is from epoch: {}'.format(i + 1), filename)

        accuracy, loss = model.test(dataset.test_image, dataset.test_label, weight_decay)
        log("test set accuracy: {:.2%}, loss: {:.4f}".format(accuracy, loss), filename)

        if (i + 1) % decay_epochs == 0:
            lr *= lr_decay

    return best_valid_accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path2data', default='./')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--hidden_dims', type=int, default=1000)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--init_lr', type=float, default=1e-4)
    parser.add_argument('--decay_epochs', type=int, default=30)
    parser.add_argument('--lr_decay', type=float, default=0.5)

    args = parser.parse_args()

    np.random.seed(1207)

    dataset = DataSet(args.path2data, args.batch_size)
    model = MLP(args.hidden_dims)

    train(dataset, model, args.init_lr, args.lr_decay, args.decay_epochs, args.weight_decay, args.num_epochs)
