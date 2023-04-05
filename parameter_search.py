from train import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path2data', default='./')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--decay_epochs', type=int, default=30)
    parser.add_argument('--lr_decay', type=float, default=0.5)

    args = parser.parse_args()

    np.random.seed(1207)

    init_lr = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    hidden_dims = [100, 500, 1000, 2000]
    weight_decay = [0.0, 1e-7, 1e-6, 1e-5]

    dataset = DataSet(args.path2data, args.batch_size)

    best_valid_accuracy = 0.0
    best_parameter = {}

    for lr in init_lr:
        for dim in hidden_dims:
            for l2 in weight_decay:
                model = MLP(dim)
                valid_accuracy = train(dataset, model, lr, args.lr_decay, args.decay_epochs, l2, args.num_epochs)
                if valid_accuracy > best_valid_accuracy:
                    best_valid_accuracy = valid_accuracy
                    best_parameter = {'init_lr': lr,
                                      'hidden_dims': dim,
                                      'weight_decay': l2}

    print('best valid accuracy:', best_valid_accuracy)
    print('best hyper-parameters:', best_parameter)




