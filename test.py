from train import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path2data', default='./')
    parser.add_argument('--path2pkl', default=None)
    parser.add_argument('--hidden_dims', type=int, default=300)

    args = parser.parse_args()

    np.random.seed(1207)

    dataset = DataSet(args.path2data, 128)
    model = MLP(args.hidden_dims)

    model.load_model(args.path2pkl)

    accuracy, _ = model.test(dataset.test_image, dataset.test_label, 0.0)
    print("test set accuracy after parameters search: {:.2%}".format(accuracy))
