


def main(argv):
    parser = argparse.ArgumentParser(description='Train a DQN to control hyperparameters.')
    parser.add_argument('--create', action="store_true", help='create the model')
    parser.add_argument('--load', action="store_true", help='load the model')
    parser.add_argument('--train', action="store_true", help='train the model')
    parser.add_argument('--test', action='store_true', help='test the model')
    parser.add_argument('--verbose', action='store_true', help='verbose')
    parser.add_argument('--steps', nargs="?", default=100000, type=int, action='store',
                        help='number of steps to train (default=100000)')
    parser.add_argument('--memory', nargs="?", default=5000, type=int, action='store',
                        help='memory size (default=5000)')
    parser.add_argument('--window', nargs="?", default=5, type=int, action='store',
                        help='window size (default=5)')
    parser.add_argument('--src', nargs="?", default=default_path, action='store',
                        help='source file (default: {})'.format(default_path))
    parser.add_argument('--dst', nargs="?", default=default_path, action='store',
                        help='destination file (default: {})'.format(default_path))
    default_test_path = "{}-test.csv".format(default_path)
    parser.add_argument('--test-dst', nargs="?", default=default_test_path, action='store',
                        help='destination file (default: {})'.format(default_test_path))
    args = parser.parse_args(argv)

if __name__ == '__main__':
    main(sys.argv[1:])
