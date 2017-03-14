import polynomial_sgd_cdqn
import polynomial_sgd_test_cdqn
import learning_rate_graph


def test_set():
    t1 = "output/polynomial-sgd-test-cdqn/test.csv"
    m1 = "output/polynomial-sgd-test-cdqn/model.h5"
    polynomial_sgd_test_cdqn.main(
        ["--create", "--train", "--steps", "5000000", "--memory", "5000", "--window", "5", "--dst", m1])


def main():
    test_set()
    """
    t1 = "output/polynomial-sgd-cdqn/test.csv"
    #polynomial_sgd_cdqn.main(["--create", "--train", "--test", "--steps", "5000000", "--memory", "5000", "--test-dst", t1])
    #learning_rate_graph.main(["--input", t1])
    cp = "output/polynomial-sgd-cdqn/cdqn.h5-cp/epoch-00009000.h5"
    polynomial_sgd_cdqn.main(
        ["--load", "--test", "--src", cp, "--test-dst", t1])
    learning_rate_graph.main(["--input", t1])
    #t2 = "output/polynomial-sgd-cdqn/test-no-regenerate.csv"
    #polynomial_sgd_cdqn.main(["--create", "--test", "--test-dst", t1])
    #polynomial_sgd_cdqn.main(["--load", "--test", "--test-dst", t2, "--no-regenerate"])
    #learning_rate_graph.main(["--input", t2])
    #
    """


if __name__ == "__main__":
    main()
