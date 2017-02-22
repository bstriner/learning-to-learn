import polynomial_sgd_cdqn
import learning_rate_graph


def main():
    t1 = "output/polynomial-sgd-cdqn/test.csv"
    polynomial_sgd_cdqn.main(["--create", "--train", "--test", "--steps", "5000000", "--memory", "5000", "--test-dst", t1])
    learning_rate_graph.main(["--input", t1])

    #t2 = "output/polynomial-sgd-cdqn/test-no-regenerate.csv"
    #polynomial_sgd_cdqn.main(["--create", "--test", "--test-dst", t1])
    #polynomial_sgd_cdqn.main(["--load", "--test", "--test-dst", t2, "--no-regenerate"])
    #learning_rate_graph.main(["--input", t2])
    #

if __name__ == "__main__":
    main()
