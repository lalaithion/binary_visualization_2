import load
import display
from sys import argv


def main(filename, n):
    data = load.load_file(filename, n=n)
    data = data ** 0.5
    display.plot(data, dims=n)


if __name__ == "__main__":
    main(argv[1], int(argv[2]))
