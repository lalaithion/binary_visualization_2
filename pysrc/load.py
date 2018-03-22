import numpy as np


def ord_safe(char):
    """
    ord_safe is the same as ord but returns None if the string is not length 1.
    """
    if len(char) != 1:
        return None

    return ord(char)


def load_stream(stream, n=2):
    """
    load_stream creates a correlation matrix from a file like object.

    stream: a file like object that yeilds bytes (not characters!)
    n: the number of bytes that are looked at at a time. The file must contain
    at least n bytes.

    If n == 1, load_stream returns a list of the frequencies of each byte in
    the stream.
    If n == 2, load_stream returns a matrix of the frequencies of the pair of
    bytes (index0, index1) in the stream.
    If n == 3, load_stream returns a tensor of the frequencies of the triplet
    of bytes (index0, index1, index2) in the stream.
    Etc.
    """

    dims = tuple(256 for i in range(n))

    count_matrix = np.zeros(dims, dtype=np.int64)

    seq = tuple(ord_safe(stream.read(1)) for i in range(n))
    total = 0
    byte = seq[-1]

    while byte is not None:
        count_matrix[seq] += 1
        total += 1

        byte = ord_safe(stream.read(1))
        seq = (*seq[1:], byte)

    return count_matrix / total


def load_file(filename, n=2):
    """
    load_file creates a correlation matrix from a file named filename.

    filename: the name of or path to a file.
    n: the number of bytes that are looked at at a time. The file must contain
    at least n bytes.

    If n == 1, load_stream returns a list of the frequencies of each byte in
    the stream.
    If n == 2, load_stream returns a matrix of the frequencies of the pair of
    bytes (index0, index1) in the stream.
    If n == 3, load_stream returns a tensor of the frequencies of the triplet
    of bytes (index0, index1, index2) in the stream.
    Etc.
    """

    with open(filename, "rb") as fd:
        res = load_stream(fd, n)

    return res
