#!/usr/bin/env python3
"""
mat_mul = __import__('8-ridin_bareback').mat_mul

mat1 = [[1, 2],
        [3, 4],
        [5, 6]]
mat2 = [[1, 2, 3, 4],
        [5, 6, 7, 8]]
print(mat_mul(mat1, mat2))
"""


def mat_mul(mat1, mat2):
    """Performs matrix multiplication between two 2D matrices.

    Args:
        mat1: A 2D list (matrix) of integers or floats.
        mat2: A 2D list (matrix) of integers or floats.

    Returns:
        A new 2D list that is the result of multiplying mat1 by mat2.
        If the matrices cannot be multiplied, returns None.
    """

    if len(mat1[0]) != len(mat2):
        return None

    result = [[0 for _ in range(len(mat2[0]))] for _ in range(len(mat1))]

    for i in range(len(mat1)):
        for j in range(len(mat2[0])):
            for k in range(len(mat2)):
                result[i][j] += mat1[i][k] * mat2[k][j]

    return result
