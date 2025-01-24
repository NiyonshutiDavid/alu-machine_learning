#!/usr/bin/env python3
"""
matrix_transpose = __import__('3-flip_me_over').matrix_transpose

mat1 = [[1, 2], [3, 4]]
print(mat1)
print(matrix_transpose(mat1))
mat2 = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15],
        [16, 17, 18, 19, 20], [21, 22, 23, 24, 25], [26, 27, 28, 29, 30]]
print(mat2)
print(matrix_transpose(mat2))
"""


def matrix_transpose(matrix):
    """Transposes a matrix.

    Args:
        matrix: A 2D list (matrix) to transpose.

    Returns:
        A new transposed matrix where the rows and columns are swapped.
    """
    return [[row[i] for row in matrix] for i in range(len(matrix[0]))]
