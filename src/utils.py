import numpy as np
from sympy import symbols, Matrix, sympify


def augmented_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Create an augmented matrix by horizontally stacking two matrices A and B.

    Args:
        A (np.ndarray): First matrix.
        B (np.ndarray): Second matrix.

    Returns:
        np.ndarray: Augmented matrix obtained by horizontally stacking A and B.
    """
    augmented_M = np.hstack((A, B))
    return augmented_M


def back_substitution(M: np.ndarray) -> np.ndarray:
    """Perform back substitution on an augmented matrix (with unique solution) in reduced row
    echelon form to find the solution to the linear system.

    Args:
        M (np.ndarray): The augmented matrix in row echelon form with unitary pivots (n x n+1)

    Returns:
        np.ndarray: The solution vector of the linear system.
    """
    M = M.copy()
    num_rows = M.shape[0]
    for row in reversed(range(num_rows)):
        substitution_row = M[row]
        index = get_index_first_non_zero_value_from_row(M, row, True)
        for j in range(row):
            row_to_reduce = M[j]
            value = row_to_reduce[index]
            row_to_reduce -= value * substitution_row
            M[j, :] = row_to_reduce
    solution = M[:, -1]
    return solution


def get_index_first_non_zero_value_from_column(
    M: np.ndarray, column: int, starting_row: int
) -> int:
    """Retrieve the index of the first non-zero value in a specified column of the given matrix.

    Args:
        M (np.ndarray): The input matrix to search for non-zero values.
        column (int): The index of the column to search.
        starting_row (int): The starting row index for the search.

    Returns:
        int: The index of the first non-zero value in the specified column, starting from the
                given row. Returns -1 if no non-zero value is found.
    """
    column_array = M[starting_row:, column]
    for i, val in enumerate(column_array):
        if not np.isclose(val, 0, atol=1e-5):
            index = i + starting_row
            return index
    return -1


def get_index_first_non_zero_value_from_row(
    M: np.ndarray, row: int, augmented: bool = False
) -> int:
    """Find the index of the first non-zero value in the specified row of the given matrix.

    Args:
        M (np.ndarray): The input matrix to search for non-zero values.
        row (int): The index of the row to search.
        augmented (bool, optional): Pass this True if you are dealing with an augmented matrix,
                                    so it will ignore the constant values (the last column in the
                                    augmented matrix). Defaults to False.

    Returns:
        int: The index of the first non-zero value in the specified row.
                Returns -1 if no non-zero value is found.
    """
    M = M.copy()
    if augmented:
        M = M[:, :-1]
    row_array = M[row]
    for i, val in enumerate(row_array):
        if not np.isclose(val, 0, atol=1e-5):
            return i
    return -1


def swap_rows(M: np.ndarray, row_index_1: int, row_index_2: int) -> np.ndarray:
    """Swap rows in the given matrix.

    Args:
        M (np.ndarray): The input matrix to perform row swaps on.
        row_index_1 (int): Index of the first row to be swapped.
        row_index_2 (int): Index of the second row to be swapped.

    Returns:
        np.ndarray: Matrix with rows swapped.
    """
    M = M.copy()
    M[[row_index_1, row_index_2]] = M[[row_index_2, row_index_1]]
    return M


def row_echelon_form(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Utilizes elementary row operations to transform a given set of matrices,
    which represent the coefficients and constant terms of a linear system, into row echelon form.

    Args:
        A (np.ndarray): The input square matrix of coefficients.
        B (np.ndarray): The input column matrix of constat terms.

    Returns:
        np.ndarray: A new augmented matrix in row echelon form with pivots as 1.
    """
    A = A.copy()
    B = B.copy()
    A = A.astype('float64')
    B = B.astype('float64')
    num_rows = len(A)
    M = augmented_matrix(A, B)
    for row in range(num_rows):
        pivot_candidate = M[row, row]
        if np.isclose(pivot_candidate, 0):
            first_non_zero_value_below_pivot_candidate = get_index_first_non_zero_value_from_column(
                M, row, row
            )
            M = swap_rows(M, row, first_non_zero_value_below_pivot_candidate)
            pivot = M[row, row]
        else:
            pivot = pivot_candidate
        M[row] /= pivot
        for j in range(row + 1, num_rows):
            value_below_pivot = M[j, row]
            M[j] = M[j] - value_below_pivot * M[row]
    return M


def string_to_augmented_matrix(equations: str) -> tuple[str, np.ndarray, np.ndarray]:
    """Convert a system of linear equations (string format) into an augmented matrix.
    The function parses a multiline string where each line represents a linear equation
    (e.g., '1*x + 2*y + 3*z = 4') and returns:
    - The variable names detected in the system,
    - The coefficient matrix A,
    - The constant vector B.

    Args:
        equations (str): Multiline containing linear equations separated by newline characters.
                            Example:
                                '1*x + 2*y + 3*z = 4'
                                '0*x + 1*y + 1.3*z = 1.7'
                                '0*x + 0*y + 1*z = 2.33333333'

    Returns:
        tuple[str, np.ndarray, np.ndarray]:
            - Space-separated string of variable names in the order they were detected.
            - A matrix of shape (n x m) containing the coefficients of the variable.
            - A matrix of shape (n x 1) containing the constat terms.
    """
    try:
        equation_list = equations.split('\n')
        equation_list = [x for x in equation_list if x != '']
        coefficients = []
        ss = ''
        for c in equations:
            if c in 'abcdefghijklmnopqrstuvwxyz':
                if c not in ss:
                    ss += c + ' '
        ss = ss[:-1]
        variables = symbols(ss)
        for equation in equation_list:
            sides = equation.replace(' ', '').split('=')
            left_side = sympify(sides[0])
            coefficients.append([left_side.coeff(variable) for variable in variables])
            coefficients[-1].append(float(sympify(sides[1])))
        augmented_matrix = Matrix(coefficients)
        augmented_matrix = np.array(augmented_matrix).astype('float64')
        A, B = augmented_matrix[:, :-1], augmented_matrix[:, -1].reshape(-1, 1)
        return ss, A, B
    except Exception:
        return 'Invalid linear system', np.array([-1]), np.array([-1])
