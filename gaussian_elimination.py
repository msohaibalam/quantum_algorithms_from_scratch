import numpy as np


def row_echelon_form(L):
    '''
    Input: The augmented matrix of
        the system of equations to solve
        (provided as a list of lists, or a numpy array with dtype=float)
    Output: Row echelon form of the matrix L
    '''
    num_rows = len(L)
    # return None if empty list/array
    if num_rows == 0:
        return None
    num_cols = len(L[0])
    # Loop through columns, handling for non-square matrices as well
    for i in range(min(num_cols, num_rows)):
        # search for max in this column
        max_el = abs(L[i][i])
        max_row = i
        for k in range(i + 1, num_rows):
            if abs(L[k][i]) > max_el:
                max_el = abs(L[k][i])
                max_row = k

        # swap max row with current row
        for k in range(i, num_cols):
            tmp = L[max_row][k]
            L[max_row][k] = L[i][k]
            L[i][k] = tmp

        # making all cells below current cell 0
        for k in range(i + 1, num_rows):
            # calculate the multiplier
            if L[i][i] == 0:
                continue
            c = - L[k][i]/L[i][i]
            # multiply and add previous row to current row
            for j in range(i, num_cols):
                if i == j:
                    L[k][j] = 0
                else:
                    L[k][j] += c * L[i][j]
    return L


def solve_row_echelon_form(R):
    '''
    Input: A list of lists R, providing the row echelon form
        of augmented matrix obtained via allowed row-operations
    Output: Solution of the linear equations
        (provided as a list, or numpy array)
    '''
    num_rows = len(R)
    # return None if empty list/array
    if num_rows == 0:
        return None
    # check if not a square (non-augmented) matrix
    num_cols = len(R[0])
    if num_rows < num_cols - 1:

        ## There are 2 scenarios here:
        ## (a) One can complete the basis with a vector not orthogonal to s
        ##  and solve (see section 18.13.2 in Quantum Computing for the Community College)
        ## (b) We genuinely don't have enough equations to solve all variables

        # loop through diagonal to check for situation (a)
        for k in range(num_rows):
            if R[k][k] == 0:
                if type(R) == list:
                    R = R[:k] + [[1 if ((kk == k) or (kk == num_cols - 1)) else 0 for kk in range(num_cols)]] + R[k:]
                elif type(R).__module__ == np.__name__:
                    R = np.vstack((np.vstack((R[:k], np.array([1 if ((kk == k) or (kk == num_cols - 1)) else 0 for kk in range(num_cols)], dtype=float))), R[k:]))

    # recompute num_rows and num_cols
    num_rows = len(R)
    num_cols = len(R[0])
    # return list of Nones if not enough eqns for vars
    if num_rows < num_cols - 1:
        print ("Not enough equations to solve all variables")
        return [None for _ in range(num_rows)]

    # solve if enough eqns for vars
    x = [0 for _ in range(num_rows)]
    ## starting with bottom-most row, solve for all the variables
    ## by back-substitution
    for i in range(num_rows - 1, -1, -1):
        # column q defined to handle the situation where num_rows >= num_cols
        q = min(i, num_cols - 2)
        # delete any extra variable placeholder(s)
        if R[i][q] == 0:
            del x[i]
            continue
        # solve for variable x[i], and back-substitute to previous equations
        x[i] = R[i][num_cols - 1]/R[i][q]
        for j in range(i - 1, -1, -1):
            R[j][num_cols - 1] -= R[j][i] * x[i]

    return x


def gauss_eliminate(L):
    '''
    Input: A list of lists L, providing the augmented matrix of
        the system of equations to solve
    Ouput: Solution of the linear equations
    '''
    R = row_echelon_form(L)
    x = solve_row_echelon_form(R)
    return x


def rank(A):
    '''
    Computes the rank of matrix A
    '''
    R = row_echelon_form(A)
    # return 0 for empty list input
    if R is None:
        return 0
    return len([l for l in R if set(l) != set([0])])
