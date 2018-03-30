def row_echelon_form(L):
    '''
    Input: The augmented matrix of
        the system of equations to solve (provided as a list of lists)
    Output: Row echelon form of the matrix L
    '''
    num_rows = len(L)
    num_cols = len(L[0])
    # Loop through columns
    for i in range(num_cols):
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
    Input: A list of lists R, providing the upper rectangular form
        of augmented matrix obtained via allowed row-operations
    Output: Solution of the linear equations
    '''
    num_rows = len(R)
    num_cols = len(R[0])
    if num_rows < num_cols - 1:
        print ("Not enough equations to solve all variables")
        return [None for _ in range(num_rows)]
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
    return len([l for l in R if set(l) != set([0])])
