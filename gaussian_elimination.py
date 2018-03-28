def upper_rect_form(L):
    '''
    Input: A list of lists L, providing the augmented matrix of
        the system of equations to solve
    Output: Upper-rectangular form after row operations
    '''
    # Loop through columns
    for i in range(len(L)):
        # making all cells below current cell 0
        for k in range(i+1, len(L)):
            # calculate the multiplier
            c = - L[k][i]/L[i][i]
            # multiply and add previous row to current row
            for j in range(i, len(L)+1):
                if i == j:
                    L[k][j] = 0
                else:
                    L[k][j] += c * L[i][j]
    return L


def solve_upper_rect_form(R):
    '''
    Input: A list of lists R, providing the upper rectangular form
        of augmented matrix obtained via allowed row-operations
    Output: Solution of the linear equations
    '''
    n = len(R)
    x = [0 for _ in range(n)]
    # starting with bottom-most row, solve for all the variables
    # by back-substitution
    for i in range(n - 1, -1, -1):
        x[i] = R[i][n]/R[i][i]
        for j in range(i - 1, -1, -1):
            R[j][n] -= R[j][i] * x[i]
    return x


def gauss_eliminate(L):
    '''
    Input: A list of lists L, providing the augmented matrix of
        the system of equations to solve
    Ouput: Solution of the linear equations
    '''
    R = upper_rect_form(L)
    x = solve_upper_rect_form(R)
    return x
