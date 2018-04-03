import numpy as np
import unittest
from gaussian_elimination import *


class TestGaussianElimination(unittest.TestCase):

    def test_row_echelon_form_3vars(self):
        A = [[2, 1, -1, 8], [-3, -1, 2, -11], [-2, 1, 2, -3]]
        result = row_echelon_form(A)
        compare = np.array([[-3, -1, 2, -11], [0, 1.6666666666666665, 0.6666666666666667, 4.333333333333333], [0, 0, 0.19999999999999987, -0.19999999999999984]])
        self.assertTrue(np.allclose(result, compare))

    def test_solve_row_echelon_form_3vars(self):
        R = [[2, 1, -1, 8], [0, 0.5, 0.5, 1.0], [0, 0, -1.0, 1.0]]
        result = solve_row_echelon_form(R)
        compare = [2, 3, -1]
        self.assertEqual(result, compare)

    def test_gauss_eliminate_3vars(self):
        A = [[2, 1, -1, 8], [-3, -1, 2, -11], [-2, 1, 2, -3]]
        result = gauss_eliminate(A)
        compare = np.array([2, 3, -1])
        self.assertTrue(np.allclose(result, compare))

    def test_row_echelon_form_4vars(self):
        A = [[1, 0, 1, -1, 0], [1, -1, 0, 1, 0], [1, 1, -1, 0, 0],
            [0, 1, 1, 1, 1]]
        result = row_echelon_form(A)
        compare = [[1, 0, 1, -1, 0], [0, -1.0, -1.0, 2.0, 0.0],
                   [0, 0, -3.0, 3.0, 0.0], [0, 0, 0, 3.0, 1.0]]
        self.assertEqual(result, compare)

    def test_solve_row_echelon_form_4vars(self):
        R = [[1, 0, 1, -1, 0], [0, -1.0, -1.0, 2.0, 0.0],
            [0, 0, -3.0, 3.0, 0.0], [0, 0, 0, 3.0, 1.0]]
        result = solve_row_echelon_form(R)
        compare = [0, 1/3., 1/3., 1/3.]
        self.assertEqual(result, compare)

    def test_gauss_eliminate_4vars(self):
        A = [[1, 0, 1, -1, 0], [1, -1, 0, 1, 0], [1, 1, -1, 0, 0],
            [0, 1, 1, 1, 1]]
        result = gauss_eliminate(A)
        compare = [0, 1/3., 1/3., 1/3.]
        self.assertEqual(result, compare)

    def test_gauss_eliminate_3vars_4eqns(self):
        A = [[2, 1, -1, 8], [-3, -1, 2, -11], [-2, 1, 2, -3], [-4, 2, 4, -6]]
        result = gauss_eliminate(A)
        compare = np.array([2, 3, -1])
        self.assertTrue(result, compare)

    def test_gauss_eliminate_3vars_5eqns(self):
        A = [[2, 1, -1, 8], [-3, -1, 2, -11], [-2, 1, 2, -3], [-4, 2, 4, -6],
            [6, 2, -4, 22]]
        result = gauss_eliminate(A)
        compare = np.array([2, 3, -1])
        self.assertTrue(np.allclose(result, compare))

    def test_rank_3vars(self):
        A = [[2, 1, -1, 8], [-3, -1, 2, -11], [-2, 1, 2, -3]]
        result = rank(A)
        compare = 3
        self.assertEqual(result, compare)

    def test_rank_4vars(self):
        A = [[1, 0, 1, -1, 0], [1, -1, 0, 1, 0], [1, 1, -1, 0, 0],
            [0, 1, 1, 1, 1]]
        result = rank(A)
        compare = 4
        self.assertEqual(result, compare)

    def test_rank_3vars_5eqns(self):
        A = [[2, 1, -1, 8], [-3, -1, 2, -11], [-2, 1, 2, -3], [-4, 2, 4, -6],
            [6, 2, -4, 22]]
        result = rank(A)
        compare = 3
        self.assertEqual(result, compare)

    def test_row_echelon_form_3vars_array(self):
        A = np.array([[2, 1, -1, 8], [-3, -1, 2, -11], [-2, 1, 2, -3]], dtype=float)
        result = row_echelon_form(A)
        compare = np.array([[-3, -1, 2, -11], [0, 1.6666666666666665, 0.6666666666666667, 4.333333333333333], [0, 0, 0.19999999999999987, -0.19999999999999984]])
        self.assertTrue(np.all(result == compare))

    def test_row_echelon_form_4vars_array(self):
        A = np.array([[1, 0, 1, -1, 0], [1, -1, 0, 1, 0], [1, 1, -1, 0, 0],
                      [0, 1, 1, 1, 1]], dtype=float)
        result = row_echelon_form(A)
        compare = np.array([[1, 0, 1, -1, 0], [0, -1.0, -1.0, 2.0, 0.0],
                            [0, 0, -3.0, 3.0, 0.0], [0, 0, 0, 3.0, 1.0]])
        self.assertTrue(np.all(result == compare))

    def test_rank_3vars_5eqns_array(self):
        A = np.array([[2, 1, -1, 8], [-3, -1, 2, -11], [-2, 1, 2, -3], [-4, 2, 4, -6],
                      [6, 2, -4, 22]], dtype=float)
        result = rank(A)
        compare = 3
        self.assertEqual(result, compare)

    def test_gauss_eliminate_3vars_4eqns_array(self):
        A = np.array([[2, 1, -1, 8], [-3, -1, 2, -11],
                      [-2, 1, 2, -3], [-4, 2, 4, -6]], dtype=float)
        result = gauss_eliminate(A)
        compare = np.array([2, 3, -1])
        self.assertTrue(np.allclose(result, compare))

    def test_rank_not_enough_eqns(self):
        A = np.array([[2, 1, -1, 8], [-3, -1, 2, -11]], dtype=float)
        result = rank(A)
        compare = 2
        self.assertEqual(result, compare)

    def test_rank_empty_list(self):
        A = []
        result = rank(A)
        compare = 0
        self.assertEqual(result, compare)

    def test_complete_basis(self):
        A = [[0, 0, 1, 0], [1, 0, 0, 0]]
        result = gauss_eliminate(A)
        compare = [0.0, 1.0, 0.0]
        self.assertEqual(result, compare)

    def test_complete_basis_array(self):
        A = np.array([[0, 0, 1, 0], [1, 0, 0, 0]], dtype=float)
        result = gauss_eliminate(A)
        compare = [0.0, 1.0, 0.0]
        self.assertEqual(result, compare)

    def test_remove_duplicate_row_echelon_form(self):
        A = [[0, 0, 1, 0], [0, 0, 1, 0]]
        result = remove_duplicate_row_echelon_form(A)
        compare = [[0, 0, 1, 0]]
        self.assertEqual(result, compare)

    def test_remove_duplicate_row_echelon_form_array(self):
        A = np.array([[0, 0, 1, 0], [0, 0, 1, 0]], dtype=float)
        result = remove_duplicate_row_echelon_form(A)
        compare = np.array([[0, 0, 1, 0]], dtype=float)
        self.assertTrue(np.allclose(result, compare))

    def test_example(self):
        # A = [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]]
        # result = row_echelon_form(A)

        # A = [[1, 1, 0, 0], [0, -1.0, 0.0, 0.0]]
        # result = solve_row_echelon_form(A)

        # A = [[1, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 1, 0, 0]]
        # result = row_echelon_form(A)

        A = [[1, 1, 1, 0], [0, 1.0, 1.0, 0.0]]
        # result = gauss_eliminate(A)
        result = row_echelon_form(A)

        print (result)

if __name__ == "__main__":
    unittest.main()
