import numpy as np
import unittest
from gaussian_elimination import *


class TestGaussianElimination(unittest.TestCase):

    def test_row_echelon_form_3vars(self):
        A = [[2, 1, -1, 8], [-3, -1, 2, -11], [-2, 1, 2, -3]]
        result = row_echelon_form(A)
        compare = [[2, 1, -1, 8], [0, 0.5, 0.5, 1.0], [0, 0, -1.0, 1.0]]
        self.assertEqual(result, compare)

    def test_solve_row_echelon_form_3vars(self):
        R = [[2, 1, -1, 8], [0, 0.5, 0.5, 1.0], [0, 0, -1.0, 1.0]]
        result = solve_row_echelon_form(R)
        compare = [2, 3, -1]
        self.assertEqual(result, compare)

    def test_gauss_eliminate_3vars(self):
        A = [[2, 1, -1, 8], [-3, -1, 2, -11], [-2, 1, 2, -3]]
        result = gauss_eliminate(A)
        compare = [2, 3, -1]
        self.assertEqual(result, compare)

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
        compare = [2, 3, -1]
        self.assertEqual(result, compare)

    def test_gauss_eliminate_3vars_5eqns(self):
        A = [[2, 1, -1, 8], [-3, -1, 2, -11], [-2, 1, 2, -3], [-4, 2, 4, -6],
            [6, 2, -4, 22]]
        result = gauss_eliminate(A)
        compare = [2, 3, -1]
        self.assertEqual(result, compare)

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
        compare = np.array([[2, 1, -1, 8], [0, 0.5, 0.5, 1.0], [0, 0, -1.0, 1.0]])
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
        self.assertTrue(np.all(result == compare))

if __name__ == "__main__":
    unittest.main()
