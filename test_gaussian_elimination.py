import numpy as np
import unittest
from gaussian_elimination import *


class TestGaussianElimination(unittest.TestCase):

    def test_new_sample_msb_equal(self):
        W = [[0, 1, 1, 0]]
        z = [0, 1, 0, 0]
        result = new_sample(W, z)
        expected = [[0, 1, 1, 0], [0, 0, 1, 0]]
        self.assertEqual(result, expected)

    def test_new_sample_msb_sandwiched(self):
        W = [[1, 0, 0, 0, 0], [0, 0, 1, 0, 0]]
        z = [0, 1, 1, 0, 0]
        result = new_sample(W, z)
        expected = [[1, 0, 0, 0, 0], [0, 1, 1, 0, 0], [0, 0, 1, 0, 0]]
        self.assertEqual(result, expected)

    def test_new_sample_beginning(self):
        W = [[0, 1, 1, 0, 0], [0, 0, 1, 0, 0]]
        z = [1, 0, 1, 0, 0]
        result = new_sample(W, z)
        expected = [[1, 0, 1, 0, 0], [0, 1, 1, 0, 0], [0, 0, 1, 0, 0]]
        self.assertEqual(result, expected)

    def test_new_sample_end(self):
        W = [[0, 1, 1, 0, 0, 0], [0, 0, 1, 0, 1, 0]]
        z = [0, 0, 0, 0, 1, 0]
        result = new_sample(W, z)
        expected = [[0, 1, 1, 0, 0, 0], [0, 0, 1, 0, 1, 0], [0, 0, 0, 0, 1, 0]]
        self.assertEqual(result, expected)

    def test_new_sample_no_add(self):
        W = [[0, 1, 1, 0, 0], [0, 0, 1, 0, 0]]
        z = [0, 1, 0, 0, 0]
        result = new_sample(W, z)
        expected = W
        self.assertEqual(result, expected)

    def test_complete_basis_middle(self):
        A = [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0]]
        result = complete_basis(A)
        expected = [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 1], [0, 0, 0, 0, 1, 0]]
        self.assertEqual(result, expected)

    def test_complete_basis_end(self):
        A = [[1, 0, 0, 0], [0, 1, 0, 0]]
        result = complete_basis(A)
        expected = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1]]
        self.assertEqual(result, expected)

    def test_complete_basis_beginning(self):
        A = [[0, 1, 1, 0], [0, 0, 1, 0]]
        result = complete_basis(A)
        expected = [[1, 0, 0, 1], [0, 1, 1, 0], [0, 0, 1, 0]]
        self.assertEqual(result, expected)

    def test_rank_empty_list(self):
        A = []
        result = rank(A)
        compare = 0
        self.assertEqual(result, compare)

    def test_rank_4by5(self):
        A = [[1, 0, 1, -1, 0], [1, -1, 0, 1, 0], [1, 1, -1, 0, 0],
            [0, 1, 1, 1, 0]]
        result = rank(A)
        compare = 4
        self.assertEqual(result, compare)

    def test_rank_zeroRow(self):
        A = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]]
        result = rank(A)
        expected = 2
        self.assertEqual(result, expected)

    def test_period_001(self):
        W = []
        samples = [[0, 0, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0]]
        for s in samples:
            W = new_sample(W, s)
        result = solve_reduced_row_echelon_form(W)
        expected = [0, 0, 1]
        self.assertEqual(result, expected)

    def test_period_010(self):
        W = []
        samples = [[0, 0, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [1, 0, 1, 0]]
        for s in samples:
            W = new_sample(W, s)
        result = solve_reduced_row_echelon_form(W)
        expected = [0, 1, 0]
        self.assertEqual(result, expected)

    def test_period_011(self):
        W = []
        samples = [[0, 0, 0, 0], [0, 1, 1, 0], [1, 0, 0, 0], [1, 1, 1, 0]]
        for s in samples:
            W = new_sample(W, s)
        result = solve_reduced_row_echelon_form(W)
        expected = [0, 1, 1]
        self.assertEqual(result, expected)

    def test_period_100(self):
        W = []
        samples = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 1, 1, 0]]
        for s in samples:
            W = new_sample(W, s)
        result = solve_reduced_row_echelon_form(W)
        expected = [1, 0, 0]
        self.assertEqual(result, expected)

    def test_period_101(self):
        W = []
        samples = [[0, 0, 0, 0], [0, 1, 0, 0], [1, 0, 1, 0], [1, 1, 1, 0]]
        for s in samples:
            W = new_sample(W, s)
        result = solve_reduced_row_echelon_form(W)
        expected = [1, 0, 1]
        self.assertEqual(result, expected)

    def test_period_110(self):
        W = []
        samples = [[0, 0, 0, 0], [0, 0, 1, 0], [1, 1, 0, 0], [1, 1, 1, 0]]
        for s in samples:
            W = new_sample(W, s)
        result = solve_reduced_row_echelon_form(W)
        expected = [1, 1, 0]
        self.assertEqual(result, expected)

    def test_period_111(self):
        W = []
        samples = [[0, 0, 0, 0], [0, 1, 1, 0], [1, 0, 1, 0], [1, 1, 0, 0]]
        for s in samples:
            W = new_sample(W, s)
        result = solve_reduced_row_echelon_form(W)
        expected = [1, 1, 1]
        self.assertEqual(result, expected)

    def test_period_0001(self):
        W = []
        samples = [[0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 1, 0, 0, 0], [0, 1, 1, 0, 0],
                   [1, 0, 0, 0, 0], [1, 0, 1, 0, 0], [1, 1, 0, 0, 0], [1, 1, 1, 0, 0]]
        for s in samples:
            W = new_sample(W, s)
        result = solve_reduced_row_echelon_form(W)
        expected = [0, 0, 0, 1]
        self.assertEqual(result, expected)

    def test_period_0010(self):
        W = []
        samples = [[0, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 1, 0, 0, 0], [0, 1, 0, 1, 0],
                   [1, 0, 0, 0, 0], [1, 0, 0, 1, 0], [1, 1, 0, 0, 0], [1, 1, 0, 1, 0]]
        for s in samples:
            W = new_sample(W, s)
        result = solve_reduced_row_echelon_form(W)
        expected = [0, 0, 1, 0]
        self.assertEqual(result, expected)

    def test_period_0101(self):
        W = []
        samples = [[0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 1, 0, 1, 0], [0, 1, 1, 1, 0],
                   [1, 0, 0, 0, 0], [1, 0, 1, 0, 0], [1, 1, 0, 1, 0], [1, 1, 1, 1, 0]]
        for s in samples:
            W = new_sample(W, s)
        result = solve_reduced_row_echelon_form(W)
        expected = [0, 1, 0, 1]
        self.assertEqual(result, expected)

    def test_period_1011(self):
        W = []
        samples = [[1, 1, 1, 0, 0], [1, 1, 0, 1, 0], [1, 0, 1, 0, 0], [1, 0, 0, 1, 0],
                   [0, 1, 1, 1, 0], [0, 1, 0, 0, 0], [0, 0, 1, 1, 0], [0, 0, 0, 0, 0]]
        for s in samples:
            W = new_sample(W, s)
        result = solve_reduced_row_echelon_form(W)
        expected = [1, 0, 1, 1]
        self.assertEqual(result, expected)

if __name__ == "__main__":
    unittest.main()
