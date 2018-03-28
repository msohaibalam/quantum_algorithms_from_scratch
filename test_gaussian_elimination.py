import unittest
from gaussian_elimination import *


class TestGaussianElimination(unittest.TestCase):

    def test_upper_rect_form_3vars(self):
        A = [[2, 1, -1, 8], [-3, -1, 2, -11], [-2, 1, 2, -3]]
        result = upper_rect_form(A)
        compare = [[2, 1, -1, 8], [0, 0.5, 0.5, 1.0], [0, 0, -1.0, 1.0]]
        self.assertEqual(result, compare)

    def test_solve_upper_rect_form_3vars(self):
        R = [[2, 1, -1, 8], [0, 0.5, 0.5, 1.0], [0, 0, -1.0, 1.0]]
        result = solve_upper_rect_form(R)
        compare = [2, 3, -1]
        self.assertEqual(result, compare)

    def test_gauss_eliminate_3vars(self):
        A = [[2, 1, -1, 8], [-3, -1, 2, -11], [-2, 1, 2, -3]]
        result = gauss_eliminate(A)
        compare = [2, 3, -1]
        self.assertEqual(result, compare)

    def test_upper_rect_form_4vars(self):
        A = [[1, 0, 1, -1, 0], [1, -1, 0, 1, 0], [1, 1, -1, 0, 0],
            [0, 1, 1, 1, 1]]
        result = upper_rect_form(A)
        compare = [[1, 0, 1, -1, 0], [0, -1.0, -1.0, 2.0, 0.0],
                   [0, 0, -3.0, 3.0, 0.0], [0, 0, 0, 3.0, 1.0]]
        self.assertEqual(result, compare)

    def test_solve_upper_rect_form_4vars(self):
        R = [[1, 0, 1, -1, 0], [0, -1.0, -1.0, 2.0, 0.0],
            [0, 0, -3.0, 3.0, 0.0], [0, 0, 0, 3.0, 1.0]]
        result = solve_upper_rect_form(R)
        compare = [0, 1/3., 1/3., 1/3.]
        self.assertEqual(result, compare)

    def test_gauss_eliminate_4vars(self):
        A = [[1, 0, 1, -1, 0], [1, -1, 0, 1, 0], [1, 1, -1, 0, 0],
            [0, 1, 1, 1, 1]]
        result = gauss_eliminate(A)
        compare = [0, 1/3., 1/3., 1/3.]
        self.assertEqual(result, compare)

if __name__ == "__main__":
    unittest.main()
