import unittest
from data import UserMovieRatings, UserMovieCategoriesRatings


class UserMovieRatingsDatasetTest(unittest.TestCase):

    def test_explicit(self):
        data = UserMovieRatings(train=False, implicit=False)
        point = data[0]

        self.assertEqual(len(point), 2)


    def test_implicit(self):
        data = UserMovieRatings(train=False, implicit=True)
        point = data[0]

        self.assertEqual(len(point), 1)


class UserMovieCategoriesRatingsDatasetTest(unittest.TestCase):
    def test_explicit(self):
        data = UserMovieCategoriesRatings(train=False, implicit=False)
        point = data[0]

        self.assertEqual(len(point), 2)


    def test_implicit(self):

        data = UserMovieCategoriesRatings(train=False, implicit=True)
        point = data[0]

        self.assertEqual(len(point), 1)


if __name__ == "__main__":
    unittest.main()