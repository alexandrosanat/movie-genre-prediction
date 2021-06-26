"""
pytest for movie_classifier
utf-8
"""
import unittest
import movie_classifier


class TestMovieClassifier(unittest.TestCase):

    """
    Unit testing for the movie_classifier module
    """

    ############################
    # Setup and Teardown ####
    ############################

    # Executed prior to all tests
    @classmethod
    def setUpClass(cls):
        cls.route_finder = route_finder.RouteFinder(load=True)

    ###############
    # test ####
    ###############

    def test_movie_classifier(self):
        test_input = {
            'title': 'Avengers',
            'description': 'Earths mightiest heroes must come together and learn to fight as a '
                           'team if they are going to stop the mischievous Loki and his alien '
                           'army from enslaving humanity.',
            'threshold': 0.5
        }

        output = movie_classifier.main(test_input.get('title'),
                                       test_input.get('description'),
                                       test_input.get('threshold'))

        # Check that the returned genres match the expected
        self.assertEqual(output.get('title'), test_input.get('title'))
        self.assertEqual(output.get('description'), test_input.get('description'))
        self.assertEqual(output.get('genre'), ['Action', 'Science Fiction'])


if __name__ == "__main__":
    unittest.main()
