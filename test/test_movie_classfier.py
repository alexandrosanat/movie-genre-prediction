"""
pytest for movie_classifier
utf-8
"""
import unittest
import movie_classifier
import json


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
        cls.pickle = movie_classifier.load_model()

    ###############
    # Tests ####
    ###############
    def test_pickle_contents(self):
        """
        Test that the pickle contains all necessary data.
        """
        self.assertEqual([item for item in self.pickle], ['pipeline', 'genres'])

    def test_pipeline_steps(self):
        """
        Test that the model pipeline contains all expected steps.
        """
        pipeline = self.pickle.get('pipeline')
        self.assertEqual([step[0] for step in pipeline.steps], ['Vectorizer', 'Classifier'])

    def test_movie_classifier(self):
        """
        Test that for given input the output is as expected.
        """
        test_input = {
            'title': 'Avengers',
            'description': 'Earths mightiest heroes must come together and learn to fight as a '
                           'team if they are going to stop the mischievous Loki and his alien '
                           'army from enslaving humanity.',
            'threshold': 0.5
        }

        output = json.loads(movie_classifier.main(test_input.get('title'),
                                                  test_input.get('description'),
                                                  test_input.get('threshold')))

        # Check that the returned genres match the expected
        self.assertEqual(output.get('title'), test_input.get('title'))
        self.assertEqual(output.get('description'), test_input.get('description'))
        self.assertEqual(output.get('genre'), 'Action, Science Fiction')


if __name__ == "__main__":
    unittest.main()
