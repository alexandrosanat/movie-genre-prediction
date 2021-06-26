import typer
import text_processing as tp
import joblib
import numpy as np
import os
import json


def load_model():
    """
    Loads the pre-trained model and the model classes
    :return: model: (Pipeline) A scikit learn pipeline object with the trained model
    :return: genres: (np.array) A numpy array of all the classes available for prediction
    """

    model_file = "movie_classifier_trained_model.pkl"  # Name to the trained model file
    file_path = os.path.dirname(os.path.realpath(__file__))  # Get parent directory
    model_path = os.path.join(file_path, model_file)  # Join directory to the model name

    try:
        with open(model_path, 'rb') as filename:
            pickle = joblib.load(filename)  # Loads the pre-trained pipeline and genres
            return pickle

    except FileNotFoundError:
        typer.echo("** Error: A trained model doesn't exist in the current directory. "
                   "Please rerun the model_training.ipynb notebook to train and save the model.**")
        exit()


# Called when a request is received
def predict(title: str, description: str, threshold: float):
    """
    Uses the pre-trained model to predict the movie genre given a title and a description
    :param title: The name of the movie (str)
    :param description: The description of the movie (str)
    :param threshold: The threshold above which probabilities will be considered (float)
    :return: output: A dictionary with the title, description and genre of the movie (dict)
    """
    pre_processed_text = tp.transform_text(title, description)  # Pre-process user input
    pickle = load_model()  # Load the model
    model = pickle.get('pipeline')  # Get the pre-trained pipeline
    genres = pickle.get('genres')  # Get the available genres
    probabilities = model.predict_proba([pre_processed_text])  # Get the prediction from the model
    top_genres = np.where(probabilities >= threshold, 1, 0)  # Return genres with prob > threshold
    prediction = ", ".join(list(genres[top_genres[0] > 0]))  # Convert array of genres to list
    prediction = prediction if len(prediction) > 0 else 'Unknown'  # Return unknown if no prediction

    return json.dumps({"title": title,
                       "description": description,
                       "genre": prediction}, indent=4)


def main(title: str = typer.Option(..., help="The name of the movie."),
         description: str = typer.Option(..., help="The description of the movie."),
         threshold: float = typer.Option(0.5, help="Threshold for probabilities to include.")):
    """
    DESCRIPTION: Movie Genre Prediction \n
    You can use this module to predict a movie's genre given the movie's title and description. \n
    e.g. python movie_classifier.py --title "Movie Title" --description "Movie Description"
    """
    # Use inputs to predict the genre
    prediction = predict(title, description, threshold)
    # Print output
    typer.echo(prediction)
    return prediction


if __name__ == '__main__':
    typer.run(main)
