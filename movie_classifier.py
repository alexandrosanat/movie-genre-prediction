import typer
import text_processing as tp
import joblib
import numpy as np
import os
import json


def load_model():
    """
    Loads the pre-trained model and the model classes from file
    :return: pickle: (pickle) A collection of objects including the model and other
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
def predict(title: str, description: str, top: bool = True, threshold: float = None):
    """
    Uses the pre-trained model to predict the movie genre given a title and a description
    :param title: (str) The name of the movie
    :param description: (str) The description of the movie
    :param top: (bool) Whether to return the top result only or not
    :param threshold: (float) The threshold above which probabilities will be considered
    :return: output: (dict) A dictionary with the title, description and genre of the movie
    """

    pre_processed_text = tp.transform_text(title, description)  # Pre-process user input

    pickle = load_model()  # Load the model
    model = pickle.get('pipeline')  # Get the pre-trained pipeline
    genres = pickle.get('genres')  # Get the available genres
    threshold = pickle.get('threshold') if threshold is None else threshold  # The model threshold
    probabilities = model.predict_proba([pre_processed_text])  # Get the prediction from the model

    sorted_index = (-probabilities).argsort()   # Sort indices in descending order
    probabilities_sorted = probabilities[0][sorted_index[::-1]]  # Sort probabilities in desc order
    sorted_genres = genres[sorted_index[::-1]]  # Sort genres in descending order
    top_genres = np.where(probabilities_sorted >= threshold, 1, 0)  # Genres with prob > threshold
    if top:
        prediction = sorted_genres[0].tolist()[0]
    else:
        # Convert genre array to list
        prediction = ", ".join(list(sorted_genres[0][top_genres[0] > 0]))

    prediction = prediction if len(prediction) > 0 else 'Unknown'  # Return unknown if no prediction

    return json.dumps({"title": title,
                       "description": description,
                       "genre": prediction}, indent=4)


def main(title: str = typer.Option(..., help="The name of the movie."),
         description: str = typer.Option(..., help="The description of the movie."),
         top: bool = typer.Option(True, help="Only return the top result."),
         threshold: float = typer.Option(0.5, help="The confidence threshold for the model.")):
    """
    DESCRIPTION: Movie Genre Prediction \n
    You can use this module to predict a movie's genre given the movie's title and description. \n
    e.g. python movie_classifier.py --title "Movie Title" --description "Movie Description"
    """
    # Use inputs to predict the genre
    prediction = predict(title, description, top, threshold)
    # Print output
    typer.echo(prediction)
    return prediction


if __name__ == '__main__':
    typer.run(main)
