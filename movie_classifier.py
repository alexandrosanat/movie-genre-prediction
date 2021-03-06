import typer
import text_processing as tp
import joblib
import os
import json


def load_model():
    """
    Loads the pre-trained model and the model classes from file
    :return: pickle: (pickle) A collection of objects including the model and the movie genres
    """

    model_file = "movie_classifier_trained_model.pkl"  # Name to the trained model file
    file_path = os.path.dirname(os.path.realpath(__file__))  # Get parent directory
    model_path = os.path.join(file_path, 'model', model_file)  # Join directory to the model name

    try:
        with open(model_path, 'rb') as filename:
            pickle = joblib.load(filename)  # Loads the pre-trained pipeline and genres
            return pickle

    except FileNotFoundError:
        typer.echo("** Error: A trained model doesn't exist in the current directory. Please "
                   "rerun the model_training.ipynb notebook to train and save the model.**")
        exit()


def predict(title: str, description: str):
    """
    Uses the pre-trained model to predict the movie genre given a title and a description
    :param title: (str) The name of the movie
    :param description: (str) The description of the movie
    :return: output: (json) A dictionary with the title, description and genre of the movie
    """

    pickle = load_model()  # Load pickle contents
    model = pickle.get('pipeline')  # Get the pre-trained pipeline
    genres = pickle.get('genres')  # Get the available genres

    pre_processed_text = tp.transform_text(title, description)  # Pre-process user input
    probabilities = model.predict_proba([pre_processed_text])  # Get the model prediction

    sorted_index = (-probabilities).argsort()   # Sort indices in descending order
    sorted_genres = genres[sorted_index[::-1]]  # Sort genres according to the indices
    prediction = sorted_genres[0][0]  # Get the most likely genre

    return json.dumps({"title": title,
                       "description": description,
                       "genre": prediction}, indent=4)


def main(title: str = typer.Option(..., help="The name of the movie."),
         description: str = typer.Option(..., help="The description of the movie.")):
    """
    DESCRIPTION: Movie Genre Prediction \n
    You can use this module to predict a movie's genre given a movie title and description. \n
    e.g. python movie_classifier.py --title "Movie Title" --description "Movie Description"
    """
    if len(title) == 0 and len(description) == 0:
        print("The title and description fields can't be empty. "
              "Please provide values and try again.")
        exit()
    elif len(title) == 0:
        print("The title field can't be empty. Please input a title and try again.")
        exit()
    elif len(description) == 0:
        print("The description field can't be empty. Please input a description and try again.")
        exit()
    else:
        prediction = predict(title, description)  # Use inputs to predict the genre
        typer.echo(prediction)  # Print output
        return prediction


if __name__ == '__main__':
    typer.run(main)
