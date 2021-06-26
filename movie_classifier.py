import typer
import text_processing as tp
import joblib
import numpy as np


def load_model():
    """
    Loads the pre-trained model and the model classes
    :return: model: (Pipeline) A scikit learn pipeline object with the trained model
    :return: genres: (np.array) A numpy array of all the classes available for prediction
    """
    model_path = "./movie_classifier_trained_model.pkl"  # Path to the trained model file

    with open(model_path, 'rb') as filename:
        loaded_data = joblib.load(filename)  # Loads the pre-trained pipeline and genres
        model = loaded_data.get('pipeline')  # Get the pre-trained pipeline
        genres = loaded_data.get('genres')  # Get the available genres

    return model, genres


# Called when a request is received
def predict(title: str, description: str, threshold: int):
    """
    Uses the pre-trained model to predict the movie genre given a title and a description
    :param title: The name of the movie (str)
    :param description: The description of the movie (str)
    :param threshold: The threshold above which probabilities will be considered (float)
    :return: output: A dictionary with the title, description and genre of the movie (dict)
    """
    pre_processed_text = tp.transform_text(title, description)  # Pre-process user input
    model, genres = load_model()  # Load the model
    probabilities = model.predict_proba([pre_processed_text])  # Get the prediction from the model
    top_genres = np.where(probabilities >= threshold, 1, 0)  # Return genres with prob > threshold
    prediction = list(genres[top_genres[0] > 0])  # Convert array of genres to list

    return {"title": title,
            "description": description,
            "genre": prediction}


def main(title: str = typer.Option(..., help="The name of the movie."),
         description: str = typer.Option(..., help="The description of the movie."),
         threshold: float = typer.Option(0.5, help="Threshold for probabilities to include.")):
    # Use inputs to predict the genre
    prediction = predict(title, description, threshold)
    # Print output
    typer.echo(prediction)


if __name__ == '__main__':
    typer.run(main)


