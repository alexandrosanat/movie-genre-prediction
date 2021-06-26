import typer
import text_processing as tp
import joblib
import numpy as np


# Create a new typer app
app = typer.Typer()


# Called when the app is loaded
def load_model():
    # Get the path to the trained model file and load it
    model_path = "./movie_classifier_trained_model.pkl"

    with open(model_path, 'rb') as filename:
        loaded_data = joblib.load(filename)
        model = loaded_data['pipeline']
        genres = loaded_data['genres']

    return model, genres


# Called when a request is received
def predict(title: str, description: str):
    """
    :param title:
    :param description:
    :return:
    """
    # Pre-process user input
    pre_processed_text = tp.transform_text(title, description)
    # Load the model
    model, genres = load_model()
    # Get a prediction from the model
    probabilities = model.predict_proba([pre_processed_text])
    threshold = 0.5
    top_genres = np.where(probabilities > threshold, 1, 0)
    prediction = list(genres[top_genres[0] > 0])

    # Return the prediction
    output = {"title": title,
              "description": description,
              "genre": prediction}
    return output


@app.command()
def main(title: str = typer.Option(...,
                                   "--title",
                                   help="The name of the movie. (str)"),
         description: str = typer.Option(...,
                                         "--description",
                                         help="The description of the movie. (str)")):
    # Use inputs to predict the genre
    prediction = predict(title, description)

    typer.echo(prediction)


if __name__ == '__main__':
    app()


