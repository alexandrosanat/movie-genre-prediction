import typer
import text_processing as tp
import joblib


# Create a new typer app
app = typer.Typer()


# Called when the app is loaded
def load_model():
    # Get the path to the trained model file and load it
    model_path = "./movie_genre_classifier.pkl"

    with open(model_path, 'rb') as filename:
        model = joblib.load(filename)
    return model


# Called when a request is received
def predict(title: str, description: str):
    """
    :param title:
    :param description:
    :return:
    """
    # Pre-process user input
    pre_processed_text = tp.transform_text(title, description)
    vectorised_text = tp.vectorise_text(pre_processed_text)
    # Load the model
    model = load_model()
    # Get a prediction from the model
    genre_array = model.predict(vectorised_text)

    vectoriser_path = "./movie_genre_vectoriser.pkl"
    with open(vectoriser_path, 'rb') as filename:
        vectoriser = joblib.load(filename)

    genre = vectoriser.inverse_transform(genre_array)

    # Return the prediction
    output = {"title": title,
              "description": description,
              "genre": genre}
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


