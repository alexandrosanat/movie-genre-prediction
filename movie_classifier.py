import typer


# Create a new typer app
app = typer.Typer()


# Called when the service is loaded
def init():
    global model
    # Get the path to the trained model file and load it
    model_path = ""
    model = None


# Called when a request is received
def predict(title, description: str):
    """
    :param title:
    :param description:
    :return:
    """
    # Get a prediction from the model
#    genre = model.predict(title, description)
    genre = "Drama"
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

    output = predict(title, description)

    typer.echo(output)


if __name__ == '__main__':
    app()


