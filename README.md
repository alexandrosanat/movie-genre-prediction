# Movie Genre Classifier

A command-line application that predicts a movie's top genre given its title and description.

## Description

An in-depth paragraph about your project and overview of use.

## Getting Started

### Dependencies

* Before running the app, you will need to download and install
 [python 3.8](https://www.python.org/downloads/release/python-380/) for your operating system.
* To install the packages that are used by the app you will also need to install
 [pip](https://pip.pypa.io/en/stable/installing/) which is a package installer for Python.
* For a complete list of all the packages used, refer to the [project
 dependencies](https://github.com/alexandrosanat/movie-genre-prediction/network/dependencies):
    - pandas: used for data manipulation
    - seaborn: used for visualisations  
    - nltk: used for text processing
    - numpy: used for manipulating array objects
    - scikit-learn: used for the classification algorithms
    - tensorflow: used for machine learning algorithms
    - joblib: used to save and load the model
    - typer: used as an alternative to the native argparse to build the CLI
* If you plan to run the app using [Docker](https://docs.docker.com/get-docker/)
 you will also need to install it first.
 
### Executing program

To run the app in your workspace follow the steps below:

#### Option 1 - using Python

* Navigate to a folder where you want to clone the repo
* Clone the repo using:
    ```
    git clone https://github.com/alexandrosanat/movie-genre-prediction.git
    ```
* Create and activate a new
 [virtual environment](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/).
* Install the required packages by running:
    ```
    pip install -r requirements.txt
    ```
* Open a new terminal, cd into the movie-genre-prediction repo and run the app using:
    ```
    python movie_classifier.py --title "<title>" -- description "<description>"
    ```
  
 ![Alt Text](./images/running_python.gif)

#### Option 2 - using Docker

* Navigate to a folder where you want to clone the repo
* Clone the repo using:
    ```
    git clone https://github.com/alexandrosanat/movie-genre-prediction.git
    ```
* Open a new terminal, cd into the movie-genre-prediction repo and build the docker image using:
    ```
    docker build -t genre_classifier --rm .
    ```
  Alternatively you can download the build image directly from
   [here](https://hub.docker.com/repository/docker/alexandrosanat/movie-genre-prediction).
* Once the image is build run the app using:
    ```
    docker run -it --name my_app --rm genre_classifier
    ```

![Alt Text](./images/running_docker.gif)

## Help

Any advise for common problems or issues.
```
command to run if program contains helper info
```

## Authors

Contributors names and contact info

ex. Dominique Pizzie  
ex. [@DomPizzie](https://twitter.com/dompizzie)

## Version History

* 0.1
    * Initial Release


## Acknowledgments

* [Predicting Movie Genres using NLP](https://www.analyticsvidhya.com/blog/2019/04/predicting-movie-genres-nlp-multi-label-classification/)


* Add logging