Practical Session of the Machine Learning Course
==============================

    This project is part of the Machine Learning Course at the Mines de Saint-Etienne Engineering School.
    It showcases many Machine Learning methods, unsupervised as well as supervised ones.

    The dataset used is the well-known MNIST digits Dataset.

    You can find the complete report of the project in the `/reports` folder.

    The project tackles many algorithms in the following order :
        - Principal Component Analysis
        - K-Means Clustering
        - EM Gaussian Mixture
        - Decision Tree
        - Support Vector Machines
        - Gaussian Naive Bayes
        - Multilayer Perceptron (MLP)
        - Convolutional Neural Network (CNN)

    If you want to reproduce the project, you can recreate the same environnment using pip or conda and the `requirements.txt` file.
    Then, you should be able to execute every notebook in the `/notebooks` folder.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

// TODO : Refractor the notebooks' code in the `/src` folder\
// TODO : Save the trained models in the `/models` folder

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
