import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler


# # -*- coding: utf-8 -*-
# import click
# import logging
# from pathlib import Path
# from dotenv import find_dotenv, load_dotenv


# @click.command()
# @click.argument('input_filepath', type=click.Path(exists=True))
# @click.argument('output_filepath', type=click.Path())
# def main(input_filepath, output_filepath):
#     """ Runs data processing scripts to turn raw data from (../raw) into
#         cleaned data ready to be analyzed (saved in ../processed).
#     """
#     logger = logging.getLogger(__name__)
#     logger.info('making final data set from raw data')


# if __name__ == '__main__':
#     log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
#     logging.basicConfig(level=logging.INFO, format=log_fmt)

#     # not used in this stub but often useful for finding various files
#     project_dir = Path(__file__).resolve().parents[2]

#     # find .env automagically by walking up directories until it's found, then
#     # load up the .env entries as environment variables
#     load_dotenv(find_dotenv())

#     main()

X_raw = np.load("../data/external/MNIST_X_28x28.npy")
X = X_raw.flatten().reshape(70000, 784)  # Flatten the data for convenience
Y = np.load("../data/external/MNIST_y.npy")


def get_split_data(features=X, labels=Y, test_size=0.2):
    return train_test_split(
        features, labels, test_size=0.2, random_state=42, shuffle=True
    )


def getNormalizedProjectedData(components=2):
    steps = Pipeline(
        [
            ("minmax", MinMaxScaler()),
            ("pca", PCA(n_components=components, random_state=42)),
        ]
    )
    X_transformed = steps.fit_transform(X)
    return X_transformed
