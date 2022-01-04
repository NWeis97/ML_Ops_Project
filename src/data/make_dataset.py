# -*- coding: utf-8 -*-
import logging
import pdb
from pathlib import Path

import click
import numpy as np
import torch
from dotenv import find_dotenv, load_dotenv

# I mports for data prep
from numpy.lib.npyio import load


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    # Load data
    folderpath = "data/raw/"
    Test = dict(np.load(folderpath + "test.npz"))

    # Concatenate training datasets
    # Inspiration from https://coderedirect.com/questions/615101/how-to-merge-very-large-numpy-arrays

    # Define train datasets to load and dicts
    data_files = [
        "train_0.npz",
        "train_1.npz",
        "train_2.npz",
        "train_3.npz",
        "train_4.npz",
    ]
    n_items = {"images": 0, "labels": 0, "allow_pickle": 0}
    rows = {"images": None, "labels": None, "allow_pickle": None}
    cols = {"images": None, "labels": None, "allow_pickle": None}
    dtype = {"images": None, "labels": None, "allow_pickle": None}

    # Load all train files and check for size
    for data_file in data_files:
        with np.load(folderpath + data_file) as data:
            keys = list(rows.keys())
            for i in keys:
                chunk = data[i]
                # pdb.set_trace()
                try:
                    n_items[i] += chunk.shape[0]
                except:
                    n_items[i] = 1
                    # Set to 1
                try:
                    rows[i] = chunk.shape[1]
                except:
                    rows[i] = 1
                    # Do nothing
                try:
                    cols[i] = chunk.shape[2]
                except:
                    cols[i] = 1
                    # Do nothing
                dtype[i] = chunk.dtype

    # Initialize training dataset
    Train = {}

    # Once the size is know create concatenated version of data
    keys_new = keys[0:2]
    for i in keys_new:
        merged = np.empty(shape=(n_items[i], rows[i], cols[i]), dtype=dtype[i])
        merged = np.squeeze(merged)

        idx = 0
        for data_file in data_files:
            with np.load(folderpath + data_file) as data:
                chunk = data[i]
                merged[idx : idx + len(chunk)] = chunk
                idx += len(chunk)
        Train[i] = merged

    # Convert to dataloader object
    train_images = torch.Tensor(Train['images']).view(1, Train['images'].shape[0], Train['images'].shape[1])
    test_images = torch.Tensor(Test['images']).view(1, Train['images'].shape[0], Train['images'].shape[1])
    Train = torch.utils.data.TensorDataset(train_images, torch.Tensor(Train['labels']).type(torch.LongTensor))
    Test = torch.utils.data.TensorDataset(test_images, torch.Tensor(Test['labels']).type(torch.LongTensor))

    # Save datasets in data/processed
    torch.save(Train, "data/processed/train_dataset.pt")
    torch.save(Test, "data/processed/test_dataset.pt")

    # Exit notes
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
