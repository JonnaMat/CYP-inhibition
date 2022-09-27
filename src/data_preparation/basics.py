"""Data preparation and exploration."""

import os
from pathlib import Path
from typing import List, Literal, Optional
import pandas as pd
from tdc.single_pred import ADME

# pylint: disable=no-name-in-module
from rdkit.Chem import Descriptors, MolFromSmiles
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator


def calculate_fingerprints(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate a number of fingerprints and add to 'data'."""


def calculate_descriptors(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate all available descriptors for the given molecules and add to 'data'."""

    molecules = [MolFromSmiles(smiles) for smiles in list(data["Drug"])]
    # pylint: disable=protected-access
    calculator = MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
    header = calculator.GetDescriptorNames()

    descriptors = [calculator.CalcDescriptors(molecule) for molecule in molecules]

    descriptor_data = pd.DataFrame(descriptors, columns=header)
    for column in data.columns:
        descriptor_data.insert(loc=0, column=column, value=list(data[column]))
    return descriptor_data


def extract_null(data: pd.DataFrame):
    """Returns all row/column pairs with NaN values."""
    data = data.iloc[1:]  # drop first row
    null_columns_data = data[data.columns[data.isnull().any()]]
    return pd.merge(
        data[["Drug", "Drug_ID", "Y"]],
        null_columns_data[null_columns_data.isna().any(axis=1)],
        left_index=True,
        right_index=True,
    )


def load_tdc_dataset_split(
    task: Literal["CYP2C19", "CYP2D6", "CYP3A4", "CYP1A2", "CYP2C9"],
    method: Literal["random", "scaffold"] = "random",
    frac: Optional[List[float]] = None,
):
    """Load dataset train-, val-, test-split given a `task`."""
    data = ADME(name=task + "_Veith")
    split = data.get_split(
        method=method, seed=1, frac=[0.7, 0.1, 0.2] if frac is None else frac
    )
    return split


def get_full_dataset(task):
    """Return the raw dataset consisting of all 208 descriptors including NaN values and saves dataset."""

    filename = f"data/{task.lower()}/raw_dataset_descriptors.csv"  # TODO

    # create directories if necessary
    Path(f"data/{task.lower()}").mkdir(parents=True, exist_ok=True)

    # check if dataset already exists
    if os.path.isfile(filename):
        return pd.read_csv(filename)

    tdc_data = load_tdc_dataset_split(task, frac=[1, 0, 0])["train"]
    descriptor_data = calculate_descriptors(tdc_data)
    # TODO add fingerprints
    descriptor_data.to_csv(filename)
    return descriptor_data


# TODO
def read_train_data(filename):
    x = []
    y = []
    max_len = 0
    with open(filename) as infile:
        infile.readline()
        for line in infile:
            line = line.strip("\n\r ")
            line = line.split(",")
            y.append(line[len(line) - 1])
            fingerprint_bit_vector = list(map(int, line[12].strip()))
            line = line[1:12] + fingerprint_bit_vector
            max_len = max(max_len, len(line))
            x.append(line)
    print("max_len: {}".format(max_len))
    x = np.array(x)
    x = x.astype(np.float)
    y = np.array(y)
    y = y.astype(np.int)
    return x, y, max_len
