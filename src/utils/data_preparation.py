"""Data preparation and exploration."""
import os
from pathlib import Path
from typing import List, Literal, Dict, Optional
from time import sleep
from copy import deepcopy
import pandas as pd
import numpy as np

from tdc.single_pred import ADME

# pylint: disable=no-name-in-module
from rdkit.Chem import Descriptors, MolFromSmiles, MolToSmiles
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from rdkit.Chem.MolStandardize import rdMolStandardize

from molvs.standardize import Standardizer


def calculate_fingerprints(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate a number of fingerprints and add to 'data'."""

# pylint: disable=protected-access
CALCULATOR = MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])

def calculate_descriptors(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate all available descriptors for the given molecules and add to 'data'."""

    header = CALCULATOR.GetDescriptorNames()

    descriptors = [
        CALCULATOR.CalcDescriptors(MolFromSmiles(smiles))
        for smiles in data["Drug"].values
    ]

    descriptor_data = pd.DataFrame(descriptors, columns=header)
    for column in data.columns:
        descriptor_data.insert(loc=0, column=column, value=list(data[column]))
    return descriptor_data

def summarize_descriptors(descriptors: List[str]):
    """Generate summaries of descriptors."""
    desc_names = CALCULATOR.GetDescriptorNames()
    summaries = CALCULATOR.GetDescriptorSummaries()

    return {descriptor: summaries[desc_names.index(descriptor)] for descriptor in descriptors}



def extract_null(data: pd.DataFrame):
    """Return all row/column pairs with NaN values."""
    null_columns_data = data[data.columns[data.isnull().any()]]
    return pd.merge(
        data[["Drug", "Drug_ID", "Y"]],
        null_columns_data[null_columns_data.isna().any(axis=1)],
        left_index=True,
        right_index=True,
    )


def dataset_split(
    data: pd.DataFrame, frac: Optional[List[float]] = None
) -> Dict[Literal["train", "val", "test"], pd.DataFrame]:
    """Shuffle the dataset and create random split of the dataset."""

    train_frac, val_frac, _ = [0.7, 0.1, 0.2] if frac is None else frac
    n_samples = len(data)
    # pylint: disable=unbalanced-tuple-unpacking
    train, val, test = np.split(
        data.sample(frac=1, random_state=42, ignore_index=False),
        [int(train_frac * n_samples), int((train_frac + val_frac) * n_samples)],
    )
    return {
        "train": train.reindex(),
        "val": val,
        "test": test,
    }


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


def load_tdc_dataset_full(
    task: Literal["CYP2C19", "CYP2D6", "CYP3A4", "CYP1A2", "CYP2C9"],
):
    """Load dataset train-, val-, test-split given a `task`."""
    data = ADME(name=task + "_Veith")
    return data.get_data()


MOL_STANDARDIZER = Standardizer()
# pylint: disable=c-extension-no-member
LARGEST_FRAGMENT_CHOOSER = rdMolStandardize.LargestFragmentChooser()


def normalize_smiles(smiles: str) -> str:
    """Return normalized smiled string given `smiles`."""

    molecule = MolFromSmiles(smiles)
    std_molecule = MOL_STANDARDIZER.standardize(molecule)
    largest_mol = LARGEST_FRAGMENT_CHOOSER.choose(std_molecule)
    if MolToSmiles(largest_mol) != smiles:
        print()
        print(smiles)
        print(MolToSmiles(largest_mol))

    return MolToSmiles(largest_mol)


def remove_small_molecules(data: pd.DataFrame) -> pd.DataFrame:
    """Remove small molecules according to user input."""

    print(data["Drug"][lambda drugs: [len(drug) <= 9 for drug in drugs]].to_string())
    sleep(1)
    remove_molecules = []
    num_remove_molecules = int(input("Enter number of molecules to remove: "))

    for _ in range(num_remove_molecules):
        remove_molecules.append(int(input("Index of molecule to be removed: ")))

    print("Removing the following molecules: ")
    print(data.iloc[remove_molecules]["Drug"].to_string())

    return deepcopy(data).drop(remove_molecules)


def data_preprocessing(
    task: Literal["CYP2C19", "CYP2D6", "CYP3A4", "CYP1A2", "CYP2C9"]
) -> pd.DataFrame:
    """
    Return the raw dataset consisting of all 208 descriptors including NaN
    values and saves dataset.

    The following steps are performed:

        1. Fetch dataset from TDC
        2. Normalize smiles strings
        3. Calculate Descriptors
        4. Normalize Descriptor values
        5. TODO Calculate Fingerprints
    """
    filename = f"data/{task.lower()}/raw_dataset_descriptors.csv"  # TODO

    # create directories if necessary
    Path(f"data/{task.lower()}").mkdir(parents=True, exist_ok=True)

    # check if dataset already exists
    if os.path.isfile(filename):
        # pylint: disable=logging-fstring-interpolation
        print(f"Dataset already exists, returning {filename}.")
        return pd.read_csv(filename).drop("Unnamed: 0", axis=1)

    print("Fetching Dataset from TDC...")
    tdc_data = load_tdc_dataset_full(task)

    print("Normalizing smiles strings...")
    tdc_data["Drug"] = tdc_data["Drug"].map(normalize_smiles)

    print("Calculating descriptors...")
    descriptor_data = calculate_descriptors(tdc_data)
    # 4. TODO
    # 5. TODO

    # pylint: disable=logging-fstring-interpolation
    print(f"Save dataset to {filename}.")
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
