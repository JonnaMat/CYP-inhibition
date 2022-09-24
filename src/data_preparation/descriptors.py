'''Descriptor calculation.'''

from typing import List
import pandas as pd
from rdkit.Chem import Descriptors, MolFromSmiles
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator


def calculate_descriptors(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate all available descriptors for the given molecules.

    :param smiles_list: List of SMILES strings
    :type smiles_list: List[str]
    :rtype: pd.DataFrame
    """

    molecules = [MolFromSmiles(smiles) for smiles in list(df["Drug"])]
    calculator = MolecularDescriptorCalculator(
        [x[0] for x in Descriptors._descList])
    header = calculator.GetDescriptorNames()


    descriptors = [calculator.CalcDescriptors(
        molecule) for molecule in molecules]

    descriptor_df = pd.DataFrame(descriptors, columns=header)
    descriptor_df.insert(loc=0, column='smiles', value=list(df["Drug"]))
    descriptor_df.insert(loc=0, column='Drug_ID', value=list(df["Drug_ID"]))
    descriptor_df.insert(loc=0, column='Y', value=list(df["Y"]))

    return descriptor_df

def extract_null(df: pd.DataFrame):
    df = df.iloc[1:] # drop first row
    null_columns_df = df[df.columns[df.isnull().any()]]
    return pd.merge(df[['smiles', 'Drug_ID', 'Y']], null_columns_df[null_columns_df.isna().any(axis=1)], left_index=True, right_index=True)

    