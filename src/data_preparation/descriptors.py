'''Descriptor calculation.'''

from typing import List
import pandas as pd
from rdkit.Chem import Descriptors, MolFromSmiles
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator


def calculate_descriptors(smiles_list: List[str]) -> pd.DataFrame:
    """Calculate all available descriptors for the given molecules.

    :param smiles_list: List of SMILES strings
    :type smiles_list: List[str]
    :rtype: pd.DataFrame
    """
    molecules = [MolFromSmiles(smiles) for smiles in smiles_list]
    calculator = MolecularDescriptorCalculator(
        [x[0] for x in Descriptors._descList])
    header = calculator.GetDescriptorNames()

    descriptions = [calculator.GetDescriptorSummaries()]

    descriptors = [calculator.CalcDescriptors(
        molecule) for molecule in molecules]

    descriptor_df = pd.DataFrame(descriptions+descriptors, columns=header)
    descriptor_df.insert(loc=0, column='smiles', value=["description"] + smiles_list)

    return descriptor_df
