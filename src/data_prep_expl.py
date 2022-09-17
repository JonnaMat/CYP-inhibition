'''Data preparation and exploration.'''

from typing import List, Literal, Optional
from tdc.single_pred import ADME
from rdkit import Chem
from rdkit.Chem import Draw


def load_dataset_split(
    task: Literal['CYP2C19', 'CYP2D6', 'CYP3A4', 'CYP1A2', 'CYP2C9'],
    method: Literal['random', 'scaffold'] = 'random',
    frac: Optional[List[float]] = None
):
    """Load dataset train-, val-, test-split given a `task`."""
    data = ADME(name=task+'_Veith')
    split = data.get_split(
        method=method,
        seed=1,
        frac=[0.7, 0.1, 0.2] if frac is None else frac)
    return split

def draw_molecule(smiles: str):
    """Draw molecule from SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    return Draw.MolToImage(mol)