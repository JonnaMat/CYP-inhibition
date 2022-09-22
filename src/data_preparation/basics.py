'''Data preparation and exploration.'''

from typing import List, Literal, Optional
from tdc.single_pred import ADME
from rdkit import Chem
from rdkit.Chem import Draw


def load_tdc_dataset_split(
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

# TODO 
def read_train_data(filename):
    x = []
    y = []
    max_len = 0
    with open(filename) as infile:
        infile.readline()
        for line in infile:
            line = line.strip('\n\r ')
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