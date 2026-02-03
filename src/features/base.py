from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from rdkit import Chem
from torch_geometric.data import Data

class BaseFeaturizer(ABC):
    """
    Abstract base class for converting RDKit molecules into PyG Data objects.
    """

    @property
    @abstractmethod
    def atom_dim(self) -> int:
        """Dimensionality of atom features."""
        pass

    @property
    @abstractmethod
    def bond_dim(self) -> int:
        """Dimensionality of bond features."""
        pass

    @property
    @abstractmethod
    def is_discrete(self) -> bool:
        """True if features are integer indices, False if continuous."""
        pass

    @abstractmethod
    def featurize(self, mol: Chem.Mol, 
                  labels: Optional[Dict[Tuple[int, int], float]] = None,
                  smiles: str = "") -> Optional[Data]:
        """Convert a molecule to a graph data object."""
        pass
    
    def prepare_data(self, smiles_list: List[str]):
        """Hook to learn from data (e.g., build vocab)."""
        pass

    def save(self, filepath: str):
        """Saves the featurizer's internal state."""
        pass

    def load(self, filepath: str):
        """Loads the featurizer's internal state."""
        pass
