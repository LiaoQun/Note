from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from rdkit import Chem
from torch_geometric.data import Data

class BaseFeaturizer(ABC):
    """
    Abstract base class for converting RDKit molecules into PyG Data objects.
    Enables swapping different featurization strategies (e.g., Token-based, One-Hot, ChemProp).
    """

    @property
    @abstractmethod
    def atom_dim(self) -> int:
        """
        Returns the dimensionality (or number of classes) of the atom features.
        Used for initializing model input layers.
        """
        pass

    @property
    @abstractmethod
    def bond_dim(self) -> int:
        """
        Returns the dimensionality (or number of classes) of the bond features.
        Used for initializing model input layers.
        """
        pass

    @property
    @abstractmethod
    def is_discrete(self) -> bool:
        """
        True if features are integer indices (requires Embedding).
        False if features are continuous vectors (requires Linear projection).
        """
        pass

    @abstractmethod
    def featurize(self, mol: Chem.Mol, 
                  labels: Optional[Dict[Tuple[int, int], float]] = None,
                  smiles: str = "") -> Optional[Data]:
        """
        Main method to convert a molecule into a PyG Data object.

        Args:
            mol (Chem.Mol): The RDKit molecule object (should have Hs added).
            labels (Dict, optional): Dictionary mapping canonical bond indices to BDE values.
                                     Format: {(min_idx, max_idx): bde_float}
            smiles (str, optional): The SMILES string for tracking/debugging.

        Returns:
            Data: PyTorch Geometric Data object with x, edge_index, edge_attr, etc.
            None: If featurization fails or molecule has no valid components.
        """
        pass
    
    def prepare_data(self, smiles_list: List[str]):
        """
        Hook for stateful featurizers to learn from data (e.g., building vocabulary).
        
        Args:
            smiles_list (List[str]): List of SMILES strings to learn from.
        """
        pass

    def save(self, filepath: str):
        """
        Saves the featurizer's internal state (e.g., vocabulary) to a file.
        
        Args:
            filepath (str): Path to save the state.
        """
        pass

    def load(self, filepath: str):
        """
        Loads the featurizer's internal state from a file.
        
        Args:
            filepath (str): Path to load the state from.
        """
        pass
