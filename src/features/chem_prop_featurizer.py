from enum import Enum, auto
from typing import Sequence, TypeVar, Generic, List, Dict, Any, NamedTuple, Optional, Union, Tuple
from abc import abstractmethod
from collections.abc import Sized

import numpy as np
from rdkit.Chem.rdchem import Atom, Bond, HybridizationType, BondType

# --- Start: Shims for missing chemprop base classes ---

class MolGraph(NamedTuple):
    """A :class:`MolGraph` represents the graph featurization of a molecule."""
    V: np.ndarray
    E: np.ndarray
    edge_index: np.ndarray
    rev_edge_index: np.ndarray

S = TypeVar("S")
T = TypeVar("T")

class Featurizer(Generic[S, T]):
    """An :class:`Featurizer` featurizes inputs type ``S`` into outputs of type ``T``."""
    @abstractmethod
    def __call__(self, input: S, *args, **kwargs) -> T:
        """featurize an input"""

class VectorFeaturizer(Featurizer[S, np.ndarray], Sized):
    ...

class GraphFeaturizer(Featurizer[S, MolGraph]):
    @property
    @abstractmethod
    def shape(self) -> Tuple[int, int]:
        ...

class EnumMapping(Enum):
    """Helper for Enum mapping, mimicking chemprop's structure."""
    @classmethod
    def get(cls, value: Any):
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            try:
                return cls[value.upper()]
            except KeyError:
                pass
        raise ValueError(f"Invalid value '{value}' for {cls.__name__}")

# --- End: Shims ---


class MultiHotAtomFeaturizer(VectorFeaturizer[Atom]):
    """A :class:`MultiHotAtomFeaturizer` uses a multi-hot encoding to featurize atoms."""
    def __init__(
        self,
        atomic_nums: Sequence[int],
        degrees: Sequence[int],
        formal_charges: Sequence[int],
        chiral_tags: Sequence[int],
        num_Hs: Sequence[int],
        hybridizations: Sequence[int],
    ):
        self.atomic_nums = {j: i for i, j in enumerate(atomic_nums)}
        self.degrees = {i: i for i in degrees}
        self.formal_charges = {j: i for i, j in enumerate(formal_charges)}
        self.chiral_tags = {i: i for i in chiral_tags}
        self.num_Hs = {i: i for i in num_Hs}
        self.hybridizations = {ht: i for i, ht in enumerate(hybridizations)}

        self._subfeats: List[Dict] = [
            self.atomic_nums, self.degrees, self.formal_charges,
            self.chiral_tags, self.num_Hs, self.hybridizations,
        ]
        subfeat_sizes = [
            1 + len(self.atomic_nums), 1 + len(self.degrees), 1 + len(self.formal_charges),
            1 + len(self.chiral_tags), 1 + len(self.num_Hs), 1 + len(self.hybridizations),
            1, 1,
        ]
        self.__size = sum(subfeat_sizes)

    def __len__(self) -> int:
        return self.__size

    def __call__(self, a: Optional[Atom]) -> np.ndarray:
        x = np.zeros(self.__size)
        if a is None: return x

        feats = [
            a.GetAtomicNum(), a.GetTotalDegree(), a.GetFormalCharge(),
            int(a.GetChiralTag()), int(a.GetTotalNumHs()), a.GetHybridization(),
        ]
        i = 0
        for feat, choices in zip(feats, self._subfeats):
            j = choices.get(feat, len(choices))
            x[i + j] = 1
            i += len(choices) + 1
        x[i] = int(a.GetIsAromatic())
        x[i + 1] = 0.01 * a.GetMass()
        return x

    @classmethod
    def v1(cls, max_atomic_num: int = 100):
        return cls(
            atomic_nums=list(range(1, max_atomic_num + 1)), degrees=list(range(6)),
            formal_charges=[-1, -2, 1, 2, 0], chiral_tags=list(range(4)), num_Hs=list(range(5)),
            hybridizations=[HybridizationType.SP, HybridizationType.SP2, HybridizationType.SP3, HybridizationType.SP3D, HybridizationType.SP3D2],
        )

    @classmethod
    def v2(cls):
        return cls(
            atomic_nums=list(range(1, 37)) + [53], degrees=list(range(6)),
            formal_charges=[-1, -2, 1, 2, 0], chiral_tags=list(range(4)), num_Hs=list(range(5)),
            hybridizations=[HybridizationType.S, HybridizationType.SP, HybridizationType.SP2, HybridizationType.SP2D, HybridizationType.SP3, HybridizationType.SP3D, HybridizationType.SP3D2],
        )

    @classmethod
    def organic(cls):
        return cls(
            atomic_nums=[1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53], degrees=list(range(6)),
            formal_charges=[-1, -2, 1, 2, 0], chiral_tags=list(range(4)), num_Hs=list(range(5)),
            hybridizations=[HybridizationType.S, HybridizationType.SP, HybridizationType.SP2, HybridizationType.SP3],
        )

class AtomFeatureMode(EnumMapping):
    V1 = auto()
    V2 = auto()
    ORGANIC = auto()

def get_multi_hot_atom_featurizer(mode: Union[str, AtomFeatureMode]) -> MultiHotAtomFeaturizer:
    mode_enum = AtomFeatureMode.get(mode)
    if mode_enum == AtomFeatureMode.V1:
        return MultiHotAtomFeaturizer.v1()
    elif mode_enum == AtomFeatureMode.V2:
        return MultiHotAtomFeaturizer.v2()
    elif mode_enum == AtomFeatureMode.ORGANIC:
        return MultiHotAtomFeaturizer.organic()
    else:
        raise RuntimeError("unreachable code reached!")


class MultiHotBondFeaturizer(VectorFeaturizer[Bond]):
    def __init__(
        self, bond_types: Optional[Sequence[BondType]] = None, stereos: Optional[Sequence[int]] = None
    ):
        self.bond_types = bond_types or [BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE, BondType.AROMATIC]
        self.stereo = stereos or range(6)

    def __len__(self):
        return 1 + len(self.bond_types) + 2 + (len(self.stereo) + 1)

    def __call__(self, b: Bond) -> np.ndarray:
        x = np.zeros(len(self), int)
        if b is None:
            x[0] = 1
            return x

        i = 1
        bond_type = b.GetBondType()
        try:
            bt_bit = self.bond_types.index(bond_type)
            x[i + bt_bit] = 1
        except ValueError:
            pass # Keep as all-zero if bond type is not in the list
        i += len(self.bond_types)

        x[i] = int(b.GetIsConjugated())
        i += 1
        x[i] = int(b.IsInRing())
        i += 1
        
        try:
            stereo_bit = self.stereo.index(int(b.GetStereo()))
            x[i + stereo_bit] = 1
        except ValueError:
            x[i + len(self.stereo)] = 1 # Unknown stereo
        
        return x
