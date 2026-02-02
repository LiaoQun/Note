from src.config import DataConfig
from src.features.base import BaseFeaturizer
from src.features.featurizer import TokenFeaturizer

def get_featurizer(config: DataConfig) -> BaseFeaturizer:
    """
    Factory method to get the configured featurizer.
    """
    if config.featurizer_type == 'TokenFeaturizer':
        return TokenFeaturizer(vocab_filepath=config.vocab_path)
    elif config.featurizer_type == 'ChemPropFeaturizer':
        # Local import to avoid circular dependencies or loading errors if not used
        from src.features.chem_prop_wrapper import ChemPropFeaturizer
        return ChemPropFeaturizer(mode='V2') # Can add mode to config later if needed
    else:
        raise ValueError(f"Unknown featurizer type: {config.featurizer_type}")
