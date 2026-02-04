from src.config import DataConfig
from src.features.base import BaseFeaturizer

def get_featurizer(config: DataConfig) -> BaseFeaturizer:
    """
    Factory method to get the configured featurizer based on the config.
    Imports are done inside the function to avoid circular dependencies.
    """
    if config.featurizer_type == 'TokenFeaturizer':
        from src.features.featurizer import TokenFeaturizer
        return TokenFeaturizer(vocab_filepath=config.vocab_path)
    
    elif config.featurizer_type == 'ChemPropFeaturizer':
        from src.features.chemprop_adapter import ChemPropPyGFeaturizer
        # Here we could pass modes from config if needed, e.g., config.chemprop_mode
        return ChemPropPyGFeaturizer(atom_featurizer_mode="v2")
    
    else:
        raise ValueError(f"Unknown featurizer type specified in config: {config.featurizer_type}")