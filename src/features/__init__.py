def get_featurizer(config):
    """
    Factory method to get the configured featurizer based on the config.
    Imports are done inside the function to avoid circular dependencies.
    """
    # Defer imports to function scope
    from src.features.featurizer import TokenFeaturizer
    from src.features.chem_prop_wrapper import ChemPropFeaturizer

    if config.featurizer_type == 'TokenFeaturizer':
        return TokenFeaturizer(vocab_filepath=config.vocab_path)
    
    elif config.featurizer_type == 'ChemPropFeaturizer':
        return ChemPropFeaturizer(mode='V2')
    
    else:
        raise ValueError(f"Unknown featurizer type specified in config: {config.featurizer_type}")