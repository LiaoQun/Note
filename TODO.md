# some need to modify
- () multiple data input
- () save predictions data sets (train/valid/test), need record original smiles and fragments
- () two evaluate function need to be removed
- () save config.py in every training runs
- () implement the BDE predictor: input smiles list to predict all BDEs, drop duplicate, BDE append issue(fragment_bond_indices = [frag['bond_index'] for frag in fragments])
- Molecule class need canon first, then do other thing
- 整合更多針對大規模計算的性能優化（例如批次處理）

