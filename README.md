# Transformer-based z3 tactic selection

Current workflow is composed of:
- Generating strategies using `z3alpha`. Produced this way strategies serve as training data for the later components.
- Encoding `*.smt` files using `smt-select`.
- Feeding both the vector representation of smt files and `z3alpha`'s strategies into the transformer-based encoder-decoder network.
- Using the network to produce linear strategy chains.

