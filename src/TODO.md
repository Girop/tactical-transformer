# What do we need to do?

- Increase the model capacity, it is very small.
Input doesn't necesarliy need to be couple to output size of GIN, we can add  learnable projection between Transformer's input and GIN's output. 
- Use GIN + Text model.
- Think how to use GIN across multiple logics.
- Better benchmark split, do the same thing as z3alpha did.
- More logics.
- Better data generation schema.
- Subset of tactics per logic generation.
- beam search + top-k later.


# Benchmarks used in z3alpha paper
QF_BV - Sage2, core
QF_NIA - AProVE, leipzig
QF_NRA - hycomp
QF_LIA, QF_LRA, QF_S - whole logic
