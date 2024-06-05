# Benchmarking

If you want to benchmark the performance of the code you could use the function `examples/benchmarks.py` which is a simple script that runs the code multiple times and calculates the average time taken to run the code.

# Comparison with other projects

Compared to other related software like [`FourierGPE.jl`](https://github.com/AshtonSBradley/FourierGPE.jl/tree/master), `NLSE` is much faster as it uses a much simpler solver (the latter uses the awesome [`DifferentialEquations.jl`](https://docs.sciml.ai/DiffEqDocs/stable/)).

Of course this gives you less control over the numerical accuracy of your model, however we think it's worth it since performance often limits the possible physical scenarii.

A comparison between the two projects can be found in [`comparison_juliaGPE.py`](../../examples/comparison_juliaGPE.py) in our examples.

# Testing

The tests have been written to work in tandem with [`pytest`](https://docs.pytest.org/en/8.2.x/) and are run on each repository push using a GitHub action.

These tests essentially check for the physical definitions of all the class methods.

Here is an example of such a test file:

::: tests.test_nlse
    options:
        show_source: true

# Examples

Minimum working examples for each class can be found in the [examples](../../examples) folder of the repo, as well as the code needed to benchmark performance on your machine and comparisons with other NLSE/GPE solvers.