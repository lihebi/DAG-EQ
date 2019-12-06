using Distributions
using CausalInference
using LightGraphs

using DelimitedFiles, LinearAlgebra, Statistics

p = 0.01

# Download data
run(`wget http://nugget.unisa.edu.au/ParallelPC/data/real/NCI-60.csv`)

# Read data and compute correlation matrix
X = readdlm("NCI-60.csv", ',')
d, n = size(X)
C = Symmetric(cor(X, dims=2))

# Compute skeleton `h` and separating sets `S`
h, S = skeleton(d, gausscitest, (C, n), quantile(Normal(), 1-p/2))

# Compute the CPDAG `g`
g = pcalg(d, gausscitest, (C, n), quantile(Normal(), 1-p/2))
