# Supervised DAG Structure Learning with Equivariant Neural Networks

This repository is the official implementation of [Supervised DAG Structure
Learning with Equivariant Neural Networks](https://examples.com).


## Requirements

DAG-EQ is implemented in Julia. Julia has a built-in package manager, so it is
very easy to install dependencies and run the code. Our most recent development
is tested on Julia version 1.6, but should also work on previous versions. You
can install Julia on your machine following instructions on
https://julialang.org/, or use a docker container to run the code. A GPU with
proper driver setup are assumed as the training and inference is most
efficient on GPUs. Running the code on CPUs should be possible but is not tested.

To install requirements:

```sh
cd DAG-EQ
julia --project
]instantiate
]precompile
]build
```

Test if CUDA works:

```julia
import CUDA
CUDA.version()
CUDA.has_cuda()
CUDA.has_cutensor()
```

## Data

The synthetic data will be generated as you run the training or evaluation.


## Pre-trained Models

Pre-trained models are also avaiable, including (d=10, 20, 50, 100, 200, 400).
The pre-trained model file is in `bson` (binary json) format.
To use the pre-trained models, put the `saved_models` in the same directory as
you run the evaluation script.

The content of the saved models:

```
saved_models
├── EQ2-ERSF-k1-d=10-ensemble
│   └── step-50000.bson
├── EQ2-ERSF-k1-d=20-ensemble
│   └── step-50000.bson
├── EQ2-ERSF-k1-d=50-ensemble
│   └── step-50000.bson
├── EQ2-ERSF-k1-d=100-ensemble
│   └── step-50000.bson
├── EQ2-ERSF-k1-d=200-ng=1000-N=1-ensemble
│   └── step-50000.bson
└── EQ2-ERSF-k1-d=400-ng=1000-N=1-ensemble
    └── step-50000.bson
```

## Training

There's a `run.jl` script on notebooks directory. To run it, simply:

```
cd notebooks
julia --project run.jl
```

The hyper-parameters (e.g., training steps, graph types, number of nodes) are
coded in that script, you can adjust based on your needs. The model will be
saved to `saved_models`.

## Evaluation

The code for testing is also in `run.jl`. So if you have run training, you
should have run testing as well. If you use the pre-trained models, you could
run `run.jl` and it will load the pre-trained model for evaluation. Run it by:

```
cd notebooks
julia --project run.jl
```

The results will be saved to `result.csv`. The notebook `plot.ipynb` will turn
the results into tables used in the paper.

## Additional experiments

Additional experiments are available in the following files and notebooks
- `src/main.jl`
- `notebooks/main.ipynb`
- `python/`: this folder contains all the code for running baseline methods for comparison.

## Results
Please see our paper for details.

## Contributing

Our code is available under MIT license.