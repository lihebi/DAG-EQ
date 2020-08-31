using PackageCompiler

# FIXME this probably would compile all functions in these packages? If that's
# slow, try identify the key functions to compile
create_sysimage([:CUDA, :Zygote,
                 :Distributions,
                 :LightGraphs, :MetaGraphs,
                 :GraphPlot,
                 :Flux,
                 :TensorBoardLogger,
                 :CSV, :Plots, :DataFrames, :HDF5,
                 :TensorOperations],
                # sysimage_path="myimage.so",
                replace_default=true)

## run this file to create the image
# julia --project precompile.jl

## to use the image
# julia --sysimage myimage.so --project main.jl
restore_default_sysimage()
