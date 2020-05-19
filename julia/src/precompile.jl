using PackageCompiler

# FIXME this probably would compile all functions in these packages? If that's
# slow, try identify the key functions to compile
create_sysimage([:CuArrays, :Zygote,
                 :Distributions,
                 :LightGraphs, :MetaGraphs,
                 :CSV, :Plots, :DataFrames, :HDF5,
                 :TensorOperations],
                sysimage_path="myimage.so")

## run this file to create the image
# julia --project precompile.jl

## to use the image
# julia --sysimage myimage.so --project main.jl
