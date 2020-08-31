using Pkg

Pkg.build(verbose=true)

Pkg.add("GaussianProcess")
Pkg.add("Distributions")
Pkg.add("Arpack")
Pkg.add("CausalInference")
Pkg.add("Match")
Pkg.add("RecursiveArrayTools")

# for LightGraphs
Pkg.add("FileIO")
Pkg.add("LightGraphs")

Pkg.add("GraphPlot")
# a compatibility bug with Cairo
# https://github.com/GiovineItalia/Compose.jl/pull/360
# Pkg.add(PackageSpec(name="Compose", rev="master"))
Pkg.add("Compose")
Pkg.add("Cairo")
# Pkg.add("Fontconfig")
Pkg.add("MetaGraphs")

Pkg.add("Optim")
Pkg.add("NLopt")
Pkg.add("LineSearches")
Pkg.add("CSV")
Pkg.add("BSON")
Pkg.add("ForwardDiff")
Pkg.add("Tracker")

Pkg.add("Images")
Pkg.add("ProgressMeter")
Pkg.add("MLDatasets")
Pkg.add("ImageMagick")

Pkg.add("TensorBoardLogger")
Pkg.add("Plots")
Pkg.add("PyPlot")

Pkg.add("Interpolations")
Pkg.add("GaussianMixtures")
Pkg.add("StatsFuns")
Pkg.add("TensorOperations")
Pkg.add(Pkg.PackageSpec(name="TensorOperations", rev="master"))
Pkg.add(Pkg.PackageSpec(url="https://github.com/mcabbott/TensorGrad.jl", rev="master"))
Pkg.free("TensorOperations")
Pkg.status()
Pkg.update()
Pkg.build("CUDA")
Pkg.rm("Zygote")
Pkg.add("Zygote")
Pkg.build()
import GPUArrays
Pkg.add("FillArrays")
Pkg.add("DataFrames")

# FIXME I have to manually specify the version
# Pkg.update does not update the package
Pkg.add(Pkg.PackageSpec(name="Flux", version="0.11"))


#!!!
Pkg.add("Distributions")

##############################
## debugging (not needed)
Pkg.add("Colors")
Pkg.add("BenchmarkTools")
Pkg.add("GPUArrays")

Pkg.develop("Zygote")
Pkg.update()
Pkg.status()
Pkg.build(verbose=true)


##############################
## Using Flux#master and Zygote

Pkg.add("Flux")
Pkg.rm("Flux")
Pkg.rm("Tracker")
Pkg.add(PackageSpec(name="Flux", rev="master"))
Pkg.free("Flux")
Pkg.add(Pkg.PackageSpec(name="Zygote", rev="master"))

##############################
## Or using Flux 0.9 and Tracker
Pkg.rm("Flux")
Pkg.rm("GPUArrays")
Pkg.rm("Distributions")
Pkg.add(PackageSpec(name="Flux", version="0.9"))
Pkg.pin(PackageSpec(name="Flux", version="0.9"))
Pkg.add("Tracker")
Pkg.add("Distributions")

Pkg.add("HDF5")
Pkg.build("HDF5")

Pkg.add("PackageCompiler")

Pkg.Registry.update()

Pkg.instantiate()

Pkg.status()
Pkg.update()

# Pkg.build("Arpack")
