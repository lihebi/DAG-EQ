using Pkg

Pkg.add("Distributions")
Pkg.add("Arpack")
Pkg.add("CausalInference")

# for LightGraphs
Pkg.add("FileIO")
Pkg.add("LightGraphs")

Pkg.add("GraphPlot")
# a compatibility bug with Cairo
# https://github.com/GiovineItalia/Compose.jl/pull/360
Pkg.add(PackageSpec(name="Compose", rev="master"))
Pkg.add("Cairo")
Pkg.add("Fontconfig")
Pkg.add("MetaGraphs")

Pkg.add("Optim")
Pkg.add("NLopt")
Pkg.add("LineSearches")
Pkg.add("CSV")

Pkg.add("Images")
Pkg.add("ProgressMeter")
Pkg.add("MLDatasets")
Pkg.add("ImageMagick")

Pkg.add("TensorBoardLogger")
Pkg.add("Plots")
Pkg.add("PyPlot")


# Flux
Pkg.add("CUDAnative")
Pkg.rm("CUDAnative")
Pkg.rm("CuArrays")
#!!!
Pkg.rm("Distributions")

Pkg.add("Flux")
Pkg.rm("Flux")
Pkg.rm("Tracker")
Pkg.add(PackageSpec(name="Flux", version="0.9"))
Pkg.pin(PackageSpec(name="Flux", version="0.9"))
Pkg.add(PackageSpec(name="Flux", rev="master"))
Pkg.free("Flux")
Pkg.add("Tracker")

Pkg.instantiate()

Pkg.status()
Pkg.update()

# Pkg.build("Arpack")
