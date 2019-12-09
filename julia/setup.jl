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

Pkg.instantiate()

Pkg.status()
Pkg.update()

# Pkg.build("Arpack")
