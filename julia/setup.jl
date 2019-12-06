using Pkg

Pkg.add("Distributions")
Pkg.add("Arpack")
Pkg.add("CausalInference")

# for LightGraphs
Pkg.add("FileIO")
Pkg.add("LightGraphs")

Pkg.add("GraphPlot")
Pkg.add("Compose")
Pkg.add("Cairo")
Pkg.add("Fontconfig")

Pkg.instantiate()

Pkg.status()
Pkg.update()

# Pkg.build("Arpack")
