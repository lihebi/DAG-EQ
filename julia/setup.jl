using Pkg

Pkg.add("Distributions")
Pkg.add("Arpack")
Pkg.add("CausalInference")
Pkg.add("LightGraphs")

Pkg.instantiate()

Pkg.update()


# Pkg.build("Arpack")
