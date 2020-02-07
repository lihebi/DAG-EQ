# For 1070
# FIXME not usable again?
g=5.8
# For 2080 Ti
# g=9.0
ENV["CUARRAYS_MEMORY_LIMIT"] = convert(Int, round(g * 1024 * 1024 * 1024))
