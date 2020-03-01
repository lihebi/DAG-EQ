# For 1070
# FIXME not usable again?
g=5.8
# For 2080 Ti
# g=9.0
ENV["CUARRAYS_MEMORY_LIMIT"] = convert(Int, round(g * 1024 * 1024 * 1024))

try
    # This is a weird error. I have to load libdl and dlopen libcutensor first, then
    # CuArrays to have cutensor. Otherwise, if CuArrays is loaded first, libcutensor
    # cannot be even dlopen-ed
    using Libdl
    Libdl.dlopen("libcutensor")
catch ex
    @warn "Cannot open libcutensor library"
end

using CuArrays: allowscalar

allowscalar(false)
# allowscalar(true)
