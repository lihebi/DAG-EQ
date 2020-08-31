import Base.show
using CUDA: CuArray

# There is a data loader PR: https://github.com/FluxML/Flux.jl/pull/450
mutable struct DataSetIterator
    raw_x::AbstractArray
    raw_y::AbstractArray
    index::Int
    batch_size::Int
    nbatch::Int
end
function DataSetIterator(x,y,batch_size)
    DataSetIterator(x, y, 1, batch_size,
                    convert(Int, floor(size(x)[end]/batch_size)))
end

mutable struct CuDataSetIterator
    gpu_x::Array{CuArray}
    gpu_y::Array{CuArray}
    index::Int
    batch_size::Int
    nbatch::Int
end

function Base.convert(::Type{CuDataSetIterator}, ds::DataSetIterator)
    CuDataSetIterator(ds.raw_x, ds.raw_y, ds.batch_size)
end

function CuDataSetIterator(ds::DataSetIterator)
    CuDataSetIterator(ds.raw_x, ds.raw_y, ds.batch_size)
end

function CuDataSetIterator(x,y,batch_size)
    xdims = collect(repeated(:, length(size(x)) - 1))
    ydims = collect(repeated(:, length(size(y)) - 1))

    N = size(x)[end]
    nbatch = Int(floor(N/batch_size))

    # reshuffle
    indices = shuffle(1:size(x)[end])
    x = x[xdims...,indices]
    y = y[ydims...,indices]

    gpu_x = map(1:nbatch) do i
        start = (i-1) * batch_size + 1
        stop = i * batch_size
        x[xdims..., start:stop]
    end
    gpu_y = map(1:nbatch) do i
        start = (i-1) * batch_size + 1
        stop = i * batch_size
        y[ydims..., start:stop]
    end
    # FIXME .|>
    gpu_x = gpu_x |> gpu
    gpu_y = gpu_y |> gpu
    CuDataSetIterator(gpu_x, gpu_y, 1, batch_size, nbatch)
end

Base.show(io::IO, x::DataSetIterator) = begin
    println(io, "DataSetIterator:")
    println(io, "  batch size: ", x.batch_size)
    println(io, "  number of batches: ", x.nbatch)
    println(io, "  x data shape: ", size(x.raw_x))
    println(io, "  y data shape: ", size(x.raw_y))
    print(io, "  current index: ", x.index)
end

Base.show(io::IO, x::CuDataSetIterator) = begin
    println(io, "CuDataSetIterator:")
    println(io, "  batch size: ", x.batch_size)
    println(io, "  number of batches: ", x.nbatch)
    println(io, "  x data shape: ", size(x.gpu_x[1]))
    println(io, "  y data shape: ", size(x.gpu_y[1]))
    print(io, "  current index: ", x.index)
end

function next_batch!(ds::DataSetIterator)
    if ds.index > ds.nbatch
        ds.index = 1
    end

    xdims = collect(repeated(:, length(size(ds.raw_x)) - 1))
    ydims = collect(repeated(:, length(size(ds.raw_y)) - 1))

    if ds.index == 1
        indices = shuffle(1:size(ds.raw_x)[end])
        ds.raw_x = ds.raw_x[xdims...,indices]
        ds.raw_y = ds.raw_y[ydims...,indices]
    end

    a = (ds.index - 1) * ds.batch_size + 1
    b = ds.index * ds.batch_size

    ds.index += 1
    # FIXME use view instead of copy
    # FIXME properly slice with channel-last format
    return ds.raw_x[xdims...,a:b], ds.raw_y[ydims...,a:b]
end

function next_batch!(ds::CuDataSetIterator)
    if ds.index > ds.nbatch
        ds.index = 1
    end
    ds.index += 1
    return ds.gpu_x[ds.index-1], ds.gpu_y[ds.index-1]
end

_next_batch_step = 0
function next_batch!(dses::Array{T, N}
                     where T<:Union{DataSetIterator, CuDataSetIterator}
                     where N)
    # keep an internal step?
    global _next_batch_step
    _next_batch_step += 1
    next_batch!(dses[_next_batch_step % length(dses) + 1])
end

"""
1. shuffle and divide *all* data into batches
2. move *all* data to GPU
3. each time next_batch! is called, a pointer to the GPU array is returned
"""
function move_to_gpu!(ds::DataSetIterator)
    xdims = collect(repeated(:, length(size(ds.raw_x)) - 1))
    ydims = collect(repeated(:, length(size(ds.raw_y)) - 1))

    N = size(ds.raw_x)[end]

    # reshuffle
    indices = shuffle(1:size(ds.raw_x)[end])
    ds.raw_x = ds.raw_x[xdims...,indices]
    ds.raw_y = ds.raw_y[ydims...,indices]

    gpu_x = map(1:Int(floor(N/ds.batch_size))) do i
        start = (i-1) * ds.batch_size + 1
        stop = i * ds.batch_size
        ds.raw_x[xdims..., start:stop]
    end
    gpu_y = map(1:Int(floor(N/ds.batch_size))) do i
        start = (i-1) * ds.batch_size + 1
        stop = i * ds.batch_size
        ds.raw_y[ydims..., start:stop]
    end
    ds.gpu_x = gpu_x |> gpu
    ds.gpu_y = gpu_y |> gpu
    ds.gpu_index = 1
    nothing
end
