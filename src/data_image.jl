import Base.show
import Base.display

include("display.jl")
include("data.jl")

using FileIO
using FileIO: @format_str

using Flux: onehotbatch, cpu
using Random

using MLDatasets

struct MyImage
    arr::AbstractArray
end

function Base.show(io::IO, ::MIME"image/png", x::MyImage)
    img = cpu(x.arr)
    size(img)[3] in [1, 3] || error("Unsupported channel size: ", size(img)[3])
    if size(img)[3] == 3
        getRGB(X) = colorview(RGB, permutedims(X, (3,1,2)))
        img = getRGB(img)
    end
    FileIO.save(FileIO.Stream(format"PNG", io), img)
end

function Base.display(d::EmacsDisplay, mime::MIME"image/png", x::MyImage)
    # path, io = Base.Filesystem.mktemp()
    # path * ".png"
    path = tempname() * ".png"
    open(path, "w") do io
        show(io, mime, x)
    end

    println("$(path)")
    println("#<Image: $(path)>")
end

# FIXME use Zygote.jl
# using Tracker: TrackedArray

function sample_and_view(x, y=nothing, model=nothing)
    if length(size(x)) < 4
        x = x[:,:,:,:]
    end
    # if typeof(x) <: Flux.Tracker.TrackedArray
    #     x = x.data
    # end
    size(x)[1] in [28,32,56] ||
        error("Image size $(size(x)[1]) not correct size. Currently support 28 or 32.")
    num = min(size(x)[4], 10)
    @info "Showing $num images .."
    if num == 0 return end
    imgs = cpu(hcat([x[:,:,:,i] for i in 1:num]...))
    # viewrepl(imgs)
    display(MyImage(imgs))
    if y != nothing
        labels = onecold(cpu(y[:,1:num]), 0:9)
        @show labels
    end
    if model != nothing
        preds = onecold(cpu(model(gpu(x))[:,1:num]), 0:9)
        @show preds
    end
    nothing
end

function load_MNIST_ds(;batch_size)
    # Ideally, it takes a batch size, and should return an iterator that repeats
    # infinitely, and shuffle before each epoch. The data should be moved to GPU
    # on demand. I prefer to just move online during training.
    train_x, train_y = MLDatasets.MNIST.traindata();
    test_x,  test_y  = MLDatasets.MNIST.testdata();
    # normalize into -1, 1
    train_x = (train_x .- 0.5) .* 2
    test_x = (test_x .- 0.5) .* 2

    # reshape to add channel, and onehot y encoding
    #
    # I'll just permute dims to make it column major
    train_ds = DataSetIterator(reshape(permutedims(train_x, [2,1,3]), 28,28,1,:),
                               onehotbatch(train_y, 0:9), batch_size);
    # FIXME test_ds should repeat only once
    # TODO add iterator interface, for x,y in ds end
    test_ds = DataSetIterator(reshape(permutedims(test_x, [2,1,3]), 28,28,1,:),
                              onehotbatch(test_y, 0:9), batch_size);
    # x, y = next_batch!(train_ds);
    return train_ds, test_ds
end
