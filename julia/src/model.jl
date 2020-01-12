using Flux
import Zygote
using CuArrays
# CuArrays.has_cutensor()
using LinearAlgebra: Diagonal
using TensorOperations
# using TensorGrad

function param_count(model)
    ps = Flux.params(model)
    res = 0
    for p in keys(ps.params.dict)
        res += prod(size(p))
    end
    res
end

struct Flatten end

function (l::Flatten)(input)
    reshape(input, :, size(input, 4))
end

struct Sigmoid end
function (l::Sigmoid)(input)
    σ.(input)
end

struct ReLU end

function (l::ReLU)(input)
    relu.(input)
end

struct LeakyReLU end
function (l::LeakyReLU)(input)
    leakyrelu.(input)
end

struct Tanh end
function (l::Tanh)(input)
    tanh.(input)
end

struct Equivariant{T}
    λ::AbstractArray{T}
    γ::AbstractArray{T}
    w::AbstractArray{T}
    # FIXME how Flux decides ch is not a trainable parameter?
    ch::Pair{<:Integer,<:Integer}
end
Flux.@functor Equivariant

mynfan(dims...) = prod(dims[1:end-2]) .* (dims[end-1], dims[end])
my_glorot_uniform(dims...) = (rand(Float32, dims...) .- 0.5f0) .* sqrt(24.0f0 / sum(mynfan(dims...)))

function Equivariant(ch::Pair{<:Integer,<:Integer})
    # FIXME NOW init function
    return Equivariant(
        my_glorot_uniform(ch[1],ch[2]),
        my_glorot_uniform(ch[1],ch[2]),
        my_glorot_uniform(ch[1],ch[2]),
        ch)
end

function eqfn(X::AbstractArray, λ::AbstractArray, w::AbstractArray, γ::AbstractArray)
    d = size(X,1)
    # convert to Float32, move this to GPU
    # FIXME performance
    one = ones(Float32, d, d)
    # support CPU as well
    # FIXME performance
    if typeof(X) <: CuArray
        one = gpu(one)
    end
    # FIXME CUTENSOR_STATUS_ARCH_MISMATCH error: cutensor only supports RTX20
    # series (with compute capability 7.0+) see:
    # https://developer.nvidia.com/cuda-gpus
    @tensor X1[a,b,ch2,batch] := X[a,b,ch1,batch] * λ[ch1,ch2]
    # TODO how to do max pooling?
    @tensor X2[a,c,ch2,batch] := one[a,b] * X[b,c,ch1,batch] * w[ch1,ch2]
    @tensor X3[a,c,ch2,batch] := X[a,b,ch1,batch] * one[b,c] * w[ch1,ch2]
    @tensor X4[a,d,ch2,batch] := one[a,b] * X[b,c,ch1,batch] * one[c,d] * γ[ch1,ch2]
    Y = X1 + X2 ./ d + X3 ./ d + X4 ./ (d * d)
end

# from https://github.com/mcabbott/TensorGrad.jl
Zygote.@adjoint function eqfn(X::AbstractArray, λ::AbstractArray, w::AbstractArray, γ::AbstractArray)
    d = size(X,1)
    one = ones(Float32, d, d)
    if typeof(X) <: CuArray
        one = gpu(one)
    end
    eqfn(X, λ, w, γ), function (ΔY)
        # ΔY is FillArray.Fill, and this is not handled in @tensor. Convert it
        # to normal array here. FIXME performance
        # ΔY = Array(ΔY)
        #
        # FIXME However, this will change CuArray to regular Array. Removing
        # this would work for CuArray, but not on CPU. I probably won't
        # calculate gradient on CPU anyway.
        @tensor ΔX1[a,b,ch1,batch] := ΔY[a,b,ch2,batch] * λ[ch1,ch2]
        @tensor ΔX2[a,c,ch1,batch] := one[a,b] * ΔY[b,c,ch2,batch] * w[ch1,ch2]
        @tensor ΔX3[a,c,ch1,batch] := ΔY[a,b,ch2,batch] * one[b,c] * w[ch1,ch2]
        @tensor ΔX4[a,d,ch1,batch] := one[a,b] * ΔY[b,c,ch2,batch] * one[c,d] * γ[ch1,ch2]
        @tensor Δλ[ch1,ch2] := X[a,b,ch1,batch] * ΔY[a,b,ch2,batch]
        @tensor Δw1[ch1,ch2] := one[a,b] * X[b,c,ch1,batch] * ΔY[a,c,ch2,batch]
        @tensor Δw2[ch1,ch2] := X[a,b,ch1,batch] * one[b,c] * ΔY[a,c,ch2,batch]
        @tensor Δγ[ch1,ch2] := one[a,b] * X[b,c,ch1,batch] * one[c,d] * ΔY[a,d,ch2,batch]
        return (ΔX1 + ΔX2 ./ d + ΔX3 ./ d + ΔX4 ./ (d*d),
                Δλ,
                (Δw1+Δw2) ./ d,
                Δγ ./ (d*d))
    end
end

function (l::Equivariant)(X)
    # X = X[:,:,:,:]
    eqfn(X, l.λ, l.w, l.γ)
end

function generator()
    Chain(ConvTranspose((4,4), 100=>512, stride=1, pad=1),
          BatchNorm(512),
          ReLU(),
          ConvTranspose((4,4), 512=>256, stride=2, pad=1),
          BatchNorm(256),
          ReLU(),
          ConvTranspose((4,4), 256=>128, stride=2, pad=1),
          BatchNorm(128),
          ReLU(),
          ConvTranspose((4,4), 128=>64, stride=2, pad=1),
          BatchNorm(64),
          ReLU(),
          ConvTranspose((4,4), 64=>3, stride=2, pad=1),
          BatchNorm(3),
          Tanh())
end

function discriminator()
    Chain(Conv((4,4), 3=>64, stride=2, pad=1),
          LeakyReLU(),
          Conv((4,4), 64=>128, stride=2, pad=1),
          BatchNorm(128),
          LeakyReLU(),
          Conv((4,4), 128=>256, stride=2, pad=1),
          BatchNorm(256),
          LeakyReLU(),
          Conv((4,4), 256=>512, stride=2, pad=1),
          BatchNorm(512),
          LeakyReLU(),
          Conv((4,4), 512=>1, stride=1, pad=1))
end

# FIXME Zygote currently does not differentiate this code, neither fill! nor
# ones https://github.com/FluxML/Zygote.jl/issues/150
function upsample(x)
    ratio = (2, 2, 1, 1)
    (h, w, c, n) = size(x)
    y = similar(x, (1, ratio[1], 1, ratio[2], 1, 1))
    # FIXME fill! does not work?
    # fill!(y, 1)
    y2 = ones(size(y))
    z = reshape(x, (h, 1, w, 1, c, n))  .* y2
    reshape(permutedims(z, (2,1,4,3,5,6)), size(x) .* ratio)
end

# BROKEN due to upsample undifferentiable
function generator_dcgan()
    Chain(Dense(100, 128 * 7 * 7),
          x -> reshape(x, 7, 7, 128, :),
          BatchNorm(128),
          upsample,
          Conv((3,3), 128=>128, stride=1, pad=1),
          BatchNorm(128),
          LeakyReLU(),
          upsample,
          Conv((3,3), 128=>64, stride=1, pad=1),
          BatchNorm(64),
          LeakyReLU(),
          Conv((3,3), 64=>1, stride=1, pad=1),
          Tanh())
end

function generator_dcgan_ct()
    Chain(Dense(100, 128 * 7 * 7),
          x -> reshape(x, 7, 7, 128, :),
          BatchNorm(128),
          # since upsampling does not work, I'm still going to use ConvT
          ConvTranspose((4,4), 128=>128, stride=2, pad=1),
          BatchNorm(128),
          LeakyReLU(),
          ConvTranspose((4,4), 128=>64, stride=2, pad=1),
          BatchNorm(64),
          LeakyReLU(),
          ConvTranspose((3,3), 64=>1, stride=1, pad=1),
          Tanh())
end

# from https://github.com/eriklindernoren/PyTorch-GAN dcgan/
function discriminator_dcgan()
    Chain(Conv((3,3), 1=>16, stride=2, pad=1),
          LeakyReLU(),
          Dropout(0.25),
          # NOTE: no dropout for the first conv
          Conv((3,3), 16=>32, stride=2, pad=1),
          LeakyReLU(),
          # CAUTION dropout turns out to be very important
          # NOTE: using both dropout and batchnorm
          Dropout(0.25),
          BatchNorm(32),
          Conv((3,3), 32=>64, stride=2, pad=1),
          LeakyReLU(),
          Dropout(0.25),
          BatchNorm(64),
          Conv((3,3), 64=>128, stride=2, pad=1),
          LeakyReLU(),
          Dropout(0.25),
          BatchNorm(128),
          Flatten(),
          Dense(128 * 2 * 2, 1))
end


# https://github.com/FluxML/model-zoo/pull/111
function generator_111()
    generator = Chain(
        Dense(100, 1024, leakyrelu),
        BatchNorm(1024),
        Dense(1024, 7 * 7 * 128, leakyrelu),
        BatchNorm(7 * 7 * 128),
        x->reshape(x, 7, 7, 128,:),
        ConvTranspose((4,4), 128=>64, relu; stride=(2,2), pad=(1,1)),
        BatchNorm(64),
        ConvTranspose((4,4), 64=>1, tanh; stride=(2,2), pad=(1,1)))
end

function discriminator_111()
    discriminator = Chain(
        Conv((3,3), 1=>32, leakyrelu;pad = 1),
        MaxPool((2,2)),
        Conv((3,3), 32=>64, leakyrelu;pad = 1),
        MaxPool((2,2)),
        x->reshape(x,7*7*64,:),
        Dense(7*7*64, 1024, leakyrelu),
        BatchNorm(1024),
        Dense(1024, 1))
end

# from https://www.tensorflow.org/tutorials/generative/dcgan
function generator_MNIST()
    Chain(Dense(100, 7*7*256),
          x -> reshape(x, 7, 7, 256, :),
          BatchNorm(256),
          LeakyReLU(),
          ConvTranspose((4,4), 256=>128, stride=2, pad=1),
          BatchNorm(128),
          LeakyReLU(),
          # FIXME this is not 14,14,64, but 13,13,64
          # ConvTranspose((5,5), 128=>64, stride=2, pad=2),
          ConvTranspose((4,4), 128=>64, stride=2, pad=1),
          BatchNorm(64),
          LeakyReLU(),
          ConvTranspose((3,3), 64=>1, stride=1, pad=1),
          Tanh())
end

function discriminator_MNIST()
    Chain(Conv((5,5), 1=>64, stride=2, pad=2),
          LeakyReLU(),
          Dropout(0.3),
          BatchNorm(64),
          Conv((5,5), 64=>128, stride=2, pad=2),
          LeakyReLU(),
          Dropout(0.3),
          BatchNorm(128),
          Flatten(),
          Dense(7*7*128, 1))
end

# FIXME use_bias = false
function denseG()
    Chain(Dense(100, 1024, relu),
          Dense(1024, 1024, relu),
          Dense(1024, 784, tanh),
          x -> reshape(x, 28, 28, 1, :))
end

# size((x -> reshape(x, 28, 28, 1, :))(randn(784, 3)))

function denseD()
    Chain(x -> reshape(x, 784, :),
          Dense(784, 256, leakyrelu),
          Dense(256, 256, leakyrelu),
          Dense(256, 1))
end

function test()
    g = denseG()
    d = denseD()

    g = generator()
    d = discriminator()

    g = generator_MNIST()
    d = discriminator_MNIST()
end


function graph_generator()
    # FIXME dim z
    Chain(Dense(100, 1024, relu),
          Dense(1024, 1024, relu),
          Dense(1024, 784, tanh),
          # FIXME number of nodes in adj matrix
          x -> reshape(x, 2, 2, :))
end

function graph_discriminator()
    Chain(x -> reshape(x, 2*2, :),
          Dense(4, 256, leakyrelu),
          Dense(256, 256, leakyrelu),
          Dense(256, 1))
end

# z: hidden noise dim
# d: graph node size
function graph_data_generator(z, d)
    Chain(Dense(z, 100, relu),
          Dense(100, 100, relu),
          Dense(100, d))
end

# d: graph node size
function graph_data_discriminator(d)
    Chain(Dense(d, 100, relu),
          Dense(100, 100, relu),
          Dense(100, 1))
end
