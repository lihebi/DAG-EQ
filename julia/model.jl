using Flux

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

using Flux: expand

struct SamePad end

# from https://github.com/FluxML/Flux.jl/pull/901
calc_padding(pad, k::NTuple{N,T}, dilation, stride) where {T,N}= expand(Val(2*N), pad)
function calc_padding(::SamePad, k::NTuple{N,T}, dilation, stride) where {N,T}
    #Formula from Relationship 14 in
    #http://deeplearning.net/software/theano_versions/dev/tutorial/conv_arithmetic.html

    # Effective kernel size, including dilation
    k_eff = @. k + (k - 1) * (dilation - 1)
    # How much total padding needs to be applied?
    pad_amt = @. k_eff - 1
    # In case amount of padding is odd we need to apply different amounts to each side.
    return Tuple(mapfoldl(i -> [ceil(Int, i/2), i ÷ 2], vcat, pad_amt))
end

function Conv(w::AbstractArray{T,N}, b::AbstractVector{T}, σ = identity;
              stride = 1, pad = 0, dilation = 1) where {T,N}
  stride = expand(Val(N-2), stride)
  # pad = expand(Val(2*(N-2)), pad)
  dilation = expand(Val(N-2), dilation)
  pad = calc_padding(pad, size(w)[1:N-2], dilation, stride)
  return Conv(σ, w, b, stride, pad, dilation)
end

function ConvTranspose(w::AbstractArray{T,N}, b::AbstractVector{T}, σ = identity;
              stride = 1, pad = 0, dilation = 1) where {T,N}
  stride = expand(Val(N-2), stride)
  # pad = expand(Val(2*(N-2)), pad)
  dilation = expand(Val(N-2), dilation)
  pad = calc_padding(pad, size(w)[1:N-2], dilation, stride)
  return ConvTranspose(σ, w, b, stride, pad, dilation)
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

# from https://www.tensorflow.org/tutorials/generative/dcgan
function generator_MNIST()
    Chain(Dense(100, 7*7*256),
          x -> reshape(x, 7, 7, 256, :),
          BatchNorm(256),
          LeakyReLU(),
          ConvTranspose((5,5), 256=>128, stride=1, pad=SamePad()),
          BatchNorm(128),
          LeakyReLU(),
          # FIXME this is not 14,14,64, but 13,13,64
          # ConvTranspose((5,5), 128=>64, stride=2, pad=SamePad()),
          ConvTranspose((4,4), 128=>64, stride=2, pad=1),
          BatchNorm(64),
          LeakyReLU(),
          ConvTranspose((4,4), 64=>1, stride=2, pad=1),
          Tanh())
end

function discriminator_MNIST()
    Chain(Conv((5,5), 1=>64, stride=2, pad=SamePad()),
          LeakyReLU(),
          Dropout(0.3),
          Conv((5,5), 64=>128, stride=2, pad=SamePad()),
          LeakyReLU(),
          Dropout(0.3),
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
