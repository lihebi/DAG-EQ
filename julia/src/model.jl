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
