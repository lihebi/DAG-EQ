using Distributions
using CausalInference
using LightGraphs

using DelimitedFiles, LinearAlgebra

# mean
using Statistics

using Flux: @epochs

function test_CI()

    p = 0.01

    # Download data
    run(`wget http://nugget.unisa.edu.au/ParallelPC/data/real/NCI-60.csv`)

    # Read data and compute correlation matrix
    X = readdlm("NCI-60.csv", ',')
    d, n = size(X)
    C = Symmetric(cor(X, dims=2))

    # Compute skeleton `h` and separating sets `S`
    h, S = skeleton(d, gausscitest, (C, n), quantile(Normal(), 1-p/2))

    # Compute the CPDAG `g`
    g = pcalg(d, gausscitest, (C, n), quantile(Normal(), 1-p/2))

end


include("model.jl")
include("train.jl")

function test()
    # create models
    g = generator() |> gpu
    d = discriminator() |> gpu

    g = generator_MNIST() |> gpu
    d = discriminator_MNIST() |> gpu

    g = denseG() |> gpu
    d = denseD() |> gpu

    # DEBUG testing the model
    out = d(gpu(randn(28,28,1,32)));
    label = Float32.(zeros(size(out))) |> gpu;

    mean(Flux.binarycrossentropy.(out, label))
    bce(out, label)

    # this errors
    gradient(()->mean(Flux.binarycrossentropy.(out, label)))
    # scalar version works if allow scalar is true
    gradient(()->Flux.logitbinarycrossentropy(out[1], label[1]))
    # this works
    gradient(()->mean(Flux.logitbinarycrossentropy.(out, label)))
    # wrap above in a function
    gradient(()->bce(out, label))

    # sample untrained G
    noise = randn(100, 32) |> gpu;
    size(noise)
    # g = generator_MNIST() |> gpu
    fake = g(noise);
    size(fake)
    typeof(fake)
    # d(fake)
    sample_and_view(fake)
    # D dimension test
    # d = discriminator_MNIST() |> gpu
    size(d(fake))


    # Loading MNIST data
    ds, test_ds = load_MNIST_ds(batch_size=50)
    x, y = next_batch!(ds);

    # visualizing MNIST
    size(x)
    img = x[:,:,:,1];
    MyImage(img)
    sample_and_view(x)


    # Training GAN
    gopt = ADAM(1e-4)
    dopt = ADAM(1e-4)
    @epochs 5 train_GAN!(g, d, gopt, dopt, ds)

    # visualize the output
    noise = randn(100, 32) |> gpu;
    fake = g(noise);
    sample_and_view(fake)

    # random testing
    gradient(x -> 3x^2 + 2x + 1, 5)
    gradient(x -> 3x^2 + 2x + eps(0.1), 5)
    gradient(()->3)

    gs = gradient(model -> sum(model(noise)), g);
    Flux.Optimise.update!(gopt, gs)
    typeof(gs)
    length(gs)
    typeof(gs[1])
    length(gs[1][:layers])

    gs[1][:layers][1][:W]

    gs[:layers]
end

