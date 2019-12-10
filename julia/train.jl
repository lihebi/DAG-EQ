using ProgressMeter
using CuArrays: allowscalar
using Flux: @epochs, onecold
# mean
using Statistics


allowscalar(false)
# allowscalar(true)

# something like tf.keras.metrics.Mean
# a good reference https://github.com/apache/incubator-mxnet/blob/master/julia/src/metric.jl
mutable struct MeanMetric
    sum::Float64
    n::Int
    MeanMetric() = new(0.0, 0)
end
function add!(m::MeanMetric, v)
    m.sum += v
    m.n += 1
end
get(m::MeanMetric) = m.sum / m.n
function get!(m::MeanMetric)
    res = m.sum / m.n
    reset!(m)
    res
end
function reset!(m::MeanMetric)
    m.sum = 0.0
    m.n = 0
end

my_accuracy(ŷ, y) = mean(onecold(cpu(ŷ)) .== onecold(cpu(y)))

# https://github.com/soumith/ganhacks
function train_GAN!(g, d, gopt, dopt, ds;
                    train_steps=ds.nbatch)
    ps_g=Flux.params(g)
    ps_d=Flux.params(d)

    loss_g_metric = MeanMetric()
    loss_d_metric = MeanMetric()
    dx_metric = MeanMetric()
    dgz1_metric = MeanMetric()
    dgz2_metric = MeanMetric()

    @showprogress 0.1 "Training..." for step in 1:train_steps
        x, y = next_batch!(ds)
        x = gpu(x)
        y = gpu(y)

        # for calculating real/fake labels
        out = d(x)
        noise = gpu(Float32.(randn(100, size(x)[end])))
        real_label = gpu(Float32.(ones(size(out))))
        fake_label = gpu(Float32.(zeros(size(out))))

        # update D
        gs = gradient(ps_d) do
            real_out = d(x)
            l1 = bce(real_out, real_label)

            add!(dx_metric, mean(σ.(real_out)))

            fake = g(noise)
            fake_out = d(fake)
            l2 = bce(fake_out, fake_label)

            add!(dgz1_metric, mean(σ.(fake_out)))

            loss = l1 + l2
            add!(loss_d_metric, loss)

            loss
        end
        Flux.Optimise.update!(dopt, ps_d, gs)

        # update G
        #
        # FIXME NOT using a new copy of noise
        # noise = gpu(Float32.(randn(100, size(x)[end])))
        gs = gradient(ps_g) do
            # FIXME I have to regenerate the fake, otherwise the gradient
            # calculation by Zygote will not consider it?
            fake = g(noise)
            fake_out = d(fake)
            loss = bce(fake_out, real_label)

            add!(dgz2_metric, mean(σ.(fake_out)))
            add!(loss_g_metric, loss)

            loss
        end
        Flux.Optimise.update!(gopt, ps_g, gs)

        # loss_D, loss_G, D(x), D(G(z))
        if step % 200 == 0
            println()
            @info("data",
                  get!(loss_g_metric),
                  get!(loss_d_metric),
                  get!(dx_metric),
                  get!(dgz1_metric),
                  get!(dgz2_metric))
            # also print out a sample
            # noise = randn(100, 32) |> gpu;
            fake = g(noise);
            sample_and_view(fake)
            # I want to print out the real and fake accuracy by D
            @show my_accuracy(d(x), real_label)
            @show my_accuracy(d(g(noise)), fake_label)
        end
    end
end

function bce(ŷ, y)
    # binarycrossentropy(ŷ, y; ϵ=eps(ŷ)) = -y*log(ŷ + ϵ) - (1 - y)*log(1 - ŷ + ϵ)
    # FIXME use mean(Flux.binarycrossentropy.(out, label))
    # mean(-y.*log.(ŷ) - (1  .- y .+ 1f-10).*log.(1 .- ŷ .+ 1f-10))
    #
    # FIXME now I have a reason to use logit version: the non-logit version
    # throw errors in calculating gradient
    mean(Flux.logitbinarycrossentropy.(ŷ, y))
end
