using ProgressMeter
using CuArrays: allowscalar

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


function train_GAN!(g, d, gopt, dopt, ds;
                    train_steps=ds.nbatch)
    ps_g=Flux.params(g)
    ps_d=Flux.params(d)

    loss_g_metric = MeanMetric()
    loss_d_metric = MeanMetric()

    @showprogress 0.1 "Training..." for step in 1:train_steps
        x, y = next_batch!(ds)
        x = gpu(x)
        y = gpu(y)

        # for calculating real/fake labels
        real_out = d(x)
        noise = gpu(Float32.(randn(100, size(x)[end])))
        real_label = gpu(Float32.(ones(size(real_out))))
        fake_label = gpu(Float32.(zeros(size(real_out))))

        # update D
        gs = gradient(ps_d) do
            real_out = d(x)
            l1 = bce(real_out, real_label)
            fake = g(noise)
            fake_out = d(fake)
            l2 = bce(fake_out, fake_label)

            loss = l1 + l2
            add!(loss_d_metric, loss)
            loss
        end
        Flux.Optimise.update!(dopt, ps_d, gs)

        # update G
        # FIXME using a new copy of noise
        noise = gpu(Float32.(randn(100, size(x)[end])))
        gs = gradient(ps_g) do
            fake = g(noise)
            fake_out = d(fake)
            loss = bce(fake_out, real_label)
            add!(loss_g_metric, loss)
            loss
        end
        Flux.Optimise.update!(gopt, ps_g, gs)

        # loss_D, loss_G, D(x), D(G(z))
        if step % 100 == 0
            println()
            loss_g = get!(loss_g_metric)
            loss_d = get!(loss_d_metric)
            @info "data" loss_g loss_d
            # also print out a sample
            noise = randn(100, 32) |> gpu;
            fake = g(noise);
            sample_and_view(fake)
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
