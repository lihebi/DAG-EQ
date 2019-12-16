using ProgressMeter
using CuArrays: allowscalar
using Flux: @epochs, onecold
# mean
using Statistics

using Logging
using TensorBoardLogger



allowscalar(false)
# allowscalar(true)

# FIXME default to number
# MeanMetric() = MeanMetric{Number}()

# something like tf.keras.metrics.Mean
# a good reference https://github.com/apache/incubator-mxnet/blob/master/julia/src/metric.jl
mutable struct MeanMetric{T}
    # FIXME syntax for T
    sum::T
    n::Int
    # CAUTION for T, you must define Base.convert(::T, v::Int64)
    MeanMetric{T}() where {T} = new(0, 0)
end
function add!(m::MeanMetric{T}, v) where {T}
    m.sum += v
    m.n += 1
end
get(m::MeanMetric) = if m.n != 0 m.sum / m.n else m.sum / 1 end
function get!(m::MeanMetric)
    res = get(m)
    reset!(m)
    res
end
function reset!(m::MeanMetric{T}) where {T}
    m.sum = 0
    m.n = 0
end

struct GraphMetric
    # All float64, because when I average, the / operator, it will be float
    nnz::Float64
    nny::Float64
    tpr::Float64
    fpr::Float64
    fdr::Float64
    shd::Float64
    prec::Float64
    recall::Float64
    # GraphMetric() = new(0,0,0,0,0,0)
end

# FIXME this only custom @show and REPL, but does not affect logger like @info
# function Base.show(io::IO, ::MIME"text/plain", m::GraphMetric)
#     for fname in fieldnames(typeof(m))
#         println(io, "$fname = $(getfield(m, fname))")
#     end
# end

# from https://discourse.julialang.org/t/get-fieldnames-and-values-of-struct-as-namedtuple/8991
to_named_tuple(p) = (; (v=>getfield(p, v) for v in fieldnames(typeof(p)))...)

# I should have used "value type", but that seems to be less performant. I'm
# setting it to 0 for whatever v.
Base.convert(::Type{GraphMetric}, v::Int64) = GraphMetric(0,0,0,0,0,0,0,0)

import Base.+
function +(a::GraphMetric, b::GraphMetric)
    # FIXME can I automate this?
    GraphMetric(a.nnz + b.nnz,
                a.nny + b.nny,
                a.tpr + b.tpr,
                a.fpr + b.fpr,
                a.fdr + b.fdr,
                a.shd + b.shd,
                a.prec + b.prec,
                a.recall + b.recall)
end
import Base./
function /(a::GraphMetric, n::Int64)
    GraphMetric(a.nnz / n,
                a.nny / n,
                a.tpr / n,
                a.fpr / n,
                a.fdr / n,
                a.shd / n,
                a.prec / n,
                a.recall / n)
end

# FIXME overwrite TBLogger
import TensorBoardLogger.log_value
function log_value(logger::TBLogger, name::AbstractString, value::GraphMetric; step=nothing)
    # FIXME loop through all fields, and log value for each of them
    fn = propertynames(value)

    # for f=fn
    #     prop = getfield(value, f)
    #     log_value(logger, name*"/$f", prop, step=step)
    # end

    # FIXME it turns out the order is random, but at least the prefix works
    #
    # I want to control the order of the fields
    log_value(logger, name*"/tpr", value.tpr, step=step)
    log_value(logger, name*"/fpr", value.fpr, step=step)
    log_value(logger, name*"/fdr", value.fdr, step=step)
    log_value(logger, name*"/prec", value.prec, step=step)
    log_value(logger, name*"/recall", value.recall, step=step)
    # tfboard has good support to log scale something near 0
    log_value(logger, name*"/1-prec", 1-value.prec, step=step)
    log_value(logger, name*"/1-recall", 1-value.recall, step=step)
    # use a different prefix
    log_value(logger, name*"/v/shd", value.shd, step=step)
    # FIXME I want to plot nnz and nny in the same plot
    log_value(logger, name*"/v/nnz", value.nnz, step=step)
    log_value(logger, name*"/v/nny", value.nny, step=step)
end

function test()
    m = MeanMetric()

    m = MeanMetric{Float64}()
    add!(m, 1.3)
    get!(m)
    reset!(m)

    m = MeanMetric{GraphMetric}()
    add!(m, GraphMetric(1,1,1,1,3,1,0,0))
    get(m)
    reset!(m)
end

my_accuracy(ŷ, y) = mean(onecold(cpu(ŷ)) .== onecold(cpu(y)))

# https://github.com/soumith/ganhacks
function train_GAN!(g, d, gopt, dopt, ds;
                    train_steps=ds.nbatch)
    ps_g=Flux.params(g)
    ps_d=Flux.params(d)

    # FIXME default type?
    loss_g_metric = MeanMetric{Float64}()
    loss_d_metric = MeanMetric{Float64}()
    dx_metric = MeanMetric{Float64}()
    dgz1_metric = MeanMetric{Float64}()
    dgz2_metric = MeanMetric{Float64}()

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



function threshold(W, ε, one=false)
    res = copy(W)
    res[abs.(res) .< ε] .= 0
    # FIXME this .= is scalar operation, must be run on CPU
    # a = copy(gpu(randn(20)))
    # a[abs.(a) .< 0.2] .= 0.3
    #
    # threshold(gpu(randn(20)), 0.2)
    if one
        res[abs.(res) .>= ε] .= 1
    end
    res
end

# sup_graph_metrics([1 1 0 1], [1 0 0 1])

function sup_graph_metrics(out, y)
    # TODO try enforce some of this metrics in loss?
    d = convert(Int, sqrt(size(y,1)))
    # FIXME threshold value
    mat(y) = threshold(reshape(y, d, d, :), 0.3, true)
    mout = mat(out)
    my = mat(y)
    nnz = sum(mout .!= 0)
    nny = sum(my .!= 0)

    tp = sum(mout[mout .== my] .== 1)
    fp = sum(mout[mout .== 1] .!= my[mout .== 1])
    tt = sum(my .== 1)
    ff = sum(my .== 0)

    prec = tp / (tp + fp)
    recall = tp / tt

    tpr = tp / sum(my .== 1)

    fpr = fp / sum(my .== 0)
    # FIXME devide by 0, ???
    fdr = fp / sum(mout .== 1)

    shd = sum(my .!= mout)

    GraphMetric(nnz, nny, tpr, fpr, fdr, shd, prec, recall)
end

function sup_create_test_cb(model, test_ds, msg; logger=nothing)
    function test_cb(step)
        test_run_steps = 20

        println()
        @info "testing for $test_run_steps steps .."
        gm = MeanMetric{GraphMetric}()
        loss_metric = MeanMetric{Float64}()

        # FIXME testmode!
        @showprogress 0.1 "Inner testing..." for i in 1:test_run_steps
            x, y = next_batch!(test_ds) |> gpu
            out = model(x)

            # FIXME performance on CPU
            metric = sup_graph_metrics(cpu(out), cpu(y))
            loss = Flux.mse(out, y)

            add!(gm, metric)
            add!(loss_metric, loss)
        end

        g_v = get!(gm)
        loss_v = get!(loss_metric)
        @info msg loss_v to_named_tuple(g_v)...
        if typeof(logger) <: TBLogger
            # log_value(logger, "graphM/" * msg, value, step=step)
            # DEBUG using tuple
            log_value(logger, "loss", loss_v, step=step)
            log_value(logger, "graph", g_v, step=step)
            # log_value(logger, "graph", to_named_tuple(g_v), step=step)
        end
    end
    # execute every 10 seconds
    # Flux.throttle(test_cb, 10)
end

function sup_create_print_cb(logger=nothing)
    function f(step, ms)
        println()
        # evaluate to extract from metrics. This will only call every seconds
        values = map(ms) do x
            x[1]=>get!(x[2])
        end |> Dict
        # @info "data" values["loss"] to_named_tuple(values["graph"])...
        @info "data" values["loss"]
        if typeof(logger) <: TBLogger
            # FIXME hard coded data names
            log_value(logger, "loss", values["loss"], step=step)
            log_value(logger, "graph", values["graph"], step=step)
        end
    end
end

# for x in ("hel"=>1, "wo"=>2)
#     @show x[1]
# end


# TODO add tensorboard logger
# TODO better GPU utilization
function sup_train!(model, opt, ds, test_ds;
                    train_steps=ds.nbatch,
                    print_cb=(i, m)->(),
                    test_cb=(i)->())
    ps=Flux.params(model)
    loss_metric = MeanMetric{Float64}()
    gm = MeanMetric{GraphMetric}()

    @info "training for $(train_steps) steps .."
    @showprogress 0.1 "Training..." for step in 1:train_steps
        x, y = next_batch!(ds)
        x = gpu(x)
        y = gpu(y)

        # FIXME evaluating this too frequently might be inefficient
        #
        # FIXME and it turns out I cannot do this here, otherwise Zygote
        # complains. But now I have to run one more forward pass to get the
        # data.
        metric = sup_graph_metrics(cpu(model(x)), cpu(y))
        add!(gm, metric)

        gs = gradient(ps) do
            out = model(x)
            loss = Flux.mse(out, y)

            add!(loss_metric, loss)

            loss
        end

        Flux.Optimise.update!(opt, ps, gs)

        print_cb(step, ("loss"=>loss_metric, "graph"=>gm))

        test_cb(step)
    end
end
