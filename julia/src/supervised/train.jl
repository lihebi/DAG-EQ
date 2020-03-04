using ProgressMeter
using CuArrays: allowscalar
using Flux: @epochs, onecold
# ???
using Statistics: mean
using BSON: @save, @load

# FIXME dir structure?
include("../train_utils.jl")

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
    # d = convert(Int, sqrt(size(y,1)))
    d = size(y,1)
    nbatch = size(y, 3)
    # FIXME threshold value
    #
    # this should be 0.5 for binary sigmoid xent results
    mat(y) = threshold(reshape(y, d, d, :), 0.3, true)

    mout = mat(out)
    my = mat(y)
    nnz = sum(mout .!= 0) / nbatch
    nny = sum(my .!= 0) / nbatch

    # tp = sum(mout[mout .== my] .== 1)
    # fp = sum(mout[mout .== 1] .!= my[mout .== 1])
    tp = sum(mout[my .== 1] .== 1)
    fp = sum(mout[my .== 0] .== 1)
    tt = sum(my .== 1)
    ff = sum(my .== 0)

    mydiv(a,b) = if a == 0 0 else a / b end

    prec = mydiv(tp, tp+fp)
    recall = mydiv(tp, tt)

    tpr = mydiv(tp, tt)
    fpr = mydiv(fp, ff)
    fdr = mydiv(fp, sum(mout .== 1))

    shd = sum(my .!= mout) / nbatch

    GraphMetric(nnz, nny, tpr, fpr, fdr, shd, prec, recall)
end


function create_print_cb(;logger=nothing)
    function f(step, ms)
        # println()
        # evaluate to extract from metrics. This will only call every seconds
        values = map(ms) do x
            x[1]=>get!(x[2])
        end # |> Dict
        # DEBUG removing stdout data logging
        # @info "data" values
        if typeof(logger) <: TBLogger
            for value in values
                log_value(logger, value[1], value[2], step=step)
            end
        end
    end
end

function myσxent(logŷ, y)
    # FIXME performance the gradient of this cost a tons of time, and seems to
    # be moving data around
    #
    # Related:
    # https://github.com/JuliaGPU/CuArrays.jl/issues/611
    # https://github.com/JuliaGPU/CuArrays.jl/pull/602
    # https://github.com/JuliaGPU/CuArrays.jl/issues/141
    xent = Flux.logitbinarycrossentropy.(logŷ, y)
    loss = sum(xent)
    return loss * 1 // size(y, 2)
end

function create_test_cb(model, test_ds, msg; logger=nothing)
    function test_cb(step)
        test_run_steps = 20

        # println()
        # @info "testing for $test_run_steps steps .."
        gm = MeanMetric{GraphMetric}()
        loss_metric = MeanMetric{Float64}()

        # FIXME testmode!
        # @showprogress 0.1 "Inner testing..."
        for i in 1:test_run_steps
            x, y = next_batch!(test_ds) |> gpu
            out = model(x)

            loss = Flux.mse(out, y)
            # loss = Flux.mse(σ.(out), y)
            # loss = myσxent(out, y)

            # FIXME performance on CPU
            # metric = sup_graph_metrics(cpu(σ.(out)), cpu(y))
            metric = sup_graph_metrics(cpu(out), cpu(y))

            add!(gm, metric)
            add!(loss_metric, loss)
        end

        g_v = get!(gm)
        loss_v = get!(loss_metric)
        # @info msg loss_v to_named_tuple(g_v)...
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


function sup_train!(model, opt, ds, test_ds;
                    train_steps=ds.nbatch,
                    print_cb=(i, m)->(),
                    test_cb=(i)->())
    ps=Flux.params(model)

    # weight decay or all params decay?
    weights = weight_params(model)

    loss_metric = MeanMetric{Float64}()
    gm = MeanMetric{GraphMetric}()

    @info "training for $(train_steps) steps .."
    @showprogress 0.1 "Training..." for step in 1:train_steps
        # CAUTION this actually cost half of the computing time.
        x, y = next_batch!(ds) |> gpu

        # FIXME evaluating this too frequently might be inefficient
        #
        # FIXME and it turns out I cannot do this here, otherwise Zygote
        # complains. But now I have to run one more forward pass to get the
        # data.
        #
        # FIXME performance hell!
        # metric = sup_graph_metrics(cpu(σ.(model(x))), cpu(y))
        # add!(gm, metric)

        gs = gradient(ps) do
            out = model(x)
            # use cross entropy loss
            # loss_mse = Flux.mse(out, y)

            # loss = Flux.mse(σ.(out), y)
            loss = Flux.mse(out, y)
            # loss = myσxent(out, y)

            # add a weight decay
            # l2 = sum((x)->sum(x.^2), weights)
            # show l2?
            # loss = loss + 1e-5 * l2
            add!(loss_metric, loss)

            loss
        end

        Flux.Optimise.update!(opt, ps, gs)

        # "graph"=>gm
        print_cb(step, ["loss"=>loss_metric])

        test_cb(step)
    end
end
