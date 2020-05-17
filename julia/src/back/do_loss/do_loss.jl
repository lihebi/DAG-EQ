using Distributions
using CausalInference
using LightGraphs

include("data_graph.jl")
include("notears.jl")
include("model.jl")
include("train.jl")


function random_intervention(X)
    index = rand(1:size(X, 1), size(X, 2))
    # this index to CartesianIndex
    index = map(zip(index, 1:size(x,2))) do p
        CartesianIndex(p...)
    end
    delta = zeros(size(X))
    # FIXME this is not parallel
    # map(zip(index, 1:size(X,2))) do p
    #     delta[p...] = randn()
    # end
    delta[index] .= randn(size(delta,2))
    # delta[index] .= randn(size(delta,2))
    index, delta
end

# 7. implement do-effect
function do_effect(w, X, index, delta)
    # FIXME get which index is mutated by delta
    X2 = copy(X)
    X2[index] .= delta[index]
    X2 = w * X2 + randn(size(X2))
    X2[index] .= delta[index]
    X2
end

# 8. implement interventional loss
function do_loss_fn(w, X, d)
    # FIXME this randomness should come from outside
    index, delta = random_intervention(X)

    # FIXME post do should set instead of +
    # post_do = X + delta
    post_do = copy(X)
    post_do[index] = delta[index]
    post_effect = do_effect(w, X, index, delta)

    ls_loss = ls_loss_fn(w, X)
    # FIXME d() returns a scalar
    # @show size(ls_loss)
    # @show size(d(post_effect))
    # FIXME mean or sum
    loss = ls_loss - mean(d(post_effect))
    loss
end

function test()

    gradient(()->do_loss_fn(w, x, cpu(D)))
    gradient(()->begin
             # FIXME Mutating arrays is not supported
             index, delta = random_intervention(x)
             ls_loss_fn(w, x)
             # sum(cpu(D)(x))
             end)
    # FIXME MethodError: no method matching getproperty(::NamedTuple{(:W, :b, :σ)
    # see https://github.com/FluxML/Zygote.jl/issues/111
    gradient(()->sum(cpu(D)(x)))


    gradient(()->ls_loss_fn(w, x))

    # do_loss_fn(w, x, cpu(D))
end


function ls_loss_fn(w, X)
    l = mean((X - w * X) .^ 2)

    # The L1 regularizer does not look right
    l1 = sum(abs.(w))
    l2 = sum(w .^ 2)

    # l + l2
    l
end

# TODO maybe test inforce the NOTEARS acyclic constraint in Flux
function train_w!(w, ds;
                  cb_fn=(m)->(), train_steps=ds.nbatch)
    # train w (the model) with X
    opt = ADAM(1e-4)
    ps = Flux.params(w)
    m = MeanMetric{Float64}()

    diag_index = Matrix(I, size(w)...)
    @showprogress 0.1 "Training.." for step in 1:train_steps
        X, _ = next_batch!(ds)
        gs = gradient(ps) do
            loss = ls_loss_fn(w, X)
            add!(m, loss)
            loss
        end
        Flux.Optimise.update!(opt, ps, gs)
        # FIXME I should add constraint that diagonal is always 0
        w[diag_index] .= 0
        # @show get!(m)
        cb_fn(m)
    end
end


function train_do!(w, ds, d)
    opt = ADAM(1e-4)
    ps = Flux.params(w)
    m = MeanMetric{Float64}()
    @showprogress 0.1 "Training.." for step in 1:100
        X, _ = next_batch!(ds)
        gs = gradient(ps) do
            # FIXME use GPU
            loss = do_loss_fn(w, X, cpu(d))
            add!(m, loss)
            loss
        end
        # Flux.Optimise.update!(opt, ps, gs)
    end
end

function test()
    train_do!(w, ds, D)
    # train_do!(w, ds, cpu(D))
    x
    gradient(()->do_loss_fn(w, x, cpu(D)))
    # gradient(()->1)
end



function test()
    # using a bivariate graph
    g = MetaDiGraph(2)
    add_edge!(g, 1, 2)
    W = gen_weights(g)
    # CAUTION: transpose to put batch to second dim
    X = gen_data(g, W, 10000)'
    # 3. generate model (a W)
    w = zeros(size(W))
    # 4. fit the model with likelihood
    # put X into dataset iterator
    ds = DataSetIterator(X, zeros(size(X)), 100)

    x, _ = next_batch!(ds)

    train_w!(w, ds,
             cb_fn=Flux.throttle((m)->@show(get!(m)), 1),
             train_steps=3e4)
    # test the model


    # 6. fit a GAN
    G = graph_data_generator(10, 2) |> gpu
    D = graph_data_discriminator(2) |> gpu
    gopt = ADAM(2e-4)
    dopt = ADAM(2e-4)
    # how about ds?
    # FIXME monitor training loss
    train_GAN!(G, D, gopt, dopt, ds, train_steps=4000)
    # test the generator
    noise = randn(10) |> gpu
    fake = G(noise)
    σ.(D(fake))

    # 9. optimize interventional loss
    train_do!(w, ds, D)
end


function test_setup()
    # 1. generate ground truth graph
    g = gen_ER_dag(5)
    is_cyclic(g) == false || error("not DAG")

    # using a bivariate graph
    g = MetaDiGraph(2)
    add_edge!(g, 1, 2)

    # 2. generate data
    @info "Generating weights .."
    W = gen_weights(g)

    @info "Generating 100 data points .."
    # more data is faster and performs better
    X = gen_data(g, W, 100)

    @info size(W)
    @info size(X)

    # 5. fit the model with NOTEARS
    # run notears
    @info "Optimizing .."
    Ŵ = notears(X)
    # compare the resulting W estimation

    # visualize gragh
    is_cyclic(DiGraph(threshold(Ŵ, 0.3)))
    @info "Results:"
    @show W
    @show sparse(threshold(Ŵ, 0.3))

    @info "Evaluating .."
    graphical_metrics(W, threshold(Ŵ, 0.3))
end
