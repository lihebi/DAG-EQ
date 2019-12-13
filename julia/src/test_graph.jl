using Statistics
include("data_graph.jl")
include("notears.jl")

# TODO test a A->B model to see if NOTEARS works for equivalent class

function test_bi()
    g = MetaDiGraph(2)
    # add_edge!(g, 1, 2)
    add_edge!(g, 2, 1)
    # generate data
    @info "Generating weights .."
    W = gen_weights(g)
    @show Matrix(W)
    @info "Generating 100 data points .."
    X = gen_data(g, W, 100)
    # X = gen_data2(W, 100)
    # now learn
    @info "Optimizing .."
    Ŵ = notears(X)
    @info "Results:"
    @show Matrix(W)
    @show Matrix(Ŵ)
    @info "Evaluating .."
    graphical_metrics(W, threshold(Ŵ, 0.3))
    # graphical_metrics(W, Ŵ)
end

function test_bi_seed(W, X)
    Ŵ = notears(X)
    @info "Results:"
    @show Matrix(W)
    @show Matrix(Ŵ)
    @info "Evaluating .."
    graphical_metrics(W, threshold(Ŵ, 0.3))
end

function test_bibi()
    g = MetaDiGraph(2)
    add_edge!(g, 1, 2)
    # add_edge!(g, 2, 1)
    W = gen_weights(g)
    X = gen_data(g, W, 100)
    Ŵ = notears(X)
    Ŵ2 = notears(X, true)
    @info "Results:"
    @show Matrix(W)
    @show Matrix(Ŵ)
    @show Matrix(Ŵ2)
    # @info "Evaluating .."
    # graphical_metrics(W, threshold(Ŵ, 0.3))
    # evaluating loss_fn
    @show loss_fn(X, Ŵ)[1]
    @show loss_fn(X, Ŵ2)[1]
    nothing
end

function test()
    test_bi_seed(W, X)
    for _ in 1:20
        @info "=============================="
        test_bibi()
    end
    for _ in 1:50
        @info "=============================="
        test_bi()
    end
    for _ in 1:2
        @info "=============================="
        test_notears()
    end
end

function test_notears()
    # generate data
    # N = 100
    # d = 20
    # FIXME number of edges, not used
    # m = 20

    # FIXME try NOTEARS's data generation
    g = gen_ER_dag(20)
    is_cyclic(g) == false || error("not DAG")

    @info "Generating weights .."
    W = gen_weights(g)

    @info "Generating 100 data points .."
    # more data is faster and performs better
    X = gen_data(g, W, 100)
    # X = gen_data2(W, 100)

    @info size(W)
    @info size(X)

    # this is the data [x1,x2,...,xd]
    # X[:,1]
    X[1,:]

    # run notears
    @info "Optimizing .."
    Ŵ = notears(X)
    # compare the resulting W estimation

    # visualize gragh
    DiGraph(W)
    DiGraph(Ŵ)
    is_cyclic(DiGraph(Ŵ))
    @info "Results:"
    @show W
    @show sparse(threshold(Ŵ, 0.3))

    @info "Evaluating .."
    graphical_metrics(W, threshold(Ŵ, 0.3))
end

using CSV

function load_csv(fname)
    csv = CSV.read(fname, header=false)
    typeof(csv)
    size(csv)
    # csv[1:10]

    # notears(csv)
    # csv[!,1]
    convert(Matrix, csv)
end

function test_notears_csv()
    X = load_csv("/home/hebi/git/reading/notears/X.csv")
    W = load_csv("/home/hebi/git/reading/notears/W_true.csv")
    Ŵ = notears(X)

    DiGraph(W)
    DiGraph(Ŵ)
    is_cyclic(ge)
    nv(ge)
    ne(ge)

    function threshold(W, ε)
        res = copy(W)
        res[abs.(res) .< ε] .= 0
        res
    end

    size(threshold(Ŵ, 0.3))

    size(Ŵ[abs.(Ŵ) .> 0.3])
    size(Ŵ)

    graphical_metrics(W, Ŵ)
    graphical_metrics(W, threshold(Ŵ, 0.1))
    graphical_metrics(W, threshold(Ŵ, 0.15))
    graphical_metrics(W, threshold(Ŵ, 0.2))
    graphical_metrics(W, threshold(Ŵ, 0.3))
    graphical_metrics(W, threshold(Ŵ, 0.4))
    graphical_metrics(W, threshold(Ŵ, 0.5))
    graphical_metrics(W, threshold(Ŵ, 0.6))
    graphical_metrics(W, threshold(Ŵ, 0.7))
    graphical_metrics(W, threshold(Ŵ, 0.8))
    graphical_metrics(W, threshold(Ŵ, 0.9))
    graphical_metrics(W, threshold(Ŵ, 1))

end


# TODO time to test the model!!
function inf()
end

include("model.jl")
include("train.jl")

function test_sup()
    # TODO test weight=1
    # FIXME test on different instance of graph
    # TODO test on different type of graph

    # using dd in REPL to catch undefined d problems
    dd = 5
    ds, test_ds = gen_sup_ds(ng=10000, N=10, d=dd, batch_size=1000)
    x, y = next_batch!(test_ds)

    model = sup_model(dd) |> gpu
    # model(gpu(x))
    opt = ADAM(1e-4)
    print_cb = sup_create_print_cb()
    # test_cb = create_test_cb(model, test_ds)
    test_cb1 = Flux.throttle(sup_create_test_cb(model, ds, "ds"), 10)
    test_cb2 = Flux.throttle(sup_create_test_cb(model, test_ds, "test_ds"), 10)
    function test_cb()
        test_cb1()
        test_cb2()
    end
    Flux.@epochs 50 sup_train!(model, opt, ds, test_ds,
                               print_cb=print_cb,
                               test_cb=test_cb,
                               train_steps=ds.nbatch*100)

    # TODO enforcing sparsity, and increase the loss weight of 1 edges, because
    # there are much more 0s, and they can take control of loss and make 1s not
    # significant. As an extreme case, the model may simply report 0 everywhere
end
