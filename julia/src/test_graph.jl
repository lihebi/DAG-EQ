using Statistics
using Dates: now

include("data_graph.jl")

# for notears exp
include("notears.jl")

# for supervised exp
include("model.jl")
include("train.jl")


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


function exp_sup(d; ng=10000, N=10)
    # TODO test on different type of graph

    ds, test_ds = gen_sup_ds_cached(ng=ng, N=N, d=d, batch_size=100)
    x, y = next_batch!(test_ds)

    # FIXME parameterize the model
    model = sup_model(d) |> gpu
    # TODO lr decay
    opt = ADAM(1e-4)

    expID = "d=$d-ng=$ng-N=$N"

    logger = TBLogger("tensorboard_logs/train-$expID-$(now())", tb_append, min_level=Logging.Info)
    test_logger = TBLogger("tensorboard_logs/test-$expID-$(now())", tb_append, min_level=Logging.Info)

    print_cb = Flux.throttle(sup_create_print_cb(logger), 1)
    test_cb = Flux.throttle(sup_create_test_cb(model, test_ds, "test_ds", logger=test_logger), 10)

    sup_train!(model, opt, ds, test_ds,
               print_cb=print_cb,
               test_cb=test_cb,
               train_steps=1e5)

    # TODO enforcing sparsity, and increase the loss weight of 1 edges, because
    # there are much more 0s, and they can take control of loss and make 1s not
    # significant. As an extreme case, the model may simply report 0 everywhere

    # do the inference
    # x, y = next_batch!(test_ds)
    # sup_view(model, x[:,8], y[:,8])
end

function test()
    exp_sup(3)
    exp_sup(5)
    exp_sup(7)
    exp_sup(10)
    exp_sup(10, ng=100000, N=1)
    exp_sup(10, ng=1000, N=100)
    exp_sup(20, ng=10000, N=2)
end

function sup_view(model, x, y)
    length(size(x)) == 1 || error("x must be one data")
    # visualize ground truth y
    d = convert(Int, sqrt(length(y)))
    g = DiGraph(reshape(y, d, d))
    display(g)

    out = cpu(model)(x)
    reshape(threshold(out, 0.3), d, d) |> DiGraph |> display
    nothing
end
