using Statistics
using Dates: now

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
