using Statistics
using Dates: now
using CSV

include("data_graph.jl")
include("notears.jl")
# for threshold for now
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

using Printf

function test_pairs(i)
    # test NOTEARS on pairs from http://webdav.tuebingen.mpg.de/cause-effect/
    # 1. read data (X and ground truth)
    s = @sprintf "%04d" i
    csv = CSV.read("data/pair$(s).txt", header=false)
    X = convert(Matrix, csv)
    # read meta data
    meta = CSV.read("data/pairmeta.txt", header=false)
    cause = meta[i,2]
    effect = meta[i,4]
    @show cause
    @show effect
    # 2. perform notears
    Ŵ = notears(X)
    # 3. compare results
    @show Ŵ
    @show sparse(threshold(Ŵ, 0.3))
    nothing
end

function test()
    # NOTEARS does not work on Tuebingen pairs
    for i in 1:30
        test_pairs(i)
    end
end

function test_eq()
    # let me directly compute the likelihood score
    g = MetaDiGraph(2)
    add_edge!(g, 1, 2)
    W = gen_weights(g, ()->0.2)
    X = gen_data(g, W, 100)

    g2 = MetaDiGraph(2)
    add_edge!(g2, 2, 1)
    W2 = gen_weights(g2, ()->0.2)
    X = gen_data(g, W2, 100)

    # compute score
    loss_fn(X, W)
    loss_fn(X, W2)
end

function likelihood_loss(z)
    g = MetaDiGraph(2)
    add_edge!(g, 1, 2)
    W = gen_weights(g, ()->z)
    X = gen_data(g, W, 100)

    g2 = MetaDiGraph(2)
    add_edge!(g2, 2, 1)
    W2 = gen_weights(g2, ()->z)


    @show sum(abs.(X - X * W))
    @show sum(abs.(X - X * W2))

    # compute score
    loss_fn(X, W)[1], loss_fn(X, W2)[1]
end

# using Plots
function test_plot_likelihood()
    # collect(0:0.1:2)
    ys = map(0:0.1:2) do z
        likelihood_loss(z)
    end
    p = plot(0:0.1:2, [map((y)->y[1], ys) map((y)->y[2], ys)])
    plot(0:0.1:2, map((y)->y[2], ys));
    y = map((y)->y[2], ys)
    x = collect(0:0.1:2)
    plot(rand(10))
    p = plot(x, y)
    savefig(p, "a.pdf")
end

function test_PC()
    p = 0.01

    g = MetaDiGraph(2)
    # add_edge!(g, 1, 2)
    add_edge!(g, 2, 1)
    # generate data
    W = gen_weights(g)
    X = gen_data(g, W, 100)

    # Read data and compute correlation matrix
    C = Symmetric(cor(X, dims=1))

    # Compute skeleton `h` and separating sets `S`
    h, S = skeleton(2, gausscitest, (C, 100), quantile(Normal(), 1-p/2))

    # Compute the CPDAG `g`
    g = pcalg(2, gausscitest, (C, 100), quantile(Normal(), 1-p/2))
    nv(g)
    g
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
    for _ in 1:10
        @info "=============================="
        @show test_notears()
    end
end

function test_notears_W(W)
    g = DiGraph(W) |> MetaDiGraph
    X = gen_data(g, W, 100)
    @info "Optimizing .."
    Ŵ = notears(X)
    # compare the resulting W estimation

    # visualize gragh
    DiGraph(W)
    DiGraph(Ŵ)
    is_cyclic(DiGraph(threshold(Ŵ, 0.3))) && error("result is cyclic")

    @info "Results:"
    @show sparse(W)
    @show sparse(threshold(Ŵ, 0.3))

    @info "Evaluating .."
    graphical_metrics(W, threshold(Ŵ, 0.3))
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
