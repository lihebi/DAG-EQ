include("data_graph.jl")
include("notears.jl")

function test_notears()
    # generate data
    # N = 100
    # d = 20
    # FIXME number of edges, not used
    # m = 20

    # FIXME try NOTEARS's data generation
    g = gen_ER_dag(20)
    is_cyclic(g) == false || error("not DAG")

    W = gen_weights(g)

    # more data is faster and performs better
    X = gen_data(g, W, 100)
    # X = gen_data2(W, 100)

    size(W)
    size(X)

    # this is the data [x1,x2,...,xd]
    # X[:,1]
    X[1,:]

    # run notears
    Ŵ = notears(X)
    # compare the resulting W estimation

    # visualize gragh
    DiGraph(W)
    DiGraph(Ŵ)
    is_cyclic(DiGraph(Ŵ))

    graphical_metrics(W, Ŵ)
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

    graphical_metrics(W, Ŵ)
end


