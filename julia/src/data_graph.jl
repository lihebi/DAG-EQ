# CAUTION there are several problems about these two packages:
#
# 1. The Cairo and Fontconfig packages must be loaded before ImageMagick.
#
# 2. And I must import Cairo first, then Fontconfig.
#
# 3. Also import Cairo is needed for Compose to draw an image.
import Cairo
import Fontconfig

# FIXME It looks like ImageMagick is not useful.
using ImageMagick

using ProgressMeter

using LightGraphs
# using LightGraphs.SimpleGraphs
using MetaGraphs
using Base.Iterators

using Statistics: I
# cov, etc
using Statistics

using GraphPlot: gplot, circular_layout
using Compose: PNG, draw

using Random

using SparseArrays: sparse

import Base.show
import Base.display
using Distributions

using HDF5


include("display.jl")
include("data.jl")

"""Display a PNG by writing it to tmp file and show the filename. The filename
would be replaced by an Emacs plugin.

"""
function Base.display(d::EmacsDisplay, mime::MIME"image/png", x::AbstractGraph)
    # path, io = Base.Filesystem.mktemp()
    # path * ".png"
    path = tempname() * ".png"

    # Using GraphLayout plotting backend
    #
    # [random_layout, circular_layout, spring_layout, stressmajorize_layout, shell_layout, spectral_layout]
    # default: spring_layout, which is random
    draw(PNG(path), gplot(x,
                          layout=circular_layout,
                          # TODO pass optional label
                          nodelabel=1:nv(x),
                          NODELABELSIZE = 4.0 * 2,
                          # nodelabeldist=2,
                          # nodelabelc="darkred",
                          arrowlengthfrac = is_directed(x) ? 0.15 : 0.0))

    # loc_x, loc_y = layout_spring_adj(x)
    # draw_layout_adj(x, loc_x, loc_y, filename=path)

    println("$(path)")
    println("#<Image: $(path)>")
end

function display_graph_with_label(x::AbstractGraph, labels)
    path = tempname() * ".png"
    draw(PNG(path), gplot(x,
                          layout=circular_layout,
                          nodelabel=labels,
                          NODELABELSIZE = 4.0 * 1,
                          arrowlengthfrac = is_directed(x) ? 0.15 : 0.0))
    println("$(path)")
    println("#<Image: $(path)>")
end



##############################
## Generating DAG
##############################


function gen_ER(d=10)
    n = d
    m = d
    erdos_renyi(n, m, is_directed=true)
end

function gen_SF(d=10)
    n = d
    m = d
    # static_scale_free(n, m, 2)
    barabasi_albert(n, convert(Int, round(m/n)), is_directed=true)
end

function gen_ER_dag(d=10)
    gen_ER(d)  |> Graph |> random_orientation_dag |> MetaDiGraph
end

function gen_SF_dag(d=10)
    gen_SF(d)  |> Graph |> random_orientation_dag |> MetaDiGraph
end

function test()
    g = gen_ER()
    g = gen_SF()

    is_directed(g)
    is_cyclic(g)

    g2 = LightGraphs.random_orientation_dag(Graph(g))
    is_cyclic(g)
    is_cyclic(g2)

end

""" Get the weight adj matrix.
"""
function dag_W(dag::AbstractGraph{U}) where U
    # adjacency_matrix(g)
    n_v = nv(dag)
    nz = ne(dag)
    # colpt = ones(U, n_v + 1)

    W = zeros(n_v, n_v)

    # rowval = sizehint!(Vector{U}(), nz)
    for j in 1:n_v  # this is by column, not by row.
        dsts = sort(inneighbors(dag, j))
        for i in dsts
            W[i,j] = get_prop(dag, i, j, :weight)
        end
        # colpt[j + 1] = colpt[j] + length(dsts)
        # append!(rowval, dsts)
    end

    # FIXME why init to 1?
    # spmx = SparseMatrixCSC(n_v, n_v, colpt, rowval, ones(Int, nz))
    # return spmx
    return sparse(W)
end


""" Generate model parameters according to the DAG structure and causal mechanism

"""
function gen_weights(dag, zfunc=()->1)
    # dag is a meta graph, I'll need to attach parameters to each edge
    # 1. go through all edges
    # 2. attach edges
    # FIXME to ensure pureness, copy dag
    for e in edges(dag)
        # DEBUG trying different weights
        #
        # z = randn()
        # MOTE using uniform in [-2,-0.5] and [0.5, 2], better performance, make sense because it is easier
        # z = (rand() * 1.5 + 0.5) * rand([1,-1])
        # z = (rand() + 0.5) * rand([1,-1])
        # z = (rand() + 0.5)
        # z = 1 * rand([1,-1])
        # z = 0.5
        set_prop!(dag, e, :weight, zfunc())
    end
    W = dag_W(dag)
    for e in edges(dag)
        rem_prop!(dag, e, :weight)
    end
    W
end

poisson_d = nothing

function gen_data2(W, noise, n)
    d = size(W, 1)
    X = zeros(n, d)
    g = DiGraph(W)

    if noise == :Gaussian
        noise_fn = randn
    elseif noise == :Poisson
        global poisson_d
        if isnothing(poisson_d)
            # DEBUG this should only be called once
            @info "generating poisson distribution"
            # FIXME but this does not seem to make it faster
            poisson_d = Poisson(1)
        end
        noise_fn = n->rand(poisson_d,n)
    else
        error("Noise model $noise not supported.")
    end

    # topological sort
    # for vertices in order
    for v in topological_sort_by_dfs(g)
        parents = inneighbors(g, v)
        # FIXME
        X[:, v] = X[:, parents] * W[parents, v] + noise_fn(n)
    end
    X
end

# TODO more variables
function gen_sup_data(g, spec)
    d = nv(g)
    ds = map(1:spec.N) do i
        # DEBUG different weights
        # W = gen_weights(g)
        # W = gen_weights(g, ()->((rand() + 0.5) * rand([1,-1])))
        # W = gen_weights(g, ()->((rand() * 1.5 + 0.5) * rand([1,-1])))
        W = gen_weights(g, ()->((rand() * (0.5+spec.k) + 0.5) * rand([1,-1])))
        # W = gen_weights(g, ()->((rand() * 1.5 + 0.5)))

        X = gen_data2(W, spec.noise, 1000)
        # cor(X), W
        # DEBUG one-hot encoding
        # cor(X), Flux.onehotbatch(W_bin, [0,1])

        # FIXME I'm just recording whether this is causal, not the linear
        # direction
        W[W .> 0] .= 1
        W[W .< 0] .= 1
        cor(X), W
    end
    input = map(ds) do x x[1] end
    output = map(ds) do x x[2] end
    cat(input..., dims=3), cat(output..., dims=3)
end

function mycat(aoa)
    # array of array
    arr = similar(aoa[1], size(aoa[1])..., length(aoa))
    for i in 1:length(aoa)
        arr[:,:,:,i] = aoa[i]
    end
    # combine the last two dims FIXME the order unchanged?
    reshape(arr, size(arr)[1:end-2]..., :)
end

function gen_sup_data_with_graph(spec, gs)
    # train data
    ds = @showprogress 0.1 "Generating.." map(gs) do g
        x, y = gen_sup_data(g, spec)
    end
    input = map(ds) do x x[1] end
    output = map(ds) do x x[2] end

    # FIXME this splat is too much and cause stack overflow. But looks like I
    # have only the way of cat() ..
    #
    # cat(input..., dims=3), cat(output..., dims=3)

    # instead I'm going to pre-allocate an array and fill
    mycat(input), mycat(output)
    # This repo might be an alternative:
    # https://github.com/JuliaDiffEq/RecursiveArrayTools.jl
end

# catch the datasets to avoid generation
function gen_sup_data(spec)
    # generate graphs first
    graphs = gen_graphs_hard(spec)

    # ratio
    index = convert(Int, round(length(graphs) * 4 / 5))
    train_gs = graphs[1:index]
    test_gs = graphs[index:end]

    # TODO use different graphs for ds and test
    @info "generating training ds .."
    train_x, train_y = gen_sup_data_with_graph(spec, train_gs)
    @info  "generating testing ds .."
    test_x, test_y = gen_sup_data_with_graph(spec, test_gs)

    train_x, train_y, test_x, test_y
end

struct DataSpec
    d
    # weight range: [0.5, 0.5+k]
    # k: 1, 2, 4, 8
    k
    # gtype: :ER :SF
    gtype
    # noise: :Gaussian :Poisson
    noise

    ng
    N
end

function DataSpec(;d, k, gtype, noise, ng=10000, N=10)
    DataSpec(d, k, gtype, noise, ng, N)
end

function dataspec_to_id(spec)
    join(["d=$(spec.d)",
          "k=$(spec.k)",
          "gtype=$(spec.gtype)",
          "noise=$(spec.noise)"],
         "_")
end

function test()
    dataspec_to_id(DataSpec(d=10, k=1, gtype=:SF, noise=:Gaussian))
end

"""create data into file
"""
function create_sup_data(spec)
    fname = "data/" * dataspec_to_id(spec) * ".hdf5"
    if ispath(fname)
        @info "Data already exist: $fname"
        return
    end
    train_x, train_y, test_x, test_y = gen_sup_data(spec)
    h5open(fname, "w") do file
        write(file, "train_x", train_x)
        write(file, "train_y", train_y)
        write(file, "test_x", test_x)
        write(file, "test_y", test_y)
    end
    @info "Saved to $fname"
end

function load_sup_ds(spec, batch_size)
    fname = "data/" * dataspec_to_id(spec) * ".hdf5"
    # load into dataSET
    train_x = h5read(fname, "train_x")
    train_y = h5read(fname, "train_y")
    test_x = h5read(fname, "test_x")
    test_y = h5read(fname, "test_y")
    return (DataSetIterator(train_x, train_y, batch_size),
            DataSetIterator(test_x, test_y, batch_size))
end


function gen_graphs(spec)
    # generate n
    if spec.gtype == :ER
        f = gen_ER_dag
    elseif spec.gtype == :SF
        f = gen_SF_dag
    else
        error("Not supported graph type.")
    end
    all_graphs = map(1:spec.ng) do i
        f(spec.d)
    end
    # @show length(all_graphs)
    # filter
    unique_graphs = unique(all_graphs) do g
        # FIXME can this be compared directly?
        adjacency_matrix(g)
    end
    # @show length(unique_graphs)
    unique_graphs
end

function gen_graphs_hard(spec)
    # CAUTION this function will loop forever to generate enough unique graph
    all_gs = []
    @showprogress 0.1 "generating graph.." for i in 1:10
        # @info "graph iter $i .."
        gs = gen_graphs(spec)
        append!(all_gs, gs)
        all_gs = unique(all_gs) do g
            # FIXME can this be compared directly?
            adjacency_matrix(g)
        end

        if length(all_gs) >= spec.ng
            all_gs = all_gs[1:spec.ng]
            break
        end
    end
    spec.ng == length(all_gs) || @warn "Not enougth unique graphs, $(length(all_gs)) < $n"
    all_gs
end
