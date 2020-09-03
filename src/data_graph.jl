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
using Match
using RecursiveArrayTools


using BSON: @save, @load

# export gen_graphs_hard, DataSpec, myplot, dataspec_to_id

include("display.jl")
include("data.jl")


function myplot(g)
#     gplot(g, nodelabel=1:nv(g), layout=circular_layout)
    gplot(g,
          layout=circular_layout,
          # TODO pass optional label
          nodelabel=1:nv(g),
          NODELABELSIZE = 4.0 * 2,
          # nodelabeldist=2,
          # nodelabelc="darkred",
          arrowlengthfrac = is_directed(g) ? 0.15 : 0.0)
end

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


function gen_ER(d, e)
    erdos_renyi(d, e, is_directed=true)
end

function gen_SF(d, e)
    # static_scale_free(n, m, 2)
    barabasi_albert(d, convert(Int, round(e/d)), is_directed=true)
end

function gen_ER_dag(d, e=d)
    gen_ER(d, e)  |> Graph |> random_orientation_dag |> MetaDiGraph
end

function gen_SF_dag(d, e=d)
    gen_SF(d, e)  |> Graph |> random_orientation_dag |> MetaDiGraph
end

function test()
    g = gen_ER(10, 10)
    g = gen_SF(10, 10)

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
        set_prop!(dag, e, :weight, zfunc())
    end
    W = dag_W(dag)
    for e in edges(dag)
        rem_prop!(dag, e, :weight)
    end
    W
end

function gen_data2(W, noise, n)
    d = size(W, 1)
    X = zeros(n, d)
    g = DiGraph(W)

    if noise == :Gaussian
        noise_fn = randn
    elseif noise == :Poisson
        # FIXME fixed hyper-parameter \lambda
        d = Poisson(1)
        noise_fn = n->rand(d, n)
    elseif noise == :Exp
        d = Exponential(1)
        noise_fn = n->rand(d, n)
    elseif noise == :Gumbel
        d = Gumbel(0, 1)
        noise_fn = n->rand(d, n)
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

function my_uniform_init(din, dout)
    (rand(din, dout) .* 1.5 .+ 0.5) .* rand([1,-1], din, dout)
end

function test()
    # maybe this MLP weights are too small
    Dense(10, 10, initW=my_uniform_init).W
end

function gen_MLP(g, noise, n)
    adj = adjacency_matrix(g)
    d = size(adj, 1)
    # CAUTION MLP requires channel last
    X = zeros(d, n)

    dist = @match noise begin
        :Gaussian => Normal(0,1)
        # FIXME fixed hyper-parameter \lambda
        :Poisson => Poisson(1)
        :Exp => Exponential(1)
        :Gumbel => Gumbel(0, 1)
        _ => error("Noise model $noise not supported.")
    end
    noise_fn = n->rand(dist, n)

    # topological sort
    # for vertices in order
    # 1. get causal order
    for v in topological_sort_by_dfs(g)
        # 2. for causal order nodes
        #    2.1 get parents
        parents = inneighbors(g, v)
        if length(parents) == 0
            X[v, :] = noise_fn(n)
        else
            # 2.2 construct a random Dense Layer, with sigmoid activation. The
            # weights should be in some range, say [0.5,2], and TODO this might
            # be configurable by the spec.k as well
            mlp = Chain(Dense(length(parents), 10, σ,
                              # FIXME maybe this default is better
                              initW = my_uniform_init),
                        Dense(10, 1),
                        x->dropdims(x, dims=1))
            # 3.3 get result by applying the MLP
            X[v, :] = mlp(X[parents, :]) .+ noise_fn(n)
        end
    end
    # CAUTION and this is transposed to the final result
    X'
end

function test()
    # construct a G
    g = gen_ER_dag(13)
    # from g to binary adj matrix
    adj = adjacency_matrix(g)
    # genereate linear X
    W = gen_weights(g, ()->((rand() * 1.5 + 0.5) * rand([1,-1])))
    # generate linear X
    X = gen_data2(W, :Gaussian, 1000)
    # generate quadratic X
    size(gen_MLP(g, :Gaussian, 100))
end


# TODO more variables
function gen_sup_data_internal(g, spec)
    d = nv(g)
    ds = map(1:spec.N) do i
        # DEBUG different weights
        # W = gen_weights(g)
        # W = gen_weights(g, ()->((rand() + 0.5) * rand([1,-1])))
        # W = gen_weights(g, ()->((rand() * 1.5 + 0.5) * rand([1,-1])))
        # W = gen_weights(g, ()->((rand() * (0.5+spec.k) + 0.5) * rand([1,-1])))
        # W = gen_weights(g, ()->((rand() * 1.5 + 0.5)))

        X = @match spec.mechanism begin
            :Linear => begin
                W = gen_weights(g, ()->((rand() * (0.5+spec.k) + 0.5) * rand([1,-1])))
                # FIXME the number of data points generated
                gen_data2(W, spec.noise, 1000)
            end
            :MLP =>
                # FIXME performance
                gen_MLP(g, spec.noise, 1000)
            _ => error("Unsupported mechanism.")
        end

        # cor(X), W
        # DEBUG one-hot encoding
        # cor(X), Flux.onehotbatch(W_bin, [0,1])

        # FIXME I should record both COR and COV
        # or, maybe I should just record W. g can be recovered from W, and
        @match spec.mat begin
            # DEBUG using adjacency_matrix(g) instead of W
            :COR => (cor(X), adjacency_matrix(g))
            :COV => (cov(X), adjacency_matrix(g))
            _ => error("Unsupported matrix type")
        end
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

function gen_sup_data_with_graphs(spec, gs)
    # train data
    ds = @showprogress 0.1 "Generating.." map(gs) do g
        x, y = gen_sup_data_internal(g, spec)
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

function gen_raw_data_with_graphs(spec, gs)
    ds = @showprogress 0.1 "Generating raw XY .." map(gs) do g
        d = nv(g)
        W = gen_weights(g, ()->((rand() * (0.5+spec.k) + 0.5) * rand([1,-1])))
        X = gen_data2(W, spec.noise, 1000)
        W[W .> 0] .= 1
        W[W .< 0] .= 1
        X, W
    end
    input = map(ds) do x x[1] end
    output = map(ds) do x x[2] end
    # mycat(input), mycat(output)
    # input, output
    # FIXME mycat is not robust enough for this
    cat(input..., dims=3), cat(output..., dims=3)
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

    # :COV or :COR
    mat
    # :Linear, :GP, :Quad, :MLP
    mechanism

    # I'll be training on different d, k, gtype, mat, mechanism
    #
    # for noise, I'll be only testing on different noise

    ng
    N
    bsize
    seed
end

# FIXME previous 10000, 10
function DataSpec(;d, k, gtype, noise, mat=:COR, mechanism=:Linear, ng=3000, N=3, seed=1234)
    # FIXME maybe check error here
    #
    # UPDATE set ng and N based on d
    if d <= 20
        ng = 3000
        N = 3
        bsize=128
    elseif d <= 40
        ng = 2000
        N = 2
        bsize=64
    elseif d <= 80
        # FIXME this might be too small. But should be more than enough if we
        # just use it as testing data
        ng = 1000
        N = 1
        bsize=32
    elseif d <= 150
        ng = 1000
        N = 1
        # DEBUG testing memory limit during training (pullback of EQ layer seems
        # to consume lots of memory)
        bsize = 16
    elseif d <= 200
        ng = 500
        N = 1
        bsize = 8
    elseif d <= 300
        ng = 300
        N = 1
        bsize = 4
    elseif d <= 400
        ng = 300
        N = 1
        bsize = 2
    end
    DataSpec(d, k, gtype, noise, mat, mechanism, ng, N, bsize, seed)
end

function dataspec_to_id(spec::DataSpec)
    a = join(["d=$(spec.d)",
              "k=$(spec.k)",
              "gtype=$(spec.gtype)",
              "noise=$(spec.noise)",
              "mat=$(spec.mat)",
            "mec=$(spec.mechanism)"
              ],
             "_")
end

function dataspec_to_id(specs::Array{DataSpec, N} where N)
    # "[" * join(dataspec_to_id.(specs), "+") * "]"

    # FIXME however, for ensemble model, I actually just want a single model
    "ensemble"
end

function test()
    # this will always connect to the last dimension
    convert(Array, VectorOfArray([randn(10,5,3), randn(10,5,3)]))
end

function gs2hdf5mat(graphs)
    # from array of graphs to matrix suitable for saving as hdf5
    mats = Matrix.(adjacency_matrix.(graphs))
    # this seems to be slow
    @info "concating .."
    # onemat = cat(mats..., dims=3)
    onemat = convert(Array, VectorOfArray(mats))
    @info "done"
    onemat
end

function hdf5mat2gs(onemat)
    # and transfer back
    MetaDiGraph.([onemat[:,:,i] for i in 1:size(onemat, 3)])
end

function test()
    graphs = gen_graphs_hard(DataSpec(d=10, k=1, gtype=:ER,
                                      noise=:Gaussian))[1:11]
    recs = hdf5mat2gs(gs2hdf5mat(graphs))
    for (g,r) in zip(graphs, recs)
        g == r || error("Error")
    end
end

function tmp_convert_bson_hdf5()
    # loop through all the g.bson, and generate a g.hdf5
    # walk through the file system
    for dir in readdir("data")
        fname = joinpath("data", dir, "g.bson")
        fnew = joinpath(dirname(fname), "g.hdf5")
        if isfile(fname) && !isfile(fnew)
            @info "converting $fname to $fnew"
            @load fname train_gs test_gs
            h5open(fnew, "w") do file
                write(file, "train_gs", gs2hdf5mat(train_gs))
                write(file, "test_gs", gs2hdf5mat(test_gs))
            end
        end
    end
end

function load_sup_ds(spec, batch_size=100; use_raw=false)
    # create "data/" folder is not already there
    if !isdir("data") mkdir("data") end
    # 1. generate graph
    gdir = "data/$(spec.gtype)-$(spec.d)-$(spec.seed)"
    if !isdir(gdir) mkdir(gdir) end
    gfile = "$gdir/g.hdf5"
    if isfile(gfile)
        # DEBUG loading
        # FIXME I don't need to load the graphs
        train_gs = hdf5mat2gs(h5read(gfile, "train_gs"))
        test_gs = hdf5mat2gs(h5read(gfile, "test_gs"))
    else
        @info "Generating graphs for " spec
        # generate graphs first
        # CAUTION this seed! might overwrite existing seeding for training and experiments.
        # I probably want to generate graphs before-hand?
        Random.seed!(spec.seed)
        graphs = gen_graphs_hard(spec)

        # ratio
        index = convert(Int, round(length(graphs) * 4 / 5))
        train_gs = graphs[1:index]
        test_gs = graphs[index:end]
        # DEBUG writing
        h5open(gfile, "w") do file
            write(file, "train_gs", gs2hdf5mat(train_gs))
            write(file, "test_gs", gs2hdf5mat(test_gs))
        end
    end

    fname = "$gdir/$(dataspec_to_id(spec)).hdf5"
    if ispath(fname)
        train_x = h5read(fname, "train_x")
        train_y = h5read(fname, "train_y")
        test_x = h5read(fname, "test_x")
        test_y = h5read(fname, "test_y")
        raw_x = h5read(fname, "raw_x")
        raw_y = h5read(fname, "raw_y")
    else
        # generate according to train_gs and test_gs
        # TODO use different graphs for ds and test
        @info "generating training ds .."
        train_x, train_y = gen_sup_data_with_graphs(spec, train_gs)
        @info  "generating testing ds .."
        test_x, test_y = gen_sup_data_with_graphs(spec, test_gs)

        # generate raw data for use with other methods
        #
        # FIXME the raw_x should not be just for test graphs. When training
        # NCC/RCC, I need them as training data
        raw_x, raw_y = gen_raw_data_with_graphs(spec, test_gs)

        h5open(fname, "w") do file
            write(file, "train_x", train_x)
            write(file, "train_y", train_y)
            write(file, "test_x", test_x)
            write(file, "test_y", test_y)
            write(file, "raw_x", raw_x)
            write(file, "raw_y", raw_y)
        end
        @info "Saved to $fname"
    end
    if use_raw
        return DataSetIterator(raw_x, raw_y, batch_size)
    else
        return (DataSetIterator(train_x, train_y, batch_size),
                DataSetIterator(test_x, test_y, batch_size))
    end
end

function gen_graphs(spec)
    # generate n
    f = @match spec.gtype begin
        :ER => gen_ER_dag
        :SF => gen_SF_dag
        :ER2 => (d)->gen_ER_dag(d, 2*d)
        :ER4 => (d)->gen_ER_dag(d, 4*d)
        :SF2 => (d)->gen_SF_dag(d, 2*d)
        :SF4 => (d)->gen_SF_dag(d, 4*d)
        :Bern => error("Not implemented")
        _ => error("Not supported graph type.")
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
