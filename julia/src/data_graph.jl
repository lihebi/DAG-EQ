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

using GraphPlot: gplot, circular_layout
using Compose: PNG, draw

using Random

using SparseArrays: sparse

import Base.show
import Base.display

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



##############################
## Generating DAG
##############################


# (HEBI: FIXME NOW no <->)
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

""" TODO ensure W and dag are consistent
"""
function is_consistent(dag, W)
    false
end


""" Generate data according to a model. Assuming dag has :weight parameters.

"""
function gen_data_old(dag, mechanism="default")
    # 1. find roots
    roots = []
    for v in vertices(dag)
        parents = inneighbors(dag, v)
        if length(parents) > 0
            push!(roots, v)
        end
    end
    # 2. generate randn for roots
    # 3. remove roots, find new roots
    # 4. generate randn + parents
end

function gen_data!(dag, W, N)
    # v = select_vertex(dag)
    for v in vertices(dag)
        gen_data!(dag, W, N, v)
    end
end

function gen_data!(dag, W, N, v)
    # start from a random node that does not have data, i.e. 1
    if ! has_prop(dag, v, :data)
        parents = inneighbors(dag, v)
        # if it has parents, run gen_data for its parents
        if length(parents) > 0
            for p in parents
                gen_data!(dag, W, N, p)
            end
        end
        # gen data for itself
        if length(parents) > 0
            pdata = map(parents) do p
                get_prop(dag, p, :data)
            end
            # FIXME ensure correctness (not zero)
            weights = map(parents) do p
                # get_prop(dag, p, v, :weight)
                W[p,v]
            end
            # causal mechanism
            data = hcat(pdata...) * weights + randn(N)
            set_prop!(dag, v, :data, data)
        else
            # generate 20 samples
            set_prop!(dag, v, :data, randn(N))
        end
    end
end

function dag_data(dag)
    d = nv(dag)
    N = length(get_prop(dag, 1, :data))
    # FIXME column or row major?
    X = []
    for i in 1:d
        xi = get_prop(dag, i, :data)
        push!(X, xi)
    end
    # FIXME get X into a matrix
    # transpose(hcat(X...))
    hcat(X...)
end

function gen_data(dag, W, N)
    d = nv(dag)
    # 2. gen_data!
    gen_data!(dag, W, N)
    # 3. get_data
    X = dag_data(dag)
    # 4. remove data from dag
    for i in 1:d
        rem_prop!(dag, i, :data)
    end
    X
end

function gen_data2(W, n)
    d = size(W, 1)
    X = zeros(n, d)
    g = DiGraph(W)
    # topological sort
    # for vertices in order
    for v in topological_sort_by_dfs(g)
        parents = inneighbors(g, v)
        # FIXME
        X[:, v] = X[:, parents] * W[parents, v] + randn(n)
    end
    X
end

function print_weights(dag)
    # for e in edges(dag)
    #     @show props(dag, e)
    # end
    for e in edges(dag)
        @show get_prop(dag, e, :weight)
    end
end

function print_data(dag)
    for v in vertices(dag)
        # if has_prop(dag, v, :data)
        #     @show v, get_prop(dag, v, :data)
        # end
        @show get_prop(dag, v, :data)
    end
end

""" Intervene on a node, the do-notation semantic.

FIXME are we setting v to a fixed value, or a distribution of values?
TODO multiple interventions
TODO hard and soft interventions

1. I should not set it to a fixed value. If setting to a fixed value, which
value are we talking about? If we select a predefined value, the mixture
distribution for this variable will have a fixed value. In calculating
interventional loss, we have no clue how to set the same intervention, thus the
interventional loss would be hard to fit.

UPDATE: I can probably still use this, and use a random value for the fixed value.

2. Instead, I should probably use a distribution, or simply just N(0,1), and
this can be simulated in computing interventional loss easily. This provides two
signals to the interventional loss:
  - the parents of the node has been cut
  - the descendants, in case of arrows

3. I can also use soft intervention, where instead of removing old mechanism, we
add a new W*randn(). This signal should be weaker, because we lose the signal of
parents.

"""
function intervene!(dag, v)
    # FIXME should we have noise for the fixed v value?
    # FIXME should we generate new noise variable?
    # FIXME should I test if v has parents or not?

    if length(inneighbors(dag, v)) == 0
        # FIXME should I avoid such interventions?
        @warn "The node $(v) has no parents, the intervention does not make sense."
    end

    # 1. get data for v, set it
    N = length(get_prop(dag, v, :data))
    set_prop!(dag, v, :data, randn(N))

    # 2. for all descendants of v, recompute the value
    des = descendants(dag, v)
    @info "Recomputing for $(length(des)) nodes .."
    # clear
    for d in des
        rem_prop!(dag, d, :data)
    end
    # recompute
    # FIXME should we use the same noise? This should not matter.
    for d in des
        gen_data!(dag, d, N)
    end
end
""" Intervene a random node.

"""
function intervene!(dag)
    l = collect(vertices(dag))
    v = l[rand(1:length(l))]
    @info "intervening on node $v .."
    intervene!(dag, v)
end

""" Return a list of descendant vertex of v
"""
function descendants(dag, v)
    g = bfs_tree(dag, v)
    # HACK [] will have type Array{Any}, while [v] will have Array{Int64}
    ret = [v]
    for e in edges(g)
        push!(ret, e.src)
        push!(ret, e.dst)
    end
    # ret |> unique |> (r)->filter((x)->x!=v, r) |> sort
    setdiff(Set(ret), [v]) |> collect |> sort
end

function test()
    g = gen_ER() |> Graph |> random_orientation_dag |> MetaDiGraph
    is_cyclic(g)
    g

    set_weights!(g)
    print_weights(g)

    gen_data!(g)
    gen_data!(g, overwrite=true)
    gen_data!(g, overwrite=true, N=5)
    print_data(g)

    for i in 1:10
        intervene!(g)
    end

    intervene!(g, 2)


    for v in vertices(g)
        @show v
        @show ! has_prop(g, v, :data)
    end

    bfs_tree(g, 1)
    bfs_tree(g, 2)
    bfs_tree(g, 3)
    bfs_parents(g, 2)

    descendants(g, 1)
    descendants(g, 2)
    descendants(g, 3)
    descendants(g, 4)
end


function myeye(n)
    1 * Matrix(I, n, n)
end

# TODO more variables
function gen_sup_data(g, N)
    d = nv(g)
    ds = map(1:N) do i
        # DEBUG different weights
        # W = gen_weights(g)
        # W = gen_weights(g, ()->((rand() + 0.5) * rand([1,-1])))
        W = gen_weights(g, ()->((rand() * 1.5 + 0.5) * rand([1,-1])))
        # X = gen_data(g, W, N)

        # compute from generated data
        # μ = mean(X, dims=1)
        # σ = var(X, dims=1)

        # compute the μ and σ analytically
        μ = zeros(d)
        Σ = inv(myeye(d) - W) * inv((myeye(d) - W)')

        Σ, W
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

function gen_sup_data_all_with_graph(N, gs)
    # train data
    ds = @showprogress 0.1 "Generating.." map(gs) do g
        x, y = gen_sup_data(g, N)
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

function gen_sup_ds_with_graph(N, gs; batch_size)
    x, y = gen_sup_data_all_with_graph(N, gs)
    # DataSetIterator(x, y, batch_size)

    # DEBUG testing Cu datasets
    CuDataSetIterator(x, y, batch_size)
end

dscache = Dict()
# catch the datasets to avoid generation
function gen_sup_ds_cached(;ng, N, d, batch_size)
    ID = "$ng, $N, $d, $batch_size"
    # 1e4 is float
    ng = convert(Int, ng)
    if ! haskey(dscache, ID)
        # generate graphs first
        graphs = gen_graphs_hard(d, ng)
        # ratio
        index = convert(Int, round(length(graphs) * 4 / 5))
        train_gs = graphs[1:index]
        test_gs = graphs[index:end]

        # TODO use different graphs for ds and test
        @info "generating training ds .."
        ds = gen_sup_ds_with_graph(N, train_gs, batch_size=batch_size)
        @info  "generating testing ds .."
        test_ds = gen_sup_ds_with_graph(N, test_gs, batch_size=batch_size)

        # ds, test_ds = gen_sup_ds(ng=ng, N=N, d=d, batch_size=100)
        # dscache[ID] = (ds, test_ds)
    end
    # dscache[ID]
    ds, test_ds
end

# generate different types of graph for training and testing
function gen_sup_ds_cached_diff(;ng, N, d, batch_size)
    train_gs = gen_graphs_hard(d, ng, :ER)
    # different graphs for testing
    test_gs = gen_graphs_hard(d, ng, :SF)
    # FIXME filter out same graphs if any

    # DEBUG using a cheap test instead of filtering
    @show size(train_gs)
    @show size(test_gs)
    all_gs = vcat(train_gs, test_gs)
    @show size(all_gs)
    unique_all_gs = unique(all_gs) do g
        adjacency_matrix(g)
    end
    length(all_gs) == length(unique_all_gs) || begin
        @show length(all_gs)
        @show length(unique_all_gs)
        error("test gs has overlap")
    end

    # FIXME setdiff not working
    # diff = setdiff(test_gs, train_gs)
    # @show size(diff)
    # FIXME will they have the the same number of edges?

    # TODO use different graphs for ds and test
    ds = gen_sup_ds_with_graph(N, train_gs, batch_size=batch_size)
    test_ds = gen_sup_ds_with_graph(N, test_gs, batch_size=batch_size)

    # ds, test_ds = gen_sup_ds(ng=ng, N=N, d=d, batch_size=100)
    # dscache[ID] = (ds, test_ds)
    ds, test_ds
end


function gen_graphs(d, n, type=:ER)
    # generate n
    if type == :ER
        f = gen_ER_dag
    elseif type == :SF
        f = gen_SF_dag
    else
        error("Not supported graph type.")
    end
    all_graphs = map(1:n) do i
        f(d)
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

function gen_graphs_hard(d, n, type=:ER)
    # CAUTION this function will loop forever to generate enough unique graph
    all_gs = []
    @showprogress 0.1 "generating graph.." for i in 1:10
        # @info "graph iter $i .."
        gs = gen_graphs(d, n, type)
        append!(all_gs, gs)
        all_gs = unique(all_gs) do g
            # FIXME can this be compared directly?
            adjacency_matrix(g)
        end

        if length(all_gs) >= n
            all_gs = all_gs[1:n]
            break
        end
    end
    # @show length(all_gs)
    n == length(all_gs) || @warn "Not enougth unique graphs, $(length(all_gs)) < $n"
    # hcat(all_gs)
    all_gs
end


function test()
    # there are totally about 10000 different graphs for d=5
    gs = gen_graphs(5, 500000)
    gs = gen_graphs(5, 500)
    gs = gen_graphs_hard(5, 5000)
    gs = gen_graphs_hard(7, 5000, :SF)
    # test if unique
    gen_ER_dag(5)
    ds, test_ds = gen_sup_ds_cached_diff(ng=5000, d=7, N=2, batch_size=100)

    gs1 = [gen_ER_dag(3) for _ in 1:10]
    gs2 = [gen_ER_dag(3) for _ in 1:10]
    setdiff(gs1, gs2)
    unique(gs1) do g adjacency_matrix(g) end
    for g in gs1
        display(g)
    end
    gs1[3] == gs1[9]
    unique(gs1)
    unique((gs1[3], gs1[9]))
    isequal(gs1[3], gs1[9])
    intersect([gs1[3]], [gs1[9]])
    gs1[3] in gs1[7:9]
    intersect(gs1[3:5], gs1[7:9])
    setdiff!(copy(gs1[3:5]), gs1[7:9])
    delete!(copy(gs1[3:5]), gs1[9])
    typeof(gs1) <: AbstractSet
end

# Base.isequal(g1, g2) = adjacency_matrix(g1) == adjacency_matrix(g2)


function dag_seq_recur(n)
    res = 0
    if n == 0 return 1 end
    for k in 1:n
        # FIXME DP
        res += (-1)^(k-1) * binomial(k, n) * 2^(k * (n-k)) * dag_seq_recur(n-k)
    end
    res
end

function test_dat_seq()
    # FIXME number of ER graphs
    dag_seq(0) == 1
    dag_seq(1) == 1
    dag_seq(2) == 3
    dag_seq(3) == 25
    dag_seq(4) == 543
    dag_seq(5) == 29281
    dag_seq(6) == 3781503
    dag_seq(7) == 1138779265
end
