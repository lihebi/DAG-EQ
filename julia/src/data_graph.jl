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

using LightGraphs
# using LightGraphs.SimpleGraphs
using MetaGraphs
using Base.Iterators

using GraphPlot: gplot, circular_layout
using Compose: PNG, draw

using GraphPlot
using Compose

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


# FIXME DAG
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
function gen_weights(dag)
    # dag is a meta graph, I'll need to attach parameters to each edge
    # 1. go through all edges
    # 2. attach edges
    # FIXME to ensure pureness, copy dag
    for e in edges(dag)
        # z = randn()
        # MOTE using uniform in [-2,-0.5] and [0.5, 2], better performance, make sense because it is easier
        z = (rand() * 1.5 + 0.5) * rand([1,-1])
        set_prop!(dag, e, :weight, z)
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

