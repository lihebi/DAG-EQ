using LightGraphs
using LightGraphs.SimpleGraphs
using Base.Iterators

import Base.show
import Base.display
using FileIO
using FileIO: @format_str

using GraphPlot
using Compose
import Cairo, Fontconfig

using MetaGraphs

struct EmacsDisplay <: AbstractDisplay end

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
function Base.display(d::EmacsDisplay, x)
    display(d, "image/png", x)
end

function register_EmacsDisplay_backend()
    for d in Base.Multimedia.displays
        if typeof(d) <: EmacsDisplay
            return
        end
    end
    # register as backend
    pushdisplay(EmacsDisplay())
    # popdisplay()
    nothing
end

register_EmacsDisplay_backend()


##############################
## Generating DAG
##############################


# FIXME DAG
function gen_ER()
    n = 10
    m = 10
    erdos_renyi(n, m, is_directed=true)
end

function gen_SF()
    n = 10
    m = 10
    # static_scale_free(n, m, 2)
    barabasi_albert(n, convert(Int, round(m/n)), is_directed=true)
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

""" Generate model parameters according to the DAG structure and causal mechanism

"""
function set_weights!(dag, mechanism="default")
    # dag is a meta graph, I'll need to attach parameters to each edge
    # 1. go through all edges
    # 2. attach edges
    for e in edges(dag)
        set_prop!(dag, e, :weight, randn())
    end
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

function gen_data!(dag; N=20, overwrite=false)
    # v = select_vertex(dag)
    if overwrite
        for v in vertices(dag)
            rem_prop!(dag, v, :data)
        end
    end
    for v in vertices(dag)
        gen_data!(dag, v, N)
    end
end

function gen_data!(dag, v, N)
    # start from a random node that does not have data, i.e. 1
    if ! has_prop(dag, v, :data)
        parents = inneighbors(dag, v)
        # if it has parents, run gen_data for its parents
        if length(parents) > 0
            for p in parents
                gen_data!(dag, p, N)
            end
        end
        # gen data for itself
        if length(parents) > 0
            pdata = map(parents) do p
                get_prop(dag, p, :data)
            end
            weights = map(parents) do p
                get_prop(dag, p, v, :weight)
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
