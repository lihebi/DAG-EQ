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

function test()
    g = gen_ER() |> Graph |> random_orientation_dag |> MetaDiGraph
    is_cyclic(g)

    set_weights!(g)
    print_weights(g)

    gen_data!(g)
    gen_data!(g, overwrite=true)
    gen_data!(g, overwrite=true, N=5)
    print_data(g)

    for v in vertices(g)
        @show v
        @show ! has_prop(g, v, :data)
    end

    randn(20,2) * [1,2]

    hcat([randn(20), randn(20)]) * [1,2]

    [randn(20), randn(20)] * [1,2]
end
