using Interpolations
using Plots

function display_plot(p)
    path = tempname() * ".png"
    savefig(p, path)
    println("$(path)")
    println("#<Image: $(path)>")
end

function test_plot()
    xs = -10:1:10
    ys = 2 * xs
    display_plot(plot(xs, ys))
end


# this should return a model, that when applying to X, returns
function random_spline_scm(span=8., num_anchors=8)
    xs = LinRange(-span, span, num_anchors)
    # xs = -span:1:span
    ys = rand(Float64, size(xs)) * span * 2 .- span
    extrapolate(Interpolations.scale(interpolate(ys, BSpline(
        # Quadratic
        Cubic(
            # FIXME what should be the option here?
            Line(
                # > `OnGrid` means that the boundary condition "activates" at the
                # > first and/or last integer location within the interpolation
                # > region,
                #
                # > `OnCell` # means the interpolation extends a half-integer beyond
                # > the edge before # activating the boundary condition
                OnGrid())))), xs),
                # FIXME not sure what's the behavior of splev's ext=0, return
                # the "extrapolated value".
                Line())
end

function plot_spline(scm)
    # generate random x, around 0
    p = plot()
    mu = [-4 0 4]
    # mu = [-2 0]
    for m in mu
        X = m .+ 2 * randn(1000)
        # regularize X into [-8,8]
        # X = X[X .> -8]
        # X = X[X .< 8]
        y = scm(X) + randn(size(X))
        plot!(p, X, y, seriestype=:scatter)
    end
    display_plot(p)
end

function test_spline()
    scm = random_spline_scm(8, 8)
    scm(0.2)
    scm(8.8)
    scm(18.8)

    plot_spline(scm)
end

function test()
    test_spline()
end
