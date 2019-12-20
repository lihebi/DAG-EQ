include("data.jl")
include("model.jl")
include("train.jl")

function test_setup()
    scm = random_spline_scm(8, 10)
    plot_spline(scm)

    model_x2y = MDN(32, 10)
    model_y2x = MDN(32, 10)

    ps = Flux.params(model_x2y)
    length(ps)
    ps[1]

    dist_fn = () -> rand(Normal(0, 2), (1,20))
    X = dist_fn()
    Y = scm.(X)
    size(X)
    size(Y)
end

function test_dist()
    # See issue https://github.com/FluxML/Zygote.jl/issues/436
    using Distributions
    import Zygote
    import Tracker
    Zygote.gradient((μ, σ) -> loglikelihood(Normal(μ, σ), [1,2,3]), 0, 1)
    # => (nothing, nothing)
    Tracker.gradient((μ, σ) -> loglikelihood(Normal(μ, σ), [1,2,3]), 0, 1)
    # => (6.0 (tracked), 11.0 (tracked))
end


function test_mdn_nll()
    out = model_x2y(X)

    # FIXME out is TrackerArray, and it cannot go through mdn_nll
    mdn_nll(out..., Y)
    # the clean array version works
    mdn_nll(map((o)->o.data, out)..., Y)

    # TODO test if Zygote.jl can differentiate across mdn_nll
    Tracker.gradient(mdn_nll, out..., Y)
    import Zygote

    # if I'm using Flux#master, what would be the output?
    gs = gradient(mdn_nll, out..., Y)
    length(gs)
    gs[1]
    gs[2]
    gs[3]
    gs[4]
end

function test()
    xx = mdn_nll(out..., Y).data
    xxx = mdn_nll(out..., Y).data

    log(sum(exp.(xx)))
    logsumexp(xx)
end


function test_meta()
    scm = random_spline_scm(8, 10)
    plot_spline(scm)

    model_x2y = MDN(32, 10)
    model_y2x = MDN(32, 10)

    ps = Flux.params(model_x2y)
    length(ps)
    ps[1]

    # X = train_dist()
    # length(model_x2y(X))
    # model_x2y(X)[1]

    # size(X)
    # scm.(X)

    # train_dist = () -> rand(Normal(0, 2), (1,1000))
    # trans_dist = () -> rand(Normal(rand(-4:4), 2), (1,1000))

    train_nll!(model_x2y, scm, "X2Y")
    train_nll!(model_y2x, scm, "Y2X")

    # alpha is a tensor, train_alpha returns a frame of alpha values
    alpha = 0.5
    frames = train_alpha!(model_x2y, model_y2x, alpha)

    # TODO plot frames
end

