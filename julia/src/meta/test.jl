
function test()
    scm = random_spline_scm(8, 10)
    plot_spline(scm)

    model_x2y = MDN(32, 10)
    model_y2x = MDN(32, 10)

    ps = Flux.params(model_x2y)
    length(ps)
    ps[1]

    train_nll!(model_x2y, scm, "X2Y")
    train_nll!(model_y2x, scm, "Y2X")

    dist_fn = () -> rand(Normal(0, 2), (1,50))
    X = dist_fn()
    Y = scm.(X)
    out = model_x2y(X)
    mdn_nll(out..., Y)
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

