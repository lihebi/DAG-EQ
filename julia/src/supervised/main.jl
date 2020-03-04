include("exp.jl")

fc_model_fn(d) = fc_model(d=d, z=1024, nlayer=3)
fc_dropout_model_fn(d) = fc_model(d=d, z=1024, reg=true, nlayer=3)

deep_fc_model_fn(d) = fc_model_deep(d=d, z=1024, nlayer=6)
deep_fc_dropout_model_fn(d) = fc_model_deep(d=d, z=1024, reg=true, nlayer=6)

eq_model_fn(d) = eq_model(d=d, z=300, reg=false, nlayer=3)
eq_dropout_model_fn(d) = eq_model(d=d, z=300, reg=true, nlayer=3)

# TODO wide model
deep_eq_model_fn(d) = eq_model(d=d, z=300, reg=false, nlayer=6)
deep_eq_dropout_model_fn(d) = eq_model(d=d, z=300, reg=true, nlayer=6)

function main_eq()
    ds = [5,7,10,15,20,25,30]
    # exp_sup(5, eq_model_fn, prefix="EQ", ng=5e3, N=20, train_steps=3e4)
    # exp_sup(7, eq_model_fn, prefix="EQ", ng=1e4, N=20, train_steps=3e4)
    for d in ds
        exp_sup(d, eq_model_fn, prefix="EQ", ng=1e4, N=20, train_steps=3e4,
                test_throttle=20)
        # exp_sup(d, eq_dropout_model_fn, prefix="EQ-reg", ng=1e4, N=20, train_steps=3e4)
        exp_sup(d, deep_eq_model_fn, prefix="EQ-deep", ng=5e3, N=20,
                train_steps=3e4, test_throttle=20)
        # exp_sup(d, deep_eq_dropout_model_fn, prefix="EQ-deep-reg", ng=5e3, N=20, train_steps=3e4)
    end
end

function main_fc()
    ds = [5,7,10,15,20,25,30]
    for d in ds
        # FIXME after using w=1 model, the N won't matter: all of them will be
        # exactly the same
        exp_sup(d, fc_model_fn, prefix="FC", ng=1e4, N=20, train_steps=3e5)
        # exp_sup(d, fc_dropout_model_fn, prefix="FC-reg", ng=1e4, N=20, train_steps=3e5)
        exp_sup(d, fc_model_deep_fn, prefix="FC-deep", ng=1e4, N=20, train_steps=3e5)
        # exp_sup(d, deep_fc_dropout_model_fn, prefix="FC-deep-reg", ng=1e4, N=20, train_steps=3e5)
    end
end

function main()
    # TODO try this
    exp_sup(10, deep_eq_model_fn, prefix="EQ-deep-noreg", ng=1e4, N=20, train_steps=3e4)
    exp_sup(10, deep_eq_dropout_model_fn, prefix="EQ-deep-dropout", ng=1e4, N=20, train_steps=3e4)

    # exp_sup(10, deep_eq_model_fn, prefix="EQ-deep-noreg", ng=1e4, N=20, train_steps=3e4)
    # exp_sup(10, eq_dropout_model_fn, prefix="EQ-dropout", ng=1e4, N=20, train_steps=3e4)
    # exp_sup(10, eq_model_fn, prefix="EQ-noreg", ng=1e4, N=20, train_steps=3e4)


    # exp_sup(30, deep_eq_dropout_model_fn, prefix="EQ-deep-dropout", ng=1e4, N=20, train_steps=3e4)
    # exp_sup(20, deep_eq_dropout_model_fn, prefix="EQ-deep-dropout", ng=1e4, N=20, train_steps=3e4)
end

main_fc()
main_eq()
