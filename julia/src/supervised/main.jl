include("exp.jl")

fc_model_fn(d) = fc_model(d=d, hidden_size=1024, nlayer=3)
fc_dropout_model_fn(d) = fc_model(d=d, hidden_size=1024, reg=true, nlayer=3)

deep_fc_model_fn(d) = fc_model_deep(d=d, hidden_size=1024, nlayer=6)
deep_fc_dropout_model_fn(d) = fc_model_deep(d=d, hidden_size=1024, reg=true, nlayer=6)

eq_model_fn(d) = eq_model(d=d, z=300, reg=false, nlayer=3)
eq_dropout_model_fn(d) = eq_model(d=d, z=300, reg=true, nlayer=3)

# TODO wide model
deep_eq_model_fn(d) = eq_model(d=d, z=300, reg=false, nlayer=6)
deep_eq_dropout_model_fn(d) = eq_model(d=d, z=300, reg=true, nlayer=6)

function main_eq()
    # FIXME why d=5 is even slower?
    exp_sup(5, eq_model_fn, prefix="EQ-noreg", ng=5e3, N=20, train_steps=3e4)
    exp_sup(7, eq_model_fn, prefix="EQ-noreg", ng=1e4, N=20, train_steps=3e4)

    # TODO try this
    exp_sup(10, eq_model_fn, prefix="EQ-noreg", ng=1e4, N=20, train_steps=3e4)
    exp_sup(15, eq_model_fn, prefix="EQ-noreg", ng=1e4, N=20, train_steps=3e4)
    exp_sup(20, eq_model_fn, prefix="EQ-noreg", ng=1e4, N=20, train_steps=3e4)
    exp_sup(25, eq_model_fn, prefix="EQ-noreg", ng=1e4, N=20, train_steps=3e4)
    exp_sup(30, eq_model_fn, prefix="EQ-noreg", ng=1e4, N=20, train_steps=3e4)

    exp_sup(10, eq_dropout_model_fn, prefix="EQ-dropout", ng=1e4, N=20, train_steps=3e4)
    exp_sup(15, eq_dropout_model_fn, prefix="EQ-dropout", ng=1e4, N=20, train_steps=3e4)
    exp_sup(20, eq_dropout_model_fn, prefix="EQ-dropout", ng=1e4, N=20, train_steps=3e4)

    # TODO try this
    exp_sup(5, deep_eq_model_fn, prefix="EQ-deep-noreg", ng=5e3, N=20, train_steps=3e4)
    exp_sup(10, deep_eq_model_fn, prefix="EQ-deep-noreg", ng=1e4, N=20, train_steps=3e4)
    exp_sup(15, deep_eq_model_fn, prefix="EQ-deep-noreg", ng=1e4, N=20, train_steps=3e4)

    exp_sup(5, deep_eq_dropout_model_fn, prefix="EQ-deep-dropout", ng=5e3, N=20, train_steps=3e4)
    exp_sup(10, deep_eq_dropout_model_fn, prefix="EQ-deep-dropout", ng=1e4, N=20, train_steps=3e4)
    exp_sup(15, deep_eq_dropout_model_fn, prefix="EQ-deep-dropout", ng=1e4, N=20, train_steps=3e4)
end

function main_fc()

    # FIXME after using w=1 model, the N won't matter: all of them will be
    # exactly the same
    exp_sup(5, fc_model_fn, prefix="FC", ng=1e4, N=20, train_steps=3e5)
    exp_sup(7, fc_model_fn, prefix="FC", ng=1e4, N=20, train_steps=3e5)
    exp_sup(10, fc_model_fn, prefix="FC", ng=1e4, N=20, train_steps=3e5)
    exp_sup(15, fc_model_fn, prefix="FC", ng=1e4, N=20, train_steps=3e5)
    exp_sup(20, fc_model_fn, prefix="FC", ng=1e4, N=20, train_steps=3e5)
    exp_sup(25, fc_model_fn, prefix="FC", ng=1e4, N=20, train_steps=3e5)
    exp_sup(30, fc_model_fn, prefix="FC", ng=1e4, N=20, train_steps=3e5)

    # TODO try this
    exp_sup(5, fc_dropout_model_fn, prefix="FC-dropout", ng=1e4, N=20, train_steps=3e5)
    exp_sup(7, fc_dropout_model_fn, prefix="FC-dropout", ng=1e4, N=20, train_steps=3e5)
    exp_sup(10, fc_dropout_model_fn, prefix="FC-dropout", ng=1e4, N=20, train_steps=3e5)
    exp_sup(15, fc_dropout_model_fn, prefix="FC-dropout", ng=1e4, N=20, train_steps=3e5)

    exp_sup(5, fc_model_deep_fn, prefix="FC-deep", ng=1e4, N=20, train_steps=3e5)
    exp_sup(7, fc_model_deep_fn, prefix="FC-deep", ng=1e4, N=20, train_steps=3e5)
    exp_sup(10, fc_model_deep_fn, prefix="FC-deep", ng=1e4, N=20, train_steps=3e5)
    exp_sup(15, fc_model_deep_fn, prefix="FC-deep", ng=1e4, N=20, train_steps=3e5)
    exp_sup(20, fc_model_deep_fn, prefix="FC-deep", ng=1e4, N=20, train_steps=3e5)
    exp_sup(25, fc_model_deep_fn, prefix="FC-deep", ng=1e4, N=20, train_steps=3e5)
    exp_sup(30, fc_model_deep_fn, prefix="FC-deep", ng=1e4, N=20, train_steps=3e5)

    # TODO try this
    exp_sup(5, deep_fc_dropout_model_fn, prefix="FC-deep-dropout", ng=1e4, N=20, train_steps=3e5)
    exp_sup(7, deep_fc_dropout_model_fn, prefix="FC-deep-dropout", ng=1e4, N=20, train_steps=3e5)
    exp_sup(10, deep_fc_dropout_model_fn, prefix="FC-deep-dropout", ng=1e4, N=20, train_steps=3e5)
    exp_sup(15, deep_fc_dropout_model_fn, prefix="FC-deep-dropout", ng=1e4, N=20, train_steps=3e5)
end
