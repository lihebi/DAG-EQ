include("exp.jl")

function main_eq()
    eq_model_fn = (d)->eq_model(d, 300)
    exp_sup(5, eq_model_fn, prefix="EQ", ng=5e3, N=20, train_steps=3e4)
    exp_sup(7, eq_model_fn, prefix="EQ", ng=1e4, N=20, train_steps=3e4)
    exp_sup(10, eq_model_fn, prefix="EQ", ng=1e4, N=20, train_steps=5e3)
    exp_sup(15, eq_model_fn, prefix="EQ", ng=1e4, N=20, train_steps=5e3)
    exp_sup(20, eq_model_fn, prefix="EQ", ng=1e4, N=20, train_steps=5e3)
    exp_sup(25, eq_model_fn, prefix="EQ", ng=1e4, N=20, train_steps=5e3)
    exp_sup(30, eq_model_fn, prefix="EQ", ng=1e4, N=20, train_steps=5e3)

    eq_model_deep_fn = (d)->eq_model_deep(d, 300)
    exp_sup(5, eq_model_deep_fn, prefix="EQ-deep", ng=5e3, N=20, train_steps=3e4)
    exp_sup(10, eq_model_deep_fn, prefix="EQ-deep", ng=1e4, N=20, train_steps=3e4)
    exp_sup(15, eq_model_deep_fn, prefix="EQ-deep", ng=1e4, N=20, train_steps=3e4)
end

function main_fc()
    model_fn = (d)->sup_model(d)

    # FIXME after using w=1 model, the N won't matter: all of them will be
    # exactly the same

    exp_sup(10, model_fn, prefix="FC", ng=1e4, N=20, train_steps=5e4)
    exp_sup(15, model_fn, prefix="FC", ng=1e4, N=20, train_steps=5e4)
    exp_sup(20, model_fn, prefix="FC", ng=1e4, N=20, train_steps=5e4)
    exp_sup(5, model_fn, prefix="FC", ng=5e3, N=20, train_steps=5e4)
    exp_sup(7, model_fn, prefix="FC", ng=1e4, N=20, train_steps=5e4)
    exp_sup(25, model_fn, prefix="FC", ng=1e4, N=20, train_steps=5e4)
    exp_sup(30, model_fn, prefix="FC", ng=1e4, N=20, train_steps=5e4)
end

main_fc()
