# I must always load config first
include("config.jl")

using Statistics
using Dates: now
try
    # This is a weird error. I have to load libdl and dlopen libcutensor first, then
    # CuArrays to have cutensor. Otherwise, if CuArrays is loaded first, libcutensor
    # cannot be even dlopen-ed
    using Libdl
    Libdl.dlopen("libcutensor")
catch ex
    @warn "Cannot open libcutensor library"
end

include("data_graph.jl")
include("model_sup.jl")

import CuArrays
CuArrays.has_cutensor()

include("train.jl")

using Profile

function exp_sup(d, model_fn; prefix="", ng=10000, N=10, train_steps=1e5)
    ng = convert(Int, ng)

    ds, test_ds = gen_sup_ds_cached(ng=ng, N=N, d=d, batch_size=100)
    # ds, test_ds = gen_sup_ds_cached_diff(ng=ng, N=N, d=d, batch_size=100)
    x, y = next_batch!(test_ds) |> gpu

    model = model_fn(d) |> gpu

    @info "warming up model with x .."
    model(x)
    @info "warming up gradient .."
    gradient(()->sum(model(x)), params(model))

    # TODO lr decay
    opt = ADAM(1e-4)

    expID = "$prefix-d=$d-ng=$ng-N=$N"

    logger = TBLogger("tensorboard_logs/train-$expID-$(now())", tb_append, min_level=Logging.Info)
    test_logger = TBLogger("tensorboard_logs/test-$expID-$(now())", tb_append, min_level=Logging.Info)

    print_cb = Flux.throttle(sup_create_print_cb(logger), 1)
    test_cb = Flux.throttle(sup_create_test_cb(model, test_ds, "test_ds", logger=test_logger), 10)

    @info "training .."

    @profile sup_train!(model, opt, ds, test_ds,
                        print_cb=print_cb,
                        test_cb=test_cb,
                        train_steps=train_steps)

    # TODO enforcing sparsity, and increase the loss weight of 1 edges, because
    # there are much more 0s, and they can take control of loss and make 1s not
    # significant. As an extreme case, the model may simply report 0 everywhere

    # do the inference
    # x, y = next_batch!(test_ds)
    # sup_view(model, x[:,8], y[:,8])
end



function main()
    model_fn = (d)->sup_model(d)

    # FIXME after using w=1 model, the N won't matter: all of them will be
    # exactly the same
    exp_sup(5, model_fn, ng=5e3, N=50, train_steps=1e5)
    exp_sup(7, model_fn, ng=1e4, N=50, train_steps=1e5)
    exp_sup(10, model_fn, ng=1e4, N=50, train_steps=1e5)
    exp_sup(15, model_fn, ng=1e4, N=50, train_steps=1e5)


    Profile.clear()
    Profile.init(n = 10^7, delay = 0.01)

    eq_model_fn = (d)->eq_model(d, 300)

    exp_sup(5, eq_model_fn, prefix="EQ", ng=5e3, N=50, train_steps=2e3)
    exp_sup(7, eq_model_fn, prefix="EQ", ng=1e4, N=50, train_steps=1e5)
    exp_sup(10, eq_model_fn, prefix="EQ", ng=1e4, N=50, train_steps=1e5)
    exp_sup(15, eq_model_fn, prefix="EQ", ng=1e4, N=50, train_steps=1e5)


    open("/tmp/prof.txt", "w") do s
        # I want to ignore all low-cost operations (e.g. <10)
        Profile.print(IOContext(s, :displaysize => (24, 200)), mincount=10)
    end

    # other visualization methods
    #
    # Profile.print()
    # import ProfileView
    # ProfileView.view()
end

function test()
    ds, test_ds = gen_sup_ds_cached(ng=5e3, N=20, d=5, batch_size=100)
    x, y = next_batch!(test_ds) |> gpu

    model = eq_model_fn(5) |> gpu


    # warm up the model
    model(x)
    gradient(()->sum(model(x)), params(model))
    σ.(model(x))

    # actually training
    opt = ADAM(1e-4)
    print_cb = Flux.throttle(sup_create_print_cb(), 1)
    test_cb = Flux.throttle(sup_create_test_cb(model, test_ds, "test_ds"), 10)

    sup_train!(model, opt, ds, test_ds,
               print_cb=print_cb,
               test_cb=test_cb,
               train_steps=1e4)
end
