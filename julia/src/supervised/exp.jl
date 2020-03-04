# I must always load config first
include("../config.jl")

using Statistics
using Dates: now

include("../data_graph.jl")
include("model.jl")

import CuArrays
CuArrays.has_cutensor()

include("train.jl")

using Profile
using BenchmarkTools: @btime

function exp_sup(d, model_fn; prefix="", ng=1e4, N=10, train_steps=1e5, test_throttle=10)
    ng = convert(Int, ng)

    expID = "$prefix-d=$d-ng=$ng-N=$N"
    @show expID


    # detect if the expID already tested

    for dir in readdir("tensorboard_logs")
        # occursin("EQ-d=5-ng=10000-N=20", "test-EQ-d=5-ng=10000-N=20-2020-03-01T16:32:34.56")
        if occursin(expID, dir)
            @info "Already trained in $dir"
            return
        end
    end


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


    logger = TBLogger("tensorboard_logs/train-$expID-$(now())", tb_append, min_level=Logging.Info)
    test_logger = TBLogger("tensorboard_logs/test-$expID-$(now())", tb_append, min_level=Logging.Info)

    print_cb = Flux.throttle(create_print_cb(logger=logger), 1)
    test_cb = Flux.throttle(create_test_cb(model, test_ds, "test_ds", logger=test_logger),
                            test_throttle)

    @info "training .."

    sup_train!(model, opt, ds, test_ds,
               print_cb=print_cb,
               test_cb=test_cb,
               train_steps=train_steps)

    @info "Saving model .."
    if !isdir("trained_model") mkdir("trained_model") end
    @time @save "trained_model/$expID-$(now()).bson" model=cpu(model)

    # TODO enforcing sparsity, and increase the loss weight of 1 edges, because
    # there are much more 0s, and they can take control of loss and make 1s not
    # significant. As an extreme case, the model may simply report 0 everywhere

    # do the inference
    # x, y = next_batch!(test_ds)
    # sup_view(model, x[:,8], y[:,8])
end


function test_profile()
    Profile.clear()
    Profile.init(n = 10^7, delay = 0.01)

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
