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

function load_most_recent(model_dir)
    if !isdir(model_dir) return nothing, 1 end
    # get the most recent one
    files = readdir(model_dir)
    if length(files) == 0 return nothing, 1 end

    # step-1000.bson
    steps = map(files) do fname
        step_str = match(r"step-(\d+).bson", fname).captures[1]
        parse(Int, step_str)
    end
    max_step = maximum(steps)
    most_recent = files[argmax(steps)]
    @load joinpath(model_dir, most_recent) model
    return model, max_step
end

function exp_sup(d, model_fn;
                 prefix="", suffix="$(now())",
                 ng=1e4, N=10, train_steps=1e5, test_throttle=10)
    ng = Int(ng)
    train_steps=Int(train_steps)

    expID = join([prefix, "d=$d", "ng=$ng", "N=$N", suffix], "-")
    @show expID

    model_dir = joinpath("saved_models", expID)

    # FIXME load model if already trained
    most_recent_model, from_step = load_most_recent(model_dir)
    if !isnothing(most_recent_model)
        @info "using trained model, starting at step $from_step"
        model = most_recent_model |> gpu
    else
        model = model_fn(d) |> gpu
        from_step = 1
    end

    if from_step >= train_steps
        @info "already trained"
        return
    end

    ds, test_ds = gen_sup_ds_cached(ng=ng, N=N, d=d, batch_size=100)
    # FIXME .|>
    ds, test_ds = (ds, test_ds) .|> CuDataSetIterator
    # ds, test_ds = gen_sup_ds_cached_diff(ng=ng, N=N, d=d, batch_size=100)
    x, y = next_batch!(test_ds) |> gpu

    @info "warming up model with x .."
    model(x)
    @info "warming up gradient .."
    gradient(()->sum(model(x)), params(model))

    # TODO lr decay
    opt = ADAM(1e-4)

    # when continual training, new files are created
    # CAUTION there will be some overlapping
    logger = TBLogger("tensorboard_logs/train-$expID", tb_append, min_level=Logging.Info)
    test_logger = TBLogger("tensorboard_logs/test-$expID", tb_append, min_level=Logging.Info)

    print_cb = create_print_cb(logger=logger)
    test_cb = create_test_cb(model, test_ds, "test_ds", logger=test_logger)
    save_cb = create_save_cb("saved_models/$expID", model)

    @info "training .."

    sup_train!(model, opt, ds,
               # print tensorboard logs every 1 second
               print_cb=Flux.throttle(print_cb, 1),
               # test throttle = 10
               test_cb=Flux.throttle(test_cb, test_throttle),
               # save every 60 seconds TODO use a ultra large number
               # I'll save all the model during training, if that's too large, I'll consider:
               # 1. delete old ones, i.e. rotating
               # 2. use a larger interval
               save_cb=Flux.throttle(save_cb, 60),
               from_step=from_step,
               train_steps=train_steps)

    # final save and test
    test_cb(train_steps)
    save_cb(train_steps)
end


function exp_mixed(model_fn; prefix="", suffix="$(now())", train_steps=3e4)
    train_steps = Int(train_steps)
    # expID = "TMP-$(now())"
    expID = "$prefix-mixed-d=[10,15,20]-$suffix"

    model_dir = joinpath("saved_models", expID)

    # FIXME load model if already trained
    most_recent_model, from_step = load_most_recent(model_dir)
    if !isnothing(most_recent_model)
        @info "using trained model, starting at step $from_step"
        model = most_recent_model |> gpu
    else
        # 100 is not used here
        model = model_fn(100) |> gpu
        from_step = 1
    end

    if from_step >= train_steps
        @info "already trained"
        return
    end

    # FIXME enough memory?
    # looks like the memory garbage collection is going to consume a lot of time
    ds10, test_ds10 = gen_sup_ds_cached(ng=10000, N=20, d=10, batch_size=100) .|> CuDataSetIterator
    ds15, test_ds15 = gen_sup_ds_cached(ng=10000, N=20, d=15, batch_size=100) .|> CuDataSetIterator
    ds20, test_ds20 = gen_sup_ds_cached(ng=10000, N=20, d=20, batch_size=100) .|> CuDataSetIterator

    # testing the data
    x10, y10 = next_batch!(ds10) |> gpu;
    x15, y15 = next_batch!(ds15) |> gpu;
    x20, y20 = next_batch!(ds20) |> gpu;
    model(x10)
    model(x15)
    model(x20)

    # TODO different lr for different data?
    opt = ADAM(1e-4)

    logger = TBLogger("tensorboard_logs/train-$expID", tb_append, min_level=Logging.Info)
    test_logger = TBLogger("tensorboard_logs/test-$expID", tb_append, min_level=Logging.Info)

    print_cb = create_print_cb(logger=logger)
    # NOTE different function
    test_cb = create_test_cb_mixed(model,
                                   [test_ds10, test_ds15, test_ds20],
                                   "test_ds", logger=test_logger)
    save_cb = create_save_cb("saved_models/$expID", model)

    @info "training .."
    # NOTE different function
    mixed_sup_train!(model, opt, [ds10, ds15, ds20],
                     # print every 1 second
                     print_cb=Flux.throttle(print_cb, 1),
                     # test every 20 seconds
                     test_cb=Flux.throttle(test_cb, 20),
                     # save every 10 minutes
                     save_cb=Flux.throttle(save_cb, 600),
                     from_step=from_step,
                     train_steps=train_steps)

    # final save and test
    test_cb(train_steps)
    save_cb(train_steps)
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
