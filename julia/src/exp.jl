# I must always load config first
include("config.jl")
include("utils.jl")

using Statistics
using Dates

include("data_graph.jl")
include("model.jl")

import CuArrays
CuArrays.has_cutensor()

include("train.jl")

using Profile
using BenchmarkTools: @btime
import CSV
using DataFrames: DataFrame

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
    @info "Loading $most_recent .."
    @load joinpath(model_dir, most_recent) model
    return model, max_step
end


"""These functions takes care of data creation. It is fine as long as the data
is created only once, before first use.

"""
function spec2ds(spec::DataSpec)
    batch_size = spec.bsize
    ds, test_ds = load_sup_ds(spec, batch_size)
    ds, test_ds = (ds, test_ds) .|> CuDataSetIterator
    return ds, test_ds
end

function spec2ds(specs::Array{DataSpec, N} where N)
    dses = map(specs) do spec
        ds, test_ds = (ds, test_ds) .|> CuDataSetIterator
        ds, test_ds = load_sup_ds(spec)
    end
    return [ds[1] for ds in dses], [ds[2] for ds in dses]
end

function exp_train(spec, model_fn;
                   prefix,
                   train_steps=1e5, test_throttle=10)
    train_steps=Int(train_steps)

    # setting expID
    expID = "$prefix-$(dataspec_to_id(spec))"

    model_dir = joinpath("saved_models", expID)

    # FIXME load model if already trained
    most_recent_model, from_step = load_most_recent(model_dir)
    if !isnothing(most_recent_model)
        @info "using trained model, starting at step $from_step"
        # FIXME it does not seem to be smooth at the resume point
        model = most_recent_model |> gpu
    else
        model = model_fn() |> gpu
        from_step = 1
    end

    if from_step >= train_steps
        @info "already trained"
        return expID
    end

    ds, test_ds = spec2ds(spec)
    x, y = next_batch!(test_ds) |> gpu

    @info "warming up model with x .."
    model(x)
    @info "warming up gradient .."
    gradient(()->sum(model(x)), Flux.params(model))

    # TODO lr decay
    opt = ADAM(1e-4)

    # when continual training, new files are created
    # CAUTION there will be some overlapping
    logger = TBLogger("tensorboard_logs/train-$expID",
                      tb_append, min_level=Logging.Info)
    test_logger = TBLogger("tensorboard_logs/test-$expID",
                           tb_append, min_level=Logging.Info)

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
               #
               # I'll save all the model during training, if that's too large,
               # I'll consider:
               #
               # 1. delete old ones, i.e. rotating
               # 2. use a larger interval
               save_cb=Flux.throttle(save_cb, 60),
               from_step=from_step,
               train_steps=train_steps)

    # final save and test
    test_cb(train_steps)
    save_cb(train_steps)

    return expID
end


# this should be a map, from (expID, testID) to values
# - expID is the model name, manually specified
# - testID is the DataSpec's string representation
# Also, I should save it as a text file, and support external editing
# I should also record the time of the test
#
# so, better just use a csv file
#
# expID, testID, date, prec, recall, shd
_results = nothing
function load_results!()
    global _results
    if isfile("result.csv")
        df = CSV.read("result.csv")
        df.date
        _results = Dict((df.expID .=> df.testID)
                        .=>
                        zip(df.prec, df.recall, df.shd, df.date))
    else
        _results = Dict()
    end
end

# DEBUG
function pretty_print_result()
    global _results

    load_results!()

    df = DataFrame(model=String[], d=Int[], k=Int[],
                   gtype=String[], noise=String[], mat=String[],
                   prec=Float64[], recall=Float64[], shd=Float64[])
    for item in _results
        m = match(r"d=(\d+)_k=(\d+)_gtype=(.*)_noise=(.*)_mat=(.*)", item[1][2])
        # m = match(r"d=(\d+)_k=(\d+)_gtype=(.*)_noise=(.*)_mat=(.*)",
        #           "d=15_k=2_gtype=ER_noise=Gaussian_mat=COR")
        model = item[1][1]
        d = parse(Int, m.captures[1])
        k = parse(Int, m.captures[2])
        gtype, noise, mat = m.captures[3:end]
        prec, recall, shd = round3.(item[2][1:end-1])
        push!(df, (model, d, k, gtype, noise, mat, prec*100, recall*100, shd))
    end

    df
    df[df.model .== "deep-EQ", :]
    describe(df)
    unique(df.model)
    sort(df[df.model .== "deep-EQ-ER", :], [:model, :gtype, :d, :k])
    sort(df[df.model .== "deep-EQ-SF", :], [:model, :gtype, :d, :k])
    sort(df[df.model .== "deep-EQ-ER-COV", :], [:model, :gtype, :d, :k])
    sort(df[df.model .== "deep-FC-ER-d=10", :], [:model, :gtype, :d, :k])
    sort(df[df.model .== "deep-FC-ER-d=15", :], [:model, :gtype, :d, :k])
    sort(df[df.model .== "deep-FC-ER-d=20", :], [:model, :gtype, :d, :k])
    sort(df[df.model .== "deep-FC-SF-d=10", :], [:model, :gtype, :d, :k])
    sort(df[df.model .== "deep-FC-SF-d=15", :], [:model, :gtype, :d, :k])
    sort(df[df.model .== "deep-FC-SF-d=20", :], [:model, :gtype, :d, :k])
end

# However, although the date is string, it is saved without quotes. Then, the
# read CSV.read() is too smart to that it converts that into DateTime. However,
# it cannot convert DateTime to string when it saves it.
Base.convert(String, date::DateTime) = "$date"

function result2csv(res, fname)
    df = DataFrame(expID=String[], testID=String[],
                   prec=Float64[], recall=Float64[],
                   shd=Float64[], date=String[])
    for (key,value) in res
        push!(df, (key[1], key[2], value[1], value[2], value[3], value[4]))
    end
    # save df
    CSV.write(fname, df)
end

function save_results!()
    global _results
    result2csv(_results, "result.csv")
end

function test()
    result = Dict(("hello"=>"world")=>(0.2, 0.3, 3),
                  ("1"=>"2")=>(0.2, 0.3, 3))'
    result2csv(result, "result.csv")
end


function exp_test(expID, spec)
    global _results
    if isnothing(_results)
        load_results!()
    end

    testID = dataspec_to_id(spec)

    if !haskey(_results, expID=>testID)
        model_dir = joinpath("saved_models", expID)
        model, _ = load_most_recent(model_dir)
        if isnothing(model)
            error("Cannot load model in $model_dir")
        end

        model = gpu(model)
        # FIXME smaller batch size to avoid out-of-mem error (for d=80)
        ds, test_ds = spec2ds(spec, batch_size=32)

        # DEBUG TODO not using all data for testing
        metrics = sup_test(model, test_ds, nbatch=16)

        # add this to result
        _results[expID=>testID] = (metrics.prec, metrics.recall,
                                   metrics.shd,
                                   now())

        # finally, save the results
        @info "Saving results .."
        save_results!()
    end
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

