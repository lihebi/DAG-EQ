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
using DataFrames: DataFrame, Not, All

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

function parse_spec_string(s)
    re = r"d=(\d+)_k=(\d+)_gtype=(.*)_noise=(.*)_mat=([^_]*)(_mec=MLP)?"
    m = match(re, s)
    if isnothing(m) @show s end
    d = parse(Int, m.captures[1])
    k = parse(Int, m.captures[2])
    gtype, noise, mat, mec = m.captures[3:end]
    if isnothing(mec)
        mec = "Linear"
    else
        mec = "MLP"
    end
    # FIXME nothing seems to match any expressions, see
    # https://github.com/kmsquire/Match.jl/issues/60
    #
    # mec = @match mec begin
    #     nothing => "Linear"
    #     # CAUTION This is substring, thus this won't match
    #     # var::String => "MLP"
    #     _ => "MLP"
    # end
    return d, k, gtype, noise, mat, mec
end

# FIXME these type definitions do not seem to be allowed inside function scope
MInt = Union{Missing, Int}
MString = Union{Missing, String}

# DEBUG
function pretty_print_result()
    global _results

    load_results!()

    df = DataFrame(model=String[],
                   train_d=MInt[], train_k=MInt[],
                   train_gtype=MString[],
                   train_noise=MString[],
                   train_mat=MString[],
                   train_mec=MString[],

                   test_d=Int[], test_k=Int[],
                   test_gtype=String[],
                   test_noise=String[],
                   test_mat=String[],
                   test_mec=MString[],

                   prec=Float32[], recall=Float32[], shd=Float32[])
    for item in _results
        # 1. find the model name
        m = match(r"(.*)-d=.*", item[1][1])
        if isnothing(m)
            model = item[1][1]
            train_d = missing
            train_k = missing
            train_gtype = missing
            train_noise = missing
            train_mat = missing
            train_mec = missing
        else
            # model = item[1][1]
            model = m.captures[1]
            (train_d, train_k, train_gtype, train_noise, train_mat,
             train_mec) = parse_spec_string(item[1][1])
        end

        (test_d, test_k, test_gtype, test_noise, test_mat,
             test_mec) = parse_spec_string(item[1][2])

        prec = round3(item[2][1]) * 100
        recall = round3(item[2][2]) * 100
        shd = round(item[2][3], digits=1)
        # prec, recall, shd = round3.(item[2][1:end-1])

        push!(df, (model,
                   train_d, train_k, train_gtype, train_noise, train_mat, train_mec,
                   test_d, test_k, test_gtype, test_noise, test_mat, test_mec,
                   prec, recall, shd))
    end

    df

    # I don't need train_k, because all of them are 1s
    df = df[!, Not(All(:train_k, :train_noise))]

    unique(df.train_mat)
    unique(df.model)

    # 1. all the models in separate mode
    select_sep_df(df, "deep-FC")
    select_sep_df(df, "flat-CNN")
    select_sep_df(df, "bottle-CNN")
    select_sep_df(df, "deep-EQ")

    # DONE close comparison between flat-CNN and deep-EQ
    # header=map((x)->replace(x,"_"=>"/"),
    #            names(tbl_baseline))
    # quotestrings=true,
    CSV.write("results/cmp-baseline.csv", table_cmp_baseline(df))
    CSV.write("results/er4.csv", table_ER24(df))

    # DONE add MLP testing results

    # 2. transfering
    #
    # - train on k=1, test on k=1,2,4
    CSV.write("results/transfer_k.csv", table_transfer_k(df))
    # - train on Gaussian, test on Exp, Gumbel
    CSV.write("results/transfer_noise.csv", table_transfer_noise(df))
    # - train on ER, test on SF
    CSV.write("results/transfer_gtype.csv", table_transfer_gtype(df))

    # 3. the ensemble models. The training setting is in model name
    select_ensemble_df(df, "flat-CNN-ER-ensemble")
    select_ensemble_df(df, "flat-CNN-SF-ensemble")
    select_ensemble_df(df, "bottle-CNN-ER-ensemble")
    select_ensemble_df(df, "bottle-CNN-SF-ensemble")

    select_ensemble_df(df, "deep-EQ-ER-ensemble")
    select_ensemble_df(df, "deep-EQ-SF-ensemble")

    select_ensemble_df(df, "deep-EQ-ER-COV-ensemble")
    select_ensemble_df(df, "deep-EQ-SF-COV-ensemble")

    CSV.write("results/ensemble.csv", table_ensemble(df))
end

function table_transfer_k(df)
    # - train on k=1, test on k=1,2,4
    selector = ((df.model .== "deep-EQ")
                .& (df.test_noise .== "Gaussian")
                .& (df.train_gtype .== "SF")
                .& (df.test_gtype .== "SF")
                .& (df.train_mec .== "Linear")
                .& (df.test_mec .== "Linear")
                .& (df.train_mat .== "COR")
                .& (df.test_mat .== "COR"))
    sort(df[selector,
            Not(All(:train_k, :train_noise,
                    :test_d,
                    :train_mec, :test_mec, :test_noise,
                    :train_mat, :test_mat))],
         [:train_gtype, :model, :test_gtype, :train_d, :test_k])
end

function table_transfer_noise(df)
    # - train on Gaussian, test on Exp, Gumbel
    selector = ((df.model .== "deep-EQ")
                .& in.(df.train_gtype, Ref(["SF", "ER"]))
                .& (df.train_gtype .== df.test_gtype)
                .& (df.test_k .== 1)
                .& (df.train_mec .== "Linear")
                .& (df.test_mec .== "Linear")
                .& (df.train_mat .== "COR")
                .& (df.test_mat .== "COR"))
    sort(df[selector,
            Not(All(:train_k,
                    # CAUTION I need to show this to tell that the training and
                    # testing noise are different :train_noise,
                    :test_d, :test_k,
                    :train_mec, :test_mec,
                    :test_gtype,
                    :train_mat, :test_mat))],
         [:train_gtype, :model, :train_d, :test_noise])
end


function table_transfer_gtype(df)
    # - train on ER, test on SF
    selector = ((df.model .== "deep-EQ")
                # TODO ER2 ER4 SF2 SF4
                .& in.(df.train_gtype, Ref(["ER", "SF"]))
                .& (df.train_gtype .!= df.test_gtype)
                .& (df.test_noise .== "Gaussian")
                .& (df.test_k .== 1)
                .& (df.train_mec .== "Linear")
                .& (df.test_mec .== "Linear")
                .& (df.train_mat .== "COR")
                .& (df.test_mat .== "COR"))
    sort(df[selector,
            Not(All(:train_k, :train_noise,
                    :test_d, :test_k,
                    :train_mec, :test_mec, :test_noise,
                    :train_mat, :test_mat))],
         [:train_gtype, :model, :test_gtype, :train_d])
end


function table_cmp_baseline(df)
    # comparison of EQ with FC and CNN (and random) baselines
    # only show d=10,15,20, SF and ER graphs, k=1
    sel1 = in.(df.model, Ref(["deep-FC", "deep-EQ", "flat-CNN"]))
    sel2 = in.(df.train_d, Ref([10, 15, 20]))
    selector = (sel1 .& sel2
                .& (df.train_gtype .== df.test_gtype)
                .& in.(df.train_gtype, Ref(["ER", "SF"]))
                .& (df.test_noise .== "Gaussian")
                .& (df.train_mec .== "Linear")
                .& (df.test_mec .== "Linear")
                .& (df.train_mat .== "COR")
                .& (df.test_mat .== "COR")
                .& (df.test_k .== 1))

    sort(df[selector,
            # test_d is same as train_d
            Not(All(:train_k,
                    :train_noise,
                    :test_d,
                    :test_k,
                    :train_mec, :test_mec,
                    :test_noise,
                    :train_mat,
                    :test_mat))],
         [:train_gtype, :model, :test_gtype, :train_d])
end

function table_ER24(df)
    # - ER2 ER4 SF2 SF4
    # TODO CNN, FC data for these graphs
    selector = ((df.model .== "deep-EQ")
                .& in.(df.train_gtype, Ref(["SF", "SF2", "SF4", "ER", "ER2", "ER4"]))
                .& (df.train_gtype .== df.test_gtype)
                .& (df.test_k .== 1)
                .& (df.test_noise .== "Gaussian")
                .& (df.train_mec .== "Linear")
                .& (df.test_mec .== "Linear")
                .& (df.train_mat .== "COR")
                .& (df.test_mat .== "COR"))
    sort(df[selector,
            Not(All(:train_k, :train_noise,
                    :test_d, :test_k,
                    :train_mec, :test_mec, :test_noise,
                    :train_mat, :test_mat))],
         [:train_gtype, :model, :test_gtype, :train_d])
end



function select_sep_df(df, model)
    subdf = sort(df[df.model .== model,
                    # test_d is same as train_d
                    Not(All(:train_k, :train_noise, :test_d))],
                 [:train_mec, :train_gtype, :test_gtype, :train_d, :test_k])
    show(subdf, allrows=true)
    nothing
end

function table_ensemble(df)
    selector = (in.(df.model, Ref(["deep-EQ-ER-ensemble",
                                   "deep-EQ-SF-ensemble"]))
                .& (df.test_k .== 1)
                .& (df.test_mec .== "Linear")
                .& (df.test_mat .== "COR"))
    sort(df[selector,
            Not(All(:train_k, :train_noise,
                    :test_k,
                    r"train_",
                    :train_mec, :test_mec, :test_noise,
                    :train_mat, :test_mat))],
         [:model, :test_gtype, :test_d])
end

function select_ensemble_df(df, model)
    subdf = sort(df[df.model .== model, Not(All(:train_k, :train_noise, r"train_"))],
                 [:test_gtype, :test_d, :test_k])
    show(subdf, allrows=true)
    nothing
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
        ds, test_ds = spec2ds(spec)

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

