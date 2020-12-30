import CSV
using DataFrames: DataFrame, Not, All
include("utils.jl")


function load_results()
    if isfile("result.csv")
        df = CSV.read("result.csv")
        df.date
        Dict((df.expID .=> df.testID)
                        .=>
                        zip(df.prec, df.recall, df.shd,
                            df.time, df.date))
    else
        @warn "No result.csv"
        Dict()
    end
end

function parse_spec_string(s)
    re = r"d=(\d+)_k=(\d+)_gtype=(.*)_noise=(.*)_mat=([^_]*)(_mec=MLP)?(_raw)?"
    m = match(re, s)
    if isnothing(m) @show s end
    d = parse(Int, m.captures[1])
    k = parse(Int, m.captures[2])
    gtype, noise, mat, mec, raw = m.captures[3:end]
    if isnothing(mec)
        mec = "Linear"
    else
        mec = "MLP"
    end
    if isnothing(raw)
        raw = false
    else
        raw = true
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
    return d, k, gtype, noise, mat, mec, raw
end

# FIXME these type definitions do not seem to be allowed inside function scope
MInt = Union{Missing, Int}
MString = Union{Missing, String}

function results2df(results)
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
                   raw=Bool[],

                   prec=Float32[], recall=Float32[], shd=Float32[],
                   time=Float32[])
    for item in results
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
             train_mec, _) = parse_spec_string(item[1][1])
        end

        (test_d, test_k, test_gtype, test_noise, test_mat,
         test_mec, raw) = parse_spec_string(item[1][2])

        prec = round3(item[2][1]) * 100
        recall = round3(item[2][2]) * 100
        shd = round(item[2][3], digits=1)
        t = round(item[2][4], digits=1)
        # prec, recall, shd = round3.(item[2][1:end-1])

        push!(df, (model,
                   train_d, train_k, train_gtype, train_noise, train_mat, train_mec,
                   test_d, test_k, test_gtype, test_noise, test_mat, test_mec, raw,
                   prec, recall, shd, t))
    end
    return df
end




function table_raw(df)
    selector = (
        # (df.raw .== 1) .&
        (df.train_gtype .== "SF")
        .& in.(df.test_d, Ref([10, 20, 50, 100, 200, 300, 400]))
        # .& in.(df.train_d, Ref([10, 20, 50, 100, 200, 300, 400]))
    )
    sort(df[selector,
            All(:model, :train_d, :prec, :recall, :shd, :time, :test_d)],
         [:train_d])
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
    sel1 = in.(df.model, Ref(["FC", "EQ", "CNN"]))
    # sel2 = in.(df.train_d, Ref([10, 15, 20,8,16,32]))
    sel2 = in.(df.train_d, Ref([10, 20, 50, 100]))
    selector = (sel1 .& sel2
                .& (df.train_gtype .== df.test_gtype)
                .& in.(df.train_gtype, Ref(["SF"]))
                .& (df.test_noise .== "Gaussian")
                .& (df.train_mec .== "Linear")
                .& (df.test_mec .== "Linear")
                .& (df.train_mat .== "COR")
                .& (df.test_mat .== "COR")
                .& (df.test_k .== 1))

    sort(df[selector,
            All(:model, :train_d, :prec, :recall, :shd)],
         [:model, :train_d])
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

function test()
    # DONE close comparison between flat-CNN and deep-EQ
    # header=map((x)->replace(x,"_"=>"/"),
    #            names(tbl_baseline))
    # quotestrings=true,
    table_cmp_baseline(df)
    CSV.write("results/cmp-baseline.csv", table_cmp_baseline(df))
    CSV.write("results/er4.csv", table_ER24(df))

    # DONE add MLP testing results

    # 1.5 raw
    CSV.write("results/raw.csv", table_raw(df))

    # 2. transfering
    #
    # - train on k=1, test on k=1,2,4
    CSV.write("results/transfer_k.csv", table_transfer_k(df))
    # - train on Gaussian, test on Exp, Gumbel
    CSV.write("results/transfer_noise.csv", table_transfer_noise(df))
    # - train on ER, test on SF
    CSV.write("results/transfer_gtype.csv", table_transfer_gtype(df))

    # 3. the ensemble models. The training setting is in model name
    select_ensemble_df(df, "deep-EQ-ER-ensemble")
    select_ensemble_df(df, "deep-EQ-SF-ensemble")

    select_ensemble_df(df, "deep-EQ-ER-COV-ensemble")
    select_ensemble_df(df, "deep-EQ-SF-COV-ensemble")

    CSV.write("results/ensemble.csv", table_ensemble(df))
end