import Printf

include("exp.jl")


function gen_row(row, d)
    one = [Printf.@sprintf("%.3f / %.3f", item.prec, item.recall) for item in row]
    join(["model-"*string(d), one...], ", ")
end

function transfer_modelX_dY(modelX, dY, Y)
    @info "creating model .."
    tmp_model = Chain(x->reshape(x, Y, Y, 1, :),
                      modelX[2:end-1]...,
                      x->reshape(x, Y, Y, :))
    # FIXME use multile batches
    x, y = next_batch!(dY) |> cpu
    @info "testing .."
    m = sup_graph_metrics(tmp_model(x), y)
    @info "result" Y m.prec, m.recall
    m
end

function transfer_modelX(fname, d5, d10, d15, d20)
    @info "loading model .."
    @load ("trained_model/" * fname) model
    [transfer_modelX_dY(model, d5, 5),
     transfer_modelX_dY(model, d10, 10),
     transfer_modelX_dY(model, d15, 15),
     transfer_modelX_dY(model, d20, 20)]
end

function test_transfer()
    # bson_files = filter(x->occursin("EQ-deep", x), readdir("trained_model"))
    # fd = parse(Int, match(r"d=(\d+)", bson_files[1]).captures[1])
    f5 = "NEW-EQ-deep-d=5-ng=5000-N=20-2020-03-04T23:48:40.762.bson"
    f10 = "NEW-EQ-deep-d=10-ng=5000-N=20-2020-03-05T02:08:38.134.bson"
    f15 = "NEW-EQ-deep-d=15-ng=5000-N=20-2020-03-05T06:13:58.645.bson"
    f20 = "NEW-EQ-deep-d=20-ng=5000-N=20-2020-03-05T13:08:06.271.bson"
    # identify the d=X
    d5, _ = gen_sup_ds_cached(ng=10000, N=20, d=5, batch_size=100)
    d10, _ = gen_sup_ds_cached(ng=10000, N=20, d=10, batch_size=100)
    d15, _ = gen_sup_ds_cached(ng=10000, N=20, d=15, batch_size=100)
    d20, _ = gen_sup_ds_cached(ng=10000, N=20, d=20, batch_size=100)

    row5 = transfer_modelX(f5, d5, d10, d15, d20)
    row10 = transfer_modelX(f10, d5, d10, d15, d20)
    row15 = transfer_modelX(f15, d5, d10, d15, d20)
    row20 = transfer_modelX(f20, d5, d10, d15, d20)

    # generate table
    println(join(["model\\data", "graph-5", "graph-10", "graph-15", "graph-20"], ", "))
    println(gen_row(row5, 5))
    println(gen_row(row10, 10))
    println(gen_row(row15, 15))
    println(gen_row(row20, 20))
end

function test_universal()
    @load "saved_models/EQ-deep-mixed-d=[10,15,20]-ONE/step-100000.bson" model

    x, y = next_batch!(d15)
    sup_graph_metrics(model(x), y)

    # for d in [5,7,10,13,15,17,20,23,25,27,30,35,40]
    # for d in [5,7,10,13,15]
    # for d in [17,20,23,25]
    for d in [27,30,35,40]
        ds, _ = gen_sup_ds_cached(ng=2000, N=2, d=d, batch_size=100)
        prec = MeanMetric{Float64}()
        recall = MeanMetric{Float64}()
        shd = MeanMetric{Float64}()
        @showprogress 0.1 "evaluating .." for i in 1:10
            x, y = next_batch!(ds)
            m = sup_graph_metrics(model(x), y)
            add!(prec, m.prec)
            add!(recall, m.recall)
            add!(shd, m.shd)
        end
        @info "result" d get!(prec) get!(recall) get!(shd)
    end
end

function test()
    # read d=20 bson
    model_file = "./trained_model/NEW-EQ-deep-d=20-ng=5000-N=20-2020-03-05T13:08:06.271.bson"
    @load model_file model
    # generate d=20 data and monitor results
    ds, test_ds = gen_sup_ds_cached(ng=10000, N=20, d=20, batch_size=100)
    x, y = next_batch!(ds) |> cpu
    model(x)
    # gpu(model)(gpu(x))
    m = sup_graph_metrics(model(x), y)
    m.prec
    m.recall
    # adapt to d=10
    d10_model = Chain(x->reshape(x, 10, 10, 1, :),
                      model[2:end-1]...,
                      x->reshape(x, 10, 10, :))

    # generate d=10 data and monitor results
    ds10, _ = gen_sup_ds_cached(ng=10000, N=20, d=10, batch_size=100)
    x10, y10 = next_batch!(ds10) |> cpu
    m10 = sup_graph_metrics(d10_model(x10), y10)
    m10.prec
    m10.recall

end
