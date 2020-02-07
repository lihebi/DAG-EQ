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
include("model.jl")

import CuArrays
CuArrays.has_cutensor()

include("train.jl")


# TODO time to test the model!!
function inf()
end

function sup_model(d)
    # first reshape
    Chain(
        x -> reshape(x, d * d, :),
        Dense(d*d, 1024, relu),
        Dense(1024, 1024, relu),
        Dense(1024, d*d),
        # reshape back
        x -> reshape(x, d, d, :),
    )
end

function test()
    x = randn(5,5,1,10)
    reshape(x, 25, 10)

    # !!!!!!!!
    x = randn(5,5,1,10)
    y = reshape(x, 5, 5, 10)
    Flux.mse(x,y)
    Flux.mse(x,x)
    Flux.mse(y,y)
end

function eq_model(d, z)
    # eq model
    # TODO put on GPU, I probably need to write GPU kernel for this
    model = Chain(
        x->reshape(x, d, d, 1, :),
        Equivariant(1=>z),
        ReLU(),
        Equivariant(z=>z),
        ReLU(),
        Equivariant(z=>z),
        ReLU(),
        Equivariant(z=>z),
        ReLU(),
        Equivariant(z=>1),
        # IMPORTANT drop the second-to-last dim 1
        x->reshape(x, d, d, :)
    )
end

function test_eq()
    d=5
    # TODO try more data
    ng=1000
    N=20
    train_steps=1e4
    ds, test_ds = gen_sup_ds_cached(ng=ng, N=N, d=d, batch_size=100)

    X, Y = next_batch!(ds) |> gpu
    # add channel 1
    # X = reshape(X, size(X)[1:end-1]..., 1, size(X)[end])
    size(X)

    # the previous sup model
    model = sup_model(5) |> gpu
    param_count(model)             # 1,106,969

    # model = eq_model(100) |> gpu
    # ┌ Info: test_ds
    # │   loss_v = 0.20647460743784904
    # │   nnz = 721.2
    # │   nny = 445.5
    # │   tpr = 0.6815749975000338
    # │   fpr = 0.20325712696593506
    # │   fdr = 0.5787666837875985
    # │   shd = 559.4
    # │   prec = 0.4212333162124013
    # └   recall = 0.6815749975000338
    #
    # 60600 params
    model = eq_model(5, 100)

    # Trained a little bit
    # ┌ Info: test_ds
    # │   loss_v = 0.23060775846242904
    # │   nnz = 578.9
    # │   nny = 446.5
    # │   tpr = 0.5019395303062226
    # │   fpr = 0.1727644088246259
    # │   fdr = 0.6126408084090456
    # │   shd = 577.1
    # │   prec = 0.38735919159095444
    # └   recall = 0.5019395303062226
    #
    # 541,800 params
    model = eq_model(5, 300)
    model = eq_model(5, 1024)
    model = gpu(model)
    param_count(model)          # 28

    # run on X
    model(X)

    model(X[:,:,1,1])
    cpu(model)(cpu(X))
    size(X)

    sup_graph_metrics(cpu(model(X)), cpu(Y))

    # gpu test
    model(gpu(X))
    gpu(model)(gpu(X))
    model(gpu(X))
    cpu(model)(X)

    # test gradient
    gradient(()->sum(model(X)), params(model))
    # gradient(()->sum(cpu(model)(cpu(X))), params(cpu(model)))

    # really train
    opt = ADAM(1e-5)
    print_cb = Flux.throttle(sup_create_print_cb(), 1)
    test_cb = Flux.throttle(sup_create_test_cb(model, test_ds, "test_ds", use_gpu=true), 10)
    sup_train!(model, opt, ds, test_ds,
               print_cb=print_cb,
               test_cb=test_cb,
               use_gpu=true,
               train_steps=train_steps * 10)
end


function exp_sup(d; ng=10000, N=10, train_steps=1e5)
    # for scitific notation
    ng = convert(Int, ng)
    # TODO test on different type of graph

    ds, test_ds = gen_sup_ds_cached(ng=ng, N=N, d=d, batch_size=100)
    # ds, test_ds = gen_sup_ds_cached_diff(ng=ng, N=N, d=d, batch_size=100)
    x, y = next_batch!(test_ds)

    # FIXME parameterize the model
    model = sup_model(d) |> gpu
    # TODO lr decay
    opt = ADAM(1e-4)

    expID = "d=$d-ng=$ng-N=$N"

    logger = TBLogger("tensorboard_logs/train-$expID-$(now())", tb_append, min_level=Logging.Info)
    test_logger = TBLogger("tensorboard_logs/test-$expID-$(now())", tb_append, min_level=Logging.Info)

    print_cb = Flux.throttle(sup_create_print_cb(logger), 1)
    test_cb = Flux.throttle(sup_create_test_cb(model, test_ds, "test_ds", logger=test_logger), 10)

    sup_train!(model, opt, ds, test_ds,
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

function test()
    # so it does not work
    exp_sup(5, ng=1e3, N=2)     # prec=0.58, recall=0.80
    exp_sup(5, ng=1e3, N=20)    # prec=0.80, recall=0.95
    exp_sup(5, ng=5e3, N=20)    # prec=0.93, recall=0.98
    # this actually generates 99xx graphs
    exp_sup(5, ng=1e4, N=20)    # prec=0.92, recall=0.97
    # with increased N, it works much better
    exp_sup(5, ng=5e3, N=50)    # prec=0.96, recall=0.98
    exp_sup(5, ng=5e3, N=100)   # prec=0.97, recall=0.99
    # exp_sup(5, ng=1e4, N=10)
    # exp_sup(5, ng=1e4, N=20)
    # prec=0.94, recall=0.98
    exp_sup(5, ng=5e3, N=50, train_steps=5e5) # prec=0.98, recall=0.99
    # prec=0.87, recall=0.93
    exp_sup(7, ng=1e4, N=50, train_steps=5e5) # prec=0.92, recall=0.96
    exp_sup(10, ng=1e4, N=100, train_steps=5e5) # prec=0.83, recall=0.87
    exp_sup(10, ng=1e5, N=50, train_steps=5e5)  # prec=0.85, recall=0.88
    exp_sup(15, ng=1e4, N=100, train_steps=5e5) # prec=0.68, recall=0.72
    exp_sup(20, ng=1e4, N=100, train_steps=5e5)
    exp_sup(20, ng=1e5, N=50, train_steps=5e5)
end

function sup_view(model, x, y)
    length(size(x)) == 1 || error("x must be one data")
    # visualize ground truth y
    d = convert(Int, sqrt(length(y)))
    g = DiGraph(reshape(y, d, d))
    display(g)

    out = cpu(model)(x)
    reshape(threshold(out, 0.3), d, d) |> DiGraph |> display
    nothing
end
