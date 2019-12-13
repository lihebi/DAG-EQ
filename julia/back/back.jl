
# def _random_acyclic_orientation(B_und):
#     return np.tril(_random_permutation(B_und), k=-1)
function random_acyclic_orientation(M)
    M |> random_permutation |> m -> tril(m, -1)
end

using LinearAlgebra
using Random

# def _random_permutation(M):
#     # np.random.permutation permutes first axis only
#     P = np.random.permutation(np.eye(M.shape[0]))
#     return P.T @ M @ P
function random_permutation(M)
    eye = 1 * Matrix(I, size(M)...)
    # P = Random.randperm(eye)
    P = eye[shuffle(1:end), :]
    transpose(P) * M * P
end

function test()
    randn(3,4) * randn(4,3)

    shuffle([1 2 3;4 5 6])
    [1 2 3;4 5 6][shuffle(1:end), :]
    shuffle([1,2,3; 4,5,6])
    randperm(3)

    random_permutation(randn(3,3))
    random_acyclic_orientation(randn(3,3))
end


function ensure_dag(g)
    # get adj matrix
    m = Matrix(adjacency_matrix(g))
    # FIXME this will remove many edges
    m = random_acyclic_orientation(m)
    m = random_permutation(m)
    # restore adj matrix
    DiGraph(m)
end

function sup_train!(model, opt, x, y, test_x, test_y)
    # FIXME mse does not seem to fit for high dim
    loss(x, y) = Flux.mse(model(x), y)

    function cb_fn1()
        @show Flux.mse(model(x), y)
        # @show accuracy(model(x), y)
        # @show accuracy(model(test_x), test_y)
        @time metrics = sup_graph_metrics(model(x), y)
        @show metrics
    end

    function cb_fn2()
        @time test_metrics = sup_graph_metrics(model(test_x), test_y)
        @info "test metrics $(test_metrics)"
        # @show test_metrics
    end

    f1 = Flux.throttle(cb_fn1, 1)
    # I want test metrics to be less frequent
    f2 = Flux.throttle(cb_fn2, 10)

    function cb_fn()
        f1()
        f2()
    end

    Flux.train!(loss, Flux.params(model), Iterators.repeated((x,y), 100), opt, cb=cb_fn)
end

function sup_model()
    Chain(Dense(4, 100, relu),
          Dense(100, 100, relu),
          Dense(100, 4))
end

# NOTE sup exp should not use accuracy, but fpr, tpr
accuracy(out, y) = mean(Flux.onecold(abs.(out)) .== Flux.onecold(abs.(y)))
# tpr(out, y) = sum(reshape(y, 20, 20, :) .== adjacency_matrix(ĝ) .== 1)


function test_sup_bi()
    # Training
    # 1. generate model g
    g = MetaDiGraph(2)
    add_edge!(g, 1, 2)

    x, y = gen_sup_data(g)
    test_x, test_y = gen_sup_data(g)

    # generate data for the other direction
    g2 = MetaDiGraph(2)
    add_edge!(g2, 2, 1)

    x2, y2 = gen_sup_data(g2)
    test_x2, test_y2 = gen_sup_data(g2)


    model = sup_model()
    # opt = Descent(0.1)
    opt = ADAM(1e-4)
    Flux.@epochs 20 sup_train!(model, opt,
                               hcat(x, x2), hcat(y, y2),
                               hcat(test_x, test_x2), hcat(test_y, test_y2))
    Flux.@epochs 5 sup_train!(model, opt, x2, y2)

    accuracy(model(test_x), test_y)
    accuracy(model(x2), y2)
    accuracy(model(test_x2), test_y2)

    # test whether it works
    model(ds[1][:,1:10])
    ds[2][:,1]
    res = model(ds[1][:,1:10])
    @show sum([x[1] == 3 for x in argmax(abs.(model(ds[1])), dims=1)]) / size(ds[1],2)
    @show sum([x[1] == 3 for x in argmax(abs.(model(test_ds[1])), dims=1)]) / size(test_ds[1],2)
    @show [x[1] for x in argmax(abs.(res), dims=1)]
    res[CartesianIndex(4, 1)]
    findmax
    mapslices(indmax, model(ds[1][:,1:10]), 2)

    # TODO generate new data and test

    # 2. generate weights W
    @info "Generating weights .."
    W = gen_weights(g)
    @show Matrix(W)
    # 3. generate data X
    @info "Generating 100 data points .."
    X = gen_data(g, W, 100)
    # 4. compute μ and σ (or just use the ground truth μ and σ)
    μ = mean.(X)
    σ = var.(X)
    # 5. f(μ, σ) => W

    # Inference
    # 1. compute μ and σ
    # 2. f(μ, σ)
end
