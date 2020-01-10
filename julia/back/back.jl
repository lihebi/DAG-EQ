
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


function random_intervention(X)
    index = rand(1:size(X, 1), size(X, 2))
    # this index to CartesianIndex
    index = map(zip(index, 1:size(x,2))) do p
        CartesianIndex(p...)
    end
    delta = zeros(size(X))
    # FIXME this is not parallel
    # map(zip(index, 1:size(X,2))) do p
    #     delta[p...] = randn()
    # end
    delta[index] .= randn(size(delta,2))
    # delta[index] .= randn(size(delta,2))
    index, delta
end

# 7. implement do-effect
function do_effect(w, X, index, delta)
    # FIXME get which index is mutated by delta
    X2 = copy(X)
    X2[index] .= delta[index]
    X2 = w * X2 + randn(size(X2))
    X2[index] .= delta[index]
    X2
end

# 8. implement interventional loss
function do_loss_fn(w, X, d)
    # FIXME this randomness should come from outside
    index, delta = random_intervention(X)

    # FIXME post do should set instead of +
    # post_do = X + delta
    post_do = copy(X)
    post_do[index] = delta[index]
    post_effect = do_effect(w, X, index, delta)

    ls_loss = ls_loss_fn(w, X)
    # FIXME d() returns a scalar
    # @show size(ls_loss)
    # @show size(d(post_effect))
    # FIXME mean or sum
    loss = ls_loss - mean(d(post_effect))
    loss
end

function test()

    gradient(()->do_loss_fn(w, x, cpu(D)))
    gradient(()->begin
             # FIXME Mutating arrays is not supported
             index, delta = random_intervention(x)
             ls_loss_fn(w, x)
             # sum(cpu(D)(x))
             end)
    # FIXME MethodError: no method matching getproperty(::NamedTuple{(:W, :b, :σ)
    # see https://github.com/FluxML/Zygote.jl/issues/111
    gradient(()->sum(cpu(D)(x)))


    gradient(()->ls_loss_fn(w, x))

    # do_loss_fn(w, x, cpu(D))
end

# TODO maybe test inforce the NOTEARS acyclic constraint in Flux
function train_w!(w, ds;
                  cb_fn=(m)->(), train_steps=ds.nbatch)
    # train w (the model) with X
    opt = ADAM(1e-4)
    ps = Flux.params(w)
    m = MeanMetric{Float64}()

    diag_index = Matrix(I, size(w)...)
    @showprogress 0.1 "Training.." for step in 1:train_steps
        X, _ = next_batch!(ds)
        gs = gradient(ps) do
            loss = ls_loss_fn(w, X)
            add!(m, loss)
            loss
        end
        Flux.Optimise.update!(opt, ps, gs)
        # FIXME I should add constraint that diagonal is always 0
        w[diag_index] .= 0
        # @show get!(m)
        cb_fn(m)
    end
end

function train_do!(w, ds, d)
    opt = ADAM(1e-4)
    ps = Flux.params(w)
    m = MeanMetric{Float64}()
    @showprogress 0.1 "Training.." for step in 1:100
        X, _ = next_batch!(ds)
        gs = gradient(ps) do
            # FIXME use GPU
            loss = do_loss_fn(w, X, cpu(d))
            add!(m, loss)
            loss
        end
        # Flux.Optimise.update!(opt, ps, gs)
    end
end

function test()
    train_do!(w, ds, D)
    # train_do!(w, ds, cpu(D))
    x
    gradient(()->do_loss_fn(w, x, cpu(D)))
    # gradient(()->1)
end


function test(X)
    # using a bivariate graph
    g = MetaDiGraph(2)
    add_edge!(g, 1, 2)
    W = gen_weights(g)
    # CAUTION: transpose to put batch to second dim
    X = gen_data(g, W, 10000)'
    # 3. generate model (a W)
    w = zeros(size(W))
    # 4. fit the model with likelihood
    # put X into dataset iterator
    ds = DataSetIterator(X, zeros(size(X)), 100)

    x, _ = next_batch!(ds)

    train_w!(w, ds,
             cb_fn=Flux.throttle((m)->@show(get!(m)), 1),
             train_steps=3e4)
    # test the model


    # 6. fit a GAN
    G = graph_data_generator(10, 2) |> gpu
    D = graph_data_discriminator(2) |> gpu
    gopt = ADAM(2e-4)
    dopt = ADAM(2e-4)
    # how about ds?
    # FIXME monitor training loss
    train_GAN!(G, D, gopt, dopt, ds, train_steps=4000)
    # test the generator
    noise = randn(10) |> gpu
    fake = G(noise)
    σ.(D(fake))

    # 9. optimize interventional loss
    # train_do!(w, ds, D)
end


function test_setup()
    # 1. generate ground truth graph
    g = gen_ER_dag(5)
    is_cyclic(g) == false || error("not DAG")

    # using a bivariate graph
    g = MetaDiGraph(2)
    add_edge!(g, 1, 2)

    # 2. generate data
    @info "Generating weights .."
    W = gen_weights(g)

    @info "Generating 100 data points .."
    # more data is faster and performs better
    X = gen_data(g, W, 100)
    X = gen_data3(W, 100)

    @info size(W)
    @info size(X)

    # 5. fit the model with NOTEARS
    # run notears
    @info "Optimizing .."
    Ŵ = notears(X)
    # compare the resulting W estimation

    # visualize gragh
    DiGraph(threshold(Ŵ, 0.3))
    @info "Results:"
    @show W
    @show sparse(threshold(Ŵ, 0.3))

    @info "Evaluating .."
    graphical_metrics(W, threshold(Ŵ, 0.3))
end


function test_GMM()

    gmm = GMM(1, X, method=:kmeans)

    # create a new GMM
    gmm = GMM(1, 2, kind=:diag)
    em!(gmm, X)

    # try just generate N(0,1) and fit
    X = randn(100, 1)
    gmm = GMM(1, 1, kind=:diag)
    em!(gmm, X[:,1:1])

    dn = Normal()
    X[1]
    sum([loglikelihood(d, X[i:i]) for i in 1:10])

    loglikelihood(Normal(0,1), zeros(10))
    loglikelihood(Normal(0,1), randn(100))

    # multi-variate
    fit_mle(MvNormal, randn(100,2))
    # mixture gaussian

    for i in 1:100
        X = randn(100)

        @info "iter $i"

        d1 = fit_mle(Normal, X)
        d2 = fit_mle(Laplace, X)

        Xt = randn(100)

        @show loglikelihood(d1, X) - loglikelihood(d2, X)
        @show loglikelihood(d1, Xt) - loglikelihood(d2, Xt)
    end

    history(gmm)

    @show size(x)
    # CAUTION droping dims 1, otherwise GMM complains
    x = dropdims(x, dims=1)
    model_g = GMM(10, x, method=:kmeans)
    # The above already trained model_g with EM
    # em!(model_g, x)
    #
    # How to do inference?
    @info "Performing GMM inference .."
    pi = weights(model_g)
    @show size(pi)
    @show pi
    mu = means(model_g)
    @show size(mu)
    @show mu
    # FIXME method signature error
    # sigma = covars(model_g)
    sigma = model_g.Σ
    @show size(sigma)
    @show sigma

    @info "Computing mdn nll .."

    return mdn_nll(pi, mu, sigma, x)
end
