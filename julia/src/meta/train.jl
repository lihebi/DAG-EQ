using Distributions
using BSON: @save, @load
using ProgressMeter: @showprogress
using StatsFuns: logsumexp
using GaussianMixtures

mutable struct MeanMetric{T}
    # FIXME syntax for T
    sum::T
    n::Int
    # CAUTION for T, you must define Base.convert(::T, v::Int64)
    MeanMetric{T}() where {T} = new(0, 0)
end
function add!(m::MeanMetric{T}, v) where {T}
    m.sum += v
    m.n += 1
end
get(m::MeanMetric) = if m.n != 0 m.sum / m.n else m.sum / 1 end
function get!(m::MeanMetric)
    res = get(m)
    reset!(m)
    res
end
function reset!(m::MeanMetric{T}) where {T}
    m.sum = 0
    m.n = 0
end


function mdn_nll(pi, mu, sigma, y)
    # pi, mu, sigma = pi_mu_sigma

    # FIXME Zygote cannot differentiate through this
    ds = Normal.(mu, sigma)

    f(d) = loglikelihood(d, y)
    log_prob_y = f.(ds)

    log_prob_pi_y = log_prob_y + log.(pi)

    loss = -log.(sum(exp.(log_prob_pi_y), dims=1))
    # FIXME Tracker.jl cannot work through reduce
    loss = -logsumexp(log_prob_pi_y, dims=1)
    @show loss
    # FIXME loss is -Inf
    @show mean(loss)
    mean(loss)
end


function train_nll!(model, scm, polarity)
    @info "Training nll"
    dist_fn = () -> rand(Normal(0, 2), (1,1000))
    ps = Flux.params(model)
    opt = ADAM(1e-4)
    ml = MeanMetric{Float64}()
    @info "entering training loop"
    @showprogress 0.1 "Training.." for step in 1:100
        # generate training dist
        X = dist_fn()
        Y = scm.(X)
        if polarity == "X2Y"
            inp, tar = X, Y
        elseif polarity == "Y2X"
            inp, tar = Y, X
        else
            error("Invalid polarity $polarity")
        end
        # @show size(inp)
        # @show size(tar)
        gs = Tracker.gradient(ps) do
            # @info "here"
            out = model(inp)
            # @info "there"
            # @show size(out)
            loss = mdn_nll(out..., tar)
            @show loss
            add!(ml, loss.data)
            # loss = sum(tar)
            loss
        end
        # FIXME loss is not decreasing
        @show get!(ml)
        Flux.Optimise.update!(opt, ps, gs)
    end
end

function train_alpha!(model_x2y, model_y2x, alpha)
    param_dist = (mean) -> rand(Normal(mean, 2), (1, 1000))
    param_sampler = () -> rand() * 8 - 4

    ps = Flux.param(alpha)
    opt = ADAM(1e-4)

    res = []
    @showprogress 0.1 "Training.." for step in 1:20
        # sample paramters from transfer distribution
        p = param_sampler()
        X = param_dist(p)
        Y = scm.(X)
        metric_x2y = transfer_metric(model_x2y, X, Y)
        metric_y2x = transfer_metric(model_y2x, Y, X)
        # calculate loss
        gs = gradient(ps) do
            loss = logsumexp(logsigmoid(alpha) + metric_x2y,
                             logsigmoid(-alpha) + metric_y2x)
        end
        Flux.Optimise.update!(opt, ps, gs)
        push!(res, sigmoid(alpha))
    end
    res
end

function transfer_metric(model, x, y)
    # FIXME save load works like deep copy?
    path = tempname() * ".bson"
    # FIXME don't need to save it again and again
    @save path model=model
    @load path model
    _transfer_metric(model, x, y)
end

function marginal_nll(x)
    # model_g = GMM()
    # train_gmm!(model_g, x)
    #
    # FIXME I'm going to use Julia's implementation of GMM instead of firing my own
    # model_g = GMM(ones(10), zeros(10), zeros(10,10), nothing)
    @info "Creating and training GMM .."
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

function _transfer_metric(model, x, y)
    ps = Flux.params(model)

    opt = ADAM(1e-4)
    joint_loss = []
    # first get the marginal loss
    # loss_marginal = marginal_nll(x)
    loss_marginal = 0
    @show loss_marginal
    # then get the loss
    for step in 1:10
        gs = Tracker.gradient(ps) do
            @show size(x)
            out = model(x)
            @show length(out)
            @show size(out[1])
            @show size(y)
            loss = mdn_nll(out..., y)
            @show loss
            # FIXME why adding loss_marginal in every iteration?
            push!(joint_loss, loss + loss_marginal)
            loss
        end
        Flux.Optimise.update!(opt, ps, gs)
    end
    sum(joint_loss)
end
