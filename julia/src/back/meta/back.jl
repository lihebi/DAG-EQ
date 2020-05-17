# plotly()
# gr()

# inspectdr()
# pgfplots()
# unicodeplots()
# pyplot()


function test_mdn_nll()
    function foo(pi_mu_sigma, y)
        pi, mu, sigma = pi_mu_sigma
        # mu = pi_mu_sigma
        pi = 1
        sigma = 1
        ds = Normal.(mu, sigma)
        f(d) = sum(x -> logpdf(d, x), y)
        # f(d) = loglikelihood(d, y)
        log_prob_y = f.(ds)
        log_prob_pi_y = log_prob_y .+ log.(pi)
        loss = mean(log.(sum(exp.(log_prob_pi_y), dims=1)))
        # loss = mean(log_prob_y)
        # loss = sum(y)
        # @show loss
        loss
    end
    function bar(pi_mu_sigma, y)
        # pi, mu, sigma = pi_mu_sigma
        mu = pi_mu_sigma
        pi = 1
        sigma = 1
        all = map(pi, mu, sigma) do p,m,s
            d = Normal(m, s)
            sum(x -> logpdf(d, x), y)
        end
        sum(all)
    end
    # gfoo = Tracker.gradient(foo, ([1 1 1], [0.5 0.5 0.5], [1 1 1]), [2 2 2; 3 3 3])
    gfoo = Tracker.gradient(foo, ([1 1 1], [1 1 1], [1 1 1]), [2 2 2; 3 3 3])
    @show gfoo
    Flux.param([2 2 2; 3 3 3])
    # gbar = Tracker.gradient(bar, ([1 1 1], [0.5 0.5 0.5], [1 1 1]), Flux.param([2 2 2; 3 3 3]))
    # gbar = Tracker.gradient(bar, [1 1 1], [2 2 2; 3 3 3])
    # @show gbar
end

import Zygote

test_mdn_nll()

import ForwardDiff
import Tracker

function test()
    Zygote.gradient((mu, sigma) -> sum(sum(x -> logpdf.(Normal.(mu, sigma), x), [8, 8])), [0, 0], [1, 1])

    Zygote.gradient((mu, sigma) -> sum(x -> logpdf(Normal(mu, sigma), x), 8), 0, 1)
    Zygote.gradient((mu, sigma) -> sum(x -> logpdf(Normal(mu, sigma), x), [8,9]), 0, 1)

    # See issue https://github.com/FluxML/Zygote.jl/issues/436
    using Distributions
    import Zygote
    import Tracker
    Zygote.gradient((μ, σ) -> loglikelihood(Normal(μ, σ), [1,2,3]), 0, 1)
    # => (nothing, nothing)
    Tracker.gradient((μ, σ) -> loglikelihood(Normal(μ, σ), [1,2,3]), 0, 1)
    # => (6.0 (tracked), 11.0 (tracked))

    Tracker.gradient((mu) -> loglikelihood(Normal(mu, 1), [8, 9]), 0)

    function f(mus, sigmas)
        all = map(mus, sigmas) do mu, sigma
            sum(x -> logpdf(Normal(mu, sigma), x), 8)
        end
        sum(all)
    end
    gradient(f, [0,0], [1,1])
    f([0,0], [1,1])
    foo(mus, sigmas) = sum(logpdf.(Normal.(mus, sigmas), [8,9]))
    foo([0,0], [1,1])
    gradient(foo, [0,0], [1,1])
end


function mdn_nll5(pi_mu_sigma, y)
    pi, mu, sigma = pi_mu_sigma
    # @show size(pi)
    ds = Normal.(mu, sigma)
    # @show size(ds)
    # FIXME
    f(d) = loglikelihood(d, y)
    log_prob_y = f.(ds)
    # @show size(log_prob_y)
    # @show size(pi)
    log_prob_pi_y = log_prob_y + log.(pi)
    loss = mean(log_prob_pi_y)
    # sum(y)
end


struct GMM
    component
    pi
    mu
    sigma
end

function GMM(comp)
    pi = zeros(comp)
    mu = zeros(comp)
    sigma = zeros(comp)
    GMM(comp, pi, mu, sigma)
end

function (m::GMM)(input)
    # FIXME seem only use input's shape, but the shape is the same?
    softmax(m.pi), m.mu, sqrt.(m.sigma)
end

function train_gmm!(model, x)
    for step in 1:1000
        
    end
end


function test()
    param_dist = (mean) -> rand(Normal(mean, 2), (1, 1000))
    param_sampler = () -> rand() * 8 - 4
    p = param_sampler()
    X = param_dist(p)
    
    g = GMM(10, dropdims(X, dims=1), method=:kmeans)
    g
    weights(g)
    means(g)
    covars(g)
    
end

# from https://www.xarg.org/2016/06/the-log-sum-exp-trick-in-machine-learning/
# and https://discourse.julialang.org/t/fast-logsumexp/22827
function logsumexp(w)
    we = similar(w)
    offset = maximum(w)
    we .= exp.(w .- offset)
    s = sum(we)
    log(s) + offset
    # w .-= log(s) + offset
    # we .*= 1/s
end
size(xxx)

StatsFuns.logsumexp(xxx, dims=1)

my_logsumexp(xxx, dims=1)
xxx
xxx .- maximum(xxx, dims=1)
mapslices(sum, ones(3,2), dims=2)
@show mapslices(logsumexp, xxx, dims=1)

ones(3,2) - ones(1,2)

# DEBUG using StatsFuns.logsumexp instead of my own
# loss = -mapslices(logsumexp, log_prob_pi_y, dims=1)

function test_plot_likelihood()
    # collect(0:0.1:2)
    ys = map(0:0.1:2) do z
        likelihood_loss(z)
    end
    p = plot(0:0.1:2, [map((y)->y[1], ys) map((y)->y[2], ys)])
    plot(0:0.1:2, map((y)->y[2], ys));
    y = map((y)->y[2], ys)
    x = collect(0:0.1:2)
    plot(rand(10))
    p = plot(x, y)
    savefig(p, "a.pdf")
end

