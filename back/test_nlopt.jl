import NLopt

function myfunc(x::Vector, grad::Vector)
    if length(grad) > 0
        grad[1] = 0
        grad[2] = 0.5/sqrt(x[2])
    end
    return sqrt(x[2])
end

function myconstraint(x::Vector, grad::Vector, a, b)
    if length(grad) > 0
        grad[1] = 3a * (a*x[1] + b)^2
        grad[2] = -1
    end
    (a*x[1] + b)^3 - x[2]
end

opt = Opt(:LD_MMA, 2)
opt.lower_bounds = [-Inf, 0.]
opt.xtol_rel = 1e-4

opt.min_objective = myfunc
inequality_constraint!(opt, (x,g) -> myconstraint(x,g,2,0), 1e-8)
inequality_constraint!(opt, (x,g) -> myconstraint(x,g,-1,1), 1e-8)

(minf,minx,ret) = optimize(opt, [1.234, 5.678])
numevals = opt.numevals # the number of function evaluations
println("got $minf at $minx after $numevals iterations (returned $ret)")


function one_step(X)
    # testing one step of opt
    n, d = size(X)
    w_est = zeros(d*d)
    rho = 1.0
    alpha = 0.0
    h = Inf
    lower = zeros(d*d)
    upper = [if i==j 0. else Inf end for i in 1:d for j in 1:d]
    _f = (v)->f(v, X, rho, alpha)[1]
    _g = (v)->f(v, X, rho, alpha)[2]

    function obj_fn(v, G)
        ret, g = f(v, X, rho, alpha)
        copyto!(G, g)
        ret
    end

    opt = NLopt.Opt(:LD_LBFGS, d*d)
    opt.lower_bounds = lower
    opt.upper_bounds = upper
    opt.min_objective = obj_fn


    res = NLopt.optimize(opt, w_est)
    res = res[2]

    typeof(res)
    length(res)
    res[1]
    res[2]
    res[3]

    res
end

one_step(X)

function test_optim()
    rosenbrock(x) =  (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
    result = optimize(rosenbrock, zeros(2), BFGS())
end

function one_step(X)
    # testing one step of opt
    n, d = size(X)
    w_est = zeros(d*d)
    rho = 1.0
    alpha = 0.0
    h = Inf
    lower = zeros(d*d)
    upper = [if i==j 0. else Inf end for i in 1:d for j in 1:d]
    _f = (v)->f(v, X, rho, alpha)[1]
    _g = (v)->f(v, X, rho, alpha)[2]
    sol = optimize(_f, _g,
                   lower, upper,
                   w_est,
                   # Fminbox(LBFGS(linesearch=LineSearches.BackTracking())),
                   Fminbox(LBFGS()),
                   # LBFGS(),
                   inplace=false,
                   Optim.Options())
    # @show sol
    w_new = Optim.minimizer(sol)
    w_new[1]
end

one_step(X)
