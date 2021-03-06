using Optim
using LinearAlgebra: I
using LineSearches
import NLopt

function loss_fn(X, W)
    n, d = size(X)
    M = X * W
    R = X - M
    loss = 0.5 / n * sum(R .^ 2)
    G_loss = - 1.0 / n * transpose(X) * R
    return loss, G_loss
end


function h_fn(W)
    d, _ = size(W)
    eye = 1 * Matrix(I, d, d)
    M = eye + W .* W / d
    # FIXME E = np.linalg.matrix_power(M, d - 1)
    E = M ^ (d-1)
    h = sum(transpose(E) .* M) - d
    G_h = transpose(E) .* W * 2
    return h, G_h
end

function adj_fn(v)
    # FIXME length or size(v,1)?
    d = convert(Int, sqrt(length(v)/2))
    # NOTE: positive and negative part
    reshape(v[1:d*d] - v[d*d+1:end], d, d)
end


function f(v, X, rho, alpha)
    lambda1 = 0.1
    n, d = size(X)

    W = adj_fn(v)
    # X use 1
    loss, G_loss = loss_fn(X, W)

    # X use 2
    h, G_h = h_fn(W)

    # rho and alpha uses
    obj = loss + 0.5 * rho * h * h + alpha * h + lambda1 * sum(v)
    G_smooth = G_loss + (rho * h + alpha) * G_h

    g_obj = hcat(G_smooth .+ lambda1, - G_smooth .+ lambda1)
    return obj, g_obj
end

# this works
function my_opt_NLopt(_f, _g, lower, upper, w_est)
    d = convert(Int, sqrt(size(w_est, 1) / 2))
    function obj_fn(v, G)
        ret = _f(v)
        g = _g(v)
        copyto!(G, g)
        ret
    end

    opt = NLopt.Opt(:LD_LBFGS, 2*d*d)
    opt.lower_bounds = lower
    opt.upper_bounds = upper
    opt.min_objective = obj_fn

    res = NLopt.optimize(opt, w_est)
    res = res[2]
end

# NOT WORKING
function my_opt_Optim(_f, _g, lower, upper, w_est)
    sol = optimize(_f,
                   _g,
                   # (v)->f(v, X, rho, alpha)[2],
                   # (G, v) -> copyto!(G, f(v, X, rho, alpha)[2]),
                   lower, upper,
                   w_est,
                   # FIXME this does not throw errors, but gives NaN
                   # Fminbox(LBFGS(linesearch=LineSearches.BackTracking())),
                   # FIXME this throw errors
                   Fminbox(LBFGS()),
                   # LBFGS(),
                   inplace=false,
                   Optim.Options())
    Optim.minimizer(sol)
end

# FIXME should I check the exact number matching?
function notears(X, fix_upper=false)
    # max_iter=100
    max_iter=10
    h_tol=1e-8
    rho_max=1e+16
    w_threshold=0.3

    n, d = size(X)
    w_est = zeros(2*d*d)
    rho = 1.0
    alpha = 0.0
    h = Inf

    # bounds
    lower = zeros(2*d*d)
    # DEBUG
    if fix_upper
        upper = [if (i==j || (i==2 && j==1)) 0.1 else Inf end for _ in 1:2 for i in 1:d for j in 1:d]
    else
        upper = [if i==j 0. else Inf end for _ in 1:2 for i in 1:d for j in 1:d]
    end
    # DEBUG setting the direction
    for i in 1:max_iter
        w_new = nothing
        h_new = nothing
        @info "iter $(i) .."
        while rho < rho_max
            # @info "running optim .."
            _f = (v)->f(v, X, rho, alpha)[1]
            _g = (v)->f(v, X, rho, alpha)[2]

            # w_new = my_opt_Optim(_f, _g, lower, upper, w_est)
            w_new = my_opt_NLopt(_f, _g, lower, upper, w_est)
            h_new, _ = h_fn(adj_fn(w_new))

            # VERBOSE
            # @show loss_fn(X, adj_fn(w_new))[1]
            @show h_new

            if h_new > 0.25 * h
                rho *= 10
            else
                break
            end
        end
        w_est, h = w_new, h_new
        alpha += rho * h
        if h <= h_tol || rho >= rho_max
            break
        end
    end
    res = adj_fn(w_est)
    # DEBUG threshold or not, here or out
    # res[abs.(res) .< w_threshold] .= 0
    res
end

function graphical_metrics(W, Ŵ)
    # assuming W is dxd matrix
    d = size(W,1)

    g = DiGraph(W)
    ĝ = DiGraph(Ŵ)

    # this will make it 1 or 1
    mg = adjacency_matrix(g)
    mĝ = adjacency_matrix(ĝ)

    # nnz
    nnz = sum(mĝ)
    # shd: structural hamming distance
    shd = sum(mg .!= mĝ)

    if shd > 0
        @warn "Not same"
        @info "true graph:"
        display(g)
        @info "estimated graph:"
        display(ĝ)
    end

    tp = sum(mĝ[mg .== mĝ] .== 1)
    fp = sum(mĝ[mĝ .== 1] .!= mg[mĝ .== 1])
    tt = sum(mg .== 1)
    ff = sum(mg .== 0)

    prec = tp / (tp + fp)
    recall = tp / tt

    # tpr: true positive rate
    # the intersect of edges / total edges in ĝ
    tpr = tp / tt
    # fpr: false positive rate
    fpr = fp / ff
    # fdr: false discovery rate
    fdr = fp / sum(mĝ .== 1)

    # (:nnz=>nnz, :shd=>shd, :prec=>prec, :recall=>recall, :tpr=>tpr, :fpr=>fpr, :fdr=>fdr)
    (:shd=>shd, :prec=>prec, :recall=>recall)
end
