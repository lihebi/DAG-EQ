using Distributions
using CausalInference
using LightGraphs

using DelimitedFiles, LinearAlgebra

function test_CI()

    p = 0.01

    # Download data
    run(`wget http://nugget.unisa.edu.au/ParallelPC/data/real/NCI-60.csv`)

    # Read data and compute correlation matrix
    X = readdlm("NCI-60.csv", ',')
    d, n = size(X)
    C = Symmetric(cor(X, dims=2))

    # Compute skeleton `h` and separating sets `S`
    h, S = skeleton(d, gausscitest, (C, n), quantile(Normal(), 1-p/2))

    # Compute the CPDAG `g`
    g = pcalg(d, gausscitest, (C, n), quantile(Normal(), 1-p/2))

end

function test_graph()
    # 1. create random graph
    # 2. generate data
    # 3. learn GAN
    # 4. plugin NOTEARS with interventional loss
end


gradient(()->sum(gen_data_analytical(w, 10)))
gradient(()->begin
         d = size(w, 1)
         Σ = inv(myeye(d) - W) * myeye(d) * inv(transpose(myeye(d) - W))
         sum(my_mvnormal_sample(zeros(d), Σ, 5)')

         # d=size(w,1)
         # Σ = inv(myeye(d) - W) * myeye(d) * inv(transpose(myeye(d) - W))
         # dist = MvNormal(zeros(d), Σ)
         # my_mvnormal_sample(zeros(d), Σ, 5)
         # sum(rand(5))
         end)


randn(3,4,5) * randn(5,1)

randn(4,5) * randn(5,1)


function (l::Equivariant)(input)
    outs = map(1:l.ch[2]) do ch2
        hs = map(1:l.ch[1]) do ch1
            l.λ[ch1,ch2] .* input[:,:,ch1] .+ l.γ[ch1,ch2] * maximum(input[:,:,ch1])
        end
        sum(hs)
    end
    cat(outs..., dims=3)
    # l.λ .* input .+ l.γ * sum(input)
end

function test_gpu()
    outs = map(1:l.ch[2]) do ch2
        hs = map(1:l.ch[1]) do ch1
            l.λ[ch1,ch2] .* input[:,:,ch1] .+ l.γ[ch1,ch2] * maximum(input[:,:,ch1])
        end
        sum(hs)
    end
    cat(outs..., dims=3)
end

function broadcast_eq(eq, X)
    inner_ch2.(eq.λ, eq.γ, X)
end

function inner_ch2(λs, γs, Xs)
    inner_ch1.(λs, γs, Xs)
end

function inner_ch1(λ, γ, X)
    λ .* X .+ γ * maximum(X)
end

function gpu_eq(eq, input)
    A::GPUArray = CuArray()
    gpu_call(input, eq.λ, eq.γ) do state, f, X, λ, γ
        i = linear_index(state)
    	if i <= length(A)
            @inbounds A[i] = f(B[i])
        end
    end
end

function test_equivariant()
    x = randn(4,4)

    eq = Equivariant(1=>4)
    eq.λ[1,2] .* x[:,:,:] .+ eq.γ[1,2] * maximum(x[:,:,:])
    eq(x[:,:,:])
    eq(x)

    c = Chain(Equivariant(1=>4),
              ReLU(),
              Equivariant(4=>2),
              ReLU(),
              Equivariant(2=>1),
              ReLU())
    c(x)

    params(c)
    param_count(c)

    # compute gradients
    gradient(()->sum(c(x)), params(c))
    # train it
end

function (l::Equivariant)(input)
    if ndims(input) == 2
        # CAUTION no batch, no channel, prone to error
        X = reshape(input, size(input)..., 1, 1)
    elseif ndims(input) == 3
        # this is input without batch, we'll add 1 at the end here for output
        # channel broadcasting
        X = reshape(input, size(input)..., 1)
    elseif ndims(input) == 4
        # this is with batch, we'll add 1 before the batch
        X = reshape(input, size(input)[1:end-1]..., 1, size(input)[end])
    else
        error("dimension error: ", ndims(input))
    end
    λ = reshape(l.λ, 1, 1, size(l.λ)...)
    γ = reshape(l.γ, 1, 1, size(l.γ)...)

    # FIXME maximum is problematic on CuArray
    # gradient(()->sum(maximum(gpu(rand(5,5)), dims=2)))
    # gradient(()->sum(maximum(gpu(rand(5,5)), dims=(1,2))))
    # gradient(()->sum(maximum(rand(5,5), dims=(1,2))))
    # M = maximum(X, dims=(3,4))

    # but mean does not seem to work well
    M = mean(X, dims=(3,4))

    dropdims(sum(λ .* X .+ γ .* M, dims=3), dims=3)
end

function (l::Equivariant)(input)
    # if ndims(input) == 2
    #     # CAUTION no batch, no channel, prone to error
    #     X = reshape(input, size(input)..., 1, 1)
    # elseif ndims(input) == 3
    #     # this is input without batch, we'll add 1 at the end here for output
    #     # channel broadcasting
    #     X = reshape(input, size(input)..., 1)
    # elseif ndims(input) == 4
    #     # this is with batch, we'll add 1 before the batch
    #     # X = reshape(input, size(input)[1:end-1]..., size(input)[end])
    #     X = input
    # else
    #     error("dimension error: ", ndims(input))
    # end
    X = input[:,:,:,:]
    # λ = reshape(l.λ, 1, 1, size(l.λ)...)
    # γ = reshape(l.γ, 1, 1, size(l.γ)...)
    # w = reshape(l.w, 1, 1, size(l.w)...)

    one = ones(size(X,1), size(X,2))
    # out = λ .* X +
    #     w .* (one * X + X * one) +
    #     γ .* one * X * one
    # @tensor OUT[a,c,d] = one[a,b] * X[b,c,d]
    # @tensor OUT[a,b] := randn(5,2)[a,b] .+ 1

    λ = l.λ
    γ = l.γ
    w = l.w
    # FIXME @grad
    # FIXME gpu
    @tensor X1[a,b,ch2,batch] := X[a,b,ch1,batch] * λ[ch1,ch2]
    @tensor X2[a,c,ch2,batch] := one[a,b] * X[b,c,ch1,batch] * w[ch1,ch2]
    @tensor X3[a,c,ch2,batch] := X[a,b,ch1,batch] * one[b,c] * w[ch1,ch2]
    @tensor X4[a,d,ch2,batch] := one[a,b] * X[b,c,ch1,batch] * one[c,d] * γ[ch1,ch2]
    X1 + X2 + X3 + X4
end


# DEPRECATED
function gen_sup_data_all(ng, N, d)
    # train data
    ds = @showprogress 0.1 "Generating.." map(1:ng) do i
        g = gen_ER_dag(d)
        x, y = gen_sup_data(g, N)
    end
    input = map(ds) do x x[1] end
    output = map(ds) do x x[2] end
    hcat(input...), hcat(output...)
end

# DEPRECATED
function gen_sup_ds(;ng, N, d, batch_size)
    x, y = gen_sup_data_all(ng, N, d)
    test_x, test_y = gen_sup_data_all(ng, N, d)

    ds = DataSetIterator(x, y, batch_size)
    test_ds = DataSetIterator(test_x, test_y, batch_size)

    ds, test_ds
end



function test_beq()
    # I really should use multi-dim arrays instead of arrays of arrays.  But,
    # that would make the dot-operations much harder if not impossible to write.
    ((x)->σ.(x)).([[1,2,3],[4,5,6]])

    X = rand(5, 5, 1, 10)
    eq = Equivariant(1=>1)

    eqfn(X, eq.λ, eq.w, eq.γ)

    # some random testing
    @tensor gpu(randn(3,3))[i,j] * gpu(randn(3,3))[j,i]
    gradient(()->sum(randn(3,3) * randn(3,3)))
    gradient(()->sum(@tensor randn(3,3)[i,j] * randn(3,3)[j,i]))

    # test the layer
    size(eq(X))

    # test equivariance
    p = Matrix(Int.(Diagonal(ones(5))[randperm(5), :]))
    p * eq(X)[:,:,1,1] * p'
    eq(p * X[:,:,1,1] * p')
    p * cpu(model(X))[:,:,1,1] * p'
    model(gpu(p * cpu(X)[:,:,1,1] * p'))


    # multi dim
    X = randn(5,5,2,100)
    size(X)
    size(Equivariant(2=>4)(X))
    Equivariant(4=>8)(Equivariant(2=>4)(X))

    # test dimension
    Equivariant(1=>1)(rand(5,5)) # should error
    Equivariant(1=>1)(rand(5,5,1)) # output 5,5,1
    Equivariant(1=>1)(rand(5,5,1,100)) # output 5,5,1,100

    # gradient
    Y, back = Zygote.pullback(eqfn, X, eq.λ, eq.w, eq.γ)
    a,b,c,d = back(ones(size(Y)...))

    _, back = Zygote.pullback((a,b,c,d)->sum(eqfn(a,b,c,d)), X, eq.λ, eq.w, eq.γ)
    _, back = Zygote.pullback((a,b,c,d)->sum(eqfn(a,b,c,d)), X, eq.λ, eq.w, eq.γ)
    back(1)
    eqfn_pullback

    gradient(()->sum(eqfn(X, eq.λ, eq.w, eq.γ)))
    sum(eqfn(X, eq.λ, eq.w, eq.γ))

    eq(X)
    sum(eq(X))
    typeof(eq(X))
    gradient(()->sum(eq(X)))
    gradient(()->sum(eq(X)), params(eq))

    gradient((x)->sum(maximum(x, dims=(1,2))), rand(5,5))

    # testing GPU
    gpu(eq)(gpu(X))
end

function test_tensor()
    using Libdl
    Libdl.dlopen("libcuda")
    Libdl.dlopen("libssl")
    Libdl.dlopen("libcutensor")
    using CUDA
    CUDA.has_cutensor()
    using TensorOperations

    using CUDAdrv
    CUDAdrv.functional()
    CUDAdrv.version()

    CUDA.cuda_version()

    Pkg.build("CUDAdrv")


    using CUDAapi
    CUDAapi.has_cuda()
    CUDAapi.has_cudnn()
    CUDAapi.find_toolkit()


    A=randn(5,5,5,5,5,5)
    B=randn(5,5,5)
    C=randn(5,5,5)
    D=zeros(5,5,5)
    @tensor begin
        D[a,b,c] = A[a,e,f,c,f,g]*B[g,b,e] + *C[c,a,b]
        E[a,b,c] := A[a,e,f,c,f,g]*B[g,b,e] + *C[c,a,b]
    end

    @tensor randn(5,5)[i,j] * randn(5,5)[j,i]
    @cutensor randn(5,5)[i,j] * randn(5,5)[j,i]

    @cutensor OUT[a,b] := randn(5,5,5)[a,i,j] * randn(5,5,5)[j,i,b]

    @tensor OUT[a,b,ch2,batch] := cu(rand(5,5,5,5))[a,b,ch1,batch] * cu(rand(5,5))[ch1,ch2]

    @tensor cu(randn(5,5))[i,j] * cu(randn(5,5))[j,i]
    @cutensor cu(randn(5,5))[i,j] * cu(randn(5,5))[j,i]
end

function test_eq()
    Equivariant(1=>2).w
end

function test_xent()
    Flux.crossentropy([0.2, 0.8], [0, 1])

    Flux.crossentropy(softmax([0.2, 0.8]), [0, 1])
    Flux.logitcrossentropy([0.2, 0.8], [0, 1])
    -sum([0, 1] .* logsoftmax([0.2, 0.8]))

    Flux.binarycrossentropy.([0.2, 0.8], [0, 1])

    Flux.binarycrossentropy.(sigmoid.([0.2, 0.8]), [0, 1])
    Flux.logitbinarycrossentropy.([0.2, 0.8], [0, 1])
    Flux.binarycrossentropy.(softmax([0.2, 0.8]), [0, 1])
end

function test()
    ds, test_ds = gen_sup_ds_cached(ng=5e3, N=20, d=5, batch_size=100)
    x, y = next_batch!(test_ds) |> gpu

    x, y = next_batch!(ds)
    x = next_batch!(ds)

    model_fn = (d)->fc_model(d)
    eq_model_fn = (d)->eq_model(d, 300)
    model = model_fn(5) |> gpu
    model = eq_model_fn(5) |> gpu
    param_count(model)

    param_count(model_fn(5))
    param_count(eq_model_fn(5))

    # warm up the model
    model(x)

    @btime gradient(()->myσxent(model(x), y))
    @time gradient(()->myσxent(model(x), y))
    gradient(()->myσxent(model(x), y))

    gradient(()->sum(model(x)), params(model))
    gradient(()->sum(model(x)))
    σ.(model(x))

    model(gpu(randn(5,5,100)))

    # actually training
    opt = ADAM(1e-4)
    print_cb = Flux.throttle(sup_create_print_cb(), 1)
    test_cb = Flux.throttle(sup_create_test_cb(model, test_ds, "test_ds"), 10)

    sup_train!(model, opt, ds, test_ds,
               print_cb=print_cb,
               test_cb=test_cb,
               train_steps=1e4)
end
