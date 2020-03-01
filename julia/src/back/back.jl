function test()
    do_loss_fn(w, x)

    index = rand(1:size(x, 1), size(x, 2))
    delta = zeros(size(x))
    index
    delta[collect(zip(index, 1:100))]


    delta[(1,1)...]=randn()

    hcat(map(collect(zip(index, 1:100))) do p
         [p[1] p[2]]
         end)

    delta[(1,1)...]

    delta[index] .= randn(size(delta, 2))

    index, delta = random_intervention(x)
    post_do = do_effect(w, x, index, delta)
    x + delta
end



function test()
    gradient((X, w)->-mean((X - X * w) .^ 2), X, w)
    opt = ADAM(1e-4)
    ps = Flux.params(w)
    gs = gradient(ps) do
        -mean((X - X * w) .^ 2)
    end
    Flux.Optimise.update!(opt, ps, gs)
end

function test()
    a = randn(5,2,3)
    b = randn(5,2)
    # assert the following equal
    a .+ b
    a + repeat(b[:,:,:], 1, 1, 3)
    # test eqfn
    eq = Equivariant(1=>1)
    a = randn(5,5,1,10)
    reshape(eq(a), 5,5,:)
    eq = eq_model(5, 100)
    σ.(eq(a))
    σ.(cpu(model)(a))
    size(x)
    size(a)

    σ.(reshape(Equivariant(1=>1)(randn(5,5,1,10)),5,5,10))
end



function test()
    # test sum performance
    Profile.clear()
    data = randn(100,100,1000);
    @time sum(data)
    data_on_gpu = cu(data);
    @time sum(data_on_gpu)
    # gradient
    @time gradient(()->sum(data_on_gpu))
    @time gradient(()->sum(cu(data)))
    @time cu(data);

    Profile.clear()
    d1 = gpu(randn(100,100,100000));
    d2 = gpu(ones(100,100,100000));
    @profile gradient(()->sum(Flux.logitbinarycrossentropy.(d1, d2)))
    Profile.print()
    open("/tmp/prof2.txt", "w") do s
        Profile.print(IOContext(s, :displaysize => (24, 200)), mincount=10)
    end
end
