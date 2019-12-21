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
