function fc_model(d)
    # first reshape
    Chain(
        x -> reshape(x, d * d, :),

        Dense(d*d, 1024, relu),
        # Dropout(0.5),
        # BatchNorm(1024),

        Dense(1024, 1024, relu),
        # Dropout(0.5),
        # BatchNorm(1024),

        Dense(1024, d*d),
        # reshape back
        x -> reshape(x, d, d, :),
    )
end

function fc_model_deep(d)
    # first reshape
    Chain(
        x -> reshape(x, d * d, :),

        Dense(d*d, 1024, relu),
        # Dropout(0.5),
        # BatchNorm(1024),

        Dense(1024, 1024, relu),
        # Dropout(0.5),
        # BatchNorm(1024),
        Dense(1024, 1024, relu),
        # Dropout(0.5),
        # BatchNorm(1024),
        Dense(1024, 1024, relu),
        # Dropout(0.5),
        # BatchNorm(1024),
        Dense(1024, 1024, relu),
        # Dropout(0.5),
        # BatchNorm(1024),

        Dense(1024, d*d),
        # reshape back
        x -> reshape(x, d, d, :),
    )
end

function eq_model(d, z)
    # eq model
    model = Chain(
        x->reshape(x, d, d, 1, :),

        Equivariant(1=>z),
        LeakyReLU(),
        Dropout(0.5),
        # BatchNorm(z),

        Equivariant(z=>z),
        LeakyReLU(),
        Dropout(0.5),
        # BatchNorm(z),

        Equivariant(z=>1),
        x->reshape(x, d, d, :)
    )
end

function eq_model_deep(d, z)
    # eq model
    # TODO put on GPU, I probably need to write GPU kernel for this
    model = Chain(
        x->reshape(x, d, d, 1, :),

        Equivariant(1=>z),
        LeakyReLU(),
        # FIXME whether this is the correct drop
        Dropout(0.5),
        # BatchNorm(z),

        Equivariant(z=>z),
        LeakyReLU(),
        Dropout(0.5),
        # BatchNorm(z),
        Equivariant(z=>z),
        LeakyReLU(),
        Dropout(0.5),
        # BatchNorm(z),
        Equivariant(z=>z),
        LeakyReLU(),
        Dropout(0.5),
        # BatchNorm(z),
        Equivariant(z=>z),
        LeakyReLU(),
        Dropout(0.5),
        # BatchNorm(z),

        Equivariant(z=>1),
        # IMPORTANT drop the second-to-last dim 1
        x->reshape(x, d, d, :)
    )
end
    # TODO enforcing sparsity, and increase the loss weight of 1 edges, because
    # there are much more 0s, and they can take control of loss and make 1s not
    # significant. As an extreme case, the model may simply report 0 everywhere

function test()
    ds, test_ds = gen_sup_ds_cached(ng=1000, N=20, d=5, batch_size=100)
    ds, test_ds |> (x)->convert(CuDataSetIterator, x)
    ds, test_ds |> CuDataSetIterator
    convert(CuDataSetIterator, ds)
    ds |> (x)->convert(CuDataSetIterator, x)
end
function test()
    # detect if the expID already tested
    if !isdir("tensorboard_logs") mkdir("tensorboard_logs") end
    for dir in readdir("tensorboard_logs")
        if occursin(expID, dir)
            @info "Already trained in $dir"
            return
        end
    end
end
