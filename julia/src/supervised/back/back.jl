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
