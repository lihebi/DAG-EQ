include("model.jl")

struct Equivariant{T}
    w1::AbstractArray{T}
    w2::AbstractArray{T}
    w3::AbstractArray{T}
    w4::AbstractArray{T}
    w5::AbstractArray{T}
    # FIXME how Flux decides ch is not a trainable parameter?
    ch::Pair{<:Integer,<:Integer}
end
Flux.@functor Equivariant


mynfan(dims...) = prod(dims[1:end-2]) .* (dims[end-1], dims[end])
my_glorot_uniform(dims...) = (rand(Float32, dims...) .- 0.5f0) .* sqrt(24.0f0 / sum(mynfan(dims...)))

function Equivariant(ch::Pair{<:Integer,<:Integer})
    # FIXME NOW init function
    return Equivariant(
        my_glorot_uniform(ch[1],ch[2]),
        my_glorot_uniform(ch[1],ch[2]),
        my_glorot_uniform(ch[1],ch[2]),
        my_glorot_uniform(ch[1],ch[2]),
        Float32.(zeros(ch[2])),
        ch)
end

"""
Y = w1 X + w2 11ᵀ X + w3 X 11ᵀ + w4 11ᵀ X 11ᵀ + w5 11ᵀ
"""
function eqfn(X::AbstractArray,
              w1::AbstractArray, w2::AbstractArray, w3::AbstractArray,
              w4::AbstractArray, w5::AbstractArray)
    d = size(X,1)
    if typeof(X) <: CuArray
        # FIXME performance hell!!!
        one = CuArrays.ones(d,d)
    else
        # support CPU as well
        # DEPRECATED convert to Float32, when moving this to GPU
        one = ones(Float32, d, d)
    end
    # FIXME CUTENSOR_STATUS_ARCH_MISMATCH error: cutensor only supports RTX20
    # series (with compute capability 7.0+) see:
    # https://developer.nvidia.com/cuda-gpus
    @tensor X1[a,b,ch2,batch] := X[a,b,ch1,batch] * w1[ch1,ch2]

    # TODO how to do max pooling?
    @tensor X2[a,c,ch2,batch] := one[a,b] * X[b,c,ch1,batch] * w2[ch1,ch2]
    @tensor X3[a,c,ch2,batch] := X[a,b,ch1,batch] * one[b,c] * w3[ch1,ch2]

    @tensor X4[a,d,ch2,batch] := one[a,b] * X[b,c,ch1,batch] * one[c,d] * w4[ch1,ch2]
    @tensor X5[a,b,ch2] := one[a,b] * w5[ch2]
    # FIXME broadcasting X5
    Y = X1 + (X2 ./ d) + (X3 ./ d) + (X4 ./ (d * d)) .+ X5
end


# from https://github.com/mcabbott/TensorGrad.jl
Zygote.@adjoint function eqfn(X::AbstractArray,
                              w1::AbstractArray, w2::AbstractArray, w3::AbstractArray,
                              w4::AbstractArray, w5::AbstractArray)
    d = size(X,1)
    if typeof(X) <: CuArray
        # FIXME performance hell!!!
        one = CuArrays.ones(d,d)
    else
        one = ones(Float32, d, d)
    end
    eqfn(X, w1, w2, w3, w4, w5), function (ΔY)
        # ΔY is FillArray.Fill, and this is not handled in @tensor. Convert it
        # to normal array here. FIXME performance
        # ΔY = Array(ΔY)
        #
        # FIXME However, this will change CuArray to regular Array. Removing
        # this would work for CuArray, but not on CPU. I probably won't
        # calculate gradient on CPU anyway.
        @tensor ΔX1[a,b,ch1,batch] := ΔY[a,b,ch2,batch] * w1[ch1,ch2]

        @tensor ΔX2[a,c,ch1,batch] := one[a,b] * ΔY[b,c,ch2,batch] * w2[ch1,ch2]
        @tensor ΔX3[a,c,ch1,batch] := ΔY[a,b,ch2,batch] * one[b,c] * w3[ch1,ch2]
        @tensor ΔX4[a,d,ch1,batch] := one[a,b] * ΔY[b,c,ch2,batch] * one[c,d] * w4[ch1,ch2]

        # DEBUG new formula for ΔX[2:4]
        # @tensor ΔX2[a,b,ch1,batch] := ΔY[a,b,ch2,batch] * w[ch1,ch2]
        # @tensor ΔX3[a,b,ch1,batch] := ΔY[a,b,ch2,batch] * w[ch1,ch2]
        # @tensor ΔX4[a,b,ch1,batch] := ΔY[a,b,ch2,batch] * γ[ch1,ch2]
        # ΔX5 is 0

        # DEBUG even newer NOT WORKING
        # delta = w1 .+ w2 .+ w3 .+ w4 .+ reshape(w5, 1,:)
        # @tensor ΔX[a,b,ch1,batch] := ΔY[a,b,ch2,batch] * delta[ch1,ch2]

        @tensor Δw1[ch1,ch2] := X[a,b,ch1,batch] * ΔY[a,b,ch2,batch]
        @tensor Δw2[ch1,ch2] := one[a,b] * X[b,c,ch1,batch] * ΔY[a,c,ch2,batch]
        @tensor Δw3[ch1,ch2] := X[a,b,ch1,batch] * one[b,c] * ΔY[a,c,ch2,batch]
        @tensor Δw4[ch1,ch2] := one[a,b] * X[b,c,ch1,batch] * one[c,d] * ΔY[a,d,ch2,batch]
        # FIXME should I normalize Δb?
        @tensor Δw5[ch2,batch] := one[a,b] * ΔY[a,b,ch2,batch]
        # FIXME can I just do a mean here?
        Δw5 = dropdims(mean(Δw5, dims=2), dims=2)

        return (
            ΔX1
            + ΔX2 ./ d + ΔX3 ./ d + ΔX4 ./ (d*d),
            # + ΔX2 + ΔX3 + ΔX4,
            # ΔX,
            Δw1,
            Δw2 ./ d,
            Δw3 ./ d,
            Δw4 ./ (d*d),
            Δw5)
    end
end

function (l::Equivariant)(X)
    # X = X[:,:,:,:]
    eqfn(X, l.w1, l.w2, l.w3, l.w4, l.w5)
end


function sup_model(d)
    # first reshape
    Chain(
        x -> reshape(x, d * d, :),

        Dense(d*d, 1024, relu),
        Dropout(0.5),
        # BatchNorm(1024),

        Dense(1024, 1024, relu),
        Dropout(0.5),
        # BatchNorm(1024),

        Dense(1024, d*d),
        # reshape back
        x -> reshape(x, d, d, :),
    )
end

function eq_model(d, z)
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

        Equivariant(z=>1),
        # IMPORTANT drop the second-to-last dim 1
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
