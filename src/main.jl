# g=5.0
# ENV["JULIA_CUDA_MEMORY_LIMIT"] = convert(Int, round(g * 1024 * 1024 * 1024))

include("exp.jl")

function main()
    for d in [10,20,50,100],
        gtype in [:ER, :SF],
        mec in [:Linear],
        mat in [:maxCOV],
        k in [1],
        noise in [:Gaussian],
        (prefix, model_fn,nsteps) in [
            ("EQ", deep_eq_model_fn, 2e4),
            ("FC", ()->deep_fc_model_fn(d), 1e5),
            # FIXME the CNN always got killed
            ("CNN", flat_cnn_model, 3e4)
        ]
        
        spec = DataSpec(d=d,k=k,gtype=gtype,noise=noise,mechanism=mec,mat=mat)
        @info "training" prefix d gtype mec mat k noise
        expID = exp_train(spec, model_fn, prefix=prefix, train_steps=nsteps)
        @info "testing" expID
        exp_test(expID, spec)
    end
end

function main_ensemble()
    specs = []
    for d in [10,15],
        gtype in [:ER, :SF],
        k in [1,4]

        push!(specs, DataSpec(d=d, k=k, gtype=gtype,
                noise=:Gaussian, mat=:medCOV))
    end
    specs = Array{DataSpec}(specs)
   
    @info "Ensemble training .."
    expID = exp_train(specs, deep_eq_model_fn,
                      # TODO I'll need to increase the training steps here
                      # CAUTION feed in the gtype in the model prefix
                      prefix="ensemEQ", train_steps=3e4)
end

main()