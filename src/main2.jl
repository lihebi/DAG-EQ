# g=5.0
# ENV["JULIA_CUDA_MEMORY_LIMIT"] = convert(Int, round(g * 1024 * 1024 * 1024))

include("exp.jl")

function main2()
    for d in [10,20,50],
        gtype in [:SF, :ER],
        mec in [:Linear],
        mat in [:COV],
        k in [1],
        noise in [:Gaussian],
        (prefix, model_fn,nsteps) in [
            ("EQ", deep_eq_model_fn, 1.5e4),
#             ("FC", ()->deep_fc_model_fn(d), 1e5),
            # FIXME the CNN always got killed
#             ("CNN", flat_cnn_model, 3e4)
        ]
        
        spec = DataSpec(d=d,k=k,gtype=gtype,noise=noise,mechanism=mec,mat=mat)
        @info "training" prefix d gtype mec mat k noise
        expID = exp_train(spec, model_fn, prefix=prefix, train_steps=nsteps)
        @info "testing" expID
        exp_test(expID, spec, use_raw=true)
    end
end

# TODO ensemble
function main_ensemble()
    for gtype in [:ER, :SF],
        mec in [:Linear]

        # I'll be training just one EQ model on SF graph with d=10,15,20
        specs = map([10, 20]) do d
            # TODO mixed training of different noise models
            DataSpec(d=d, k=1, gtype=gtype, noise=:Gaussian, mechanism=mec)
        end
        expID = exp_train(specs, deep_eq_model_fn,
                          # TODO I'll need to increase the training steps here
                          # CAUTION feed in the gtype in the model prefix
                          prefix="deep-EQ-$gtype", train_steps=3e4)
        # Testing
        # get specs
        for d in [10,15,20,30],
            gtype in [:ER, :SF],
            k in [1,2,4]

            @info "Testing" expID d gtype k
            spec = DataSpec(d=d, k=k, gtype=gtype, noise=:Gaussian)
            exp_test(expID, spec)
        end
    end
end