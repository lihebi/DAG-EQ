include("exp.jl")

function main2()
    for d in [10,20],
        gtype in [:SF, :ER],
        mec in [:Linear],
        mat in [:COR, :COV],
        k in [1],
        noise in [:Gaussian],
        (prefix, model_fn,nsteps) in [
            ("EQ", deep_eq_model_fn, 1.5e4),
            ("FC", ()->deep_fc_model_fn(d), 1e5),
            ("CNN", flat_cnn_model, 3e4)
        ]
        
        spec = DataSpec(d=d,k=k,gtype=gtype,noise=noise,mechanism=mec,mat=mat)
        @info "training" prefix d gtype mec mat k noise
        expID = exp_train(spec, model_fn, prefix=prefix, train_steps=nsteps)
        @info "testing" expID
        exp_test(expID, spec, use_raw=true)
    end
end



function main2_lite()
    for d in [10,20],
        gtype in [:SF, :ER],
        mec in [:Linear],
        mat in [:COR, :COV],
        k in [1],
        noise in [:Gaussian],
        (prefix, model_fn,nsteps) in [
            ("lite-EQ", deep_eq_model_fn, 1e3),
            ("lite-FC", ()->deep_fc_model_fn(d), 1e3),
            ("lite-CNN", flat_cnn_model, 1e3)
        ]
        
        spec = DataSpec(d=d,k=k,gtype=gtype,noise=noise,mechanism=mec,mat=mat)
        @info "training" prefix d gtype mec mat k noise
        expID = exp_train(spec, model_fn, prefix=prefix, train_steps=nsteps)
        @info "testing" expID
        exp_test(expID, spec, use_raw=true)
    end
end