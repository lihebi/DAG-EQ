# g=5.0
# ENV["JULIA_CUDA_MEMORY_LIMIT"] = convert(Int, round(g * 1024 * 1024 * 1024))

include("exp.jl")


function main_ch3()
    for d in [10,20,50],
        (prefix, model_fn,nsteps) in [
            ("EQ2", eq2_model_fn, 3e4),
            ("FC", ()->fc_model(d=d, ch=2, z=1024, nlayer=6), 3e4),
            ("FCreg", ()->fc_model(d=d, ch=2, z=1024, nlayer=6, reg=true), 3e4),
            ("CNN", ()->cnn_model(2), 3e4)
        ]
        
        specs = []
        for gtype in [:ER, :SF],
            k in [1]
            push!(specs, DataSpec(d=d, k=k, gtype=gtype,
                    noise=:Gaussian, mat=:CH3))
        end
        specs = Array{DataSpec}(specs)
        
        # print more frequently for CNN and FC to get more data to print
        test_throttle = if prefix == "EQ2" 10 else 1 end

        @info "training .." prefix d
        expID = exp_train(specs, model_fn,
                          # TODO I'll need to increase the training steps here
                          # CAUTION feed in the gtype in the model prefix
                          prefix="$prefix-ERSF-k1-d=$d", train_steps=nsteps,
                          test_throttle = test_throttle,
                          merge=true)
    end
end


function main_mat()
    # test the different matrix, including: CH3, COV, COR
    for d in [20],
        (mat, model_fn) in [
            (:CH3, eq2_model_fn),
            (:COV, eq_model_fn),
            (:COR, eq_model_fn)]
        
        specs = []
        for gtype in [:ER, :SF],
            k in [1,2,4,8]
            push!(specs, DataSpec(d=d, k=k, gtype=gtype,
                    noise=:Gaussian, mat=mat))
        end
        specs = Array{DataSpec}(specs)
        
        @info "training .." d mat
        expID = exp_train(specs, eq2_model_fn,
                          # TODO I'll need to increase the training steps here
                          # CAUTION feed in the gtype in the model prefix
                          prefix="EQ-d=$d-mat=$mat", train_steps=3e4,
                          merge=true)
    end
end



function main()
    for d in [10, 50,100],
        gtype in [:SF, :ER],
        mec in [:Linear],
        mat in [:COV],
        # CAUTION using CH2, only eq2 model can be used
#         mat in [:CH2],
        k in [1],
        noise in [:Gaussian],
        (prefix, model_fn,nsteps) in [
            ("EQ", deep_eq_model_fn, 5e4),
            ("FC", ()->deep_fc_model_fn(d), 1e5),
            # FIXME the CNN always got killed
            ("CNN", cnn_model, 3e4)
        ]
        
#         prefix = "prefix-$(now())"
        
        spec = DataSpec(d=d,k=k,gtype=gtype,noise=noise,mechanism=mec,mat=mat)
        @info "training" prefix d gtype mec mat k noise
        expID = exp_train(spec, model_fn, prefix=prefix, train_steps=nsteps)
        @info "testing" expID
        exp_test(expID, spec)
        # testing other specs
        # 1. transfer with k
            @info "testing differet k .."
        for k in [1,2,4]
            test_spec = DataSpec(d=d,k=k,gtype=gtype,noise=noise,mechanism=mec,mat=mat)
            exp_test(expID, test_spec)
        end
        @info "testing different noise .."
        # 2. transfer with noise
        for noise in [:Poisson, :Exp, :Gumbel]
            test_spec = DataSpec(d=d,k=k,gtype=gtype,noise=noise,mechanism=mec,mat=mat)
            exp_test(expID, test_spec)
        end
        @info "testing different gtype .."
        # 3. transfer with gtype
        for gtype in [:ER, :SF]
            test_spec = DataSpec(d=d,k=k,gtype=gtype,noise=noise,mechanism=mec,mat=mat)
            exp_test(expID, test_spec)
        end
    end
end