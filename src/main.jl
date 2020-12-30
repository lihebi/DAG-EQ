# g=5.0
# ENV["JULIA_CUDA_MEMORY_LIMIT"] = convert(Int, round(g * 1024 * 1024 * 1024))

include("exp.jl")

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
            ("CNN", flat_cnn_model, 3e4)
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

function main_ch2()
    for d in [10, 50,100],
        gtype in [:SF, :ER],
        mec in [:Linear],
        # CAUTION using CH2, only eq2 model can be used
        mat in [:CH2],
        k in [1],
        noise in [:Gaussian],
        (prefix, model_fn,nsteps) in [
            ("EQ2", eq2_model_fn, 3e4),
        ]
        
        spec = DataSpec(d=d,k=k,gtype=gtype,noise=noise,mechanism=mec,mat=mat)
        @info "training" prefix d gtype mec mat k noise
        expID = exp_train(spec, model_fn, prefix=prefix, train_steps=nsteps)
        @info "testing" expID
        exp_test(expID, spec)
    end
end

function main_newCOV()
    "The new experiment for COV, this time, ensemble ER and SF."
    for d in [10,20,50,100],
        (prefix, model_fn,nsteps) in [
            ("EQ", deep_eq_model_fn, 3e4),
#             ("FC", ()->deep_fc_model_fn(d), 1e5),
            # FIXME the CNN always got killed
#             ("CNN", flat_cnn_model, 3e4)
        ]
        
        specs = []
        for gtype in [:ER, :SF],
            k in [1]
            push!(specs, DataSpec(d=d, k=k, gtype=gtype,
                    noise=:Gaussian, mat=:COV))
        end
        specs = Array{DataSpec}(specs)

        @info "Ensemble training .."
        expID = exp_train(specs, model_fn,
                          # TODO I'll need to increase the training steps here
                          # CAUTION feed in the gtype in the model prefix
                          prefix="$prefix-ERSF-k1-d=$d", train_steps=nsteps,
                          merge=true)
    end
end


function main_ch3()
    for d in [10,20,50,100],
        (prefix, model_fn,nsteps) in [
            ("EQ2", eq2_model_fn, 3e4),
        ]
        
        specs = []
        for gtype in [:ER, :SF],
            k in [1]
            push!(specs, DataSpec(d=d, k=k, gtype=gtype,
                    noise=:Gaussian, mat=:CH3))
        end
        specs = Array{DataSpec}(specs)

        @info "Ensemble training .."
        expID = exp_train(specs, model_fn,
                          # TODO I'll need to increase the training steps here
                          # CAUTION feed in the gtype in the model prefix
                          prefix="$prefix-ERSF-k1-d=$d", train_steps=nsteps,
                          merge=true)
    end
end

function main_ensemble()
    specs = []
    for d in [10,20],
        gtype in [:ER, :SF],
        k in [1,4,10,20]

        push!(specs, DataSpec(d=d, k=k, gtype=gtype,
                noise=:Gaussian, mat=:maxCOV))
    end
    specs = Array{DataSpec}(specs)
   
    @info "Ensemble training .."
    expID = exp_train(specs, deep_eq_model_fn,
                      # TODO I'll need to increase the training steps here
                      # CAUTION feed in the gtype in the model prefix
                      prefix="ensemEQ-ICLR-1", train_steps=1e5,
                      merge=true)
    
    # test the ensemble model
    # TODO test ensemble data
    # TODO test separate data
    @info "testing .."
    for k in [1],
        d in [10,20,30],
        gtype in [:ER, :SF],
        noise in [:Gaussian],
        mec in [:Linear],
        mat in [:maxCOV]
        
        @info k d gtype noise mec mat
        test_spec = DataSpec(d=d,k=k,gtype=gtype,noise=noise,mechanism=mec,mat=mat)
        exp_test(expID, test_spec)
    end
end

function main_ensemble_new()
    specs = []
    mat = :COR
    for d in [11],
        gtype in [:ER, :SF],
#         k in [1,4,10,20]
        k in 1:2:20

        push!(specs, DataSpec(d=d, k=k, gtype=gtype,
                noise=:Gaussian, mat=mat))
    end
    specs = Array{DataSpec}(specs)
   
    @info "Ensemble training .."
    expID = exp_train(specs, deep_eq_model_fn,
                      # TODO I'll need to increase the training steps here
                      # CAUTION feed in the gtype in the model prefix
                      prefix="ensemEQ-$(mat)-NEW-1:2:20", train_steps=1e5,
                      merge=true)
end

function main_ensemble_ch2()
    specs = []
    for mat in [:CH3],
        d in [20]
        for gtype in [:ER, :SF],
    #         k in [1,4,10,20]
            k in 1:2:20

            push!(specs, DataSpec(d=d, k=k, gtype=gtype,
                    noise=:Gaussian, mat=mat))
        end
        specs = Array{DataSpec}(specs)

        @info "Ensemble training .."
        expID = exp_train(specs, eq2_model_fn,
                          # TODO I'll need to increase the training steps here
                          # CAUTION feed in the gtype in the model prefix
                          prefix="ensemEQ-$(mat)-d=$d-1:2:20", train_steps=1e5,
                          merge=true)
    end
end
