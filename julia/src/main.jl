include("exp.jl")

import Printf

function test_size()
    # model size
    @info "FC model"
    for d in [7, 10,15,20,25,30]
        Printf.@printf "%.2f\n" param_count(fc_model_fn(d)) / 1e6
    end
    @info "FC deep model"
    for d in [7, 10,15,20,25,30]
        Printf.@printf "%.2f\n" param_count(deep_fc_model_fn(d)) / 1e6
    end
    # EQ models is independent of input size
    @info "EQ model"
    Printf.@printf "%.2f\n" param_count(eq_model_fn(10)) / 1e6
    Printf.@printf "%.2f\n" param_count(deep_eq_model_fn(10)) / 1e6
end

function main_EQ_sep()
    # seperate training
    # TODO testing code for these settings
    # CAUTION this will be super slow. That's 10hour * 6
    for d in [10,15,20]
        for gtype in [:ER, :SF]
            # UPDATE the expID is set to modelID-dataID
            expID = exp_train(DataSpec(d=d, k=1, gtype=gtype, noise=:Gaussian),
                              deep_eq_model_fn,
                              prefix="deep-EQ", train_steps=3e4)
            # testing config
            for k in [1, 2, 4]
                # This loop is for testing gtype
                # CAUTION I'm using the same variable name
                for gtype in [:ER, :SF]
                    @info "Testing" expID d k gtype
                    exp_test(expID,
                             DataSpec(d=d, k=k, gtype=gtype, noise=:Gaussian))
                    exp_test(expID,
                             DataSpec(d=d, k=k, gtype=gtype, noise=:Gaussian))
                end
            end
        end
    end
end

function main_EQ_ensemble()
    # ensemble training
    for gtype in [:ER, :SF]
        # I'll be training just one EQ model on SF graph with d=10,15,20
        specs = map([10, 15, 20]) do d
            # TODO mixed training of different noise models
            DataSpec(d=d, k=1, gtype=gtype, noise=:Gaussian)
        end
        expID = exp_train(specs, deep_eq_model_fn,
                          # TODO I'll need to increase the training steps here
                          # CAUTION feed in the gtype in the model prefix
                          prefix="deep-EQ-$gtype", train_steps=3e4)
        # Testing
        # get specs
        for d in [10,15,20,30]
            for gtype in [:ER, :SF]
                for k in [1,2,4]
                    @info "Testing" expID d gtype k
                    spec = DataSpec(d=d, k=k, gtype=gtype, noise=:Gaussian)
                    exp_test(expID, spec)
                end
            end
        end
        # really large graphs have lower ng and N
        for d in [50, 80]
            for gtype in [:ER, :SF]
                @info "Testing" expID d gtype
                spec = DataSpec(d=d, k=1,
                                gtype=gtype,
                                noise=:Gaussian,
                                ng=1000, N=1)
                exp_test(expID, spec)
            end
        end
    end
end

function main_EQ_cov()
    # TODO OVERNIGHT add :SF
    for gtype in [:ER]
        # train on COV data
        # I'll be training just one EQ model on SF graph with d=10,15,20
        specs = map([10, 15, 20]) do d
            DataSpec(d=d, k=1, gtype=gtype, noise=:Gaussian, mat=:COV)
        end
        expID = exp_train(specs, deep_eq_model_fn,
                          prefix="deep-EQ-$gtype-COV", train_steps=3e4)
        # COV data and models must be run separately
        # TODO different k
        for d in [10, 15, 20, 30]
            for gtype in [:ER, :SF]
                exp_test(expID,
                         DataSpec(d=d, k=1, gtype=gtype, noise=:Gaussian, mat=:COV))
            end
        end
    end
end

function main_FC()
    # FIXME FC performance seems to be really poor, maybe add some regularizations
    # using COV
    for d in [10, 15, 20]
        for gtype in [:ER, :SF]
            expID = exp_train(DataSpec(d=d, k=1, gtype=gtype, noise=:Gaussian),
                              ()->deep_fc_model_fn(d),
                              prefix="deep-FC", train_steps=1e5)
            # testing ..
            for k in [1, 2, 4]
                for gtype in [:ER, :SF]
                    @info "Testing" d k gtype
                    exp_test(expID,
                             DataSpec(d=d, k=k, gtype=gtype, noise=:Gaussian))
                    exp_test(expID,
                             DataSpec(d=d, k=k, gtype=gtype, noise=:Gaussian))
                end
            end
        end
    end
end

function main_FC_cov()
    for d in [10, 15, 20]
        for gtype in [:ER, :SF]
            expID = exp_train(DataSpec(d=d, k=1, gtype=gtype,
                                       noise=:Gaussian, mat=:COV),
                              ()->deep_fc_model_fn(d),
                              prefix="deep-FC", train_steps=1e5)
            exp_test(expID,
                     DataSpec(d=d, k=1, gtype=gtype, noise=:Gaussian, mat=:COV))
        end
    end
end

function main_CNN_sep()
    # FIXME use 8, 16, 32 in other models to keep consistent with CNN models?
    for d in [8,16,32]
        for gtype in [:ER, :SF]
            expID1 = exp_train(DataSpec(d=d, k=1, gtype=gtype, noise=:Gaussian),
                              bottleneck_cnn_model,
                              prefix="bottle-CNN", train_steps=1e5)

            expID2 = exp_train(DataSpec(d=d, k=1, gtype=gtype, noise=:Gaussian),
                              flat_cnn_model,
                              prefix="flat-CNN", train_steps=1e5)

            for gtype in [:ER, :SF]
                exp_test(expID1,
                         DataSpec(d=d, k=1, gtype=gtype, noise=:Gaussian))
                exp_test(expID2,
                         DataSpec(d=d, k=1, gtype=gtype, noise=:Gaussian))
            end
        end
    end
end

function main_CNN_ensemble()
    # mix training
    for gtype in [:ER, :SF]
        specs = map([8,16,32]) do d
            DataSpec(d=d, k=1, gtype=gtype, noise=:Gaussian)
        end
        expID1 = exp_train(specs,
                           bottleneck_cnn_model,
                           prefix="bottle-CNN-$gtype", train_steps=1e5)
        expID2 = exp_train(specs,
                           flat_cnn_model,
                           prefix="flat-CNN-gtype", train_steps=1e5)
        # testing
        # FIXME 64 might be too large
        for d in [8,16,32]
            for gtype in [:ER, :SF]
                for k in [1,2,4]
                    spec = DataSpec(d=d, k=k, gtype=gtype, noise=:Gaussian)
                    exp_test(expID1, spec)
                    exp_test(expID2, spec)
                end
            end
        end
    end
end

function main()
    # TODO OVERNIGHT
    # main_EQ_sep()
    main_EQ_ensemble()
    main_EQ_cov()

    main_FC()
    main_FC_cov()

    main_CNN_sep()
    main_CNN_ensemble()
end

main()
