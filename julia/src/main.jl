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

function main_train_EQ()
    function train_ER()
        # I'll be training just one EQ model on SF graph with d=10,15,20
        specs = map([10, 15, 20]) do d
            DataSpec(d=d, k=1, gtype=:ER, noise=:Gaussian)
        end
        exp_train(specs, deep_eq_model_fn,
                  expID="deep-EQ-ER", train_steps=3e4)
    end
    train_ER()
    # TODO train on individual graph size, instead of mixed

    # train on SF and test on ER
    function train_SF()
        specs = map([10, 15, 20]) do d
            # Train on :SF
            DataSpec(d=d, k=1, gtype=:SF, noise=:Gaussian)
        end
        exp_train(specs, deep_eq_model_fn,
                  expID="deep-EQ-SF", train_steps=3e4)
    end
    train_SF()

    # TODO mixed training of different noise models

    # train on COV data
    function train_ER_COV()
        # I'll be training just one EQ model on SF graph with d=10,15,20
        specs = map([10, 15, 20]) do d
            DataSpec(d=d, k=1, gtype=:ER, noise=:Gaussian, mat=:COV)
        end
        exp_train(specs, deep_eq_model_fn,
                  expID="deep-EQ-ER-COV", train_steps=3e4)
    end
    train_ER_COV()
end


function main_test_EQ()
    # TODO test on different types of data
    for expID in ["deep-EQ-ER", "deep-EQ-SF"]
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

    # COV data and models must be run separately
    @info "Testing deep-EQ-ER-COV .."
    # TODO different k
    for k in [10, 15, 20, 30]
        for gtype in [:ER, :SF]
            exp_test("deep-EQ-ER-COV",
                     DataSpec(d=20, k=1, gtype=:ER, noise=:Gaussian, mat=:COV))
        end
    end
end

function main_train_FC()
    for d in [10, 15, 20]
        for gtype in [:ER, :SF]
            spec = DataSpec(d=d, k=1, gtype=gtype, noise=:Gaussian)
            exp_train(spec,
                      ()->deep_fc_model_fn(d),
                      expID="deep-FC-$gtype-d=$d", train_steps=1e5)
        end
    end
    for d in [10, 15, 20]
        for gtype in [:ER, :SF]
            spec = DataSpec(d=d, k=1, gtype=gtype, noise=:Gaussian, mat=:COV)
            exp_train(spec,
                      ()->deep_fc_model_fn(d),
                      expID="deep-FC-$gtype-COV-d=$d", train_steps=1e5)
        end
    end
end

function main_test_FC()
    # FIXME FC performance seems to be really poor, maybe add some regularizations
    for d in [10, 15, 20]
        for gtype in [:ER, :SF]
            for k in [1, 2, 4]
                @info "Testing" d k gtype
                exp_test("deep-FC-ER-d=$d",
                         DataSpec(d=d, k=k, gtype=gtype, noise=:Gaussian))
                exp_test("deep-FC-SF-d=$d",
                         DataSpec(d=d, k=k, gtype=gtype, noise=:Gaussian))
            end
        end
    end
    # using COV
    for d in [10, 15, 20]
        for gtype in [:ER, :SF]
            exp_test("deep-FC-$gtype-COV-d=$d",
                     DataSpec(d=d, k=1, gtype=gtype, noise=:Gaussian, mat=:COV))
        end
    end
end

function test()
    # DEBUG
    exp_train(DataSpec(d=10, k=1, gtype=:ER, noise=:Gaussian),
              cnn_model,
              # eq_model,
              expID="CNN-$(now())", train_steps=3e4)
end

function main_train_CNN()
    # first, train separate models
    for d in [8,16,32]
        exp_train(DataSpec(d=d, k=1, gtype=:ER, noise=:Gaussian),
                  bottleneck_cnn_model,
                  expID="bottle-CNN-d=$d", train_steps=1e5)
        exp_train(DataSpec(d=d, k=1, gtype=:ER, noise=:Gaussian),
                  flat_cnn_model,
                  expID="flat-CNN-d=$d", train_steps=1e5)
    end
    # mix training
    specs = map([8,16,32]) do d
        DataSpec(d=d, k=1, gtype=:ER, noise=:Gaussian)
    end
    exp_train(specs,
              bottle_cnn_model,
              expID="bottle-CNN", train_steps=1e5)
    exp_train(specs,
              flat_cnn_model,
              expID="flat-CNN", train_steps=1e5)
end

function main_test_CNN()
    for d in [8,16,32]
        for gtype in [:ER, :SF]
            exp_test("bottle-CNN-d=$d",
                     DataSpec(d=d, k=1, gtype=gtype, noise=:Gaussian))
            exp_test("flat-CNN-d=$d",
                     DataSpec(d=d, k=1, gtype=gtype, noise=:Gaussian))
        end
    end
end

main_train_EQ()
main_test_EQ()

main_train_FC()
main_test_FC()

main_train_CNN()
# main_test_CNN()
