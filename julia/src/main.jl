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

function test()
    spec = DataSpec(d=11, k=1, gtype=:SF, noise=:Gaussian)
    create_sup_data(spec)
    load_sup_ds(spec, 100)
    # load the raw data part
    x, y = load_sup_raw(spec)
    size(x)
    size(y)
end

function main_gen_data()
    # Create data to use. I must create this beforehand (not online during
    # training), so that I can completely separate training and testing
    # data. Probably also validation data.

    # TODO create data with different W range
    # TODO create graphs of different types
    for d in [10,15,20,30]
        for k in [1,2,4]
            for gtype in [:ER, :SF]
                for noise in [:Gaussian, :Poisson]
                    create_sup_data(DataSpec(d=d, k=k, gtype=gtype, noise=noise))
                end
            end
        end
    end

    @info "Generating really large graphs .."
    # only for testing
    for d in [50,80]
        for k in [1]
            for gtype in [:SF]
                for noise in [:Gaussian]
                    create_sup_data(DataSpec(d=d, k=k, gtype=gtype, noise=noise, ng=1000, N=1))
                end
            end
        end
    end

    @info "Generating covariate matrix inputs .."
    for d in [10,15,20]
        for k in [1]
            for gtype in [:ER, :SF]
                for noise in [:Gaussian]
                    create_sup_data(DataSpec(d=d, k=k,
                                             gtype=gtype, noise=noise,
                                             mat=:COV))
                end
            end
        end
    end
end

function main_train_EQ()
    function train_ER()
        # I'll be training just one EQ model on SF graph with d=10,15,20
        specs = map([10, 15, 20]) do d
            DataSpec(d=d, k=1, gtype=:ER, noise=:Gaussian)
        end
        exp_train(specs, deep_eq_model_fn,
                  expID="deep-EQ", train_steps=3e4)
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
                  expID="deep-EQ-COV", train_steps=3e4)
    end
    train_ER_COV()
end


function main_test_EQ()
    # TODO test on different types of data
    for expID in ["deep-EQ", "deep-EQ-SF"]
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
            @info "Testing" expID d
            spec = DataSpec(d=d, k=1,
                            gtype=:SF,
                            noise=:Gaussian,
                            ng=1000, N=1)
            exp_test(expID, spec)
        end
    end

    # COV data and models must be run separately
    @info "Testing deep-EQ-COV .."
    spec = DataSpec(d=20, k=1, gtype=:ER, noise=:Gaussian, mat=:COV)
    exp_test("deep-EQ-COV", spec)
end

function main_train_FC()
    for d in [10, 15, 20]
        spec = DataSpec(d=d, k=1, gtype=:ER, noise=:Gaussian)
        exp_train(spec,
                  ()->deep_fc_model_fn(d),
                  expID="deep-FC-d=$d", train_steps=1e5)
    end
end


function main_test_FC()
    # FIXME FC performance seems to be really poor, maybe add some regularizations
    for d in [10, 15, 20]
        @info "Testing FC with d=$d .."
        exp_test("deep-FC-d=$d",
                 DataSpec(d=d, k=1, gtype=:SF, noise=:Poisson))
    end
end

function test()
    # DEBUG
    exp_train(DataSpec(d=10, k=1, gtype=:ER, noise=:Gaussian),
              cnn_model,
              # eq_model,
              expID="CNN-$(now())", train_steps=3e4)
end

    specs = map([10, 15, 20]) do d
        # Train on :SF
        DataSpec(d=d, k=1, gtype=:SF, noise=:Gaussian)
    end
    exp_train(()->mixed_ds_fn(specs),
              deep_eq_model_fn,
              expID="deep-EQ-SF", train_steps=3e4)
end


main_gen_data()
main_train_EQ()
main_test_EQ()
main_train_FC()
main_test_FC()
