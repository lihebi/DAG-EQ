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
    for d in [10,15,20]
        for k in [1,2,4]
            for gtype in [:ER, :SF]
                for noise in [:Gaussian, :Poisson]
                    create_sup_data(DataSpec(d=d, k=k, gtype=gtype, noise=noise))
                end
            end
        end
    end
end

function main_train_EQ()
    # I'll be training just one EQ model on SF graph with d=10,15,20
    specs = map([10, 15, 20]) do d
        DataSpec(d=d, k=1, gtype=:ER, noise=:Gaussian)
    end
    exp_train(()->mixed_ds_fn(specs),
              deep_eq_model_fn,
              expID="deep-EQ", train_steps=3e4)
    # TODO train on individual graph size, instead of mixed
    # TODO train on SF and test on ER
    # TODO mixed training of different noise models
end


function main_test_EQ()
    # TODO test on different types of data
    exp_test("deep-EQ",
             ()->spec_ds_fn(DataSpec(d=25, k=1, gtype=:ER, noise=:Gaussian)))
    exp_test("deep-EQ",
             ()->spec_ds_fn(DataSpec(d=15, k=1, gtype=:SF, noise=:Poisson)))
end

function main_train_FC()
    # train for d = 10
    for d in [10, 15, 20]
        spec = DataSpec(d=d, k=1, gtype=:ER, noise=:Gaussian)
        exp_train(()->spec_ds_fn(spec),
                  ()->deep_fc_model_fn(d),
                  expID="deep-FC-d=$d", train_steps=1e5)
    end
end

function main_test_FC()
    # FIXME FC performance seems to be really poor, maybe add some regularizations
    for d in [10, 15, 20]
        exp_test("deep-FC-d=$d",
                 ()->spec_ds_fn(DataSpec(d=d, k=1, gtype=:SF, noise=:Poisson)))
    end
end

function main_train_EQ_SF()
    # I'll be training just one EQ model on SF graph with d=10,15,20
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
main_train_EQ_SF()
