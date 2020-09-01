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
    for d in [10, 20,
              # DEBUG see how long it takes for training d=50
#               50, 100
              ],
        # I'm putting SF first because they tends to perform better
        gtype in [:SF,
                  # Anyhow I still need to show ER graphs
                  :ER,
                  # :SF2, :SF4,
                  # :ER2, :ER4
                  ],
        mec in [:Linear,
                # :MLP
                ]

        @info "Training" d gtype mec
        # UPDATE the expID is set to modelID-dataID
        expID = exp_train(DataSpec(d=d, k=1,
                                   gtype=gtype,
                                   noise=:Gaussian,
                                   mechanism=mec),
                          deep_eq_model_fn,
                          prefix="deep-EQ",
                          # FIXME I probably want to train more steps for this
                          train_steps=1.5e4)
        # testing config
        for k in [1,
                  # 2,4
                  ],
            # This loop is for testing gtype
            # CAUTION I'm using the same variable name
            # gtype in [:ER, :SF],
            noise in [:Gaussian,
                      # :Gumbel, :Exp, :Poisson
                      ]

            @info "Testing" expID d k gtype
            exp_test(expID,
                     DataSpec(d=d, k=k, gtype=gtype, noise=noise, mechanism=mec),
                     use_raw=true)
        end
    end
end

function test()
    spec = DataSpec(d=50, k=1,
                    gtype=:SF,
                    noise=:Gaussian)
    ds, test_ds = spec2ds(spec, batch_size=32)
    x, y = next_batch!(test_ds) |> gpu
    model = deep_eq_model_fn() |> gpu
    model(x)
    gradient(()->sum(model(x)), Flux.params(model))
end

function main_ultra_large_graphs()
    gtype = :SF
    mec = :Linear
    train_spec = DataSpec(d=100, k=1, gtype=gtype, noise=:Gaussian, mechanism=mec)
    expID = exp_train(train_spec, deep_eq_model_fn, prefix="deep-EQ", train_steps=3e4)
    for d in [100]
        spec = DataSpec(d=d, k=1, gtype=gtype, noise=:Gaussian, mechanism=mec)
        exp_test(expID, spec)
    end
end

function main_EQ_ensemble()
    # ensemble training
    for gtype in [:ER, :SF,
                  # DEBUG
                  # :ER2, :SF2
                  ],
        mec in [:Linear,
                # :MLP
                ]

        # I'll be training just one EQ model on SF graph with d=10,15,20
        specs = map([10, 15, 20]) do d
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
        # really large graphs have lower ng and N
        for d in [50, 80],
            gtype in [:ER, :SF]

            @info "Testing" expID d gtype
            spec = DataSpec(d=d, k=1,
                            gtype=gtype,
                            noise=:Gaussian,
                            ng=1000, N=1)
            exp_test(expID, spec)
        end
    end
end

function main_EQ_super_ensemble()
    # ensemble training
    for gtype in [:ER, :SF,
                  :ER2, :SF4],
        mec in [:Linear,
                # :MLP
                ],
        k in [1,2,4]

        # I'll be training just one EQ model on SF graph with d=10,15,20
        specs = map([10, 20, 50, 100]) do d
            DataSpec(d=d, k=1, gtype=gtype, noise=:Gaussian, mechanism=mec)
        end
        expID = exp_train(specs, deep_eq_model_fn,
                          prefix="deep-EQ-super-$gtype", train_steps=3e4)
        # Testing
        # get specs
        for d in [15,30,80],
            gtype in [:ER, :SF],
            k in [1,2,4]

            @info "Testing" expID d gtype k
            spec = DataSpec(d=d, k=k, gtype=gtype, noise=:Gaussian)
            exp_test(expID, spec)
        end
    end
end

function main_EQ_cov()
    # TODO OVERNIGHT add :SF
    for gtype in [:ER, :SF]
        # train on COV data
        # I'll be training just one EQ model on SF graph with d=10,15,20
        specs = map([10, 15, 20]) do d
            DataSpec(d=d, k=1, gtype=gtype, noise=:Gaussian, mat=:COV)
        end
        expID = exp_train(specs, deep_eq_model_fn,
                          prefix="deep-EQ-$gtype-COV", train_steps=3e4)
        # COV data and models must be run separately
        # TODO different k
        for d in [10, 15, 20, 30],
            gtype in [:ER, :SF]

            exp_test(expID,
                     DataSpec(d=d, k=1, gtype=gtype, noise=:Gaussian, mat=:COV))
        end
    end
end

function test()
    spec = DataSpec(d=10, k=1, gtype=:ER2, noise=:Gaussian, mechanism=:Linear)
    ds, test_ds = load_sup_ds(spec)
    x, y = next_batch!(ds)
    size(x)
    size(y)
end

function main_FC()
    # FIXME FC performance seems to be really poor, maybe add some regularizations
    # using COV
    for d in [10, 15, 20],
        gtype in [:ER, :SF,
                  # :ER2, :SF2,
                  # :ER4, :SF4
                  ],
        mec in [:Linear, :MLP]

        @info "Training " d gtype mec
        expID = exp_train(DataSpec(d=d, k=1,
                                   gtype=gtype, noise=:Gaussian,
                                   mechanism=mec),
                          ()->deep_fc_model_fn(d),
                          prefix="deep-FC", train_steps=1e5)
        # testing ..
        for k in [1, 2, 4],
            gtype in [:ER, :SF]

            @info "Testing" d k gtype
            exp_test(expID,
                     DataSpec(d=d, k=k, gtype=gtype, noise=:Gaussian))
            exp_test(expID,
                     DataSpec(d=d, k=k, gtype=gtype, noise=:Gaussian))
        end
    end
end

function main_FC_cov()
    for d in [10, 15, 20],
        gtype in [:ER, :SF]

        expID = exp_train(DataSpec(d=d, k=1, gtype=gtype,
                                   noise=:Gaussian, mat=:COV),
                          ()->deep_fc_model_fn(d),
                          prefix="deep-FC", train_steps=5e4)
        exp_test(expID,
                 DataSpec(d=d, k=1, gtype=gtype, noise=:Gaussian, mat=:COV))
    end
end

# TODO the GPU usage is too low for this small CNN
function main_CNN_sep()
    # FIXME use 8, 16, 32 in other models to keep consistent with CNN models?
    for d in [
        # 8,16,32,
        10, 20, 
#             50, 100
    ],
        gtype in [
            # :ER,
            :SF,
                  # :ER2, :SF2,
                  # :ER4, :SF4
                  ],
        mec in [:Linear,
                # :MLP
                ]

        spec = DataSpec(d=d, k=1,
                        gtype=gtype,
                        noise=:Gaussian,
                        mechanism=mec)

        # expID1 = exp_train(spec,
        #                    bottleneck_cnn_model,
        #                    prefix="bottle-CNN", train_steps=3e4)

        expID2 = exp_train(spec,
                           flat_cnn_model,
                           prefix="flat-CNN", train_steps=3e4)

        for gtype in [:ER, :SF],
            expID in [
                # expID1,
                expID2]

            exp_test(expID,
                     DataSpec(d=d, k=1, gtype=gtype, noise=:Gaussian, mechanism=mec))
        end
    end
end

function main_CNN_ensemble()
    # mix training
    for gtype in [:ER, :SF],
        mec in [:Linear, :MLP]

        specs = map([8,16,32]) do d
            DataSpec(d=d, k=1, gtype=gtype, noise=:Gaussian, mechanism=mec)
        end
        expID1 = exp_train(specs,
                           bottleneck_cnn_model,
                           prefix="bottle-CNN-$gtype", train_steps=1e5)
        expID2 = exp_train(specs,
                           flat_cnn_model,
                           prefix="flat-CNN-$gtype", train_steps=1e5)
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

function test()
    # generating data
    for d in [8, 16, 32, 10, 15],
        gtype in [:ER, :SF],
        mat in [:COR, :COV],
        k in [1,2,4],
        # TODO :Gumbel
        noise in [:Gaussian, :Exp]

        load_sup_ds(DataSpec(d=d, k=k, gtype=gtype,
                             noise=noise, mat=mat))
    end

    for d in [10, 15, 20],
        gtype in [:ER, :SF],
        k in [1],
        mec in [:Linear]

        load_sup_ds(DataSpec(d=d, k=k,
                             gtype=gtype,
                             noise=:Gaussian,
                             mechanism=mec))
    end

    for d in [10, 15, 20],
        gtype in [:ER, :ER2, :ER4, :SF, :SF2, :SF4],
        k in [1],
        mec in [:Linear]

        load_sup_ds(DataSpec(d=d, k=k,
                             gtype=gtype,
                             noise=:Gaussian,
                             mechanism=mec))
    end
end

function test()
    (DataSpec(10, 1, :ER, :Gaussian, :COR, :Linear, 3000, 3))
end

function test_real()
    # cite:2005-Science-Sachs-Causal
    df = CSV.read("/home/hebi/Downloads/tmp/Sachs/csv/1.cd3cd28.csv")
    df = CSV.read("/home/hebi/Downloads/tmp/Sachs/csv/12.cd3cd28icam2+psit.csv")
    df = CSV.read("/home/hebi/Downloads/tmp/Sachs/csv/7.cd3cd28+ly.csv")
    
    names(df)

    SachsG = Sachs_ground_truth()
    display_named_graph(SachsG)
    # 853, 11
    SachsX = convert(Matrix, df)


    @load "saved_models/deep-EQ-SF-ensemble/step-30000.bson" model
    @load "saved_models/deep-EQ-ER-ensemble/step-30000.bson" model
    
    @load "back/supervised/saved_models/EQ-deep-mixed-d=[10,15,20]-COR2-2020-05-11T22:09:56.734/step-40000.bson" model
    @load "back/supervised/saved_models/EQ-deep-mixed-d=[10,15,20]-ONE/step-17671.bson" model
    @load "back/supervised/saved_models/EQ-deep-mixed-d=[10,15,20]-CORR/step-41438.bson" model

    @load "saved_models/EQ-deep-mixed-d=[10,15,20]-COV/step-47208.bson" model

    @load "saved_models/deep-EQ-d=10_k=1_gtype=SF_noise=Gaussian_mat=COR/step-30000.bson"
    @load "saved_models/deep-EQ-ER-ensemble/step-30000.bson" model
    # out = inf_one(model, cov(SachsX))
    out = inf_one(model, cor(SachsX))
    # visualize the graph
    Wout = threshold(σ.(out), 0.5, true)
    # Wout = threshold(out, 25, true)
    # no, this is all 0
    display_graph_with_label(DiGraph(Wout), names(df))
    display_graph_with_label(SachsG, names(df))
    # call notears
    ne(DiGraph(Wout))
    adjacency_matrix(SachsG)

    # predicted edge, true edge, SHD
    predicted_edge = ne(DiGraph(Wout))
    @show predicted_edge
    correct_edge = sum(Wout[Wout .== 1] .== adjacency_matrix(SachsG)[Wout .== 1])
    @show correct_edge

    # metrics
    ytrue = Matrix(gen_weights(SachsG))
    sup_graph_metrics(Wout, ytrue)
end


function main()
    main_ultra_large_graphs()

    main_CNN_ensemble()
    main_CNN_sep()

    main_FC()
    main_FC_cov()

    main_EQ_ensemble()
    # TODO OVERNIGHT
    main_EQ_sep()
    main_EQ_cov()
    # DEBUG super ensemble
    main_EQ_super_ensemble()
end

# main()
