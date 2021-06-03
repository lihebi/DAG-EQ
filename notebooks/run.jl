@info "Including exp.jl .."
include("../src/exp.jl")

function main()
    for d in [10,20,50,100],
        (prefix, model_fn,nsteps) in [
            ("EQ2", eq2_model_fn, 5e4),
            ("FC", ()->fc_model(d=d, ch=2, z=1024, nlayer=6), 3e4),
            ("CNN", ()->cnn_model(2), 3e4),
        ]
        
        specs = []
        for gtype in [:ER, :SF],
            k in [1]
            push!(specs, DataSpec(d=d, k=k, gtype=gtype,
                    noise=:Gaussian, mat=:CH3))
        end
        specs = Array{DataSpec}(specs)
        
        test_throttle = if prefix == "EQ2" 10 else 1 end

        @info "training .." prefix d
        expID = exp_train(specs, model_fn,
                          prefix="$prefix-ERSF-k1-d=$d", train_steps=nsteps,
                          test_throttle = test_throttle,
                          merge=true)
        @info "testing"
        exp_test(expID, specs, "TEST-gtype=$gtype")
    end
end


@info "Running main() .."
main()