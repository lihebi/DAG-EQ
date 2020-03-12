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

function main()
    # 15/20 is already 0
    for d in [7,10,15,20]
        exp_sup(d, fc_model_fn, prefix="FC", suffix="ONE",
                ng=1e4, N=20, train_steps=3e5)
        exp_sup(d, deep_fc_model_fn, prefix="FC-deep", suffix="ONE",
                ng=1e4, N=20, train_steps=3e5)
    end
    for d in [7,10,15,20,25,30]
        exp_sup(d, eq_model_fn, prefix="EQ", suffix="ONE",
                ng=1e4, N=20, train_steps=3e4, test_throttle=20)
        exp_sup(d, deep_eq_model_fn, prefix="EQ-deep", suffix="ONE",
                ng=1e4, N=20, train_steps=3e4, test_throttle=20)
    end
end

function main_mixed()
    exp_mixed(deep_eq_model_fn, prefix="EQ-deep", suffix="ONE", train_steps=1e5)
    # exp_mixed(eq_model_fn, prefix="EQ", suffix="ONE", train_steps=3e4)
end

# main_fc()
# main_eq()
# main()
main_mixed()
