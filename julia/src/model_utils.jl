function param_count(model)
    ps = Flux.params(model)
    res = 0
    for p in keys(ps.params.dict)
        res += prod(size(p))
    end
    res
end

struct Flatten end

function (l::Flatten)(input)
    reshape(input, :, size(input, 4))
end

struct Sigmoid end
function (l::Sigmoid)(input)
    σ.(input)
end

struct ReLU end

function (l::ReLU)(input)
    relu.(input)
end

struct LeakyReLU end
function (l::LeakyReLU)(input)
    leakyrelu.(input)
end

struct Tanh end
function (l::Tanh)(input)
    tanh.(input)
end
