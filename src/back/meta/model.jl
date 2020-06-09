using Flux

struct Tanh end
function (l::Tanh)(input)
    tanh.(input)
end

# Mixture Density Network, FIXME there seems to be blog post about implementing
# in Julia https://www.janisklaise.com/post/mdn_julia/
struct MDN
    # FIXME cap and comp is fixed. How does Flux knows which are parameters?
    capacity
    component
    h
    pi
    mu
    sigma
end

# Flux.@functor MDN
Flux.@treelike MDN

function MDN(cap, comp)
    h = Chain(Dense(1, cap),
              Tanh())
    pi = Dense(cap, comp)
    mu = Dense(cap, comp)
    sigma = Dense(cap, comp)
    MDN(cap, comp, h, pi, mu, sigma)
end

function (m::MDN)(input)
    h = m.h(input)
    # FIXME dim=-1
    pi = softmax(m.pi(h))
    mu = m.mu(h)
    sigma = exp.(m.sigma(h))
    pi, mu, sigma
end
