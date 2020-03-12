using Logging
using TensorBoardLogger

# something like tf.keras.metrics.Mean
# a good reference https://github.com/apache/incubator-mxnet/blob/master/julia/src/metric.jl
mutable struct MeanMetric{T}
    # FIXME syntax for T
    sum::T
    n::Int
    # CAUTION for T, you must define Base.convert(::T, v::Int64)
    MeanMetric{T}() where {T} = new(0, 0)
end
function add!(m::MeanMetric{T}, v) where {T}
    m.sum += v
    m.n += 1
end
get(m::MeanMetric) = if m.n != 0 m.sum / m.n else m.sum / 1 end
function get!(m::MeanMetric)
    res = get(m)
    reset!(m)
    res
end
function reset!(m::MeanMetric{T}) where {T}
    m.sum = 0
    m.n = 0
end

struct GraphMetric
    # All float64, because when I average, the / operator, it will be float
    nnz::Float64
    nny::Float64
    tpr::Float64
    fpr::Float64
    fdr::Float64
    shd::Float64
    prec::Float64
    recall::Float64
    # GraphMetric() = new(0,0,0,0,0,0)
end

# FIXME this only custom @show and REPL, but does not affect logger like @info
# function Base.show(io::IO, ::MIME"text/plain", m::GraphMetric)
#     for fname in fieldnames(typeof(m))
#         println(io, "$fname = $(getfield(m, fname))")
#     end
# end

# from https://discourse.julialang.org/t/get-fieldnames-and-values-of-struct-as-namedtuple/8991
to_named_tuple(p) = (; (v=>getfield(p, v) for v in fieldnames(typeof(p)))...)

# I should have used "value type", but that seems to be less performant. I'm
# setting it to 0 for whatever v.
Base.convert(::Type{GraphMetric}, v::Int64) = GraphMetric(0,0,0,0,0,0,0,0)

import Base.+
function +(a::GraphMetric, b::GraphMetric)
    # FIXME can I automate this?
    GraphMetric(a.nnz + b.nnz,
                a.nny + b.nny,
                a.tpr + b.tpr,
                a.fpr + b.fpr,
                a.fdr + b.fdr,
                a.shd + b.shd,
                a.prec + b.prec,
                a.recall + b.recall)
end
import Base./
function /(a::GraphMetric, n::Int64)
    GraphMetric(a.nnz / n,
                a.nny / n,
                a.tpr / n,
                a.fpr / n,
                a.fdr / n,
                a.shd / n,
                a.prec / n,
                a.recall / n)
end

# FIXME overwrite TBLogger
import TensorBoardLogger.log_value
function log_value(logger::TBLogger, name::AbstractString, value::GraphMetric; step=nothing)
    # FIXME loop through all fields, and log value for each of them
    fn = propertynames(value)

    # for f=fn
    #     prop = getfield(value, f)
    #     log_value(logger, name*"/$f", prop, step=step)
    # end

    # FIXME it turns out the order is random, but at least the prefix works
    #
    # I want to control the order of the fields

    # log_value(logger, name*"/tpr", value.tpr, step=step)
    # log_value(logger, name*"/fpr", value.fpr, step=step)
    # log_value(logger, name*"/fdr", value.fdr, step=step)
    log_value(logger, name*"/prec", value.prec, step=step)
    log_value(logger, name*"/recall", value.recall, step=step)

    # tfboard has good support to log scale something near 0
    # log_value(logger, name*"/1-prec", 1-value.prec, step=step)
    # log_value(logger, name*"/1-recall", 1-value.recall, step=step)

    # use a different prefix
    log_value(logger, name*"/v/shd", value.shd, step=step)

    # FIXME I want to plot nnz and nny in the same plot
    # log_value(logger, name*"/v/nnz", value.nnz, step=step)
    # log_value(logger, name*"/v/nny", value.nny, step=step)
end

function test()
    m = MeanMetric()

    m = MeanMetric{Float64}()
    add!(m, 1.3)
    get!(m)
    reset!(m)

    m = MeanMetric{GraphMetric}()
    add!(m, GraphMetric(1,1,1,1,3,1,0,0))
    get(m)
    reset!(m)
end
