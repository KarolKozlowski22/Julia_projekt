include("autodiff.jl")

struct Dense
    W::TensorJL
    b::TensorJL
end

Dense(in_features::Int, out_features::Int) = Dense(
    TensorJL(randn(Float32, in_features, out_features) * 0.1f0, true),
    TensorJL(zeros(Float32, 1, out_features), true)
)

function (d::Dense)(x::TensorJL)
    add(matmul(x, d.W), d.b)
end

struct Sequential
    layers::Vector{Any}
    parameters::Vector{TensorJL}
end

Sequential(layers...) = begin
    params = TensorJL[]
    for layer in layers
        if isa(layer, Dense)
            push!(params, layer.W, layer.b)
        end
    end
    Sequential(collect(layers), params)
end

function (s::Sequential)(x::TensorJL)
    for layer in s.layers
        x = layer isa Function ? layer(x) : layer(x)
    end
    x
end