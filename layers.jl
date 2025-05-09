include("autodiff.jl")

struct Dense
    W::TensorJL
    b::TensorJL
end

Dense(in_features::Int, out_features::Int) = Dense(
    # TensorJL(randn(Float32, in_features, out_features) * 0.1f0, true),
    TensorJL(randn(Float32, in_features, out_features) * √(2f0 / in_features), true),
    TensorJL(zeros(Float32, 1, out_features), true)
)

# function (d::Dense)(x::TensorJL)
#     out = matmul(x, d.W)
#     y_data = out.data .+ repeat(d.b.data, size(out.data, 1), 1)
#     y = TensorJL(y_data, x.requires_grad || d.W.requires_grad || d.b.requires_grad,
#                  Set([x, d.W, d.b]))

#     function _backward()
#         if x.requires_grad
#             x.grad .+= out.grad * transpose(d.W.data)
#         end
#         if d.W.requires_grad
#             d.W.grad .+= transpose(x.data) * out.grad
#         end
#         if d.b.requires_grad
#             d.b.grad .+= sum(out.grad, dims=1)
#         end
#     end
#     y._backward = _backward
#     return y
# end

function (d::Dense)(x::TensorJL)
    return add(matmul(x, d.W), d.b)
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
        elseif isa(layer, Embedding)
            push!(params, layer.weight)
        elseif isa(layer, RNN)
            push!(params, layer.Wxh, layer.Whh, layer.bh)
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

struct Embedding
    weight::TensorJL
end

Embedding(vocab_size::Int, emb_dim::Int) = Embedding(
    TensorJL((randn(Float32, vocab_size, emb_dim) * 0.01f0), true)
)

function (e::Embedding)(x::TensorJL)
    batch_size, seq_len = size(x.data)
    emb_dim = size(e.weight.data, 2)
    out_data = Array{Float32}(undef, batch_size, seq_len, emb_dim)
    # println(">>  Embedding out size =", size(out_data))
    if size(out_data, 3) == 1        # podejrzanie − zgłoś
        @warn "Embedding dim is 1!"
    end
    index_copy = Array{Int}(undef, batch_size, seq_len)  # potrzebne do backwarda

    for i in 1:batch_size
        for j in 1:seq_len
            idx = Int(x.data[i, j])
            index_copy[i, j] = idx
            out_data[i, j, :] = e.weight.data[idx, :]
        end
    end

    out = TensorJL(out_data, true, Set([e.weight]))

    function _backward()
        if e.weight.requires_grad && out.grad !== nothing
            for i in 1:batch_size
                for j in 1:seq_len
                    idx = index_copy[i, j]
                    e.weight.grad[idx, :] .+= out.grad[i, j, :]
                end
            end
        end
    end

    out._backward = _backward
    return out
end


struct RNN
    Wxh::TensorJL
    Whh::TensorJL
    bh::TensorJL
    activation::Function
end

function RNN(input_size::Int, hidden_size::Int, activation=relu)
    RNN(
        TensorJL(randn(Float32, input_size,  hidden_size) *
                 √(2f0 / input_size),  true),     # Wxh – zostaje He
        TensorJL(randn(Float32, hidden_size, hidden_size) *
                 0.01f0, true),                   # Whh – dużo mniejszy!
        TensorJL(zeros(Float32, 1, hidden_size), true),
        activation
    )
end

function (rnn::RNN)(x::TensorJL)
    # x: (batch_size, seq_len, input_size)
    # println(">>  RNN in size      =", size(x.data))
    batch_size = size(x.data, 1)
    seq_len    = size(x.data, 2)
    input_size = size(x.data, 3) 
    hidden_size = size(rnn.Whh.data, 2)
    h = zeros(Float32, batch_size, hidden_size)
    h_tensor = TensorJL(h, rnn.Wxh.requires_grad || rnn.Whh.requires_grad, Set())

    for t in 1:seq_len
        xt = slice_along_time(x, t)
        h_tensor = rnn.activation(
                      add(add(matmul(xt, rnn.Wxh),
                              matmul(h_tensor, rnn.Whh)),
                          rnn.bh))
        h_tensor.data .= clamp.(h_tensor.data, -20f0, 20f0)
    end
    h_tensor
end

function slice_along_time(x::TensorJL, t::Int)
    out_data = @view x.data[:, t, :]
    # println(">>  slice t=", t, "  size =", size(out_data))   # pokaże (batch, ?, ?)
    out = TensorJL(out_data, x.requires_grad, Set([x]))

    function _backward()
        if x.requires_grad
            x.grad[:, t, :] .+= out.grad
        end
    end
    out._backward = _backward
    return out
end


last_output(x::TensorJL) = x

function flatten(x::TensorJL)
    orig_shape = size(x.data)           # (batch, seq_len, hidden)
    out_data   = reshape(x.data, size(x.data,1), :)   # (batch, -1)
    out = TensorJL(out_data, x.requires_grad, Set([x]))

    function _backward()
        if x.requires_grad && out.grad !== nothing
            # „rozprostowany” grad trzeba ułożyć z powrotem do oryginalnego kształtu
            x.grad .+= reshape(out.grad, orig_shape)
        end
    end
    out._backward = _backward
    return out
end
