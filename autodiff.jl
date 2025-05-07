import Base: *
import Base: +

mutable struct TensorJL
    data::Array{Float32}
    grad::Union{Array{Float32}, Nothing}
    requires_grad::Bool
    _backward::Function
    _prev::Set{TensorJL}
    
    function TensorJL(data::AbstractArray{<:Real}, requires_grad=false, _children=Set{TensorJL}())
        data_f32 = Float32.(data)
        new(data_f32, 
            requires_grad ? zeros(Float32, size(data)) : nothing,
            requires_grad,
            () -> nothing,
            _children)
    end
    
end


function *(a::TensorJL, b::TensorJL)
    return matmul(a, b)
end

function +(a::TensorJL, b::TensorJL)
    return add(a, b)
end

function Base.show(io::IO, t::TensorJL)
    print(io, "TensorJL($(size(t.data)), grad=$(t.grad !== nothing)")
end

function backward!(t::TensorJL)
    topo = TensorJL[]
    visited = Set{TensorJL}()
    
    function build_topo(v::TensorJL)
        if !(v in visited)
            push!(visited, v)
            for child in v._prev
                build_topo(child)
            end
            push!(topo, v)
        end
    end
    
    build_topo(t)
    t.grad = ones(Float32, size(t.data))
    for v in reverse(topo)
        v._backward()
    end
end

function matmul(a::TensorJL, b::TensorJL)
    out_data = a.data * b.data
    out = TensorJL(out_data, a.requires_grad || b.requires_grad, Set([a, b]))
    
    function _backward()
        if a.requires_grad
            a.grad += out.grad * transpose(b.data)
        end
        if b.requires_grad
            b.grad += transpose(a.data) * out.grad
        end
    end
    out._backward = _backward
    out
end

function add(a::TensorJL, b::TensorJL)
    a_data = a.data
    b_data = size(a_data) == size(b.data) ? b.data : repeat(b.data, size(a_data, 1), 1)

    out_data = a_data .+ b_data
    out = TensorJL(out_data, a.requires_grad || b.requires_grad, Set([a, b]))
    
    function _backward()
        if a.requires_grad
            a.grad .+= out.grad
        end
        if b.requires_grad
            # Sumujemy wzdłuż batcha, by wrócić do rozmiaru (1, features)
            b.grad .+= sum(out.grad, dims=1)
        end
    end
    out._backward = _backward
    return out
end


function relu(x::TensorJL)
    out_data = max.(0, x.data)
    out = TensorJL(out_data, x.requires_grad, Set([x]))

    function _backward()
        if x.requires_grad
            x.grad += (x.data .> 0) .* out.grad
        end
    end
    out._backward = _backward
    return out
end

function sigmoid(x::TensorJL)
    s = 1 ./ (1 .+ exp.(-x.data))
    out = TensorJL(s, x.requires_grad, Set([x]))
    
    function _backward()
        if x.requires_grad
            x.grad += (s .* (1 .- s)) .* out.grad
        end
    end
    out._backward = _backward
    out
end
