include("autodiff.jl")

function binary_cross_entropy(pred::TensorJL, target::TensorJL)
    eps = 1.0f-8
    loss_data = - ( target.data .* log.(pred.data .+ eps)
                   .+ (1 .- target.data) .* log.(1 .- pred.data .+ eps) )
    avg = sum(loss_data) / size(loss_data,1)
    out = TensorJL(reshape(Float32[avg],1,1),
                   pred.requires_grad || target.requires_grad,
                   Set([pred, target]))
    function _backward()
        if pred.requires_grad
            grad = ((pred.data .- target.data) 
                    ./ ( (pred.data .+ eps) .* (1 .- pred.data .+ eps) ))
            grad ./= size(target.data,1)
            pred.grad .+= grad .* out.grad[1] 
        end
    end
    out._backward = _backward
    return out
end


accuracy(y_pred, y_true) = mean((y_pred .> 0.5f0) .== (y_true .> 0.5f0))