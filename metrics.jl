include("autodiff.jl")

function binary_cross_entropy(pred::TensorJL, target::TensorJL)
    eps = 1f-7  # poprawna notacja Float32
    clamped = clamp.(pred.data, eps, 1f0 - eps)

    loss_data = - (target.data .* log.(clamped) .+ (1 .- target.data) .* log.(1 .- clamped))
    avg = sum(loss_data) / size(loss_data, 1)

    out = TensorJL(reshape(Float32[avg], 1, 1),
                   pred.requires_grad || target.requires_grad,
                   Set([pred, target]))

    function _backward()
        if pred.requires_grad
            # Gradient przez funkcję strat: BCE wrt pred
            grad = (clamped .- target.data) ./ (clamped .* (1 .- clamped))  # uważaj: używamy clamped
            grad ./= size(target.data, 1)
            pred.grad .+= grad .* out.grad[1]
        end
    end

    out._backward = _backward
    return out
end


accuracy(y_pred, y_true) = mean((y_pred .> 0.5f0) .== (y_true .> 0.5f0))