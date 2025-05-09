include("autodiff.jl")

# function binary_cross_entropy(pred::TensorJL, target::TensorJL)
#     eps = 1f-7  # poprawna notacja Float32
#     clamped = clamp.(pred.data, eps, 1f0 - eps)

#     loss_data = - (target.data .* log.(clamped) .+ (1 .- target.data) .* log.(1 .- clamped))
#     avg = sum(loss_data) / size(loss_data, 1)

#     out = TensorJL(reshape(Float32[avg], 1, 1),
#                    pred.requires_grad || target.requires_grad,
#                    Set([pred, target]))

#     function _backward()
#         if pred.requires_grad
#             # Gradient przez funkcję strat: BCE wrt pred
#             grad = (clamped .- target.data) ./ (clamped .* (1 .- clamped))  # uważaj: używamy clamped
#             grad ./= size(target.data, 1)
#             pred.grad .+= grad .* out.grad[1]
#         end
#     end

#     out._backward = _backward
#     return out
# end

### zupełnie nowa funkcja (zastępuje starą) ###
function binary_cross_entropy_with_logits(logits::TensorJL, target::TensorJL)
    z = clamp.(logits.data, -20f0, 20f0)

    y = target.data
    zeroz     = max.(0f0, z)
    loss_data = zeroz .- z .* y .+ log.(1f0 .+ exp.(-abs.(z)))
    avg       = sum(loss_data) / size(loss_data, 1)

    out = TensorJL(reshape(Float32[avg], 1, 1),
                   logits.requires_grad || target.requires_grad,
                   Set([logits, target]))

    function _backward()
        if logits.requires_grad
            sig  = 1f0 ./ (1f0 .+ exp.(-z))          # σ(z) policzone na z-clamp
            grad = (sig .- y) ./ size(y,1)
            logits.grad .+= grad .* out.grad[1]
        end
    end
    out._backward = _backward
    return out
end

clip!(g, thr=5f0) = (@. g = clamp(g, -thr, thr))


accuracy(y_pred, y_true) = mean((y_pred .> 0.5f0) .== (y_true .> 0.5f0))