include("autodiff.jl")

mutable struct Adam
    params::Vector{TensorJL}
    lr::Float32
    beta1::Float32
    beta2::Float32
    m::Vector{Array{Float32}}
    v::Vector{Array{Float32}}
    t::Int
end

Adam(params, lr=0.0001f0, betas=(0.9f0, 0.999f0)) = Adam(
    params,
    lr,
    betas[1],
    betas[2],
    [zeros(Float32, size(p.data)) for p in params],
    [zeros(Float32, size(p.data)) for p in params],
    0
)

function step!(opt::Adam)
    opt.t += 1
    for i in 1:length(opt.params)
        p = opt.params[i]
        opt.m[i] = opt.beta1 .* opt.m[i] .+ (1 - opt.beta1) .* p.grad
        opt.v[i] = opt.beta2 .* opt.v[i] .+ (1 - opt.beta2) .* (p.grad .^ 2)
        m_hat = opt.m[i] ./ (1 .- opt.beta1.^opt.t)
        v_hat = opt.v[i] ./ (1 .- opt.beta2.^opt.t)
        p.data .-= opt.lr .* m_hat ./ (sqrt.(v_hat) .+ 1.0f-8)
    end
end