include("autodiff.jl")
include("layers.jl")
include("metrics.jl")
include("optimizer.jl")
include("training.jl")

a = TensorJL([1f0], true)           # param 1
b = TensorJL([2f0], true)           # param 2
c = add(a, b)                       # c = a + b
loss = binary_cross_entropy_with_logits(c, TensorJL([1f0]))

for p in (a, b); p.grad .= 0; end   # wyzeruj ręcznie
backward!(loss)

@info "grad a" a.grad               # spodziewane ≠ 0
@info "grad b" b.grad
return 

X_train, y_train, X_test, y_test, vocab, embeddings, embedding_dim = load_data()
model, optimizer, epochs, batch_size = build_infra(vocab, embeddings, embedding_dim)
train_model(model, optimizer, epochs, batch_size, X_train, y_train, X_test, y_test)
