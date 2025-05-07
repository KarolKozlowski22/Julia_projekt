include("autodiff.jl")
include("layers.jl")
include("metrics.jl")
include("optimizer.jl")
include("training.jl")

X_train, y_train, X_test, y_test, vocab, embeddings, embedding_dim = load_data()
model, optimizer, epochs, batch_size = build_infra(vocab, embeddings, embedding_dim)
train_model(model, optimizer, epochs, batch_size, X_train, y_train, X_test, y_test)
