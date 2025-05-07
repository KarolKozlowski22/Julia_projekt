include("autodiff.jl")
include("layers.jl")
include("optimizer.jl")
include("metrics.jl")

using JLD2
using Random
using Statistics
using Printf
using LinearAlgebra

function load_data()
    @time begin
        println("Ładowanie danych...")
        X_train = load("data/imdb_dataset_prepared.jld2", "X_train")
        y_train = load("data/imdb_dataset_prepared.jld2", "y_train")
        X_test = load("data/imdb_dataset_prepared.jld2", "X_test")
        y_test = load("data/imdb_dataset_prepared.jld2", "y_test")
        vocab = load("data/imdb_dataset_prepared.jld2", "vocab")
        embeddings = load("data/imdb_dataset_prepared.jld2", "embeddings")
        println("Dane załadowane!")
    end
    # X_train=X_train'
    # X_test=X_test'
    # # y_train=y_train
    # # y_test=y_test
    # y_train = Float32.(Array(y_train))
    # y_test  = Float32.(Array(y_test))
    # y_train = reshape(y_train, :, 1)
    # y_test  = reshape(y_test, :, 1)
    embedding_dim = size(embeddings,1)
    y_train = Float32.(y_train)
    y_test  = Float32.(y_test)
    return X_train', y_train, X_test', y_test, vocab, embeddings, embedding_dim

end

function build_infra(vocab, embeddings, embedding_dim)
    vocab_size = length(vocab)

    emb_layer = Embedding(vocab_size, embedding_dim)
    emb_layer.weight.data .= embeddings' 

    model = Sequential(
        emb_layer,
        RNN(embedding_dim, 16, relu),
        last_output,
        flatten,
        Dense(16, 1),
        sigmoid
    )

    optimizer = Adam(model.parameters, 0.001f0)
    epochs = 5
    batch_size = 32
    # model.layers[1].weight .= embeddings;
    return model, optimizer, epochs, batch_size
end

function train_model(model, optimizer, epochs, batch_size, X_train, y_train, X_test, y_test)
    for epoch in 1:epochs
        total_loss = 0.0f0
        total_acc = 0.0f0
        num_samples = 0

        t = @elapsed begin
            perm = shuffle(1:size(X_train, 1))
            X_shuffled = X_train[perm, :]
            y_shuffled = y_train[perm]

            for i in 1:batch_size:size(X_train, 1)
                x_batch = X_shuffled[i:min(i+batch_size-1, end), :]
                y_batch = y_shuffled[i:min(i+batch_size-1, end)]

                x_tensor = TensorJL(Float32.(x_batch), true)  
                y_tensor = TensorJL(Float32.(y_batch))

                # Forward
                output = model(x_tensor)
                loss = binary_cross_entropy(output, y_tensor)

                # Reset grad
                for p in model.parameters
                    p.grad .= 0.0f0
                end

                # Backward
                backward!(loss)
                println("Sample output: ", output.data[1:5])
                println("Sample y: ", y_tensor.data[1:5])
                println("Loss: ", loss.data[1])
                for (i, p) in enumerate(model.parameters)
                    println("Grad norm param $i: ", norm(p.grad))
                end
                println("------")
                println("Embedding weight grad norm: ", norm(model.layers[1].weight.grad))

                # Opt step
                step!(optimizer)

                total_loss += loss.data[1]
                total_acc += accuracy(output.data, y_tensor.data)
                num_samples += 1
            end
        end

        # Eval
        test_output = model(TensorJL(Int.(X_test)))
        test_loss = binary_cross_entropy(test_output, TensorJL(Float32.(y_test)))
        test_acc = accuracy(test_output.data, y_test)

        train_loss = total_loss / num_samples
        train_acc = total_acc / num_samples

        println(@sprintf("Epoch: %d (%.2fs) \tTrain: (l: %.4f, a: %.4f) \tTest: (l: %.4f, a: %.4f)",
            epoch, t, train_loss, train_acc, test_loss.data[1], test_acc))
    end
end

