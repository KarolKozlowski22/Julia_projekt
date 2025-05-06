include("autodiff.jl")
include("layers.jl")
include("optimizer.jl")
include("metrics.jl")

using JLD2
using Random
using Statistics
using Printf

function load_data()
    @time begin
        println("Ładowanie danych...")
        X_train = load("data/imdb_dataset_prepared.jld2", "X_train")
        y_train = load("data/imdb_dataset_prepared.jld2", "y_train")
        X_test = load("data/imdb_dataset_prepared.jld2", "X_test")
        y_test = load("data/imdb_dataset_prepared.jld2", "y_test")
        println("Dane załadowane!")
    end
    X_train=X_train'
    X_test=X_test'
    # y_train=y_train
    # y_test=y_test
    y_train = Float32.(Array(y_train))
    y_test  = Float32.(Array(y_test))
    y_train = reshape(y_train, :, 1)
    y_test  = reshape(y_test, :, 1)
    return X_train, y_train, X_test, y_test
end

function build_infra()
    model = Sequential(
        Dense(17703, 32),
        x -> relu(x),
        Dense(32, 1),
        x -> sigmoid(x)
    )

    optimizer = Adam(model.parameters, 0.001f0)

    epochs = 5
    batch_size = 32
    return model, optimizer, epochs, batch_size
end

function train_model(model, optimizer, epochs, batch_size, X_train, y_train, X_test, y_test)
    for epoch in 1:epochs
        total_loss = 0.0f0
        total_acc = 0.0f0
        num_batches = 0

        t = @elapsed begin
            perm = shuffle(1:size(X_train, 1))
            X_shuffled = X_train[perm, :]
            y_shuffled = y_train[perm]

            for i in 1:batch_size:size(X_train, 1)
                X_batch = TensorJL(X_shuffled[i:min(i+batch_size-1, end), :])
                y_batch = TensorJL(Float32.(y_shuffled[i:min(i+batch_size-1, end)])) 

                output = model(X_batch)
                loss = binary_cross_entropy(output, y_batch)  

                for p in model.parameters
                    p.grad .= 0.0f0
                end

                backward!(loss)

                step!(optimizer)

                total_loss += loss.data[1]
                total_acc += accuracy(output.data, y_batch.data)
                num_batches += 1
            end
        end

        test_output = model(TensorJL(X_test))
        test_loss = binary_cross_entropy(test_output, TensorJL(y_test))
        test_acc = accuracy(test_output.data, y_test)

        println(@sprintf(
            "Epoch: %d (%.2fs) \tTrain: (l: %.4f, a: %.4f) \tTest: (l: %.4f, a: %.4f)",
            epoch, t, total_loss/num_batches, total_acc/num_batches, test_loss.data[1], test_acc
        ))
    end
end
