import torch

# Example tensor
tensor = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

# Get the shape of the tensor
# [2, 2, 2]
shape = tensor.shape

# Get the number of dimensions (rank) of the tensor
num_dims = tensor.ndimension()


# Define a function to recursively iterate over each dimension
def iterate_tensor(tensor, idx=(), dim=0):
    if dim == num_dims:
        print(f"Element at index {idx}: {tensor.item()}")
    else:
        for i in range(shape[dim]):
            # i 0..2
            iterate_tensor(tensor[i], idx + (i,), dim + 1)


print("starting")
# Start the recursion
iterate_tensor(tensor)

x = torch.rand(1000, 2, 2, 100)
# print(x) # prints the truncated tensor
# torch.set_printoptions(threshold=10_000)
# Create a 4-dimensional tensor

# Get the first 1,000 elements along the first dimension
first_thousand = x

# Print the first 1,000 elements


# for element in first_thousand:
#     print(element.item())

first_thousand = first_thousand.flatten()[-500:]
for i in range(0, first_thousand.numel(), 32):
    print(" ".join(f"{value:.3f}" for value in first_thousand[i : min(i + 32, first_thousand.numel())]))
    print()
