# import torch
# import matplotlib.pyplot as plt

# print("starting")

# # Example tensors
# tensor1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
# tensor2 = torch.tensor([[2, 3, 4], [5, 6, 7]])

# # Plot both tensors
# plt.figure(figsize=(10, 5))

# plt.subplot(1, 2, 1)
# plt.imshow(tensor1, cmap='viridis')
# plt.title('Tensor 1')
# plt.colorbar()

# plt.subplot(1, 2, 2)
# plt.imshow(tensor2, cmap='viridis')
# plt.title('Tensor 2')
# plt.colorbar()

# plt.tight_layout()
# plt.savefig("gpt.png")

# # import torch
# # import numpy as np
# # import matplotlib.pyplot as plt

# # # Create two 2-dimensional tensors
# # tensor1 = torch.rand(4, 4)
# # tensor2 = torch.rand(4, 4)

# # # Convert the tensors to NumPy arrays
# # array1 = tensor1.numpy()
# # array2 = tensor2.numpy()

# # # Create a figure and axes for the subplots
# # fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# # # Plot the first tensor
# # axs[0].imshow(array1, cmap="gray")
# # axs[0].set_title("Tensor 1")

# # # Plot the second tensor
# # axs[1].imshow(array2, cmap="gray")
# # axs[1].set_title("Tensor 2")

# # # Remove the x and y tick labels
# # for ax in axs:
# #     ax.set_xticks([])
# #     ax.set_yticks([])

# # # Show the plot
# # # plt.show()
# # plt.savefig("dummy_name.png")

# import torch

# # Create a tensor
# tensor = torch.tensor([1.0, 2.5, 3.1415, 100])

# # Print the tensor values
# for value in tensor:
#     print(value)
