## TT-Metal [ttnn]

### A demonstration of how multihead attention could be implemented with tt_lib using apis very similar to what you would see in pytorch.  The intent of this effort is to provide a successful demo and, more importantly, highlight differences on what one might expect to encounter when migrating from pytorch.

#### Key differences with the following operations:
* to_torch and from_torch
    * Incompatible constructor arguments.
* matmul
    * The last two dimensions must be a multiple of 32 when creating a tensor.  For example a tensor of (1, 1, 3, 4) would not be successfully multiplied to another tensor.  Instead, the developer is currently expected to pad the tensor with zeros to be viable in a multiplication.
    * Results from a matmul will not have the same precision.  This is still under investigation.
* reshape
    * The last two dimensions in the reshape must be a multiple of 32.
    * When converting a tt tensor to a pytorch tensor, the 3rd and 4th dimensions must be a multiple of 32.
    * Using -1 for reshape is not supported.
* transpose
    * SIGABRT during a transpose that swaps the last two dimensions
* add
    *
* softmax
*
