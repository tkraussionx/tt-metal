# # ######## ORIGINAL MODULE
# MATMUL_MULTI_CORE_REUSE_MCAST_EXAMPLE_SRC = $(TT_METAL_HOME)/tt_metal/programming_examples/matmul_multicore_reuse_mcast/matmul_multicore_reuse_mcast.cpp

# MATMUL_MULTI_CORE_REUSE_MCAST_EXAMPLES_DEPS = $(PROGRAMMING_EXAMPLES_OBJDIR)/matmul_multicore_reuse_mcast.d

# -include $(MATMUL_MULTI_CORE_REUSE_MCAST_EXAMPLES_DEPS)

# .PRECIOUS: $(PROGRAMMING_EXAMPLES_TESTDIR)/matmul_multicore_reuse_mcast
# $(PROGRAMMING_EXAMPLES_TESTDIR)/matmul_multicore_reuse_mcast: $(PROGRAMMING_EXAMPLES_OBJDIR)/matmul_multicore_reuse_mcast.o $(TT_METAL_LIB)
# 	@mkdir -p $(@D)
# 	$(CXX) $(CFLAGS) $(CXXFLAGS) $(PROGRAMMING_EXAMPLES_INCLUDES) -o $@ $^ $(LDFLAGS) $(PROGRAMMING_EXAMPLES_LDFLAGS)

# .PRECIOUS: $(PROGRAMMING_EXAMPLES_OBJDIR)/matmul_multicore_reuse_mcast.o
# $(PROGRAMMING_EXAMPLES_OBJDIR)/matmul_multicore_reuse_mcast.o: $(MATMUL_MULTI_CORE_REUSE_MCAST_EXAMPLE_SRC)
# 	@mkdir -p $(@D)
# 	$(CXX) $(CFLAGS) $(CXXFLAGS) $(PROGRAMMING_EXAMPLES_INCLUDES) -c -o $@ $<



# ######## LINKING CPP FILES ONE-BY-ONE
# TT_METAL_SRC = $(TT_METAL_HOME)/tt_eager/tensor/tensor.cpp \
#                $(TT_METAL_HOME)/tt_eager/tensor/tensor_utils.cpp \
#                $(TT_METAL_HOME)/tt_eager/tensor/types.cpp \
# 			   $(TT_METAL_HOME)/tt_eager/tensor/tensor_impl.cpp \
# 			   $(TT_METAL_HOME)/tt_eager/tensor/tensor_impl_wrapper.cpp \
#              $(TT_METAL_HOME)/tt_eager/tt_dnn/op_library/downsample/downsample_op.cpp \
# 			   $(TT_METAL_HOME)/tt_eager/tt_dnn/op_library/downsample/downsample_op.cpp \
# 			   $(TT_METAL_HOME)/tt_eager/tt_dnn/op_library/eltwise_unary/eltwise_unary_op.cpp \
# 			   $(TT_METAL_HOME)/tt_eager/tt_dnn/op_library/eltwise_unary/single_core/eltwise_unary_op_single_core.cpp \
# 			   $(TT_METAL_HOME)/tt_eager/tt_dnn/op_library/eltwise_unary/multi_core/eltwise_unary_op_multi_core.cpp \
# 			   $(TT_METAL_HOME)/tt_eager/tt_dnn/op_library/operation_history.cpp \
# 			   $(TT_METAL_HOME)/tt_eager/tt_dnn/op_library/auto_format.cpp

# MATMUL_MULTI_CORE_REUSE_MCAST_EXAMPLE_SRC = $(TT_METAL_HOME)/tt_metal/programming_examples/matmul_multicore_reuse_mcast/matmul_multicore_reuse_mcast.cpp

# # Combine your example source with the necessary TT_METAL sources
# COMBINED_SRC = $(MATMUL_MULTI_CORE_REUSE_MCAST_EXAMPLE_SRC) $(TT_METAL_SRC)

# MATMUL_MULTI_CORE_REUSE_MCAST_EXAMPLES_OBJS = $(COMBINED_SRC:%.cpp=$(PROGRAMMING_EXAMPLES_OBJDIR)/%.o)
# MATMUL_MULTI_CORE_REUSE_MCAST_EXAMPLES_DEPS = $(COMBINED_SRC:%.cpp=$(PROGRAMMING_EXAMPLES_OBJDIR)/%.d)

# -include $(MATMUL_MULTI_CORE_REUSE_MCAST_EXAMPLES_DEPS)

# .PRECIOUS: $(PROGRAMMING_EXAMPLES_TESTDIR)/matmul_multicore_reuse_mcast
# $(PROGRAMMING_EXAMPLES_TESTDIR)/matmul_multicore_reuse_mcast: $(MATMUL_MULTI_CORE_REUSE_MCAST_EXAMPLES_OBJS) $(TT_METAL_LIB)
# 	@mkdir -p $(@D)
# 	$(CXX) $(CFLAGS) $(CXXFLAGS) $(PROGRAMMING_EXAMPLES_INCLUDES) -o $@ $^ $(LDFLAGS) $(PROGRAMMING_EXAMPLES_LDFLAGS)

# $(PROGRAMMING_EXAMPLES_OBJDIR)/%.o: %.cpp
# 	@mkdir -p $(@D)
# 	$(CXX) $(CFLAGS) $(CXXFLAGS) $(PROGRAMMING_EXAMPLES_INCLUDES) -c -o $@ $<


######## LINKING CPP FILES RECURSIVELY

# Define paths to the source directories
TENSOR_SRC_DIR = $(TT_METAL_HOME)/tt_eager/tensor
DNN_OP_LIB_SRC_DIR = $(TT_METAL_HOME)/tt_eager/tt_dnn/op_library

# Collect all .cpp files from the specified directories
TENSOR_SRCS = $(shell find $(TENSOR_SRC_DIR) -name '*.cpp')
DNN_OP_LIB_SRCS = $(shell find $(DNN_OP_LIB_SRC_DIR) -name '*.cpp')

# Specify the path to your matmul program source file
MATMUL_MULTI_CORE_REUSE_MCAST_EXAMPLE_SRC = $(TT_METAL_HOME)/tt_metal/programming_examples/matmul_multicore_reuse_mcast/matmul_multicore_reuse_mcast.cpp

# Combine your example source with the necessary TT_METAL sources
COMBINED_SRCS = $(MATMUL_MULTI_CORE_REUSE_MCAST_EXAMPLE_SRC) $(TENSOR_SRCS) $(DNN_OP_LIB_SRCS)

# Convert full source paths to object paths in the build directory
COMBINED_OBJS = $(patsubst $(TT_METAL_HOME)%,$(PROGRAMMING_EXAMPLES_OBJDIR)%,$(COMBINED_SRCS:.cpp=.o))

# Dependency files
COMBINED_DEPS = $(COMBINED_OBJS:.o=.d)

-include $(COMBINED_DEPS)

.PRECIOUS: $(PROGRAMMING_EXAMPLES_TESTDIR)/matmul_multicore_reuse_mcast
$(PROGRAMMING_EXAMPLES_TESTDIR)/matmul_multicore_reuse_mcast: $(COMBINED_OBJS) $(TT_METAL_LIB)
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) $(CXXFLAGS) $(PROGRAMMING_EXAMPLES_INCLUDES) -o $@ $^ $(LDFLAGS) $(PROGRAMMING_EXAMPLES_LDFLAGS)

# Rule for converting .cpp to .o, taking directory structure into account
$(PROGRAMMING_EXAMPLES_OBJDIR)/%.o: $(TT_METAL_HOME)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CFLAGS) $(CXXFLAGS) $(PROGRAMMING_EXAMPLES_INCLUDES) -c $< -o $@
