######## ORIGINAL MODULE
MATMUL_MULTI_CORE_REUSE_MCAST_EXAMPLE_SRC = $(TT_METAL_HOME)/tt_metal/programming_examples/matmul_multicore_reuse_mcast/matmul_multicore_reuse_mcast.cpp

MATMUL_MULTI_CORE_REUSE_MCAST_EXAMPLES_DEPS = $(PROGRAMMING_EXAMPLES_OBJDIR)/matmul_multicore_reuse_mcast.d

-include $(MATMUL_MULTI_CORE_REUSE_MCAST_EXAMPLES_DEPS)

.PRECIOUS: $(PROGRAMMING_EXAMPLES_TESTDIR)/matmul_multicore_reuse_mcast
$(PROGRAMMING_EXAMPLES_TESTDIR)/matmul_multicore_reuse_mcast: $(PROGRAMMING_EXAMPLES_OBJDIR)/matmul_multicore_reuse_mcast.o $(TT_METAL_LIB)
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) $(CXXFLAGS) $(PROGRAMMING_EXAMPLES_INCLUDES) -o $@ $^ $(LDFLAGS) $(PROGRAMMING_EXAMPLES_LDFLAGS)

.PRECIOUS: $(PROGRAMMING_EXAMPLES_OBJDIR)/matmul_multicore_reuse_mcast.o
$(PROGRAMMING_EXAMPLES_OBJDIR)/matmul_multicore_reuse_mcast.o: $(MATMUL_MULTI_CORE_REUSE_MCAST_EXAMPLE_SRC)
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) $(CXXFLAGS) $(PROGRAMMING_EXAMPLES_INCLUDES) -c -o $@ $<

# ######## LINKING CPP FILES RECURSIVELY

# TENSOR_SRC_DIR = $(TT_METAL_HOME)/tt_eager/tensor
# DNN_OP_LIB_SRC_DIR = $(TT_METAL_HOME)/tt_eager/tt_dnn/op_library

# TENSOR_SRCS = $(shell find $(TENSOR_SRC_DIR) -name '*.cpp')
# DNN_OP_LIB_SRCS = $(shell find $(DNN_OP_LIB_SRC_DIR) -name '*.cpp')

# MATMUL_MULTI_CORE_REUSE_MCAST_EXAMPLE_SRC = $(TT_METAL_HOME)/tt_metal/programming_examples/matmul_multicore_reuse_mcast/matmul_multicore_reuse_mcast.cpp

# COMBINED_SRCS = $(MATMUL_MULTI_CORE_REUSE_MCAST_EXAMPLE_SRC) $(TENSOR_SRCS) $(DNN_OP_LIB_SRCS)

# COMBINED_OBJS = $(patsubst $(TT_METAL_HOME)%,$(PROGRAMMING_EXAMPLES_OBJDIR)%,$(COMBINED_SRCS:.cpp=.o))

# COMBINED_DEPS = $(COMBINED_OBJS:.o=.d)

# -include $(COMBINED_DEPS)

# .PRECIOUS: $(PROGRAMMING_EXAMPLES_TESTDIR)/matmul_multicore_reuse_mcast
# $(PROGRAMMING_EXAMPLES_TESTDIR)/matmul_multicore_reuse_mcast: $(COMBINED_OBJS) $(TT_METAL_LIB)
# 	@mkdir -p $(@D)
# 	$(CXX) $(CFLAGS) $(CXXFLAGS) $(PROGRAMMING_EXAMPLES_INCLUDES) -o $@ $^ $(LDFLAGS) $(PROGRAMMING_EXAMPLES_LDFLAGS)

# $(PROGRAMMING_EXAMPLES_OBJDIR)/%.o: $(TT_METAL_HOME)/%.cpp
# 	@mkdir -p $(dir $@)
# 	$(CXX) $(CFLAGS) $(CXXFLAGS) $(PROGRAMMING_EXAMPLES_INCLUDES) -c $< -o $@


# ####### LINKING CPP FILES ONE-BY-ONE
# TT_METAL_SRC = $(TT_METAL_HOME)/tt_eager/tt_dnn/op_library/operation_history.cpp \

# MATMUL_MULTI_CORE_REUSE_MCAST_EXAMPLE_SRC = $(TT_METAL_HOME)/tt_metal/programming_examples/matmul_multicore_reuse_mcast/matmul_multicore_reuse_mcast.cpp

# # Combine above sources with all necessary TT_METAL sources
# COMBINED_SRC = $(MATMUL_MULTI_CORE_REUSE_MCAST_EXAMPLE_SRC) $(TT_METAL_SRC)

# MATMUL_MULTI_CORE_REUSE_MCAST_EXAMPLES_OBJS = $(patsubst $(TT_METAL_HOME)/%.cpp,$(PROGRAMMING_EXAMPLES_OBJDIR)/%.o,$(COMBINED_SRC))
# MATMUL_MULTI_CORE_REUSE_MCAST_EXAMPLES_DEPS = $(patsubst $(TT_METAL_HOME)/%.cpp,$(PROGRAMMING_EXAMPLES_OBJDIR)/%.d,$(COMBINED_SRC))

# -include $(MATMUL_MULTI_CORE_REUSE_MCAST_EXAMPLES_DEPS)

# .PRECIOUS: $(PROGRAMMING_EXAMPLES_TESTDIR)/matmul_multicore_reuse_mcast
# $(PROGRAMMING_EXAMPLES_TESTDIR)/matmul_multicore_reuse_mcast: $(MATMUL_MULTI_CORE_REUSE_MCAST_EXAMPLES_OBJS) $(TT_METAL_LIB)
# 	@mkdir -p $(@D)
# 	$(CXX) $(CFLAGS) $(CXXFLAGS) $(PROGRAMMING_EXAMPLES_INCLUDES) -o $@ $^ $(LDFLAGS) $(PROGRAMMING_EXAMPLES_LDFLAGS)

# $(PROGRAMMING_EXAMPLES_OBJDIR)/%.o: $(TT_METAL_HOME)/%.cpp
# 	@mkdir -p $(dir $@)
# 	$(CXX) $(CFLAGS) $(CXXFLAGS) $(PROGRAMMING_EXAMPLES_INCLUDES) -c $< -o $@
