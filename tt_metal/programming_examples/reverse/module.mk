SRC = $(TT_METAL_HOME)/tt_metal/programming_examples/reverse/single_core_reverse.cpp
SRC2 = $(TT_METAL_HOME)/tt_metal/programming_examples/reverse/single_core_rw_reverse.cpp
SRC3 = $(TT_METAL_HOME)/tt_metal/programming_examples/reverse/all_core_reverse.cpp



-include $(LOOPBACK_EXAMPLES_DEPS)

.PRECIOUS: $(PROGRAMMING_EXAMPLES_TESTDIR)/reverse_single_core
$(PROGRAMMING_EXAMPLES_TESTDIR)/reverse_single_core: $(SRC) $(TT_METAL_LIB)
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) $(CXXFLAGS) $(PROGRAMMING_EXAMPLES_INCLUDES) -o $@ $(SRC) $(LDFLAGS) $(PROGRAMMING_EXAMPLES_LDFLAGS)

.PRECIOUS: $(PROGRAMMING_EXAMPLES_TESTDIR)/reverse_single_core_rw
$(PROGRAMMING_EXAMPLES_TESTDIR)/reverse_single_core_rw: $(SRC2) $(TT_METAL_LIB)
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) $(CXXFLAGS) $(PROGRAMMING_EXAMPLES_INCLUDES) -o $@ $(SRC2) $(LDFLAGS) $(PROGRAMMING_EXAMPLES_LDFLAGS)



.PRECIOUS: $(PROGRAMMING_EXAMPLES_TESTDIR)/reverse_multi_core
$(PROGRAMMING_EXAMPLES_TESTDIR)/reverse_multi_core: $(SRC3) $(TT_METAL_LIB)
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) $(CXXFLAGS) $(PROGRAMMING_EXAMPLES_INCLUDES) -o $@ $(SRC3) $(LDFLAGS) $(PROGRAMMING_EXAMPLES_LDFLAGS)
