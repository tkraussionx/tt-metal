# Every variable in subdir must be prefixed with subdir (emulating a namespace)
#TT_DISPATCH_TESTS += $(basename $(wildcard tt_dispatch/*.cpp))
TT_DISPATCH_TESTS += tests/frameworks/tt_dispatch/test_copy_descriptor \
					 tests/frameworks/tt_dispatch/test_dispatch_primitives

TT_DISPATCH_TESTS_SRCS = $(addprefix tests/frameworks/tt_dispatch/, $(addsuffix .cpp, $(TT_DISPATCH_TESTS:tests%=%)))

TT_DISPATCH_TEST_INCLUDES = $(TEST_INCLUDES) $(TT_METAL_INCLUDES) -Itt_gdb -I$(TT_METAL_HOME)/frameworks/tt_dispatch

# TT_DISPATCH_TESTS_LDFLAGS = -ltt_dispatch -ltensor -ltt_dnn -ldtx -ltt_metal_impl -ltt_metal -lllrt -ltt_gdb -ldevice -lbuild_kernels_for_riscv -ldl -lcommon -lprofiler -lstdc++fs -pthread -lyaml-cpp

TT_DISPATCH_TESTS_LDFLAGS = -ltensor -ltt_dnn -ldtx -ltt_metal_impl -ltt_metal -lllrt -ltt_gdb -ldevice -lbuild_kernels_for_riscv -ldl -lcommon -lprofiler -lstdc++fs -pthread -lyaml-cpp

TT_DISPATCH_TESTS_OBJS = $(addprefix $(OBJDIR)/, $(TT_DISPATCH_TESTS_SRCS:.cpp=.o))
TT_DISPATCH_TESTS_DEPS = $(addprefix $(OBJDIR)/, $(TT_DISPATCH_TESTS_SRCS:.cpp=.d))

-include $(TT_DISPATCH_TESTS_DEPS)

# Each module has a top level target as the entrypoint which must match the subdir name
tests/frameworks/tt_dispatch: $(TT_DISPATCH_TESTS)

tests/frameworks/tt_dispatch/all: $(TT_DISPATCH_TESTS)

tests/frameworks/tt_dispatch/test_%: $(TESTDIR)/frameworks/tt_dispatch/test_% ;

.PRECIOUS: $(TESTDIR)/frameworks/tt_dispatch/test_%
$(TESTDIR)/frameworks/tt_dispatch/test_%: $(OBJDIR)/frameworks/tt_dispatch/tests/test_%.o $(TT_DISPATCH_LIB)
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) $(CXXFLAGS) $(TT_DISPATCH_TEST_INCLUDES) -o $@ $^ $(LDFLAGS) $(TT_DISPATCH_TESTS_LDFLAGS)

.PRECIOUS: $(OBJDIR)/frameworks/tt_dispatch/tests/test_%.o
$(OBJDIR)/frameworks/tt_dispatch/tests/test_%.o: tests/frameworks/tt_dispatch/test_%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) $(CXXFLAGS) $(TT_DISPATCH_TEST_INCLUDES) -c -o $@ $<

tt_dispatch/tests: tests/frameworks/tt_dispatch
tt_dispatch/tests/all: tests/frameworks/tt_dispatch/all
