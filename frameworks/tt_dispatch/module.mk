# Every variable in subdir must be prefixed with subdir (emulating a namespace)

TT_DISPATCH_FRAMEWORK = $(LIBDIR)/libtt_dispatch.a
TT_DISPATCH_DEFINES = -DGIT_HASH=$(shell git rev-parse HEAD)
TT_DISPATCH_INCLUDES = $(COMMON_INCLUDES) -I$(TT_METAL_HOME)/frameworks/tt_dispatch
TT_DISPATCH_LDFLAGS = -L$(TT_METAL_HOME) -ltt_gdb -ldevice -lcommon
TT_DISPATCH_CFLAGS = $(CFLAGS) -Werror -Wno-int-to-pointer-cast

TT_DISPATCH_SRCS_RELATIVE = \
	tt_dispatch/impl/dispatch.cpp

TT_DISPATCH_SRCS = $(addprefix frameworks/, $(TT_DISPATCH_SRCS_RELATIVE))

TT_DISPATCH_OBJS = $(addprefix $(OBJDIR)/, $(TT_DISPATCH_SRCS:.cpp=.o))
TT_DISPATCH_DEPS = $(addprefix $(OBJDIR)/, $(TT_DISPATCH_SRCS:.cpp=.d))

-include $(TT_DISPATCH_DEPS)

# Each module has a top level target as the entrypoint which must match the subdir name
tt_dispatch: $(TT_DISPATCH_FRAMEWORK)

$(TT_DISPATCH_FRAMEWORK): $(COMMON_LIB) $(NETLIST_LIB) $(TT_DISPATCH_OBJS) $(DEVICE_LIB)
	@mkdir -p $(@D)
	ar rcs -o $@ $(TT_DISPATCH_OBJS)

$(OBJDIR)/frameworks/tt_dispatch/%.o: frameworks/tt_dispatch/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(TT_DISPATCH_CFLAGS) $(CXXFLAGS) $(STATIC_LIB_FLAGS) $(TT_DISPATCH_INCLUDES) $(TT_DISPATCH_DEFINES) -c -o $@ $<
