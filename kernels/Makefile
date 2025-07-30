# Vortex toolchain
VORTEX_TOOLDIR ?= $(abspath ../../tools)
VORTEX_LLVM ?= $(VORTEX_TOOLDIR)/llvm-vortex
VORTEX_RISCV_TOOLCHAIN_PATH ?= $(VORTEX_TOOLDIR)/riscv64-gnu-toolchain
VORTEX_RISCV_PREFIX  ?= riscv64-unknown-elf
VORTEX_RISCV_SYSROOT ?= $(VORTEX_RISCV_TOOLCHAIN_PATH)/$(VORTEX_RISCV_PREFIX)

LIBC_VORTEX ?= $(VORTEX_TOOLDIR)/libc64
LIBCRT_VORTEX ?= $(VORTEX_TOOLDIR)/libcrt64

LLVM_CFLAGS += --sysroot=$(VORTEX_RISCV_SYSROOT)
LLVM_CFLAGS += --gcc-toolchain=$(VORTEX_RISCV_TOOLCHAIN_PATH)
LLVM_CFLAGS += -Xclang -target-feature -Xclang +vortex
LLVM_CFLAGS += -Xclang -target-feature -Xclang +zicond
LLVM_CFLAGS += -mllvm -disable-loop-idiom-all

VX_CC  = $(VORTEX_LLVM)/bin/clang $(LLVM_CFLAGS)
VX_CXX = $(VORTEX_LLVM)/bin/clang++ $(LLVM_CFLAGS)
VX_DP  = $(VORTEX_LLVM)/bin/llvm-objdump
VX_CP  = $(VORTEX_LLVM)/bin/llvm-objcopy

VX_CFLAGS += -O3 -mcmodel=medany -fno-rtti -fno-exceptions -nostartfiles -nostdlib -fdata-sections -ffunction-sections -std=c++20
VX_CFLAGS += -I$(VORTEX_HOME)/kernel/include -I$(VORTEX_BUILD_DIR)/hw
VX_CFLAGS += -DXLEN_64
VX_CFLAGS += -DNDEBUG

VX_CFLAGS += -march=rv64imafd -mabi=lp64d
STARTUP_ADDR ?= 0x180000000

VX_LIBS += -L$(LIBC_VORTEX)/lib -lm -lc
VX_LIBS += $(LIBCRT_VORTEX)/lib/baremetal/libclang_rt.builtins-riscv64.a

VX_LDFLAGS += -Wl,-Bstatic,--gc-sections,-T,$(VORTEX_HOME)/kernel/scripts/link64.ld,--defsym=STARTUP_ADDR=$(STARTUP_ADDR) $(VORTEX_KN_PATH)/libvortex.a $(VX_LIBS)

# Directories
BUILD_DIR = ./build
$(shell mkdir -p $(BUILD_DIR))

# Source and object files
SRCS = $(wildcard *.cpp)
TARGETS = $(patsubst %.cpp,$(BUILD_DIR)/%.elf,$(SRCS))
VXBINS = $(patsubst %.cpp,$(BUILD_DIR)/%.vxbin,$(SRCS))
DUMPS = $(patsubst %.cpp,$(BUILD_DIR)/%.dump,$(SRCS))

# Build all kernels
build: $(TARGETS) $(VXBINS) $(DUMPS)

# Pattern rules for each kernel
$(BUILD_DIR)/%.elf: %.cpp
	$(VX_CXX) $(VX_CFLAGS) $< $(VX_LDFLAGS) -o $@

$(BUILD_DIR)/%.vxbin: $(BUILD_DIR)/%.elf
	OBJCOPY=$(VX_CP) $(VORTEX_HOME)/kernel/scripts/vxbin.py $< $@

$(BUILD_DIR)/%.dump: $(BUILD_DIR)/%.elf
	$(VX_DP) -D $< > $@

# Clean build directory
clean:
	rm -rf $(BUILD_DIR)

.PHONY: build clean