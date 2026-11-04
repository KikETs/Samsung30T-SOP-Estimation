# V322b: paper SOP predictor (Ls + V284 adaptation) in pure float32 on NUCLEO-H563ZI.
#   make            build, sizes, and the soft-double check (Build/double_check.txt)
# Board support (startup, linker script, HAL configuration, interrupt and system files) is read unchanged from ../fw (V322).
PROJECT := sop_v322b
BUILD_DIR := Build
BSP := ../fw

# GNU Arm toolchain: arm-none-eabi-* on PATH, or TOOLCHAIN=<its bin directory>
TOOLCHAIN ?=
TC := $(if $(TOOLCHAIN),$(TOOLCHAIN)/,)
CC := $(TC)arm-none-eabi-gcc
OBJCOPY := $(TC)arm-none-eabi-objcopy
OBJDUMP := $(TC)arm-none-eabi-objdump
SIZE := $(TC)arm-none-eabi-size
NM := $(TC)arm-none-eabi-nm

# STM32CubeH5 firmware package (HAL drivers and CMSIS), unpacked in mcu_port_v322/
FW_ROOT ?= ../STM32Cube_FW_H5_V1.7.0
HAL_SRC := $(FW_ROOT)/Drivers/STM32H5xx_HAL_Driver/Src
HAL_INC := $(FW_ROOT)/Drivers/STM32H5xx_HAL_Driver/Inc
CMSIS_DEVICE := $(FW_ROOT)/Drivers/CMSIS/Device/ST/STM32H5xx/Include
CMSIS_CORE := $(FW_ROOT)/Drivers/CMSIS/Include
PORT_DIR := ../c
CFLAGS := -mcpu=cortex-m33 -mthumb -mfpu=fpv5-sp-d16 -mfloat-abi=hard
CFLAGS += -std=gnu11 -O2 -ffunction-sections -fdata-sections -Wall
# no fused multiply-add, so the board runs the same operations as the host float build (x86-64, no FMA)
CFLAGS += -ffp-contract=off
CFLAGS += -fstack-usage -MMD -MP --specs=nano.specs
CFLAGS += -DUSE_HAL_DRIVER -DSTM32H563xx
CFLAGS += -I$(BSP)/Inc -I$(PORT_DIR) -I$(HAL_INC) -I$(HAL_INC)/Legacy -I$(CMSIS_DEVICE) -I$(CMSIS_CORE)
CFLAGS += $(EXTRA_CFLAGS)
# the SOP object and the harness: warn on any implicit promotion to double or narrowing of a real value
STRICT := -Wdouble-promotion -Wfloat-conversion
ASFLAGS := $(CFLAGS) -x assembler-with-cpp
LDFLAGS := -mcpu=cortex-m33 -mthumb -mfpu=fpv5-sp-d16 -mfloat-abi=hard
LDFLAGS += -T$(BSP)/STM32H563ZITX_FLASH.ld --specs=nosys.specs --specs=nano.specs
LDFLAGS += -Wl,-Map=$(BUILD_DIR)/$(PROJECT).map -Wl,--gc-sections -static

APP_SRCS := Src/main.c Src/coef_v322b.c
BSP_SRCS := \
	$(BSP)/Src/stm32h5xx_hal_msp.c \
	$(BSP)/Src/stm32h5xx_it.c \
	$(BSP)/Src/syscalls.c \
	$(BSP)/Src/sysmem.c \
	$(BSP)/Src/system_stm32h5xx.c

HAL_SRCS := \
	$(HAL_SRC)/stm32h5xx_hal.c \
	$(HAL_SRC)/stm32h5xx_hal_cortex.c \
	$(HAL_SRC)/stm32h5xx_hal_dma.c \
	$(HAL_SRC)/stm32h5xx_hal_dma_ex.c \
	$(HAL_SRC)/stm32h5xx_hal_exti.c \
	$(HAL_SRC)/stm32h5xx_hal_flash.c \
	$(HAL_SRC)/stm32h5xx_hal_flash_ex.c \
	$(HAL_SRC)/stm32h5xx_hal_gpio.c \
	$(HAL_SRC)/stm32h5xx_hal_icache.c \
	$(HAL_SRC)/stm32h5xx_hal_pwr.c \
	$(HAL_SRC)/stm32h5xx_hal_pwr_ex.c \
	$(HAL_SRC)/stm32h5xx_hal_rcc.c \
	$(HAL_SRC)/stm32h5xx_hal_rcc_ex.c \
	$(HAL_SRC)/stm32h5xx_hal_uart.c \
	$(HAL_SRC)/stm32h5xx_hal_uart_ex.c

APP_OBJS := $(patsubst %.c,$(BUILD_DIR)/%.o,$(APP_SRCS))
BSP_OBJS := $(patsubst $(BSP)/Src/%.c,$(BUILD_DIR)/BSP/%.o,$(BSP_SRCS))
PORT_OBJ := $(BUILD_DIR)/port/sop_v322b.o
HAL_OBJS := $(patsubst $(HAL_SRC)/%.c,$(BUILD_DIR)/HAL/%.o,$(HAL_SRCS))
STARTUP_OBJ := $(BUILD_DIR)/Startup/startup_stm32h563zitx.o
OBJS := $(APP_OBJS) $(BSP_OBJS) $(PORT_OBJ) $(HAL_OBJS) $(STARTUP_OBJ)
DEPS := $(OBJS:.o=.d)
ELF := $(BUILD_DIR)/$(PROJECT).elf

# software double-precision helpers of libgcc (ARM EABI and generic names)
SOFTD := '__aeabi_(d[a-z0-9]+|cd[a-z]+|f2d|i2d|ui2d|l2d|ul2d)$$|__(add|sub|mul|div|neg)df[23]$$|__(extendsfdf2|truncdfsf2|fixdfsi|fixunsdfsi|fixdfdi|fixunsdfdi|floatsidf|floatunsidf|floatdidf|floatundidf|eqdf2|nedf2|ltdf2|ledf2|gtdf2|gedf2|unorddf2|cmpdf2)$$'

.PHONY: all clean double_check
all: $(ELF) $(BUILD_DIR)/$(PROJECT).bin $(BUILD_DIR)/$(PROJECT).hex $(BUILD_DIR)/$(PROJECT).list double_check
	$(SIZE) $(ELF) | tee $(BUILD_DIR)/size.txt
	$(SIZE) $(PORT_OBJ) $(BUILD_DIR)/Src/coef_v322b.o | tee $(BUILD_DIR)/size_objs.txt

double_check: $(ELF)
	@{ echo "SOP object ($(PORT_OBJ)), undefined symbols:"; $(NM) -u $(PORT_OBJ); \
	   echo "SOP object, soft-double calls:"; $(NM) -u $(PORT_OBJ) | grep -E $(SOFTD) || echo "  none"; \
	   echo "harness object (Build/Src/main.o), soft-double calls:"; $(NM) -u $(BUILD_DIR)/Src/main.o | grep -E $(SOFTD) || echo "  none"; \
	   echo "linked image ($(ELF)), soft-double helpers present:"; $(NM) $(ELF) | grep -E $(SOFTD) || echo "  none"; \
	   echo "library members linked (libm, libgcc):"; grep -oE '(libm|libgcc)\.a\([^)]+\)' $(BUILD_DIR)/$(PROJECT).map | sort -u; } | tee $(BUILD_DIR)/double_check.txt

$(ELF): $(OBJS) $(BSP)/STM32H563ZITX_FLASH.ld
	@mkdir -p $(dir $@)
	$(CC) $(OBJS) $(LDFLAGS) -Wl,--start-group -lc -lm -Wl,--end-group -o $@

$(BUILD_DIR)/$(PROJECT).bin: $(ELF)
	$(OBJCOPY) -O binary $< $@

$(BUILD_DIR)/$(PROJECT).hex: $(ELF)
	$(OBJCOPY) -O ihex $< $@

$(BUILD_DIR)/$(PROJECT).list: $(ELF)
	$(OBJDUMP) -h -S $< > $@

$(BUILD_DIR)/Src/%.o: Src/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(STRICT) -c $< -o $@

$(BUILD_DIR)/BSP/%.o: $(BSP)/Src/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

$(PORT_OBJ): $(PORT_DIR)/sop_v322b.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(STRICT) -c $< -o $@

$(BUILD_DIR)/HAL/%.o: $(HAL_SRC)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

$(STARTUP_OBJ): $(BSP)/Startup/startup_stm32h563zitx.s
	@mkdir -p $(dir $@)
	$(CC) $(ASFLAGS) -c $< -o $@

clean:
	rm -rf Build

-include $(DEPS)
