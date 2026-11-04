# V322: paper SOP predictor (Ls + V284 adaptation) on NUCLEO-H563ZI.
#   make PREC=float    float32 arithmetic (-DSOP_FLOAT; time stamps and Ah stay double)
#   make PREC=double   double arithmetic (software double on the FPv5-SP)
PROJECT := sop_v322
PREC ?= float
BUILD_DIR := Build/$(PREC)

# GNU Arm toolchain: arm-none-eabi-* on PATH, or TOOLCHAIN=<its bin directory>
TOOLCHAIN ?=
TC := $(if $(TOOLCHAIN),$(TOOLCHAIN)/,)
CC := $(TC)arm-none-eabi-gcc
OBJCOPY := $(TC)arm-none-eabi-objcopy
OBJDUMP := $(TC)arm-none-eabi-objdump
SIZE := $(TC)arm-none-eabi-size

# STM32CubeH5 firmware package (HAL drivers and CMSIS), unpacked in mcu_port_v322/
FW_ROOT ?= ../STM32Cube_FW_H5_V1.7.0
HAL_SRC := $(FW_ROOT)/Drivers/STM32H5xx_HAL_Driver/Src
HAL_INC := $(FW_ROOT)/Drivers/STM32H5xx_HAL_Driver/Inc
CMSIS_DEVICE := $(FW_ROOT)/Drivers/CMSIS/Device/ST/STM32H5xx/Include
CMSIS_CORE := $(FW_ROOT)/Drivers/CMSIS/Include
PORT_DIR := ../c
CFLAGS := -mcpu=cortex-m33 -mthumb -mfpu=fpv5-sp-d16 -mfloat-abi=hard
CFLAGS += -std=gnu11 -O2 -ffunction-sections -fdata-sections -Wall
# no fused multiply-add, so the float build runs the same operations as the host float build (x86-64, no FMA)
CFLAGS += -ffp-contract=off
CFLAGS += -fstack-usage -MMD -MP --specs=nano.specs
CFLAGS += -DUSE_HAL_DRIVER -DSTM32H563xx
CFLAGS += -IInc -I$(PORT_DIR) -I$(HAL_INC) -I$(HAL_INC)/Legacy -I$(CMSIS_DEVICE) -I$(CMSIS_CORE)
ifeq ($(PREC),float)
CFLAGS += -DSOP_FLOAT
endif
CFLAGS += $(EXTRA_CFLAGS)
ASFLAGS := $(CFLAGS) -x assembler-with-cpp
LDFLAGS := -mcpu=cortex-m33 -mthumb -mfpu=fpv5-sp-d16 -mfloat-abi=hard
LDFLAGS += -TSTM32H563ZITX_FLASH.ld --specs=nosys.specs --specs=nano.specs
LDFLAGS += -Wl,-Map=$(BUILD_DIR)/$(PROJECT).map -Wl,--gc-sections -static

APP_SRCS := \
	Src/main.c \
	Src/coef_v322.c \
	Src/stm32h5xx_hal_msp.c \
	Src/stm32h5xx_it.c \
	Src/syscalls.c \
	Src/sysmem.c \
	Src/system_stm32h5xx.c

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
PORT_OBJ := $(BUILD_DIR)/port/sop_v322.o
HAL_OBJS := $(patsubst $(HAL_SRC)/%.c,$(BUILD_DIR)/HAL/%.o,$(HAL_SRCS))
STARTUP_OBJ := $(BUILD_DIR)/Startup/startup_stm32h563zitx.o
OBJS := $(APP_OBJS) $(PORT_OBJ) $(HAL_OBJS) $(STARTUP_OBJ)
DEPS := $(OBJS:.o=.d)
ELF := $(BUILD_DIR)/$(PROJECT).elf

.PHONY: all clean
all: $(ELF) $(BUILD_DIR)/$(PROJECT).bin $(BUILD_DIR)/$(PROJECT).hex $(BUILD_DIR)/$(PROJECT).list
	$(SIZE) $(ELF) | tee $(BUILD_DIR)/size.txt
	$(SIZE) $(PORT_OBJ) $(BUILD_DIR)/Src/coef_v322.o | tee $(BUILD_DIR)/size_objs.txt

$(ELF): $(OBJS) STM32H563ZITX_FLASH.ld
	@mkdir -p $(dir $@)
	$(CC) $(OBJS) $(LDFLAGS) -Wl,--start-group -lc -lm -Wl,--end-group -o $@

$(BUILD_DIR)/$(PROJECT).bin: $(ELF)
	$(OBJCOPY) -O binary $< $@

$(BUILD_DIR)/$(PROJECT).hex: $(ELF)
	$(OBJCOPY) -O ihex $< $@

$(BUILD_DIR)/$(PROJECT).list: $(ELF)
	$(OBJDUMP) -h -S $< > $@

$(BUILD_DIR)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

$(PORT_OBJ): $(PORT_DIR)/sop_v322.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/HAL/%.o: $(HAL_SRC)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

$(STARTUP_OBJ): Startup/startup_stm32h563zitx.s
	@mkdir -p $(dir $@)
	$(CC) $(ASFLAGS) -c $< -o $@

clean:
	rm -rf Build

-include $(DEPS)
