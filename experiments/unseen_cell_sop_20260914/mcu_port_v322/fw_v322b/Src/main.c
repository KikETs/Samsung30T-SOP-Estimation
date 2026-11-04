/* V322b firmware for NUCLEO-H563ZI (Cortex-M33, 250 MHz): the paper's SOP predictor (V287 Ls + V284 adaptation) in pure
 * float32 (c/sop_v322b.c), fed sample by sample over USART3 (ST-LINK VCP, 921600 baud). No double anywhere: samples
 * arrive as integers and a float32 temperature, replies carry float32 values and the integer charge accumulator (int64),
 * and the PC converts. Cycle counts from DWT with interrupts disabled around the measured call only. Board support
 * (clock, UART, cache, stack watermark) is copied from mcu/fw_sop, as in the V322 firmware. */
#include "main.h"
#include <string.h>
#include "sop_v322b.h"

UART_HandleTypeDef huart3;
static uint32_t *g_stack_paint_end;
extern uint32_t _sstack;
extern uint32_t _estack;
extern const fold_coef_t *const COEF_TAB;
#define STACK_PATTERN 0xA5A5A5A5UL

void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_USART3_UART_Init(void);
static void MX_ICACHE_Init(void);
static void CEMA_DWT_Init(void);
static void Stack_Watermark_Init(void);
static uint32_t Stack_Highwater_Bytes(void);

#define V322B_MAGIC  0x42323356UL               /* "V32B" */
#define CMD_INFO     0x70U
#define CMD_WIN      0x71U
#define CMD_SAMPLES  0x72U
#define CMD_QUERY    0x73U
#define CMD_AD_START 0x74U
#define CMD_AD_REQ   0x75U
#define CMD_AD_EVENT 0x76U
#define CMD_STATS    0x77U
#define MAX_BATCH    256U
#define NQ           672U
#define HIST_N       1024U
#define HIST_W       64U

static eng_t g_eng;
static int g_cell;
static feats_t g_f;
static pred_out_t g_o;
static float g_centre[NQ];
static ad_t g_ad;
static uint8_t g_rx[MAX_BATCH * 32U];
static uint32_t g_hist[HIST_N], g_hist_over, g_cnt, g_max;
static uint64_t g_sum;

static void tx(const void *d, uint32_t n) { (void)HAL_UART_Transmit(&huart3, (const uint8_t *)d, (uint16_t)n, HAL_MAX_DELAY); }
static int rx(void *d, uint32_t n) { return HAL_UART_Receive(&huart3, (uint8_t *)d, (uint16_t)n, 5000U) == HAL_OK; }
static uint32_t rd_u32(const uint8_t *p) { uint32_t v; memcpy(&v, p, 4); return v; }

static void put_f(uint8_t **w, float v) { memcpy(*w, &v, 4); *w += 4; }
static void put_i(uint8_t **w, int32_t v) { memcpy(*w, &v, 4); *w += 4; }
static void put_l(uint8_t **w, int64_t v) { memcpy(*w, &v, 8); *w += 8; }
static void put_int(uint8_t **w, const interr_t *it)
{
  put_f(w, it->I_off); put_f(w, it->off_dur_s);
  for (int k = 0; k < NTOFF; k++) put_f(w, it->Roff[k]);
  put_f(w, it->sq_R0); put_f(w, it->sq_A); put_f(w, it->Rstep1); put_f(w, it->Rstep2); put_f(w, it->Rstep10);
}
static void put_pulse(uint8_t **w, const pulse_t *p)
{
  put_f(w, p->P); put_f(w, p->Vpre); put_f(w, p->I_last); put_f(w, p->V_last); put_f(w, p->dt_last);
  put_f(w, p->R1); put_f(w, p->R2); put_f(w, p->R10);
  put_int(w, &p->it); put_f(w, p->rest_s);
}

static void h_info(void)
{
  uint32_t pm = __get_PRIMASK(); __disable_irq(); __DSB(); __ISB();
  DWT->CYCCNT = 0UL;
  uint32_t c0 = DWT->CYCCNT;                      /* cost of the measurement itself (an empty interval) */
  if (pm == 0U) __enable_irq();
  uint32_t r[11] = {V322B_MAGIC, 2U, (uint32_t)sizeof(float), SystemCoreClock, (uint32_t)sizeof(eng_t), (uint32_t)sizeof(feats_t),
                    (uint32_t)sizeof(pred_out_t), (uint32_t)sizeof(ad_t), (uint32_t)(3U * sizeof(fold_coef_t)), Stack_Highwater_Bytes(), c0};
  tx(r, sizeof r);
}

static void h_win(void)
{
  uint8_t p[8]; uint32_t r[2] = {V322B_MAGIC, 0xFFFFFFFFU};
  if (rx(p, 8)) { g_cell = (int)rd_u32(p); eng_reset(&g_eng); r[1] = rd_u32(p + 4); }
  tx(r, sizeof r);
}

static void h_samples(void)
{
  uint8_t p[4]; uint32_t r[5] = {V322B_MAGIC, 0U, 0U, 0U, 0U};
  if (!rx(p, 4)) { r[1] = 0xFFFFFFFFU; tx(r, sizeof r); return; }
  uint32_t n = rd_u32(p);
  if (n > MAX_BATCH || !rx(g_rx, n * 32U)) { r[1] = 0xFFFFFFFFU; tx(r, sizeof r); return; }
  uint32_t sum = 0, mx = 0, ck = 0;
  for (uint32_t i = 0; i < n * 32U; i++) ck += g_rx[i];
  for (uint32_t i = 0; i < n; i++) {
    const uint8_t *b = g_rx + 32U * i;
    int64_t t; uint32_t st; int32_t key, I, V; float T;
    memcpy(&t, b, 8); memcpy(&st, b + 8, 4); memcpy(&key, b + 12, 4); memcpy(&I, b + 16, 4); memcpy(&V, b + 20, 4); memcpy(&T, b + 24, 4);
    uint32_t pm = __get_PRIMASK(); __disable_irq(); __DSB(); __ISB();
    DWT->CYCCNT = 0UL;
    eng_push(&g_eng, t, st, key, I, V, T);
    uint32_t c = DWT->CYCCNT;
    if (pm == 0U) __enable_irq();
    sum += c; if (c > mx) mx = c;
    uint32_t bin = c / HIST_W; if (bin < HIST_N) g_hist[bin]++; else g_hist_over++;
    g_cnt++; g_sum += c; if (c > g_max) g_max = c;
  }
  r[1] = n; r[2] = sum; r[3] = mx; r[4] = ck;
  tx(r, sizeof r);
}

static void h_query(void)
{
  static uint8_t out[512];
  uint8_t p[12]; uint8_t *w = out;
  uint32_t qid = 0xFFFFFFFFU, c1 = 0, c2 = 0;
  if (rx(p, 12)) {
    qid = rd_u32(p); int dir = (int)rd_u32(p + 4); uint32_t hms = rd_u32(p + 8);
    uint32_t pm = __get_PRIMASK(); __disable_irq(); __DSB(); __ISB();
    DWT->CYCCNT = 0UL;
    eng_query(&g_eng, dir, &g_f);
    c1 = DWT->CYCCNT;
    sop_predict(&g_f, dir, hms, &COEF_TAB[g_cell], &g_o);
    c2 = DWT->CYCCNT;
    if (pm == 0U) __enable_irq();
    if (qid < NQ) g_centre[qid] = g_o.centre;
  }
  uint32_t hdr[5] = {V322B_MAGIC, qid, c1, c2 - c1, Stack_Highwater_Bytes()};
  memcpy(w, hdr, sizeof hdr); w += sizeof hdr;
  put_i(&w, g_f.valid); put_i(&w, g_f.a_zero); put_i(&w, g_f.rb_trunc); put_i(&w, g_f.last_sign); put_i(&w, g_o.base); put_i(&w, g_o.reg);
  put_i(&w, g_o.nfinite); put_i(&w, g_f.prev.valid); put_i(&w, g_f.prev.k0); put_i(&w, g_f.prev.has_int);
  put_i(&w, g_f.prev2.valid); put_i(&w, g_f.prev2.k0); put_i(&w, g_f.prev2.has_int);
  put_i(&w, g_f.it.has_fit); put_i(&w, g_f.prev.it.has_fit); put_i(&w, g_f.prev2.it.has_fit);
  put_l(&w, g_f.ah2_q); put_l(&w, g_f.prev.ah2); put_l(&w, g_f.prev2.ah2);
  put_f(&w, g_f.V_pre); put_f(&w, g_f.T_pre); put_f(&w, g_f.rest_s); put_f(&w, g_f.I_end); put_f(&w, g_f.last_dur_s); put_f(&w, g_f.V_end);
  put_f(&w, g_f.V_rest0); put_f(&w, g_f.dt_rest0); put_f(&w, g_f.slope_mVs);
  put_int(&w, &g_f.it); put_pulse(&w, &g_f.prev); put_pulse(&w, &g_f.prev2); put_f(&w, g_f.prev_gap_s);
  for (int k = 0; k < NPRED; k++) put_f(&w, g_o.pred[k]);
  put_f(&w, g_o.centre);
  tx(out, (uint32_t)(w - out));
}

static void h_ad_start(void)
{
  uint8_t p[4]; uint32_t r[2] = {V322B_MAGIC, 0xFFFFFFFFU};
  if (rx(p, 4)) { uint32_t c = rd_u32(p); if (c < 3U) { ad_init(&g_ad, COEF_TAB[c].s0); r[1] = c; } }
  tx(r, sizeof r);
}

static void h_ad_req(void)
{
  uint8_t p[4], out[24]; uint32_t qid = 0xFFFFFFFFU, c = 0; float s = 0.0f, req = 0.0f;
  if (rx(p, 4)) {
    qid = rd_u32(p);
    uint32_t pm = __get_PRIMASK(); __disable_irq(); __DSB(); __ISB();
    DWT->CYCCNT = 0UL;
    req = ad_request(&g_ad, qid < NQ ? g_centre[qid] : 0.0f, &s);
    c = DWT->CYCCNT;
    if (pm == 0U) __enable_irq();
  }
  uint32_t hdr[4] = {V322B_MAGIC, qid, c, 0U}; uint8_t *w = out;
  memcpy(w, hdr, sizeof hdr); w += sizeof hdr; put_f(&w, s); put_f(&w, req);
  tx(out, (uint32_t)(w - out));
}

static void h_ad_event(void)
{
  uint8_t p[4]; uint32_t r[2] = {V322B_MAGIC, 0xFFFFFFFFU};
  if (rx(p, 4)) {
    uint32_t ev = rd_u32(p);
    uint32_t pm = __get_PRIMASK(); __disable_irq(); __DSB(); __ISB();
    DWT->CYCCNT = 0UL;
    ad_update(&g_ad, ev != 0U);
    r[1] = DWT->CYCCNT;
    if (pm == 0U) __enable_irq();
  }
  tx(r, sizeof r);
}

static void h_stats(void)
{
  uint32_t r[6] = {V322B_MAGIC, g_cnt, g_max, (uint32_t)(g_sum & 0xFFFFFFFFU), (uint32_t)(g_sum >> 32), g_hist_over};
  tx(r, sizeof r); tx(g_hist, sizeof g_hist);
}

static void V322B_Loop(void)
{
  uint8_t cmd;
  for (;;) {
    if (HAL_UART_Receive(&huart3, &cmd, 1U, HAL_MAX_DELAY) != HAL_OK) continue;
    switch (cmd) {
      case CMD_INFO: h_info(); break;
      case CMD_WIN: h_win(); break;
      case CMD_SAMPLES: h_samples(); break;
      case CMD_QUERY: h_query(); break;
      case CMD_AD_START: h_ad_start(); break;
      case CMD_AD_REQ: h_ad_req(); break;
      case CMD_AD_EVENT: h_ad_event(); break;
      case CMD_STATS: h_stats(); break;
      default: break;
    }
  }
}

int main(void)
{
  HAL_Init();
  SystemClock_Config();
  MX_GPIO_Init();
  MX_USART3_UART_Init();
  MX_ICACHE_Init();
  CEMA_DWT_Init();
  Stack_Watermark_Init();
  HAL_GPIO_WritePin(LED1_GPIO_Port, LED1_Pin, GPIO_PIN_SET);
  V322B_Loop();
  while (1) { }
}

/* ---- copied from mcu/fw_sop/Src/main.c (board support) ---- */
static void CEMA_DWT_Init(void)
{
  CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
#if defined(DWT_LAR)
  DWT->LAR = 0xC5ACCE55UL;
#endif
  DWT->CYCCNT = 0UL;
  DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
  __DSB();
  __ISB();
}

static void Stack_Watermark_Init(void)
{
  uintptr_t stack_pointer;
  uint32_t *cursor;

  __asm volatile ("mrs %0, msp" : "=r" (stack_pointer));
  if (stack_pointer > ((uintptr_t)&_sstack + 512U))
  {
    g_stack_paint_end = (uint32_t *)(stack_pointer - 256U);
    for (cursor = &_sstack; cursor < g_stack_paint_end; ++cursor)
    {
      *cursor = STACK_PATTERN;
    }
  }
  else
  {
    g_stack_paint_end = &_sstack;
  }
}

static uint32_t Stack_Highwater_Bytes(void)
{
  uint32_t *cursor;
  uint32_t *lowest_used = g_stack_paint_end;

  for (cursor = &_sstack; cursor < g_stack_paint_end; ++cursor)
  {
    if (*cursor != STACK_PATTERN)
    {
      lowest_used = cursor;
      break;
    }
  }
  return (uint32_t)((uintptr_t)&_estack - (uintptr_t)lowest_used);
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the main internal regulator output voltage
  */
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE0);

  while(!__HAL_PWR_GET_FLAG(PWR_FLAG_VOSRDY)) {}

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
  RCC_OscInitStruct.HSEState = RCC_HSE_BYPASS_DIGITAL;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLL1_SOURCE_HSE;
  RCC_OscInitStruct.PLL.PLLM = 4;
  RCC_OscInitStruct.PLL.PLLN = 250;
  RCC_OscInitStruct.PLL.PLLP = 2;
  RCC_OscInitStruct.PLL.PLLQ = 2;
  RCC_OscInitStruct.PLL.PLLR = 2;
  RCC_OscInitStruct.PLL.PLLRGE = RCC_PLL1_VCIRANGE_1;
  RCC_OscInitStruct.PLL.PLLVCOSEL = RCC_PLL1_VCORANGE_WIDE;
  RCC_OscInitStruct.PLL.PLLFRACN = 0;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2
                              |RCC_CLOCKTYPE_PCLK3;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV1;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;
  RCC_ClkInitStruct.APB3CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_5) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure the programming delay
  */
  __HAL_FLASH_SET_PROGRAM_DELAY(FLASH_PROGRAMMING_DELAY_2);
}

/**
  * @brief ICACHE Initialization Function
  * @param None
  * @retval None
  */
static void MX_ICACHE_Init(void)
{

  /* USER CODE BEGIN ICACHE_Init 0 */

  /* USER CODE END ICACHE_Init 0 */

  /* USER CODE BEGIN ICACHE_Init 1 */

  /* USER CODE END ICACHE_Init 1 */

  /** Enable instruction cache in 1-way (direct mapped cache)
  */
  if (HAL_ICACHE_ConfigAssociativityMode(ICACHE_1WAY) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_ICACHE_Enable() != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN ICACHE_Init 2 */

  /* USER CODE END ICACHE_Init 2 */

}

/**
  * @brief USART3 Initialization Function
  * @param None
  * @retval None
  */
static void MX_USART3_UART_Init(void)
{

  /* USER CODE BEGIN USART3_Init 0 */

  /* USER CODE END USART3_Init 0 */

  /* USER CODE BEGIN USART3_Init 1 */

  /* USER CODE END USART3_Init 1 */
  huart3.Instance = USART3;
  huart3.Init.BaudRate = 921600;
  huart3.Init.WordLength = UART_WORDLENGTH_8B;
  huart3.Init.StopBits = UART_STOPBITS_1;
  huart3.Init.Parity = UART_PARITY_NONE;
  huart3.Init.Mode = UART_MODE_TX_RX;
  huart3.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart3.Init.OverSampling = UART_OVERSAMPLING_16;
  huart3.Init.OneBitSampling = UART_ONE_BIT_SAMPLE_DISABLE;
  huart3.Init.ClockPrescaler = UART_PRESCALER_DIV1;
  huart3.AdvancedInit.AdvFeatureInit = UART_ADVFEATURE_NO_INIT;
  if (HAL_UART_Init(&huart3) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_SetTxFifoThreshold(&huart3, UART_TXFIFO_THRESHOLD_1_8) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_SetRxFifoThreshold(&huart3, UART_RXFIFO_THRESHOLD_1_8) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_DisableFifoMode(&huart3) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN USART3_Init 2 */

  /* USER CODE END USART3_Init 2 */

}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};
  /* USER CODE BEGIN MX_GPIO_Init_1 */

  /* USER CODE END MX_GPIO_Init_1 */

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOC_CLK_ENABLE();
  __HAL_RCC_GPIOH_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();
  __HAL_RCC_GPIOD_CLK_ENABLE();

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(LED1_GPIO_Port, LED1_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pin : USER_BUTTON_Pin */
  GPIO_InitStruct.Pin = USER_BUTTON_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(USER_BUTTON_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pin : LED1_Pin */
  GPIO_InitStruct.Pin = LED1_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(LED1_GPIO_Port, &GPIO_InitStruct);

  /* USER CODE BEGIN MX_GPIO_Init_2 */

  /* USER CODE END MX_GPIO_Init_2 */
}

/* USER CODE BEGIN 4 */

/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @param None
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}
#ifdef USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
