/* V322: C port of the paper's SOP predictor (V287 Ls, all audit fixes; V284 online scale adaptation).
 * One code base for the host (parity) and the STM32H563 (cost). Arithmetic in real_t: double by default,
 * float with -DSOP_FLOAT. Times are kept in double (absolute test time in seconds).
 * Streaming: eng_push() takes one raw sample at a time from the sweep start h0; eng_query() returns the features
 * the Python extractor (specaudit_v287/src/relax_feats_audit.py, FIXED = True) computes from the window [h0, q).
 * sop_predict() applies the five local predictors of the evaluated cell's fold, their log median and the request. */
#ifndef SOP_V322_H
#define SOP_V322_H
#include <stdint.h>

#ifdef SOP_FLOAT
typedef float real_t;
#define RE_EXP expf
#define RE_LOG logf
#define RE_POW powf
#define RE_SQRT sqrtf
#define RE_FABS fabsf
#else
typedef double real_t;
#define RE_EXP exp
#define RE_LOG log
#define RE_POW pow
#define RE_SQRT sqrt
#define RE_FABS fabs
#endif

#define NTOFF 8
#define NPRED 5
#define RB_N 1024

/* ---------------------------------------------------------------- coefficients (generated header) */
typedef struct { int valid, kind, fixed, nfeat; int feat[5]; double c[5]; double c0, beta, a; } par_t;   /* kind 0 prev, 1 off, 2 step */
typedef struct { par_t p, p2; } regpar_t;                    /* p2: 'off' fallback for 'prev'-kind regions (S2) */
typedef struct { int direct, gap; double g_chg, g_dis; regpar_t reg[15]; } pred_coef_t;   /* reg = base * 3 + first */
typedef struct { int has; double kappa, beta; } prevpar_t;
typedef struct { double th[3]; prevpar_t prev[5]; pred_coef_t pr[4]; double s0; } fold_coef_t;

/* ---------------------------------------------------------------- features */
typedef struct {
    real_t I_off, off_dur_s, Roff[NTOFF];
    int has_fit; real_t sq_R0, sq_A, Rstep1, Rstep2, Rstep10;
} interr_t;

typedef struct {
    int valid;
    real_t P, Vpre, I_last, V_last, dt_last, R1, R2, R10;
    double Ah, t_k0; long k0_abs;
    int has_int; interr_t it; real_t rest_s;
} pulse_t;

typedef struct {
    int valid;                                   /* 0: no pre-rest (Python note "no pre-rest") */
    int a_zero;                                  /* diagnostic: the pre-rest starts at the window start */
    int rb_trunc;                                /* diagnostic: the slope ring buffer could not hold the last 30 s */
    interr_t it;
    real_t V_pre, T_pre, rest_s, I_end, last_dur_s, V_end, V_rest0, dt_rest0, slope_mVs;
    int last_sign;                               /* +1 charge, -1 discharge */
    double Ah_q;
    pulse_t prev, prev2; real_t prev_gap_s;
} feats_t;

/* ---------------------------------------------------------------- streaming engine */
typedef struct {
    long n, h0;
    /* previous sample */
    double t_p, st_p, T_p, ah_p, S; int32_t key_p, I5_p; int rest_p, sign_p; real_t I_p, V_p;
    /* load run (non-rest, same sign) */
    int lr_active, lr_sign; double lr_tb;
    /* current rest run */
    long rs_a; double rs_ta; real_t rs_Va; int rs_has_load; double rs_te, rs_tb; real_t rs_Ie, rs_Ve; int rs_sign_e;
    double rs_bd[NTOFF], rs_bt[NTOFF]; real_t rs_bV[NTOFF]; int rs_bset[NTOFF];
    double rb_t[RB_N]; real_t rb_V[RB_N]; int rb_head, rb_n;
    /* the rest run that ended right before the current key segment */
    double lr_len; int lr_has_int; interr_t lr_int;
    /* key segment (non-rest, same Arbin cycle/step) */
    int sg_active; long sg_k0; int32_t sg_key; double sg_tk0, sg_t0; int sg_prev_rest; double sg_pre_len; real_t sg_Vp; double sg_ah_pre;
    int sg_has_int; interr_t sg_int; real_t sg_rest_s;
    double sg_b1, sg_b2; real_t sg_R1, sg_R2, sg_R10; int sg_hasR10;
    double sg_tk1, sg_dtlast; real_t sg_Ik1, sg_Vk1; int sg_sign;
    /* valid pulses: the last two per direction (0 discharge, 1 charge) */
    pulse_t pl[2][2]; int pl_n[2];
    int rb_overflow;
} eng_t;

void eng_reset(eng_t *e, long h0);
void eng_push(eng_t *e, int64_t t_ms, uint32_t st_ms, int32_t key, int32_t I_e5, int32_t V_e4, double T);
void eng_query(const eng_t *e, int charge, feats_t *f);

/* ---------------------------------------------------------------- predictors */
typedef struct { real_t pred[NPRED]; real_t centre; int reg, base; int nfinite; } pred_out_t;
void sop_predict(const feats_t *f, int charge, double horizon_s, const fold_coef_t *cf, pred_out_t *o);

/* ---------------------------------------------------------------- online scale adaptation (V284) */
typedef struct { real_t ls; } ad_t;
void ad_init(ad_t *a, double s0);
real_t ad_request(const ad_t *a, real_t centre, real_t *scale);
void ad_update(ad_t *a, int event);

#endif
