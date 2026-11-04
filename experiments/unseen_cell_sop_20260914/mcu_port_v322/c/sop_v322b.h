/* V322b: the paper's SOP predictor (V287 Ls, all audit fixes; V284 online scale adaptation) in pure float32.
 * Same logic as sop_v322.c. No double anywhere: time is an integer in milliseconds from the first sample of the sweep
 * (all time rules are integer comparisons), charge is an exact integer accumulator (1e-5 A x ms), every other quantity is
 * float. eng_push() takes one raw sample at a time; eng_query() returns the features the Python extractor
 * (specaudit_v287/src/relax_feats_audit.py, FIXED = True) computes; sop_predict() applies the five local predictors of the
 * evaluated cell's fold and their log median. */
#ifndef SOP_V322B_H
#define SOP_V322B_H
#include <stdint.h>

#define NTOFF 8
#define NPRED 5
#define RB_N 1024

/* ---------------------------------------------------------------- coefficients (generated header) */
typedef struct { int8_t valid, kind, fixed, nfeat; int8_t feat[5]; float c[5]; float c0, beta, a; } par_t;   /* kind 0 prev, 1 off, 2 step */
typedef struct { par_t p, p2; } regpar_t;                    /* p2: 'off' fallback for 'prev'-kind regions (S2) */
typedef struct { int8_t direct, gap; float g_chg, g_dis; regpar_t reg[15]; } pred_coef_t;   /* reg = base * 3 + first */
typedef struct { int8_t has; float kappa, beta; } prevpar_t;
typedef struct { float th[3]; prevpar_t prev[5]; pred_coef_t pr[4]; float s0; } fold_coef_t;

/* ---------------------------------------------------------------- features */
typedef struct {
    float I_off, off_dur_s, Roff[NTOFF];
    int has_fit; float sq_R0, sq_A, Rstep1, Rstep2, Rstep10;
} interr_t;

typedef struct {
    int valid;
    float P, Vpre, I_last, V_last, dt_last, R1, R2, R10;
    int64_t ah2;                                 /* 2 x discharged charge at the sample before the pulse, 1e-5 A x ms */
    int32_t t_k0; int32_t k0;
    int has_int; interr_t it; float rest_s;
} pulse_t;

typedef struct {
    int valid, a_zero, rb_trunc;
    interr_t it;
    float V_pre, T_pre, rest_s, I_end, last_dur_s, V_end, V_rest0, dt_rest0, slope_mVs;
    int last_sign;
    int64_t ah2_q;                               /* 2 x discharged charge at the query, 1e-5 A x ms */
    pulse_t prev, prev2; float prev_gap_s;
} feats_t;

/* ---------------------------------------------------------------- streaming engine (times in ms from the first sample) */
typedef struct {
    int32_t n; int64_t t0;
    int32_t t_p; int64_t ah2, ah2_p; int32_t key_p, I5_p; int rest_p, sign_p; float I_p, V_p, T_p;
    int lr_active, lr_sign; int32_t lr_tb;
    int32_t rs_a, rs_ta; float rs_Va; int rs_has_load; int32_t rs_te, rs_tb; float rs_Ie, rs_Ve; int rs_sign_e;
    int32_t rs_bd[NTOFF], rs_bt[NTOFF]; float rs_bV[NTOFF]; int rs_bset[NTOFF];
    int32_t rb_t[RB_N]; float rb_V[RB_N]; int rb_head, rb_n;
    int32_t lr_len; int lr_has_int; interr_t lr_int;
    int sg_active; int32_t sg_k0; int32_t sg_key; int32_t sg_tk0, sg_t0; int sg_prev_rest; int32_t sg_pre_len; float sg_Vp; int64_t sg_ah2_pre;
    int sg_has_int; interr_t sg_int; float sg_rest_s;
    int32_t sg_b1, sg_b2; float sg_R1, sg_R2, sg_R10; int sg_hasR10;
    int32_t sg_tk1, sg_dtlast; float sg_Ik1, sg_Vk1; int sg_sign;
    pulse_t pl[2][2]; int pl_n[2];
} eng_t;

void eng_reset(eng_t *e);
void eng_push(eng_t *e, int64_t t_ms, uint32_t st_ms, int32_t key, int32_t I_e5, int32_t V_e4, float T);
void eng_query(const eng_t *e, int charge, feats_t *f);

/* ---------------------------------------------------------------- predictors */
typedef struct { float pred[NPRED]; float centre; int reg, base; int nfinite; } pred_out_t;
void sop_predict(const feats_t *f, int charge, uint32_t horizon_ms, const fold_coef_t *cf, pred_out_t *o);

/* ---------------------------------------------------------------- online scale adaptation (V284) */
typedef struct { float ls; } ad_t;
void ad_init(ad_t *a, float s0);
float ad_request(const ad_t *a, float centre, float *scale);
void ad_update(ad_t *a, int event);

#endif
