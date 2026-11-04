/* V322b: see sop_v322b.h. The same logic as sop_v322.c (Python references: specaudit_v287/src/relax_feats_audit.py,
 * leakfree_v272/src/sop_ident/tools/local_sop.py, local_sop2.py with specaudit_v287.run_audit, leakfree_v272.centres,
 * adaptive_v284.adapt), in float with integer time (ms) and integer charge. Python semantics kept: max(a, b) returns a
 * unless b > a, np.clip propagates NaN, argmin keeps the first minimum, sum() starts from 0. */
#include <math.h>
#include <string.h>
#include "sop_v322b.h"

static const int32_t T_OFF_MS[NTOFF] = {1000, 2000, 5000, 10000, 30000, 60000, 120000, 240000};
static const int32_t TOL_MS[NTOFF] = {500, 700, 1750, 3500, 10500, 21000, 42000, 84000};    /* max(0.35 tau, 0.5 s) */
#define PRE_REST_MIN_MS 60000
#define PULSE_MAX_MS 12500
#define R10_MAX_MS 10100
#define SLOPE_WIN_MS 30000
#define AD_GAMMA 0.05f
#define AD_TAU 0.05f
#define NANF ((float)NAN)
#define BIG_MS 2147483647

static float pymax(float a, float b) { return (b > a) ? b : a; }                       /* Python max(a, b) */
static float npclip(float x, float lo, float hi) { if (isnan(x)) return x; if (x < lo) x = lo; if (x > hi) x = hi; return x; }
static float npclip_lo(float x, float lo) { if (isnan(x)) return x; return x < lo ? lo : x; }
static int32_t iabs32(int32_t x) { return x < 0 ? -x : x; }
static float ms2s(int32_t ms) { return (float)ms / 1000.0f; }

static void interr_nan(interr_t *it)
{
    it->I_off = it->off_dur_s = NANF;
    for (int k = 0; k < NTOFF; k++) it->Roff[k] = NANF;
    it->has_fit = 0; it->sq_R0 = it->sq_A = it->Rstep1 = it->Rstep2 = it->Rstep10 = NANF;
}

static void pulse_nan(pulse_t *p)
{
    memset(p, 0, sizeof *p);
    p->P = p->Vpre = p->I_last = p->V_last = p->dt_last = p->R1 = p->R2 = p->R10 = NANF;
    p->k0 = -1; interr_nan(&p->it); p->rest_s = NANF;
}

/* ordinary least squares y = A x + B (np.polyfit degree 1); returns 0 if degenerate */
static int ols(const float *x, const float *y, int n, float *A, float *B)
{
    float mx = 0, my = 0, sxx = 0, sxy = 0;
    for (int i = 0; i < n; i++) { mx += x[i]; my += y[i]; }
    mx /= (float)n; my /= (float)n;
    for (int i = 0; i < n; i++) { float dx = x[i] - mx; sxx += dx * dx; sxy += dx * (y[i] - my); }
    if (!(sxx > 0)) { *A = NANF; *B = NANF; return 0; }
    *A = sxy / sxx; *B = my - *A * mx; return 1;
}

/* relax_feats_audit.interruption() from the nearest rest samples to t_e + tau (times in ms) */
static void interr_from(interr_t *it, int32_t te, int32_t tb, float Ie, float Ve, const int32_t *bt, const float *bV, const int *bset)
{
    int32_t T = te - tb; float taus[NTOFF], rec[NTOFF]; int nt = 0;
    interr_nan(it);
    it->I_off = Ie; it->off_dur_s = ms2s(T);
    for (int k = 0; k < NTOFF; k++) {
        if (!bset[k]) continue;
        int32_t dd = iabs32((bt[k] - te) - T_OFF_MS[k]);
        if (dd <= TOL_MS[k]) {
            float R = fabsf(bV[k] - Ve) / fabsf(Ie);
            it->Roff[k] = R;
            if (T_OFF_MS[k] <= 30000) { taus[nt] = ms2s(bt[k] - te); rec[nt] = R; nt++; }
        }
    }
    if (nt >= 3 && T > 0) {
        float x[NTOFF], A, B, Ts = ms2s(T), sT = sqrtf(Ts);
        for (int i = 0; i < nt; i++)                    /* sqrt(T) - sqrt(T + tau) + sqrt(tau), cancellation-free form */
            x[i] = sqrtf(taus[i]) - taus[i] / (sT + sqrtf(Ts + taus[i]));
        if (ols(x, rec, nt, &A, &B)) {
            it->has_fit = 1; it->sq_R0 = B; it->sq_A = A;
            it->Rstep1 = B + A; it->Rstep2 = B + A * sqrtf(2.0f); it->Rstep10 = B + A * sqrtf(10.0f);
        }
    }
}

void eng_reset(eng_t *e)
{
    memset(e, 0, sizeof *e);
    interr_nan(&e->lr_int); interr_nan(&e->sg_int);
    for (int d = 0; d < 2; d++) { pulse_nan(&e->pl[d][0]); pulse_nan(&e->pl[d][1]); }
}

static void seg_finalize(eng_t *e)
{
    if (!e->sg_active) return;
    e->sg_active = 0;
    if (!((e->sg_tk1 - e->sg_tk0 <= PULSE_MAX_MS) && e->sg_k0 > 0 && e->sg_prev_rest && e->sg_pre_len >= PRE_REST_MIN_MS)) return;
    pulse_t p; pulse_nan(&p);
    p.valid = 1;
    p.P = fabsf(e->sg_Vk1 * e->sg_Ik1); p.Vpre = e->sg_Vp; p.I_last = e->sg_Ik1; p.V_last = e->sg_Vk1; p.dt_last = ms2s(e->sg_dtlast);
    p.R1 = e->sg_R1; p.R2 = e->sg_R2; p.R10 = e->sg_hasR10 ? e->sg_R10 : NANF;
    p.ah2 = e->sg_ah2_pre; p.t_k0 = e->sg_tk0; p.k0 = e->sg_k0;
    p.has_int = e->sg_has_int;
    if (p.has_int) { p.it = e->sg_int; p.rest_s = e->sg_rest_s; }
    int d = e->sg_sign > 0 ? 1 : 0;
    e->pl[d][1] = e->pl[d][0]; e->pl[d][0] = p; e->pl_n[d]++;
}

static void rest_finalize(eng_t *e)                  /* the rest run [rs_a, n - 1] has just ended */
{
    e->lr_len = e->t_p - e->rs_ta;
    e->lr_has_int = e->rs_has_load;
    if (e->rs_has_load) interr_from(&e->lr_int, e->rs_te, e->rs_tb, e->rs_Ie, e->rs_Ve, e->rs_bt, e->rs_bV, e->rs_bset);
    else interr_nan(&e->lr_int);
}

void eng_push(eng_t *e, int64_t t_ms, uint32_t st_ms, int32_t key, int32_t I_e5, int32_t V_e4, float T)
{
    if (e->n == 0) e->t0 = t_ms;
    int32_t t = (int32_t)(t_ms - e->t0), st = (int32_t)st_ms;
    float I = (float)I_e5 / 1e5f, V = (float)V_e4 / 1e4f;
    int rest = (I_e5 < 2000 && I_e5 > -2000);           /* |I| < 0.02 A, exact on the 1e-5 A grid */
    int sign = (I_e5 > 0) - (I_e5 < 0);
    int64_t ah2_prev = e->ah2_p, ah2 = 0;
    if (e->n > 0) {                                     /* 2 x discharged charge since the first sample: exact trapezoid */
        e->ah2 += -((int64_t)e->I5_p + (int64_t)I_e5) * (int64_t)(t - e->t_p);
        ah2 = e->ah2;
    }
    if (rest) {
        if (e->n == 0 || !e->rest_p) {                  /* a rest run starts */
            seg_finalize(e);
            e->rs_a = e->n; e->rs_ta = t; e->rs_Va = V;
            e->rs_has_load = (e->n > 0 && !e->rest_p);
            if (e->rs_has_load) { e->rs_te = e->t_p; e->rs_tb = e->lr_tb; e->rs_Ie = e->I_p; e->rs_Ve = e->V_p; e->rs_sign_e = e->sign_p; }
            for (int k = 0; k < NTOFF; k++) { e->rs_bset[k] = 0; e->rs_bd[k] = BIG_MS; }
            e->rb_head = 0; e->rb_n = 0;
        }
        if (e->rs_has_load)
            for (int k = 0; k < NTOFF; k++) {
                int32_t dd = iabs32((t - e->rs_te) - T_OFF_MS[k]);
                if (!e->rs_bset[k] || dd < e->rs_bd[k]) { e->rs_bset[k] = 1; e->rs_bd[k] = dd; e->rs_bt[k] = t; e->rs_bV[k] = V; }
            }
        e->rb_t[e->rb_head] = t; e->rb_V[e->rb_head] = V; e->rb_head = (e->rb_head + 1) % RB_N;
        if (e->rb_n < RB_N) e->rb_n++;
        e->lr_active = 0;
    } else {
        if (e->n > 0 && e->rest_p) rest_finalize(e);
        if (!e->lr_active || sign != e->lr_sign) { e->lr_active = 1; e->lr_sign = sign; e->lr_tb = t; }
        if (!(e->sg_active && key == e->sg_key)) {     /* a new key segment */
            seg_finalize(e);
            e->sg_active = 1; e->sg_k0 = e->n; e->sg_key = key; e->sg_tk0 = t; e->sg_t0 = t - st;
            e->sg_prev_rest = (e->n > 0 && e->rest_p);
            e->sg_pre_len = e->sg_prev_rest ? e->lr_len : 0;
            e->sg_Vp = e->V_p; e->sg_ah2_pre = ah2_prev;
            e->sg_has_int = e->sg_prev_rest && e->lr_has_int;
            if (e->sg_has_int) { e->sg_int = e->lr_int; e->sg_rest_s = ms2s(e->lr_len); } else { interr_nan(&e->sg_int); e->sg_rest_s = NANF; }
            e->sg_b1 = e->sg_b2 = BIG_MS; e->sg_hasR10 = 0; e->sg_R1 = e->sg_R2 = e->sg_R10 = NANF;
        }
        int32_t dt = t - e->sg_t0;
        float R = (V - e->sg_Vp) / I;
        int32_t d1 = iabs32(dt - 1000), d2 = iabs32(dt - 2000);
        if (d1 < e->sg_b1) { e->sg_b1 = d1; e->sg_R1 = R; }
        if (d2 < e->sg_b2) { e->sg_b2 = d2; e->sg_R2 = R; }
        if (dt <= R10_MAX_MS) { e->sg_R10 = R; e->sg_hasR10 = 1; }
        e->sg_tk1 = t; e->sg_dtlast = dt; e->sg_Ik1 = I; e->sg_Vk1 = V; e->sg_sign = sign;
    }
    e->t_p = t; e->key_p = key; e->I5_p = I_e5; e->rest_p = rest; e->sign_p = sign;
    e->I_p = I; e->V_p = V; e->T_p = T; e->ah2_p = ah2; e->n++;
}

void eng_query(const eng_t *e, int charge, feats_t *f)
{
    memset(f, 0, sizeof *f);
    interr_nan(&f->it); pulse_nan(&f->prev); pulse_nan(&f->prev2);
    f->V_pre = f->T_pre = f->rest_s = f->I_end = f->last_dur_s = f->V_end = f->V_rest0 = f->dt_rest0 = f->slope_mVs = f->prev_gap_s = NANF;
    if (e->n - 1 < 1 || !e->rest_p) return;             /* no pre-rest */
    f->valid = 1;
    if (e->rs_a == 0 || !e->rs_has_load) { f->a_zero = 1; return; }
    interr_from(&f->it, e->rs_te, e->rs_tb, e->rs_Ie, e->rs_Ve, e->rs_bt, e->rs_bV, e->rs_bset);
    f->V_pre = e->V_p; f->T_pre = e->T_p; f->rest_s = ms2s(e->t_p - e->rs_ta);
    f->last_sign = e->rs_sign_e > 0 ? 1 : -1;
    f->I_end = e->rs_Ie; f->last_dur_s = ms2s(e->rs_te - e->rs_tb); f->V_end = e->rs_Ve; f->V_rest0 = e->rs_Va;
    f->dt_rest0 = ms2s(e->rs_ta - e->rs_te);
    {   /* end-of-rest slope over the last 30 s, >= 5 samples (two passes over the ring buffer, no copies) */
        int32_t lim = e->t_p - SLOPE_WIN_MS; int m = 0; float mx = 0, my = 0, sxx = 0, sxy = 0;
        int j0 = (e->rb_head - e->rb_n + RB_N) % RB_N;
        if (e->rb_n == RB_N && e->rb_t[j0] >= lim) f->rb_trunc = 1;
        for (int i = 0; i < e->rb_n; i++) {
            int j = (j0 + i) % RB_N;
            if (e->rb_t[j] >= lim) { mx += ms2s(e->rb_t[j] - e->t_p); my += e->rb_V[j]; m++; }
        }
        if (m >= 5) {
            mx /= (float)m; my /= (float)m;
            for (int i = 0; i < e->rb_n; i++) {
                int j = (j0 + i) % RB_N;
                if (e->rb_t[j] >= lim) { float dx = ms2s(e->rb_t[j] - e->t_p) - mx; sxx += dx * dx; sxy += dx * (e->rb_V[j] - my); }
            }
            if (sxx > 0) f->slope_mVs = 1e3f * (sxy / sxx);
        }
    }
    f->ah2_q = e->ah2_p;
    int d = charge ? 1 : 0;
    if (e->pl_n[d] >= 1) { f->prev = e->pl[d][0]; f->prev_gap_s = ms2s(e->t_p - f->prev.t_k0); }
    if (e->pl_n[d] >= 2) f->prev2 = e->pl[d][1];
}

/* ---------------------------------------------------------------- predictors */
enum { F_TREND10, F_TREND1, F_INVT, F_TRENDS, F_TRENDS1, F_DLT, F_SQA, F_PDIFF, F_SLOPE, F_LT, F_AMP, F_EXT, NFEAT };

typedef struct {
    int chg; float Vlim, Imax, V_pre, H, Roff, Rstep, prevR, prevRoff, I_end, prev_I_last;
    float x[NFEAT];
} rowin_t;

static float pred_ls2(const rowin_t *r, const par_t *p, int direct, float H, float Vlim)
{
    float b, ir;
    if (p->kind == 0) { b = r->prevR; ir = fabsf(r->prev_I_last); }
    else if (p->kind == 1) { b = r->Roff; ir = fabsf(r->I_end); }
    else { b = r->Rstep; ir = fabsf(r->I_end); }
    float s = 0;
    for (int j = 0; j < p->nfeat; j++) s = s + p->c[j] * r->x[p->feat[j]];
    float lin = p->c0 + s;
    if (!(isfinite(b) && b > 0 && isfinite(ir) && ir > 0 && isfinite(lin))) return NANF;
    if (H <= 0) return 0;
    float I, Rh;
    if (direct) {
        I = ir * expf(lin + p->a * logf(H / (b * ir)));
        Rh = H / I;
        if (I > r->Imax) return r->Imax * (r->V_pre + (r->chg ? 1 : -1) * r->Imax * Rh);
        return Vlim * I;
    }
    float beta = p->beta, k0 = b * expf(lin) / powf(ir, beta);
    I = powf(H / k0, 1.0f / (1.0f + beta));
    if (I > r->Imax) { Rh = k0 * powf(pymax(r->Imax, 1e-4f), beta); return r->Imax * (r->V_pre + (r->chg ? 1 : -1) * r->Imax * Rh); }
    return Vlim * I;
}

static float pred_prev(const rowin_t *r, const prevpar_t *pp)
{
    float kap = pp->has ? pp->kappa : 1.0f, beta = pp->has ? pp->beta : 0.0f;
    float B = r->prevR * r->Roff / r->prevRoff, b, ir;
    if (isfinite(B)) { b = B; ir = fabsf(r->prev_I_last); } else { b = r->Roff; ir = fabsf(r->I_end); }
    float H = r->H;
    if (!(isfinite(b) && b > 0 && isfinite(ir) && ir > 0) || H <= 0) return (H > 0) ? NANF : 0.0f;
#define RH(Ix) (b * kap * powf(pymax((Ix), 1e-3f) / ir, beta))
    float I = H / RH(ir);
    for (int it = 0; it < 50; it++) {
        float In = H / RH(I);
        if (fabsf(In - I) < 1e-6f * pymax(1.0f, I)) break;
        I = 0.5f * (I + In);
    }
    if (I > r->Imax) {
        float Vend = r->chg ? r->V_pre + r->Imax * RH(r->Imax) : r->V_pre - r->Imax * RH(r->Imax);
        return r->Imax * Vend;
    }
#undef RH
    return r->Vlim * I;
}

void sop_predict(const feats_t *f, int charge, uint32_t horizon_ms, const fold_coef_t *cf, pred_out_t *o)
{
    rowin_t r; int h10 = horizon_ms >= 5000; float hs = (float)horizon_ms / 1000.0f;
    const pulse_t *pv = &f->prev, *p2 = &f->prev2;
    r.chg = charge; r.Vlim = charge ? 4.15f : 2.55f; r.Imax = charge ? 15.0f : 30.0f;
    r.V_pre = f->V_pre; r.I_end = f->I_end; r.prev_I_last = pv->I_last;
    float drift = 1e-3f * (isnan(f->slope_mVs) ? 0.0f : f->slope_mVs) * hs;
    r.H = charge ? (r.Vlim - f->V_pre) - drift : (f->V_pre + drift) - r.Vlim;
    r.Roff = h10 ? f->it.Roff[3] : f->it.Roff[1];
    r.Rstep = h10 ? f->it.Rstep10 : f->it.Rstep2;
    r.prevR = h10 ? pv->R10 : pv->R2;
    r.prevRoff = h10 ? pv->it.Roff[3] : pv->it.Roff[1];
    float prevRstep = h10 ? pv->it.Rstep10 : pv->it.Rstep2;
    float p2R = h10 ? p2->R10 : p2->R2, p2S = h10 ? p2->it.Rstep10 : p2->it.Rstep2;
    float *x = r.x;
    x[F_TREND10] = logf(f->it.Roff[3] / pv->it.Roff[3]);
    x[F_TREND1] = logf(f->it.Roff[0] / pv->it.Roff[0]);
    x[F_PDIFF] = logf(pv->R10 / pv->R1);
    x[F_SLOPE] = f->slope_mVs;
    x[F_INVT] = 1000.0f / (f->T_pre + 273.15f) - 3.354f;
    x[F_TRENDS] = logf(r.Rstep / prevRstep);
    x[F_TRENDS1] = logf(f->it.Rstep1 / pv->it.Rstep1);
    x[F_LT] = logf(f->it.off_dur_s / 200.0f);
    x[F_DLT] = logf(f->it.off_dur_s / pv->it.off_dur_s);
    x[F_SQA] = logf(npclip_lo(f->it.sq_A * sqrtf(10.0f) / f->it.Rstep10, 0.02f));
    float obs = logf(r.prevR / p2R), trS2 = logf(prevRstep / p2S);
    /* charge ratio from exact integer differences; NaN when either earlier pulse is missing (as the Python NaN Ah) */
    float dah = (f->valid && pv->valid && p2->valid) ? (float)(f->ah2_q - pv->ah2) / (float)(pv->ah2 - p2->ah2) : NANF;
    x[F_EXT] = obs * npclip(dah, 0, 3);
    x[F_AMP] = x[F_TRENDS] * npclip(fabsf(trS2) > 0.05f ? obs / trS2 : 1.0f, 0, 3);
    /* region from the rest voltage, then the first-query sub-region (S1) */
    int base;
    if (!charge) base = (f->V_pre <= cf->th[0]) ? 0 : ((f->V_pre <= cf->th[1]) ? 1 : 2);
    else base = (f->V_pre <= cf->th[2]) ? 3 : 4;
    int first = isnan(pv->R10) ? ((f->last_sign == 1 && f->V_end >= 4.19f) ? 2 : 1) : 0;
    int reg = base * 3 + first;
    o->base = base; o->reg = reg;
    o->pred[0] = pred_prev(&r, &cf->prev[base]);
    int noprev = !(isfinite(r.prevR) && isfinite(pv->I_last));
    for (int k = 0; k < 4; k++) {
        const pred_coef_t *pc = &cf->pr[k]; const regpar_t *rp = &pc->reg[reg];
        float H = r.H, Vlim = r.Vlim;
        if (pc->gap) { float g = charge ? pc->g_chg : pc->g_dis; H = H - g; Vlim = charge ? Vlim - g : Vlim + g; }
        float P = NANF;
        if (rp->p.valid) {
            P = pred_ls2(&r, &rp->p, pc->direct, H, Vlim);
            if (rp->p.kind == 0 && isnan(P) && noprev && rp->p2.valid) P = pred_ls2(&r, &rp->p2, pc->direct, H, Vlim);
        }
        o->pred[k + 1] = P;
    }
    /* log median of the finite logs */
    float L[NPRED]; int m = 0;
    for (int k = 0; k < NPRED; k++) { float l = logf(o->pred[k]); if (isfinite(l)) L[m++] = l; }
    o->nfinite = m;
    if (m == 0) { o->centre = 0; return; }
    for (int i = 1; i < m; i++) { float v = L[i]; int j = i - 1; while (j >= 0 && L[j] > v) { L[j + 1] = L[j]; j--; } L[j + 1] = v; }
    float med = (m % 2) ? L[m / 2] : (L[m / 2 - 1] + L[m / 2]) / 2.0f;
    o->centre = expf(med);
}

/* ---------------------------------------------------------------- adaptation */
void ad_init(ad_t *a, float s0) { a->ls = logf(s0); }

float ad_request(const ad_t *a, float centre, float *scale)
{
    float s = expf(a->ls); if (scale) *scale = s;
    return s * centre;
}

void ad_update(ad_t *a, int event)
{
    static const float LO = -1.6094379124341003f, HI = 0.4054651081081644f;   /* log 0.2, log 1.5 */
    float v = a->ls + AD_GAMMA * (AD_TAU - (event ? 1.0f : 0.0f));
    a->ls = v < LO ? LO : (v > HI ? HI : v);
}
