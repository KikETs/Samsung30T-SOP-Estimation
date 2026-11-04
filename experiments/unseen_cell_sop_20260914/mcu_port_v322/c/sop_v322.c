/* V322: see sop_v322.h. Python references: specaudit_v287/src/relax_feats_audit.py (features, FIXED = True),
 * leakfree_v272/src/sop_ident/tools/local_sop.py (prev predictor), local_sop2.py with specaudit_v287.run_audit (L4, M6,
 * DN6, G1; S1, S2), leakfree_v272.centres (log median), adaptive_v284.adapt (scale adaptation).
 * Python semantics kept on purpose: max(a, b) returns a unless b > a (NaN stays), np.clip propagates NaN, argmin keeps
 * the first minimum, sum() starts from 0. */
#include <math.h>
#include <string.h>
#include "sop_v322.h"

static const double T_OFF[NTOFF] = {1, 2, 5, 10, 30, 60, 120, 240};
#define PRE_REST_MIN_S 60.0
#define AD_GAMMA 0.05
#define AD_TAU 0.05
#define NANR ((real_t)NAN)

static real_t pymax(real_t a, real_t b) { return (b > a) ? b : a; }                  /* Python max(a, b) */
static real_t npclip(real_t x, real_t lo, real_t hi) { if (isnan(x)) return x; if (x < lo) x = lo; if (x > hi) x = hi; return x; }
static real_t npclip_lo(real_t x, real_t lo) { if (isnan(x)) return x; return x < lo ? lo : x; }

static void interr_nan(interr_t *it)
{
    it->I_off = it->off_dur_s = NANR;
    for (int k = 0; k < NTOFF; k++) it->Roff[k] = NANR;
    it->has_fit = 0; it->sq_R0 = it->sq_A = it->Rstep1 = it->Rstep2 = it->Rstep10 = NANR;
}

static void pulse_nan(pulse_t *p)
{
    memset(p, 0, sizeof *p);
    p->P = p->Vpre = p->I_last = p->V_last = p->dt_last = p->R1 = p->R2 = p->R10 = NANR;
    p->Ah = NAN; p->t_k0 = NAN; p->k0_abs = -1; interr_nan(&p->it); p->rest_s = NANR;
}

/* ordinary least squares y = A x + B (np.polyfit degree 1); returns 0 if degenerate */
static int ols(const real_t *x, const real_t *y, int n, real_t *A, real_t *B)
{
    real_t mx = 0, my = 0, sxx = 0, sxy = 0;
    for (int i = 0; i < n; i++) { mx += x[i]; my += y[i]; }
    mx /= (real_t)n; my /= (real_t)n;
    for (int i = 0; i < n; i++) { real_t dx = x[i] - mx; sxx += dx * dx; sxy += dx * (y[i] - my); }
    if (!(sxx > 0)) { *A = NANR; *B = NANR; return 0; }
    *A = sxy / sxx; *B = my - *A * mx; return 1;
}

/* relax_feats_audit.interruption() from the nearest rest samples to t_e + tau */
static void interr_from(interr_t *it, double te, double tb, real_t Ie, real_t Ve, const double *bt, const real_t *bV, const int *bset)
{
    double T = te - tb; real_t taus[NTOFF], rec[NTOFF]; int nt = 0;
    interr_nan(it);
    it->I_off = Ie; it->off_dur_s = (real_t)T;
    for (int k = 0; k < NTOFF; k++) {
        double tau = T_OFF[k];
        if (!bset[k]) continue;
        double dd = fabs((bt[k] - te) - tau), tol = 0.35 * tau;
        if (0.5 > tol) tol = 0.5;
        if (dd <= tol) {
            real_t R = RE_FABS(bV[k] - Ve) / RE_FABS(Ie);
            it->Roff[k] = R;
            if (tau <= 30) { taus[nt] = (real_t)(bt[k] - te); rec[nt] = R; nt++; }
        }
    }
    if (nt >= 3 && T > 0) {
        real_t x[NTOFF], A, B, sT = RE_SQRT((real_t)T);
        for (int i = 0; i < nt; i++)                    /* sqrt(T) - sqrt(T + tau) + sqrt(tau), cancellation-free form */
            x[i] = RE_SQRT(taus[i]) - taus[i] / (sT + RE_SQRT((real_t)T + taus[i]));
        if (ols(x, rec, nt, &A, &B)) {
            it->has_fit = 1; it->sq_R0 = B; it->sq_A = A;
            it->Rstep1 = B + A; it->Rstep2 = B + A * RE_SQRT((real_t)2); it->Rstep10 = B + A * RE_SQRT((real_t)10);
        }
    }
}

void eng_reset(eng_t *e, long h0)
{
    memset(e, 0, sizeof *e); e->h0 = h0;
    interr_nan(&e->lr_int); interr_nan(&e->sg_int);
    for (int d = 0; d < 2; d++) { pulse_nan(&e->pl[d][0]); pulse_nan(&e->pl[d][1]); }
}

static void seg_finalize(eng_t *e)
{
    if (!e->sg_active) return;
    e->sg_active = 0;
    if (!((e->sg_tk1 - e->sg_tk0 <= 12.5) && e->sg_k0 > 0 && e->sg_prev_rest && e->sg_pre_len >= PRE_REST_MIN_S)) return;
    pulse_t p; pulse_nan(&p);
    p.valid = 1;
    p.P = RE_FABS(e->sg_Vk1 * e->sg_Ik1); p.Vpre = e->sg_Vp; p.I_last = e->sg_Ik1; p.V_last = e->sg_Vk1; p.dt_last = (real_t)e->sg_dtlast;
    p.R1 = e->sg_R1; p.R2 = e->sg_R2; p.R10 = e->sg_hasR10 ? e->sg_R10 : NANR;
    p.Ah = e->sg_ah_pre; p.t_k0 = e->sg_tk0; p.k0_abs = e->h0 + e->sg_k0;
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

void eng_push(eng_t *e, int64_t t_ms, uint32_t st_ms, int32_t key, int32_t I_e5, int32_t V_e4, double T)
{
    double t = (double)t_ms / 1000.0, st = (double)st_ms / 1000.0;
    real_t I = (real_t)((double)I_e5 / 1e5), V = (real_t)((double)V_e4 / 1e4);
    int rest = (I_e5 < 2000 && I_e5 > -2000);           /* |I| < 0.02 A, exact on the 1e-5 A grid */
    int sign = (I_e5 > 0) - (I_e5 < 0);
    double ah_prev = e->ah_p, ah = 0.0;
    if (e->n > 0) {                                     /* discharged Ah since h0 (np.cumsum, then / 3600) */
        double x = (-0.5 * ((double)e->I_p + (double)I)) * (t - e->t_p);
        e->S += x; ah = e->S / 3600.0;
    }
    if (rest) {
        if (e->n == 0 || !e->rest_p) {                  /* a rest run starts */
            seg_finalize(e);
            e->rs_a = e->n; e->rs_ta = t; e->rs_Va = V;
            e->rs_has_load = (e->n > 0 && !e->rest_p);
            if (e->rs_has_load) { e->rs_te = e->t_p; e->rs_tb = e->lr_tb; e->rs_Ie = e->I_p; e->rs_Ve = e->V_p; e->rs_sign_e = e->sign_p; }
            for (int k = 0; k < NTOFF; k++) { e->rs_bset[k] = 0; e->rs_bd[k] = INFINITY; }
            e->rb_head = 0; e->rb_n = 0;
        }
        if (e->rs_has_load)
            for (int k = 0; k < NTOFF; k++) {
                double dd = fabs((t - e->rs_te) - T_OFF[k]);
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
            e->sg_pre_len = e->sg_prev_rest ? e->lr_len : 0.0;
            e->sg_Vp = e->V_p; e->sg_ah_pre = ah_prev;
            e->sg_has_int = e->sg_prev_rest && e->lr_has_int;
            if (e->sg_has_int) { e->sg_int = e->lr_int; e->sg_rest_s = (real_t)e->lr_len; } else { interr_nan(&e->sg_int); e->sg_rest_s = NANR; }
            e->sg_b1 = e->sg_b2 = INFINITY; e->sg_hasR10 = 0; e->sg_R1 = e->sg_R2 = e->sg_R10 = NANR;
        }
        double dt = t - e->sg_t0;
        real_t R = (V - e->sg_Vp) / I;
        double d1 = fabs(dt - 1.0), d2 = fabs(dt - 2.0);
        if (d1 < e->sg_b1) { e->sg_b1 = d1; e->sg_R1 = R; }
        if (d2 < e->sg_b2) { e->sg_b2 = d2; e->sg_R2 = R; }
        if (dt <= 10.1) { e->sg_R10 = R; e->sg_hasR10 = 1; }
        e->sg_tk1 = t; e->sg_dtlast = dt; e->sg_Ik1 = I; e->sg_Vk1 = V; e->sg_sign = sign;
    }
    e->t_p = t; e->st_p = st; e->key_p = key; e->I5_p = I_e5; e->rest_p = rest; e->sign_p = sign;
    e->I_p = I; e->V_p = V; e->T_p = T; e->ah_p = ah; e->n++;
}

void eng_query(const eng_t *e, int charge, feats_t *f)
{
    memset(f, 0, sizeof *f);
    interr_nan(&f->it); pulse_nan(&f->prev); pulse_nan(&f->prev2);
    f->V_pre = f->T_pre = f->rest_s = f->I_end = f->last_dur_s = f->V_end = f->V_rest0 = f->dt_rest0 = f->slope_mVs = f->prev_gap_s = NANR;
    f->Ah_q = NAN;
    if (e->n - 1 < 1 || !e->rest_p) return;             /* no pre-rest */
    f->valid = 1;
    if (e->rs_a == 0 || !e->rs_has_load) { f->a_zero = 1; return; }
    interr_from(&f->it, e->rs_te, e->rs_tb, e->rs_Ie, e->rs_Ve, e->rs_bt, e->rs_bV, e->rs_bset);
    f->V_pre = e->V_p; f->T_pre = (real_t)e->T_p; f->rest_s = (real_t)(e->t_p - e->rs_ta);
    f->last_sign = e->rs_sign_e > 0 ? 1 : -1;
    f->I_end = e->rs_Ie; f->last_dur_s = (real_t)(e->rs_te - e->rs_tb); f->V_end = e->rs_Ve; f->V_rest0 = e->rs_Va;
    f->dt_rest0 = (real_t)(e->rs_ta - e->rs_te);
    {   /* end-of-rest slope over the last 30 s, >= 5 samples (two passes over the ring buffer, no copies) */
        double lim = e->t_p - 30.0; int m = 0; real_t mx = 0, my = 0, sxx = 0, sxy = 0;
        int j0 = (e->rb_head - e->rb_n + RB_N) % RB_N;
        if (e->rb_n == RB_N && e->rb_t[j0] >= lim) f->rb_trunc = 1;       /* older samples of the last 30 s were dropped */
        for (int i = 0; i < e->rb_n; i++) {
            int j = (j0 + i) % RB_N;
            if (e->rb_t[j] >= lim) { mx += (real_t)(e->rb_t[j] - e->t_p); my += e->rb_V[j]; m++; }
        }
        if (m >= 5) {
            mx /= (real_t)m; my /= (real_t)m;
            for (int i = 0; i < e->rb_n; i++) {
                int j = (j0 + i) % RB_N;
                if (e->rb_t[j] >= lim) { real_t dx = (real_t)(e->rb_t[j] - e->t_p) - mx; sxx += dx * dx; sxy += dx * (e->rb_V[j] - my); }
            }
            if (sxx > 0) f->slope_mVs = (real_t)1e3 * (sxy / sxx);
        }
    }
    f->Ah_q = e->ah_p;
    int d = charge ? 1 : 0;
    if (e->pl_n[d] >= 1) { f->prev = e->pl[d][0]; f->prev_gap_s = (real_t)(e->t_p - f->prev.t_k0); }
    if (e->pl_n[d] >= 2) f->prev2 = e->pl[d][1];
}

/* ---------------------------------------------------------------- predictors */
enum { F_TREND10, F_TREND1, F_INVT, F_TRENDS, F_TRENDS1, F_DLT, F_SQA, F_PDIFF, F_SLOPE, F_LT, F_AMP, F_EXT, NFEAT };

typedef struct {
    int chg; real_t Vlim, Imax, V_pre, H, Roff, Rstep, prevR, prevRoff, I_end, prev_I_last;
    real_t x[NFEAT];
} rowin_t;

static real_t pred_ls2(const rowin_t *r, const par_t *p, int direct, real_t H, real_t Vlim)
{
    real_t b, ir;
    if (p->kind == 0) { b = r->prevR; ir = RE_FABS(r->prev_I_last); }
    else if (p->kind == 1) { b = r->Roff; ir = RE_FABS(r->I_end); }
    else { b = r->Rstep; ir = RE_FABS(r->I_end); }
    real_t s = 0;
    for (int j = 0; j < p->nfeat; j++) s = s + (real_t)p->c[j] * r->x[p->feat[j]];
    real_t lin = (real_t)p->c0 + s;
    if (!(isfinite(b) && b > 0 && isfinite(ir) && ir > 0 && isfinite(lin))) return NANR;
    if (H <= 0) return 0;
    real_t I, Rh;
    if (direct) {
        I = ir * RE_EXP(lin + (real_t)p->a * RE_LOG(H / (b * ir)));
        Rh = H / I;
        if (I > r->Imax) return r->Imax * (r->V_pre + (r->chg ? 1 : -1) * r->Imax * Rh);
        return Vlim * I;
    }
    real_t beta = (real_t)p->beta, k0 = b * RE_EXP(lin) / RE_POW(ir, beta);
    I = RE_POW(H / k0, (real_t)1 / ((real_t)1 + beta));
    if (I > r->Imax) { Rh = k0 * RE_POW(pymax(r->Imax, (real_t)1e-4), beta); return r->Imax * (r->V_pre + (r->chg ? 1 : -1) * r->Imax * Rh); }
    return Vlim * I;
}

static real_t pred_prev(const rowin_t *r, const prevpar_t *pp)
{
    real_t kap = pp->has ? (real_t)pp->kappa : (real_t)1, beta = pp->has ? (real_t)pp->beta : (real_t)0;
    real_t B = r->prevR * r->Roff / r->prevRoff, b, ir;
    if (isfinite(B)) { b = B; ir = RE_FABS(r->prev_I_last); } else { b = r->Roff; ir = RE_FABS(r->I_end); }
    real_t H = r->H;
    if (!(isfinite(b) && b > 0 && isfinite(ir) && ir > 0) || H <= 0) return (H > 0) ? NANR : (real_t)0;
#define RH(Ix) (b * kap * RE_POW(pymax((Ix), (real_t)1e-3) / ir, beta))
    real_t I = H / RH(ir);
    for (int it = 0; it < 50; it++) {
        real_t In = H / RH(I);
        if (RE_FABS(In - I) < (real_t)1e-6 * pymax((real_t)1, I)) break;
        I = (real_t)0.5 * (I + In);
    }
    if (I > r->Imax) {
        real_t Vend = r->chg ? r->V_pre + r->Imax * RH(r->Imax) : r->V_pre - r->Imax * RH(r->Imax);
        return r->Imax * Vend;
    }
#undef RH
    return r->Vlim * I;
}

void sop_predict(const feats_t *f, int charge, double horizon_s, const fold_coef_t *cf, pred_out_t *o)
{
    rowin_t r; int h10 = horizon_s >= 5;
    const pulse_t *pv = &f->prev, *p2 = &f->prev2;
    r.chg = charge; r.Vlim = charge ? (real_t)4.15 : (real_t)2.55; r.Imax = charge ? (real_t)15 : (real_t)30;
    r.V_pre = f->V_pre; r.I_end = f->I_end; r.prev_I_last = pv->I_last;
    real_t drift = (real_t)1e-3 * (isnan(f->slope_mVs) ? (real_t)0 : f->slope_mVs) * (real_t)horizon_s;
    r.H = charge ? (r.Vlim - f->V_pre) - drift : (f->V_pre + drift) - r.Vlim;
    r.Roff = h10 ? f->it.Roff[3] : f->it.Roff[1];
    r.Rstep = h10 ? f->it.Rstep10 : f->it.Rstep2;
    r.prevR = h10 ? pv->R10 : pv->R2;
    r.prevRoff = h10 ? pv->it.Roff[3] : pv->it.Roff[1];
    real_t prevRstep = h10 ? pv->it.Rstep10 : pv->it.Rstep2;
    real_t p2R = h10 ? p2->R10 : p2->R2, p2S = h10 ? p2->it.Rstep10 : p2->it.Rstep2;
    real_t *x = r.x;
    x[F_TREND10] = RE_LOG(f->it.Roff[3] / pv->it.Roff[3]);
    x[F_TREND1] = RE_LOG(f->it.Roff[0] / pv->it.Roff[0]);
    x[F_PDIFF] = RE_LOG(pv->R10 / pv->R1);
    x[F_SLOPE] = f->slope_mVs;
    x[F_INVT] = (real_t)1000 / (f->T_pre + (real_t)273.15) - (real_t)3.354;
    x[F_TRENDS] = RE_LOG(r.Rstep / prevRstep);
    x[F_TRENDS1] = RE_LOG(f->it.Rstep1 / pv->it.Rstep1);
    x[F_LT] = RE_LOG(f->it.off_dur_s / (real_t)200);
    x[F_DLT] = RE_LOG(f->it.off_dur_s / pv->it.off_dur_s);
    x[F_SQA] = RE_LOG(npclip_lo(f->it.sq_A * RE_SQRT((real_t)10) / f->it.Rstep10, (real_t)0.02));
    real_t obs = RE_LOG(r.prevR / p2R), trS2 = RE_LOG(prevRstep / p2S);
    real_t dah = (real_t)((f->Ah_q - pv->Ah) / (pv->Ah - p2->Ah));
    x[F_EXT] = obs * npclip(dah, 0, 3);
    x[F_AMP] = x[F_TRENDS] * npclip(RE_FABS(trS2) > (real_t)0.05 ? obs / trS2 : (real_t)1, 0, 3);
    /* region from the rest voltage, then the first-query sub-region (S1) */
    int base;
    if (!charge) base = (f->V_pre <= (real_t)cf->th[0]) ? 0 : ((f->V_pre <= (real_t)cf->th[1]) ? 1 : 2);
    else base = (f->V_pre <= (real_t)cf->th[2]) ? 3 : 4;
    int first = isnan(pv->R10) ? ((f->last_sign == 1 && f->V_end >= (real_t)4.19) ? 2 : 1) : 0;
    int reg = base * 3 + first;
    o->base = base; o->reg = reg;
    o->pred[0] = pred_prev(&r, &cf->prev[base]);
    int noprev = !(isfinite(r.prevR) && isfinite(pv->I_last));
    for (int k = 0; k < 4; k++) {
        const pred_coef_t *pc = &cf->pr[k]; const regpar_t *rp = &pc->reg[reg];
        real_t H = r.H, Vlim = r.Vlim;
        if (pc->gap) { real_t g = (real_t)(charge ? pc->g_chg : pc->g_dis); H = H - g; Vlim = charge ? Vlim - g : Vlim + g; }
        real_t P = NANR;
        if (rp->p.valid) {
            P = pred_ls2(&r, &rp->p, pc->direct, H, Vlim);
            if (rp->p.kind == 0 && isnan(P) && noprev && rp->p2.valid) P = pred_ls2(&r, &rp->p2, pc->direct, H, Vlim);
        }
        o->pred[k + 1] = P;
    }
    /* log median of the finite logs */
    real_t L[NPRED]; int m = 0;
    for (int k = 0; k < NPRED; k++) { real_t l = RE_LOG(o->pred[k]); if (isfinite(l)) L[m++] = l; }
    o->nfinite = m;
    if (m == 0) { o->centre = 0; return; }
    for (int i = 1; i < m; i++) { real_t v = L[i]; int j = i - 1; while (j >= 0 && L[j] > v) { L[j + 1] = L[j]; j--; } L[j + 1] = v; }
    real_t med = (m % 2) ? L[m / 2] : (L[m / 2 - 1] + L[m / 2]) / (real_t)2;
    o->centre = RE_EXP(med);
}

/* ---------------------------------------------------------------- adaptation */
void ad_init(ad_t *a, double s0) { a->ls = (real_t)log(s0); }

real_t ad_request(const ad_t *a, real_t centre, real_t *scale)
{
    real_t s = RE_EXP(a->ls); if (scale) *scale = s;
    return s * centre;
}

void ad_update(ad_t *a, int event)
{
    static const real_t LO = (real_t)-1.6094379124341003, HI = (real_t)0.4054651081081644;   /* log 0.2, log 1.5 */
    real_t v = a->ls + (real_t)AD_GAMMA * ((real_t)AD_TAU - (real_t)(event ? 1 : 0));
    a->ls = v < LO ? LO : (v > HI ? HI : v);
}
