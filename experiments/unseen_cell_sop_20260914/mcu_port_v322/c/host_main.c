/* V322 host harness: replay stream_v322.bin through the C engine, write per-query features / predictions and the
 * adaptation's requests. usage: host_v322 stream_v322.bin ad_v322.bin out_prefix */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "sop_v322.h"
#include "coef_v322.h"

#define NQ 672
static feats_t F[NQ]; static pred_out_t P[NQ]; static int seen[NQ], qdir[NQ];

static uint32_t rd32(FILE *f) { uint32_t v; if (fread(&v, 4, 1, f) != 1) { fprintf(stderr, "read error\n"); exit(1); } return v; }

static void put_int(FILE *o, const interr_t *it)
{
    fprintf(o, ",%.17g,%.17g", (double)it->I_off, (double)it->off_dur_s);
    for (int k = 0; k < NTOFF; k++) fprintf(o, ",%.17g", (double)it->Roff[k]);
    fprintf(o, ",%d,%.17g,%.17g,%.17g,%.17g,%.17g", it->has_fit, (double)it->sq_R0, (double)it->sq_A, (double)it->Rstep1, (double)it->Rstep2, (double)it->Rstep10);
}

static void put_pulse(FILE *o, const pulse_t *p)
{
    fprintf(o, ",%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%ld,%d", p->valid, (double)p->P, (double)p->Vpre, (double)p->I_last,
            (double)p->V_last, (double)p->dt_last, (double)p->R1, (double)p->R2, (double)p->R10, p->Ah, p->k0_abs, p->has_int);
    put_int(o, &p->it);
    fprintf(o, ",%.17g", (double)p->rest_s);
}

int main(int argc, char **argv)
{
    if (argc < 4) { fprintf(stderr, "usage: %s stream.bin ad.bin out_prefix\n", argv[0]); return 2; }
    FILE *f = fopen(argv[1], "rb"); char magic[4];
    if (!f || fread(magic, 1, 4, f) != 4 || memcmp(magic, "V322", 4)) { fprintf(stderr, "bad stream\n"); return 1; }
    uint32_t nw = rd32(f); static eng_t E; long nsamp = 0;
    for (uint32_t w = 0; w < nw; w++) {
        uint32_t wid = rd32(f), cell = rd32(f), ns = rd32(f), nq = rd32(f); (void)wid;
        uint32_t qo[64], qi[64], qd[64], qh[64];
        if (nq > 64) { fprintf(stderr, "too many queries\n"); return 1; }
        for (uint32_t k = 0; k < nq; k++) { qo[k] = rd32(f); qi[k] = rd32(f); qd[k] = rd32(f); qh[k] = rd32(f); }
        eng_reset(&E, 0); uint32_t nextq = 0;
        for (uint32_t i = 0; i <= ns; i++) {
            while (nextq < nq && qo[nextq] == i) {
                uint32_t id = qi[nextq];
                eng_query(&E, (int)qd[nextq], &F[id]);
                sop_predict(&F[id], (int)qd[nextq], qh[nextq] / 1000.0, &COEF_V322[cell], &P[id]);
                seen[id] = 1; qdir[id] = (int)qd[nextq]; nextq++;
            }
            if (i == ns) break;
            unsigned char b[32];
            if (fread(b, 1, 32, f) != 32) { fprintf(stderr, "short sample\n"); return 1; }
            int64_t t; uint32_t st; int32_t key, I, V; double T;
            memcpy(&t, b, 8); memcpy(&st, b + 8, 4); memcpy(&key, b + 12, 4); memcpy(&I, b + 16, 4); memcpy(&V, b + 20, 4); memcpy(&T, b + 24, 8);
            eng_push(&E, t, st, key, I, V, T); nsamp++;
        }
        if (nextq != nq) { fprintf(stderr, "window %u: %u of %u queries not reached\n", w, nextq, nq); return 1; }
    }
    fclose(f);
    char path[512]; snprintf(path, sizeof path, "%s_rows.csv", argv[3]);
    FILE *o = fopen(path, "w");
    fprintf(o, "query_id,valid,a_zero,rb_trunc,V_pre,T_pre,rest_s,I_end,last_dur_s,V_end,V_rest0,dt_rest0,slope_mVs,Ah_q,last_sign");
    const char *ints[] = {"I_off", "off_dur_s", "Roff1", "Roff2", "Roff5", "Roff10", "Roff30", "Roff60", "Roff120", "Roff240", "has_fit", "sq_R0", "sq_A", "Rstep1", "Rstep2", "Rstep10"};
    for (int k = 0; k < 16; k++) fprintf(o, ",%s", ints[k]);
    const char *pre[] = {"prev_", "prev2_"};
    for (int q = 0; q < 2; q++) {
        const char *pf[] = {"valid", "P", "Vpre", "I_last", "V_last", "dt_last", "R1", "R2", "R10", "Ah", "k0", "has_int"};
        for (int k = 0; k < 12; k++) fprintf(o, ",%s%s", pre[q], pf[k]);
        for (int k = 0; k < 16; k++) fprintf(o, ",%s%s", pre[q], ints[k]);
        fprintf(o, ",%srest_s", pre[q]);
    }
    fprintf(o, ",prev_gap_s,base,reg,p_prev,p_L4,p_M6,p_DN6,p_G1,centre,nfinite\n");
    for (int id = 0; id < NQ; id++) {
        if (!seen[id]) { fprintf(stderr, "query %d not seen\n", id); return 1; }
        const feats_t *x = &F[id];
        fprintf(o, "%d,%d,%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%d", id, x->valid, x->a_zero, x->rb_trunc, (double)x->V_pre,
                (double)x->T_pre, (double)x->rest_s, (double)x->I_end, (double)x->last_dur_s, (double)x->V_end, (double)x->V_rest0, (double)x->dt_rest0,
                (double)x->slope_mVs, x->Ah_q, x->last_sign);
        put_int(o, &x->it); put_pulse(o, &x->prev); put_pulse(o, &x->prev2);
        fprintf(o, ",%.17g,%d,%d", (double)x->prev_gap_s, P[id].base, P[id].reg);
        for (int k = 0; k < NPRED; k++) fprintf(o, ",%.17g", (double)P[id].pred[k]);
        fprintf(o, ",%.17g,%d\n", (double)P[id].centre, P[id].nfinite);
    }
    fclose(o);
    FILE *a = fopen(argv[2], "rb"); snprintf(path, sizeof path, "%s_requests.csv", argv[3]); o = fopen(path, "w");
    fprintf(o, "query_id,order,scale,request,event\n");
    for (int c = 0; c < 3; c++) {
        uint32_t n = rd32(a); ad_t ad; ad_init(&ad, COEF_V322[c].s0);
        for (uint32_t k = 0; k < n; k++) {
            uint32_t id = rd32(a); double truth; if (fread(&truth, 8, 1, a) != 1) return 1;
            real_t s, req = ad_request(&ad, P[id].centre, &s);
            int ev = ((double)req / truth) > 1 + 1e-9;
            fprintf(o, "%u,%u,%.17g,%.17g,%d\n", id, k, (double)s, (double)req, ev);
            ad_update(&ad, ev);
        }
    }
    fclose(o); fclose(a);
    fprintf(stderr, "samples %ld, queries %d\n", nsamp, NQ);
    return 0;
}
