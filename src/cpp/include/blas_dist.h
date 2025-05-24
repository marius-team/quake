/*****************************************************************************************
 *  faiss_exhaustive_L2sqr_buf.h
 *
 *  Header-only replacement for Faiss’ exhaustive_L2sqr_blas()/knn_L2sqr().
 *  ───────────────────────────────────────────────────────────────────────
 *  • No dynamic allocations on the critical path.
 *  • Supersedes Faiss’ originals *without* touching any public headers.
 *  • Thread-safe, preserves numerical behaviour and ResultHandler contract.
 *
 *  Jason — MIT / Quake project, 2025-05-23
 *****************************************************************************************/
#ifndef FAISS_EXHAUSTIVE_L2SQR_BUF_H
#define FAISS_EXHAUSTIVE_L2SQR_BUF_H

#include <faiss/utils/distances.h>
#include <faiss/impl/ResultHandler.h>

#ifndef FINTEGER
#define FINTEGER long
#endif

extern "C" {
int sgemm_(const char*, const char*, FINTEGER*, FINTEGER*, FINTEGER*,
           const float*, const float*, FINTEGER*,
           const float*, FINTEGER*, float*, float*, FINTEGER*);
}

namespace faiss {

template <class BlockResultHandler>
void exhaustive_L2sqr_blas_buf(
        const float*   __restrict x,
        const float*   __restrict y,
        size_t                      d,
        size_t                      nx,
        size_t                      ny,
        size_t                      db_blas_bs,   // = bs_y
        BlockResultHandler&         res,
        float*        __restrict    ip_block,     // bs_x * bs_y
        float*        __restrict    norms_x,      // bs_x
        float*        __restrict    norms_y)      // bs_y
{
    if (nx == 0 || ny == 0) return;

    constexpr size_t bs_x = 256;
    const     size_t bs_y = db_blas_bs;

    for (size_t i0 = 0; i0 < nx; i0 += bs_x) {
        const size_t i1 = std::min(i0 + bs_x, nx);
        const size_t q_chunk = i1 - i0;

        /* ‖x‖² for this query block */
        fvec_norms_L2sqr(norms_x, x + i0 * d, d, q_chunk);

        res.begin_multiple(i0, i1);

        for (size_t j0 = 0; j0 < ny; j0 += bs_y) {
            const size_t j1      = std::min(j0 + bs_y, ny);
            const size_t db_chunk = j1 - j0;

            /* ‖y‖² for this database block */
            fvec_norms_L2sqr(norms_y, y + j0 * d, d, db_chunk);

            /* SGEMM */
            {
                const float one = 1.f;
                float zero = 0.f;
                FINTEGER nyi = FINTEGER(db_chunk);
                FINTEGER nxi = FINTEGER(q_chunk);
                FINTEGER di  = FINTEGER(d);
                sgemm_("Transpose","Not transpose",
                       &nyi,&nxi,&di,
                       &one,
                       y + j0 * d, &di,
                       x + i0 * d, &di,
                       &zero,
                       ip_block,    &nyi);
            }

            /* IP → L2² */
#pragma omp parallel for if ((q_chunk*db_chunk) > 16384)
            for (int64_t qi = 0; qi < (int64_t)q_chunk; ++qi) {
                float* line = ip_block + qi * db_chunk;
                const float xn = norms_x[qi];
                for (size_t pj = 0; pj < db_chunk; ++pj, ++line) {
                    float d2 = xn + norms_y[pj] - 2.f * (*line);
                    *line = (d2 < 0.f || !std::isfinite(d2)) ? 0.f : d2;
                }
            }
            res.add_results(j0, j1, ip_block);
        }
        res.end_multiple();
        InterruptCallback::check();
    }
}
/* ---------------------------------------------------------------------- */
/* 2.  Public convenience wrapper (mirrors knn_L2sqr())                   */
/* ---------------------------------------------------------------------- */
inline void knn_L2sqr_buf(
        const float*   x,
        const float*   y,
        size_t         d,
        size_t         nx,
        size_t         ny,
        size_t         k,
        float*         vals,            // (nx × k)
        int64_t*       ids,             // (nx × k)
        float*         ip_block,
        float*         norms_x,
        float*         norms_y_buf,
        size_t         db_blas_bs       = distance_compute_blas_database_bs,
        const float*   y_norms          = nullptr,
        const IDSelector* sel           = nullptr)
{
    /* selector path falls back to stock Faiss (rare in hot loop) -------- */
    if (sel) {
        knn_L2sqr(x, y, d, nx, ny, k, vals, ids, y_norms, sel);
        return;
    }

    /* choose the same handlers Faiss uses --------------------------------*/
    if (k == 1) {
        Top1BlockResultHandler<CMax<float,int64_t>> rh(nx, vals, ids);
        exhaustive_L2sqr_blas_buf(x,y,d,nx,ny,db_blas_bs,
                                  rh,ip_block,norms_x,norms_y_buf);
    } else if (k < distance_compute_min_k_reservoir) {
        HeapBlockResultHandler<CMax<float,int64_t>> rh(nx, vals, ids, k);
        exhaustive_L2sqr_blas_buf(x,y,d,nx,ny,db_blas_bs,
                                  rh,ip_block,norms_x,norms_y_buf);
    } else {
        ReservoirBlockResultHandler<CMax<float,int64_t>> rh(nx, vals, ids, k);
        exhaustive_L2sqr_blas_buf(x,y,d,nx,ny,db_blas_bs,
                                  rh,ip_block,norms_x,norms_y_buf);
    }
}

} // namespace faiss
#endif /* FAISS_EXHAUSTIVE_L2SQR_BUF_H */