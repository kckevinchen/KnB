/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexFlyLSH.h>

#include <cstdio>
#include <cstring>

#include <algorithm>

#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/hamming.h>
#include <faiss/utils/utils.h>

namespace faiss {

/***************************************************************
 * IndexLSH
 ***************************************************************/

IndexFlyLSH::IndexFlyLSH(
        idx_t d,
        idx_t d_h,
        int nbits,
        bool train_thresholds)
        : IndexFlatCodes((nbits + 7) / 8, d),
          nbits(nbits),
          train_thresholds(train_thresholds),
          rrot(d, d_h) {
    is_trained = false;
}

IndexFlyLSH::IndexFlyLSH()
        : nbits(0), train_thresholds(false) {}

const float* IndexFlyLSH::apply_preprocess(idx_t n, const float* x) const {
    float* xt = rrot.apply(n, x);
    // if (rotate_data) {
    //     // also applies bias if exists
    //     xt = rrot.apply(n, x);
    // } else if (d != nbits) {
    //     assert(nbits < d);
    //     xt = new float[nbits * n];
    //     float* xp = xt;
    //     for (idx_t i = 0; i < n; i++) {
    //         const float* xl = x + i * d;
    //         for (int j = 0; j < nbits; j++)
    //             *xp++ = xl[j];
    //     }
    // }

    if (train_thresholds) {

        float* xp = xt;
        for (idx_t i = 0; i < n; i++)
            for (int j = 0; j < nbits; j++)
                *xp++ -= thresholds[j];
    }

    return xt ? xt : x;
}


void IndexFlyLSH::load_weight(const float* x,const int* p){
    rrot.init(x);
    partition.resize(nbits+1);
    for(int i = 0; i < nbits; i ++){
        partition[i] = p[i];
    }
    is_trained = true;
}



void IndexFlyLSH::train(idx_t n, const float* x) {
    if (train_thresholds) {
        thresholds.resize(nbits);
        train_thresholds = false;
        const float* xt = apply_preprocess(n, x);
        ScopeDeleter<float> del(xt == x ? nullptr : xt);
        train_thresholds = true;

        float* transposed_x = new float[n * nbits];
        ScopeDeleter<float> del2(transposed_x);

        for (idx_t i = 0; i < n; i++)
            for (idx_t j = 0; j < nbits; j++)
                transposed_x[j * n + i] = xt[i * nbits + j];

        for (idx_t i = 0; i < nbits; i++) {
            float* xi = transposed_x + i * n;
            // std::nth_element
            std::sort(xi, xi + n);
            if (n % 2 == 1)
                thresholds[i] = xi[n / 2];
            else
                thresholds[i] = (xi[n / 2 - 1] + xi[n / 2]) / 2;
        }
    }
    is_trained = true;
}

void IndexFlyLSH::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels) const {
    FAISS_THROW_IF_NOT(k > 0);

    FAISS_THROW_IF_NOT(is_trained);
    const float* xt = apply_preprocess(n, x);
    ScopeDeleter<float> del(xt == x ? nullptr : xt);

    uint8_t* qcodes = new uint8_t[n * code_size];
    ScopeDeleter<uint8_t> del2(qcodes);

    const int* p = partition.data();
    fvecs2bitvecs_p(xt, qcodes, nbits, n,p);

    int* idistances = new int[n * k];
    ScopeDeleter<int> del3(idistances);

    int_maxheap_array_t res = {size_t(n), size_t(k), labels, idistances};

    hammings_knn_hc(&res, qcodes, codes.data(), ntotal, code_size, true);

    // convert distances to floats
    for (int i = 0; i < k * n; i++)
        distances[i] = idistances[i];
}

void IndexFlyLSH::transfer_thresholds(LinearTransform* vt) {
    if (!train_thresholds)
        return;
    FAISS_THROW_IF_NOT(nbits == vt->d_out);
    if (!vt->have_bias) {
        vt->b.resize(nbits, 0);
        vt->have_bias = true;
    }
    for (int i = 0; i < nbits; i++)
        vt->b[i] -= thresholds[i];
    train_thresholds = false;
    thresholds.clear();
}

void IndexFlyLSH::sa_encode(idx_t n, const float* x, uint8_t* bytes) const {
    FAISS_THROW_IF_NOT(is_trained);
    const float* xt = apply_preprocess(n, x);
    ScopeDeleter<float> del(xt == x ? nullptr : xt);
    const int* p = partition.data();
    fvecs2bitvecs_p(xt, bytes, nbits, n,p);
}

void IndexFlyLSH::sa_decode(idx_t n, const uint8_t* bytes, float* x) const {
    float* xt = x;
    ScopeDeleter<float> del;
    if (nbits != d) {
        xt = new float[n * nbits];
        del.set(xt);
    }
    bitvecs2fvecs(bytes, xt, nbits, n);

    if (train_thresholds) {
        float* xp = xt;
        for (idx_t i = 0; i < n; i++) {
            for (int j = 0; j < nbits; j++) {
                *xp++ += thresholds[j];
            }
        }
    }
    rrot.reverse_transform(n, xt, x);
    // if (rotate_data) {
    //     rrot.reverse_transform(n, xt, x);
    // } else if (nbits != d) {
    //     for (idx_t i = 0; i < n; i++) {
    //         memcpy(x + i * d, xt + i * nbits, nbits * sizeof(xt[0]));
    //     }
    // }
}

} // namespace faiss
