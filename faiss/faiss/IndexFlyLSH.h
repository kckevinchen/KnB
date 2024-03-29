/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#ifndef INDEX_FLY_LSH_H
#define INDEX_FLY_LSH_H

#include <vector>

#include <faiss/IndexFlatCodes.h>
#include <faiss/VectorTransform.h>

namespace faiss {

/** The sign of each vector component is put in a binary signature */
struct IndexFlyLSH : IndexFlatCodes {
    int nbits;             ///< nb of bits per vector
    bool train_thresholds; ///< whether we train thresholds or use 0

    //     RandomRotationMatrix rrot; ///< optional random rotation
    FlyMatrix rrot;

    std::vector<float> thresholds; ///< thresholds to compare with

    std::vector<int> partition;  ///< partiton to compress

    IndexFlyLSH(
            idx_t d,
            idx_t d_h,
            int nbits,
            bool train_thresholds = false);

    /** Preprocesses and resizes the input to the size required to
     * binarize the data
     *
     * @param x input vectors, size n * d
     * @return output vectors, size n * bits. May be the same pointer
     *         as x, otherwise it should be deleted by the caller
     */
    const float* apply_preprocess(idx_t n, const float* x) const;

    void load_weight(const float* x,const int* p);

    void train(idx_t n, const float* x) override;

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels) const override;

    /// transfer the thresholds to a pre-processing stage (and unset
    /// train_thresholds)
    void transfer_thresholds(LinearTransform* vt);

    ~IndexFlyLSH() override {}

    IndexFlyLSH();

    /* standalone codec interface.
     *
     * The vectors are decoded to +/- 1 (not 0, 1) */
    void sa_encode(idx_t n, const float* x, uint8_t* bytes) const override;

    void sa_decode(idx_t n, const uint8_t* bytes, float* x) const override;
};

} // namespace faiss

#endif
