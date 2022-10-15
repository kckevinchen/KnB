/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#pragma once

#include <vector>

#include <faiss/IndexFlat.h>
#include <faiss/IndexNNDescent.h>
#include <faiss/impl/NSGAdaptive.h>
#include <faiss/utils/utils.h>

namespace faiss {

/** The NSG index is a normal random-access index with a NSG
 * link structure built on top */

struct IndexNSGAdaptive : Index {
    /// the link strcuture
    NSGAdaptive nsg;

    /// the sequential storage
    bool own_fields;
    Index* storage;

    /// the index is built or not
    bool is_built;

    /// K of KNN graph for building
    int GK;

    /// indicate how to build a knn graph
    /// - 0: build NSG with brute force search
    /// - 1: build NSG with NNDescent
    char build_type;

    /// parameters for nndescent
    int nndescent_S;
    int nndescent_R;
    int nndescent_L;
    int nndescent_iter;

    explicit IndexNSGAdaptive(int d = 0, int R = 32, MetricType metric = METRIC_L2);
    explicit IndexNSGAdaptive(Index* storage, int R = 32);

    ~IndexNSGAdaptive() override;

    void build(idx_t n, const float* x, idx_t* knn_graph, int GK);

    void add(idx_t n, const float* x) override;

    /// Trains the storage if needed
    void train(idx_t n, const float* x) override;

    /// entry point for search
    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels) const override;

    void reconstruct(idx_t key, float* recons) const override;

    void reset() override;

    void check_knn_graph(const idx_t* knn_graph, idx_t n, int K) const;
    void add_with_prob(idx_t n, const float* x,const double* prob);
};

/** Flat index topped with with a NSG structure to access elements
 *  more efficiently.
 */

struct IndexNSGFlatAdaptive : IndexNSGAdaptive {
    IndexNSGFlatAdaptive();
    IndexNSGFlatAdaptive(int d, int R, MetricType metric = METRIC_L2);
};

} // namespace faiss













// /**
//  * Copyright (c) Facebook, Inc. and its affiliates.
//  *
//  * This source code is licensed under the MIT license found in the
//  * LICENSE file in the root directory of this source tree.
//  */

// // -*- c++ -*-

// #pragma once

// #include <vector>

// #include <faiss/IndexFlat.h>
// #include <faiss/IndexNNDescent.h>
// #include <faiss/impl/NSG.h>
// #include <faiss/impl/NSGAdaptive.h>
// #include <faiss/utils/utils.h>
// #include <faiss/IndexNSG.h>

// namespace faiss {

// /** The NSG index is a normal random-access index with a NSG
//  * link structure built on top */

// struct IndexNSGAdaptive : IndexNSG {
//     /// the link strcuture
//     NSGAdaptive nsg;

//     explicit IndexNSGAdaptive(int d = 0, int R = 32, MetricType metric = METRIC_L2);
//     explicit IndexNSGAdaptive(Index* storage, int R = 32);

//     ~IndexNSGAdaptive() override;

//     void add_with_prob(idx_t n, const float* x,const float* prob);
// };

// /** Flat index topped with with a NSG structure to access elements
//  *  more efficiently.
//  */

// struct IndexNSGFlatAdaptive : IndexNSGAdaptive {
//     IndexNSGFlatAdaptive();
//     IndexNSGFlatAdaptive(int d, int R, MetricType metric = METRIC_L2);
// };

// } // namespace faiss
