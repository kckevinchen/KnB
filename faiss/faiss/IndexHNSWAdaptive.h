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
#include <faiss/IndexPQ.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/impl/HNSWAdaptive.h>
#include <faiss/utils/utils.h>

namespace faiss {

struct IndexHNSWAdaptive;

/** The HNSW index is a normal random-access index with a HNSW
 * link structure built on top */

struct IndexHNSWAdaptive : Index {
    typedef HNSWAdaptive::storage_idx_t storage_idx_t;

    // the link strcuture
    HNSWAdaptive hnsw;

    // the sequential storage
    bool own_fields;
    Index* storage;
    
    explicit IndexHNSWAdaptive(int d = 0, int M = 32, MetricType metric = METRIC_L2);
    explicit IndexHNSWAdaptive(Index* storage, int M = 32);

    ~IndexHNSWAdaptive() override;

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

    void shrink_level_0_neighbors(int size);

    /** Perform search only on level 0, given the starting points for
     * each vertex.
     *
     * @param search_type 1:perform one search per nprobe, 2: enqueue
     *                    all entry points
     */
    void search_level_0(
            idx_t n,
            const float* x,
            idx_t k,
            const storage_idx_t* nearest,
            const float* nearest_d,
            float* distances,
            idx_t* labels,
            int nprobe = 1,
            int search_type = 1) const;

    /// alternative graph building
    void init_level_0_from_knngraph(int k, const float* D, const idx_t* I);

    /// alternative graph building
    void init_level_0_from_entry_points(
            int npt,
            const storage_idx_t* points,
            const storage_idx_t* nearests);

    // reorder links from nearest to farthest
    void reorder_links();

    void link_singletons();
    void add_with_prob(idx_t n, const float* x,const double* prob);
};

/** Flat index topped with with a HNSW structure to access elements
 *  more efficiently.
 */

struct IndexHNSWFlatAdaptive : IndexHNSWAdaptive {
    IndexHNSWFlatAdaptive();
    IndexHNSWFlatAdaptive(int d, int M, MetricType metric = METRIC_L2);
};
} // namespace faiss