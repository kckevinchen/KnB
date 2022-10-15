/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/IndexHNSWAdaptive.h>

#include <omp.h>
#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <queue>
#include <unordered_set>

#include <stdint.h>
#include <sys/stat.h>
#include <sys/types.h>

#ifdef __SSE__
#endif

#include <faiss/Index2Layer.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/random.h>

extern "C" {

/* declare BLAS functions, see http://www.netlib.org/clapack/cblas/ */

int sgemm_(
        const char* transa,
        const char* transb,
        FINTEGER* m,
        FINTEGER* n,
        FINTEGER* k,
        const float* alpha,
        const float* a,
        FINTEGER* lda,
        const float* b,
        FINTEGER* ldb,
        float* beta,
        float* c,
        FINTEGER* ldc);
}

namespace faiss {

using idx_t = Index::idx_t;
using MinimaxHeap = HNSWAdaptive::MinimaxHeap;
using storage_idx_t = HNSWAdaptive::storage_idx_t;
using NodeDistFarther = HNSWAdaptive::NodeDistFarther;

HNSWAdaptiveStats hnsw_adaptive_stats;

/**************************************************************
 * add / search blocks of descriptors
 **************************************************************/

namespace {

/* Wrap the distance computer into one that negates the
   distances. This makes supporting INNER_PRODUCE search easier */

struct NegativeDistanceComputer : DistanceComputer {
    /// owned by this
    DistanceComputer* basedis;

    explicit NegativeDistanceComputer(DistanceComputer* basedis)
            : basedis(basedis) {}

    void set_query(const float* x) override {
        basedis->set_query(x);
    }

    /// compute distance of vector i to current query
    float operator()(idx_t i) override {
        return -(*basedis)(i);
    }

    /// compute distance between two stored vectors
    float symmetric_dis(idx_t i, idx_t j) override {
        return -basedis->symmetric_dis(i, j);
    }

    virtual ~NegativeDistanceComputer() {
        delete basedis;
    }
};

DistanceComputer* storage_distance_computer(const Index* storage) {
    if (storage->metric_type == METRIC_INNER_PRODUCT) {
        return new NegativeDistanceComputer(storage->get_distance_computer());
    } else {
        return storage->get_distance_computer();
    }
}

void hnsw_add_vertices(
        IndexHNSWAdaptive& index_hnsw,
        size_t n0,
        size_t n,
        const float* x,
        bool verbose,
        bool preset_levels = false) {
    size_t d = index_hnsw.d;
    HNSWAdaptive& hnsw = index_hnsw.hnsw;
    size_t ntotal = n0 + n;
    double t0 = getmillisecs();
    if (verbose) {
        printf("hnsw_add_vertices: adding %zd elements on top of %zd "
               "(preset_levels=%d)\n",
               n,
               n0,
               int(preset_levels));
    }

    if (n == 0) {
        return;
    }

    int max_level = hnsw.prepare_level_tab(n, preset_levels);

    if (verbose) {
        printf("  max_level = %d\n", max_level);
    }

    std::vector<omp_lock_t> locks(ntotal);
    for (int i = 0; i < ntotal; i++)
        omp_init_lock(&locks[i]);

    // add vectors from highest to lowest level
    std::vector<int> hist;
    std::vector<int> order(n);

    { // make buckets with vectors of the same level

        // build histogram
        for (int i = 0; i < n; i++) {
            storage_idx_t pt_id = i + n0;
            int pt_level = hnsw.levels[pt_id] - 1;
            while (pt_level >= hist.size())
                hist.push_back(0);
            hist[pt_level]++;
        }

        // accumulate
        std::vector<int> offsets(hist.size() + 1, 0);
        for (int i = 0; i < hist.size() - 1; i++) {
            offsets[i + 1] = offsets[i] + hist[i];
        }

        // bucket sort
        for (int i = 0; i < n; i++) {
            storage_idx_t pt_id = i + n0;
            int pt_level = hnsw.levels[pt_id] - 1;
            order[offsets[pt_level]++] = pt_id;
        }
    }

    idx_t check_period = InterruptCallback::get_period_hint(
            max_level * index_hnsw.d * hnsw.efConstruction);

    { // perform add
        RandomGenerator rng2(789);

        int i1 = n;

        for (int pt_level = hist.size() - 1; pt_level >= 0; pt_level--) {
            int i0 = i1 - hist[pt_level];

            if (verbose) {
                printf("Adding %d elements at level %d\n", i1 - i0, pt_level);
            }

            // random permutation to get rid of dataset order bias
            for (int j = i0; j < i1; j++)
                std::swap(order[j], order[j + rng2.rand_int(i1 - j)]);

            bool interrupt = false;

#pragma omp parallel if (i1 > i0 + 100)
            {
                VisitedTable vt(ntotal);

                DistanceComputer* dis =
                        storage_distance_computer(index_hnsw.storage);
                ScopeDeleter1<DistanceComputer> del(dis);
                int prev_display =
                        verbose && omp_get_thread_num() == 0 ? 0 : -1;
                size_t counter = 0;

#pragma omp for schedule(dynamic)
                for (int i = i0; i < i1; i++) {
                    storage_idx_t pt_id = order[i];
                    dis->set_query(x + (pt_id - n0) * d);

                    // cannot break
                    if (interrupt) {
                        continue;
                    }

                    hnsw.add_with_locks(*dis, pt_level, pt_id, locks, vt);

                    if (prev_display >= 0 && i - i0 > prev_display + 10000) {
                        prev_display = i - i0;
                        printf("  %d / %d\r", i - i0, i1 - i0);
                        fflush(stdout);
                    }

                    if (counter % check_period == 0) {
                        if (InterruptCallback::is_interrupted()) {
                            interrupt = true;
                        }
                    }
                    counter++;
                }
            }
            if (interrupt) {
                FAISS_THROW_MSG("computation interrupted");
            }
            i1 = i0;
        }
        FAISS_ASSERT(i1 == 0);
    }
    if (verbose) {
        printf("Done in %.3f ms\n", getmillisecs() - t0);
    }

    for (int i = 0; i < ntotal; i++) {
        omp_destroy_lock(&locks[i]);
    }
}

} // namespace

/**************************************************************
 * IndexHNSW implementation
 **************************************************************/

IndexHNSWAdaptive::IndexHNSWAdaptive(int d, int M, MetricType metric)
        : Index(d, metric),
          hnsw(M),
          own_fields(false),
          storage(nullptr)
          {}

IndexHNSWAdaptive::IndexHNSWAdaptive(Index* storage, int M)
        : Index(storage->d, storage->metric_type),
          hnsw(M),
          own_fields(false),
          storage(storage)
          {}

IndexHNSWAdaptive::~IndexHNSWAdaptive() {
    if (own_fields) {
        delete storage;
    }
}

void IndexHNSWAdaptive::train(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(
            storage,
            "Please use IndexHNSWFlat (or variants) instead of IndexHNSW directly");
    // hnsw structure does not require training
    storage->train(n, x);
    is_trained = true;
}

void IndexHNSWAdaptive::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels) const

{
    FAISS_THROW_IF_NOT(k > 0);

    FAISS_THROW_IF_NOT_MSG(
            storage,
            "Please use IndexHNSWFlat (or variants) instead of IndexHNSW directly");
    size_t n1 = 0, n2 = 0, n3 = 0, ndis = 0, nreorder = 0;

    idx_t check_period = InterruptCallback::get_period_hint(
            hnsw.max_level * d * hnsw.efSearch);

    for (idx_t i0 = 0; i0 < n; i0 += check_period) {
        idx_t i1 = std::min(i0 + check_period, n);

#pragma omp parallel
        {
            VisitedTable vt(ntotal);

            DistanceComputer* dis = storage_distance_computer(storage);
            ScopeDeleter1<DistanceComputer> del(dis);

#pragma omp for reduction(+ : n1, n2, n3, ndis, nreorder)
            for (idx_t i = i0; i < i1; i++) {
                idx_t* idxi = labels + i * k;
                float* simi = distances + i * k;
                dis->set_query(x + i * d);

                maxheap_heapify(k, simi, idxi);
                HNSWAdaptiveStats stats = hnsw.search(*dis, k, idxi, simi, vt);
                n1 += stats.n1;
                n2 += stats.n2;
                n3 += stats.n3;
                ndis += stats.ndis;
                nreorder += stats.nreorder;
                maxheap_reorder(k, simi, idxi);
            }
        }
        InterruptCallback::check();
    }

    if (metric_type == METRIC_INNER_PRODUCT) {
        // we need to revert the negated distances
        for (size_t i = 0; i < k * n; i++) {
            distances[i] = -distances[i];
        }
    }

    hnsw_adaptive_stats.combine({n1, n2, n3, ndis, nreorder});
}

void IndexHNSWAdaptive::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(
            storage,
            "Please use IndexHNSWFlat (or variants) instead of IndexHNSW directly");
    FAISS_THROW_IF_NOT(is_trained);
    int n0 = ntotal;
    storage->add(n, x);
    ntotal = storage->ntotal;

    hnsw_add_vertices(*this, n0, n, x, verbose, hnsw.levels.size() == ntotal);
}

void IndexHNSWAdaptive::reset() {
    hnsw.reset();
    storage->reset();
    ntotal = 0;
}

void IndexHNSWAdaptive::reconstruct(idx_t key, float* recons) const {
    storage->reconstruct(key, recons);
}

void IndexHNSWAdaptive::shrink_level_0_neighbors(int new_size) {
#pragma omp parallel
    {
        DistanceComputer* dis = storage_distance_computer(storage);
        ScopeDeleter1<DistanceComputer> del(dis);

#pragma omp for
        for (idx_t i = 0; i < ntotal; i++) {
            size_t begin, end;
            hnsw.neighbor_range(i, 0, &begin, &end);

            std::priority_queue<NodeDistFarther> initial_list;

            for (size_t j = begin; j < end; j++) {
                int v1 = hnsw.neighbors[j];
                if (v1 < 0)
                    break;
                initial_list.emplace(dis->symmetric_dis(i, v1), v1);

                // initial_list.emplace(qdis(v1), v1);
            }

            std::vector<NodeDistFarther> shrunk_list;
            HNSWAdaptive::shrink_neighbor_list(
                    *dis, initial_list, shrunk_list, new_size);

            for (size_t j = begin; j < end; j++) {
                if (j - begin < shrunk_list.size())
                    hnsw.neighbors[j] = shrunk_list[j - begin].id;
                else
                    hnsw.neighbors[j] = -1;
            }
        }
    }
}

void IndexHNSWAdaptive::search_level_0(
        idx_t n,
        const float* x,
        idx_t k,
        const storage_idx_t* nearest,
        const float* nearest_d,
        float* distances,
        idx_t* labels,
        int nprobe,
        int search_type) const {
    FAISS_THROW_IF_NOT(k > 0);
    FAISS_THROW_IF_NOT(nprobe > 0);

    storage_idx_t ntotal = hnsw.levels.size();
    size_t n1 = 0, n2 = 0, n3 = 0, ndis = 0, nreorder = 0;

#pragma omp parallel
    {
        DistanceComputer* qdis = storage_distance_computer(storage);
        ScopeDeleter1<DistanceComputer> del(qdis);

        VisitedTable vt(ntotal);

#pragma omp for reduction(+ : n1, n2, n3, ndis, nreorder)
        for (idx_t i = 0; i < n; i++) {
            idx_t* idxi = labels + i * k;
            float* simi = distances + i * k;

            qdis->set_query(x + i * d);
            maxheap_heapify(k, simi, idxi);

            if (search_type == 1) {
                int nres = 0;

                for (int j = 0; j < nprobe; j++) {
                    storage_idx_t cj = nearest[i * nprobe + j];

                    if (cj < 0)
                        break;

                    if (vt.get(cj))
                        continue;

                    int candidates_size = std::max(hnsw.efSearch, int(k));
                    MinimaxHeap candidates(candidates_size);

                    candidates.push(cj, nearest_d[i * nprobe + j]);

                    HNSWAdaptiveStats search_stats;
                    nres = hnsw.search_from_candidates(
                            *qdis,
                            k,
                            idxi,
                            simi,
                            candidates,
                            vt,
                            search_stats,
                            0,
                            nres);
                    n1 += search_stats.n1;
                    n2 += search_stats.n2;
                    n3 += search_stats.n3;
                    ndis += search_stats.ndis;
                    nreorder += search_stats.nreorder;
                }
            } else if (search_type == 2) {
                int candidates_size = std::max(hnsw.efSearch, int(k));
                candidates_size = std::max(candidates_size, nprobe);

                MinimaxHeap candidates(candidates_size);
                for (int j = 0; j < nprobe; j++) {
                    storage_idx_t cj = nearest[i * nprobe + j];

                    if (cj < 0)
                        break;
                    candidates.push(cj, nearest_d[i * nprobe + j]);
                }

                HNSWAdaptiveStats search_stats;
                hnsw.search_from_candidates(
                        *qdis, k, idxi, simi, candidates, vt, search_stats, 0);
                n1 += search_stats.n1;
                n2 += search_stats.n2;
                n3 += search_stats.n3;
                ndis += search_stats.ndis;
                nreorder += search_stats.nreorder;
            }
            vt.advance();

            maxheap_reorder(k, simi, idxi);
        }
    }

    hnsw_adaptive_stats.combine({n1, n2, n3, ndis, nreorder});
}

void IndexHNSWAdaptive::init_level_0_from_knngraph(
        int k,
        const float* D,
        const idx_t* I) {
    int dest_size = hnsw.nb_neighbors(0);

#pragma omp parallel for
    for (idx_t i = 0; i < ntotal; i++) {
        DistanceComputer* qdis = storage_distance_computer(storage);
        std::vector<float> vec(d);
        storage->reconstruct(i, vec.data());
        qdis->set_query(vec.data());

        std::priority_queue<NodeDistFarther> initial_list;

        for (size_t j = 0; j < k; j++) {
            int v1 = I[i * k + j];
            if (v1 == i)
                continue;
            if (v1 < 0)
                break;
            initial_list.emplace(D[i * k + j], v1);
        }

        std::vector<NodeDistFarther> shrunk_list;
        HNSWAdaptive::shrink_neighbor_list(*qdis, initial_list, shrunk_list, dest_size);

        size_t begin, end;
        hnsw.neighbor_range(i, 0, &begin, &end);

        for (size_t j = begin; j < end; j++) {
            if (j - begin < shrunk_list.size())
                hnsw.neighbors[j] = shrunk_list[j - begin].id;
            else
                hnsw.neighbors[j] = -1;
        }
    }
}

void IndexHNSWAdaptive::init_level_0_from_entry_points(
        int n,
        const storage_idx_t* points,
        const storage_idx_t* nearests) {
    std::vector<omp_lock_t> locks(ntotal);
    for (int i = 0; i < ntotal; i++)
        omp_init_lock(&locks[i]);

#pragma omp parallel
    {
        VisitedTable vt(ntotal);

        DistanceComputer* dis = storage_distance_computer(storage);
        ScopeDeleter1<DistanceComputer> del(dis);
        std::vector<float> vec(storage->d);

#pragma omp for schedule(dynamic)
        for (int i = 0; i < n; i++) {
            storage_idx_t pt_id = points[i];
            storage_idx_t nearest = nearests[i];
            storage->reconstruct(pt_id, vec.data());
            dis->set_query(vec.data());

            hnsw.add_links_starting_from(
                    *dis, pt_id, nearest, (*dis)(nearest), 0, locks.data(), vt);

            if (verbose && i % 10000 == 0) {
                printf("  %d / %d\r", i, n);
                fflush(stdout);
            }
        }
    }
    if (verbose) {
        printf("\n");
    }

    for (int i = 0; i < ntotal; i++)
        omp_destroy_lock(&locks[i]);
}

void IndexHNSWAdaptive::reorder_links() {
    int M = hnsw.nb_neighbors(0);

#pragma omp parallel
    {
        std::vector<float> distances(M);
        std::vector<size_t> order(M);
        std::vector<storage_idx_t> tmp(M);
        DistanceComputer* dis = storage_distance_computer(storage);
        ScopeDeleter1<DistanceComputer> del(dis);

#pragma omp for
        for (storage_idx_t i = 0; i < ntotal; i++) {
            size_t begin, end;
            hnsw.neighbor_range(i, 0, &begin, &end);

            for (size_t j = begin; j < end; j++) {
                storage_idx_t nj = hnsw.neighbors[j];
                if (nj < 0) {
                    end = j;
                    break;
                }
                distances[j - begin] = dis->symmetric_dis(i, nj);
                tmp[j - begin] = nj;
            }

            fvec_argsort(end - begin, distances.data(), order.data());
            for (size_t j = begin; j < end; j++) {
                hnsw.neighbors[j] = tmp[order[j - begin]];
            }
        }
    }
}

void IndexHNSWAdaptive::link_singletons() {
    printf("search for singletons\n");

    std::vector<bool> seen(ntotal);

    for (size_t i = 0; i < ntotal; i++) {
        size_t begin, end;
        hnsw.neighbor_range(i, 0, &begin, &end);
        for (size_t j = begin; j < end; j++) {
            storage_idx_t ni = hnsw.neighbors[j];
            if (ni >= 0)
                seen[ni] = true;
        }
    }

    int n_sing = 0, n_sing_l1 = 0;
    std::vector<storage_idx_t> singletons;
    for (storage_idx_t i = 0; i < ntotal; i++) {
        if (!seen[i]) {
            singletons.push_back(i);
            n_sing++;
            if (hnsw.levels[i] > 1)
                n_sing_l1++;
        }
    }

    printf("  Found %d / %" PRId64 " singletons (%d appear in a level above)\n",
           n_sing,
           ntotal,
           n_sing_l1);

    std::vector<float> recons(singletons.size() * d);
    for (int i = 0; i < singletons.size(); i++) {
        FAISS_ASSERT(!"not implemented");
    }
}

void IndexHNSWAdaptive::add_with_prob(idx_t n, const float* x,const double* prob) {
    FAISS_THROW_IF_NOT_MSG(
            storage,
            "Please use IndexHNSWFlat (or variants) instead of IndexHNSW directly");
    FAISS_THROW_IF_NOT(is_trained);
    hnsw.preset_level(n, prob);
    int n0 = ntotal;

    FAISS_THROW_IF_NOT(hnsw.levels.size() == n0+n);

    add(n,x);
}


/**************************************************************
 * IndexHNSWFlat implementation
 **************************************************************/

IndexHNSWFlatAdaptive::IndexHNSWFlatAdaptive() {
    is_trained = true;
}

IndexHNSWFlatAdaptive::IndexHNSWFlatAdaptive(int d, int M, MetricType metric)
        : IndexHNSWAdaptive(new IndexFlat(d, metric), M) {
    own_fields = true;
    is_trained = true;
}


} // namespace faiss