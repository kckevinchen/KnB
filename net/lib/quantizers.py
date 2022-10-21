# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import division
import numpy as np
from lattices.Zn_lattice import ZnCodec
import faiss
from scipy.special import digamma
import torch
from torch import nn
from sklearn.preprocessing import QuantileTransformer
from tqdm import tqdm

def get_nearestneighbors_faiss(xq, xb, k, device, needs_exact=True, verbose=False):
    assert device in ["cpu", "cuda"]

    if verbose:
        print("Computing nearest neighbors (Faiss)")

    if needs_exact or device == 'cuda':
        index = faiss.IndexFlatL2(xq.shape[1])
    else:
        index = faiss.index_factory(xq.shape[1], "HNSW32")
        index.hnsw.efSearch = 64
    if device == 'cuda':
        index = faiss.index_cpu_to_all_gpus(index)

    index.add(xb)
    d, I = index.search(xq, k)
    return d,I

def calculate_nn_by_batch(train_data,test_data,k,n=20,device="cuda"):
    all_gt = []
    t = int(train_data.shape[0]/n)
    d = train_data.shape[1]
    for i in tqdm(range(n)):
        xt = train_data[i*t:(i+1)*t,:]
        index = faiss.IndexFlatL2(d)   # build the index
        res = faiss.StandardGpuResources()
        gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index)
        gpu_index_flat.add(xt)                  # add vectors to the index
        _, gt = gpu_index_flat.search(test_data, k) # sanity check
        all_gt.append(gt+ i*t)
        # print("round {}".format(i))
        del gpu_index_flat
    all_gt = np.hstack(all_gt)
    # return all_gt
    gt_true = []
    dist_true= []
    n = n*k
    t = int(test_data.shape[0]/n)
    N = test_data.shape[0]
    for i0 in range(0, N, t):
        i1 = min(i0 + t, N)
        data_idx = np.arange(i0,i1)
        gt_i= all_gt[data_idx ,:]
        xq = test_data[data_idx,:]
        xt = train_data[gt_i,:]
        xt = torch.from_numpy(xt).to(device)
        xq = torch.from_numpy(xq).to(device)
        xq = xq.unsqueeze(1)
        dist = torch.sqrt(torch.sum((xt - xq)**2,dim=-1))
        topk = torch.topk(dist,k,largest=False,dim=-1)
        if(device=="cuda"):
            dist = topk[0].cpu().numpy()
            gt= topk[1].cpu().numpy()
        else:
            dist = topk[0].numpy()
            gt= gt.numpy()
        for j in range(gt.shape[0]):
            gt[j,:] +=  j*n
        gt_i = gt_i.flatten()
        gt = gt_i[gt]
        gt_true.append(gt)
        dist_true.append(dist)
    dist_true = np.vstack(dist_true)
    gt_true = np.vstack(gt_true)
    return dist_true,gt_true

def estimate_lid(xt,k,batch_size=1024):
    dist,I = get_nearestneighbors_faiss(xt, xt, k+2, "cuda")
    # dist,I = calculate_nn_by_batch(xt, xt, k+2)
    N = xt.shape[0]
    lid = np.zeros(xt.shape[0])
    dist += 1e-5

        # k neighbour for imp data
    for i0 in range(0, N, batch_size):
            i1 = min(i0 + batch_size, N)
            data_idx = np.arange(i0,i1)
            cur_dist = dist[data_idx,1:]
            cur_dist = np.log(cur_dist)
            batch_lid = (np.mean(cur_dist[:,:k],axis=1)-cur_dist[:,k])
            lid[data_idx] = batch_lid
    return lid



def kl_estimate(x,x_all =None):
    if x_all is None:
      x_all = x
    n,d = x.shape
    index = faiss.IndexFlatL2(d)
    index = faiss.index_cpu_to_all_gpus(index)
    index.add(x_all)
    dist,_ = index.search(x,2)
    dist = dist[:,1] + 1e-8
    # unitball = np.pi**(d/2)/gamma(d/2 + 1)
    entropy = np.mean(d*np.log(dist)) + digamma(n)
    return entropy
def cal_m(l,x,y):
  d = x.shape[1]
  partial1 = (1-l)/l
  partial2 = 2 * (kl_estimate(y,np.concatenate([x,y])) - kl_estimate(x,np.concatenate([x,y]))) / d
  return partial1*np.exp(partial2)
def cal_k(l,x,y,c=256):
  d = x.shape[1]
  return c/(cal_m(l,x,y)**(1/(2/d+1))+1)

def allocate_budget(x,imp_idx,l=0.5,c=256):
    imp_x = x[imp_idx]
    other_x = x[~imp_idx]
    k = cal_k(l,imp_x,other_x,c)
    return round(k)



def get_nearestneighbors_imp(xq, xb,imp_idx, k, device, needs_exact=True, verbose=False,cross=False):
    assert device in ["cpu", "cuda"]

    if verbose:
        print("Computing nearest neighbors (Faiss)")

    if needs_exact or device == 'cuda':
        index_imp = faiss.IndexFlatL2(xq.shape[1])
        index_other = faiss.IndexFlatL2(xq.shape[1])
    else:
        index_imp = faiss.index_factory(xq.shape[1], "HNSW32")
        index_imp.hnsw.efSearch = 64
        index_other = faiss.index_factory(xq.shape[1], "HNSW32")
        index_other.hnsw.efSearch = 64
    if device == 'cuda':
        index_imp = faiss.index_cpu_to_all_gpus(index_imp)
        index_other = faiss.index_cpu_to_all_gpus(index_other)

    xq_imp = xq[imp_idx]
    xq_other = xq[~imp_idx]
    xb_imp = xb[imp_idx]
    xb_other = xb[~imp_idx]

    I = np.empty((xq.shape[0],k))

    index_imp.add(xb_imp)
    index_other.add(xb_other)
    org_imp = np.where(imp_idx == True)[0]
    org_other = np.where(imp_idx == False)[0]

    if(cross):
      _, I_other = index_imp.search(xq_other, k)
      _, I_imp = index_other.search(xq_imp, k)
      I_other = org_imp[I_other]
      I_imp = org_other[I_imp]
    else:
      _, I_imp = index_imp.search(xq_imp, k)
      _, I_other = index_other.search(xq_other, k)
      I_other = org_other[I_other]
      I_imp = org_imp[I_imp]



    # project back to I
    I[imp_idx,:]= I_imp
    I[~imp_idx,:] = I_other
    return I


class Quantizer:
    def __init__(self):
        self.requires_train = False
        self.asymmetric = True

    def train(self, x):
        "do nothing by default"
        pass

    def quantize(self, x):
        "return closest quantized vector"
        raise NotImplementedError("Function is not implemented")

    def __call__(self, x):
        return self.quantize(x)

class Zn(Quantizer):
    def __init__(self, r2, d):
        super(Zn, self).__init__()
        self.r2 = r2
        self.r = np.sqrt(self.r2)
        self.d = d
        self.codec = ZnCodec(self.d, self.r2)
        ntot = self.codec.nv
        self.bits = int(np.ceil(np.log2(float(ntot))))

    def quantize(self, x):
        if not np.all(np.abs(np.linalg.norm(x, axis=1) - 1) < 1e-5):
            print("WARNING: Vectors were not L2 normalized in Zn")

        return self.codec.quantize(self.r * x) / self.r


class Identity(Quantizer):
    def __init__(self, d):
        super(Identity, self).__init__()
        self.dim = d
        self.bits = d * 4

    def quantize(self, x):
        assert x.ndim == 2
        return x

try:
    import faiss
except ImportError:
    faiss = None

try:
    import faissAdaptive
except ImportError:
    faissAdaptive = None

def to_binary(x):
    n, d = x.shape
    assert d % 8 == 0
    if faiss is None:
        return ((x >= 0).reshape(n, d // 8, 8) *
                (1 << np.arange(8)).astype('uint8')).sum(2)
    else:
        y = np.empty((n, d // 8), dtype='uint8')
        faiss.real_to_binary(n * d, faiss.swig_ptr(x), faiss.swig_ptr(y))
        return y

class OPQ(Quantizer):
    def __init__(self, nbytes, d):
        super(OPQ, self).__init__()
        self.dim = d
        self.requires_train = True
        self.bits = nbytes * 8
        self.index = faiss.index_factory(
            self.dim, "OPQ%d_%d,PQ%d" % (nbytes,8*nbytes,nbytes))

    def train(self, x):
        self.index.train(x)

    def quantize(self, x):
        if not self.index.is_trained:
            print("WARNING: training OPQ inside the quantize() function")
            self.train(x[:10000])
        print("Adding vectors")
        self.index.add(x)
        return self.index.reconstruct_n(0, x.shape[0])
    def add(self,x):
        if not self.index.is_trained:
            print("WARNING: training OPQ inside the add() function")
            self.train(x[:10000])
        print("Adding vectors")
        self.index.add(x)
    def search(self,x,k):
        return self.index.search(x,k)


class PQAdaptive(Quantizer):
    def __init__(self, nbytes, d):
        super(PQAdaptive, self).__init__()
        self.dim = d
        self.requires_train = True
        self.bits = nbytes * 8
        self.index = faissAdaptive.IndexPQAdaptive(d,nbytes,8)
        self.l = 0.5
    
    def set_imp(self,imp):
        self.imp = imp

    def cal_margin(self,x,k=5):
        negative_idx_imp = get_nearestneighbors_faiss(x[self.imp], x, k, "cuda", needs_exact=False)
        self.margin = np.unique(negative_idx_imp.flatten()).astype(np.int32)
        imp_idx = np.nonzero(self.imp)[0]
        self.margin = np.setdiff1d(self.margin,imp_idx,assume_unique=True)
        print("margin",self.margin.shape)

    def set_lambda(self,l):
        self.l = l

    def train(self, x,margin=0):
        levels = np.zeros(x.shape[0]).astype(np.int32)
        if(margin):
            self.cal_margin(x,k=margin)
            imp_copy = np.copy(self.imp)
            imp_copy[self.margin] = 1
            top_level = allocate_budget(x,imp_copy,self.l)

            x_imp = x[self.imp]
            x_margin =x[self.margin]
            x_all = np.concatenate([x_imp,x_margin])
            margin_idx = np.zeros(x_all.shape[0]).astype(bool)
            margin_idx[x_imp.shape[0]:] = 1

            middle_level = allocate_budget(x_all,margin_idx,self.margin.shape[0]/(np.sum(self.imp)+self.margin.shape[0]),c=top_level)
            top_level = top_level-middle_level

            levels[self.imp] = 2
            levels[self.margin] = 1
            max_level = 3

            budget_per_level = np.array([256-top_level-middle_level,middle_level,top_level]).astype(np.int32)
            print("level",middle_level,top_level)

        else:
            levels[self.imp] = 1
            top_level = allocate_budget(x,self.imp,self.l)
            max_level = 2
            budget_per_level = np.array([256-top_level,top_level]).astype(np.int32)
            print("level",top_level)
        # imp_sum = np.sum(self.imp)
        # top_level = np.min([128,imp_sum//39])

        # other_sum = np.sum(self.margin)
        # middle_level = np.min([32,other_sum//39])

        # print(top_level,middle_level)

        # budget_per_level = np.array([256-top_level,top_level]).astype(np.int32)
        # imp_sum = np.sum(self.assign_level)
        # top_level = np.min([128,imp_sum//39])
        # budget_per_level = np.array([256-top_level,top_level]).astype(np.int32)
        self.index.preset_level(x.shape[0], faiss.swig_ptr(levels), max_level, faiss.swig_ptr(budget_per_level))
        f_p = np.zeros(x.shape[0])
        self.index.train_with_prob(x.shape[0],faiss.swig_ptr(x),faiss.swig_ptr(f_p),1)

    def quantize(self, x):
        if not self.index.is_trained:
            print("WARNING: training OPQ inside the quantize() function")
            self.train(x[:10000])
        print("Adding vectors")
        self.index.add(x)
        return self.index.reconstruct_n(0, x.shape[0])
    def add(self,x):
        if not self.index.is_trained:
            print("WARNING: training OPQ inside the add() function")
            self.train(x[:10000])
        print("Adding vectors")
        self.index.add(x)
    def search(self,x,k):
        return self.index.search(x,k)




class PCAPQAdaptive():
    def __init__(self, nbytes, d,dint=64):
        self.pca = faiss.PCAMatrix(d, dint)
        self.index = PQAdaptive(nbytes,dint)
    
    def set_imp(self,imp):
        self.index.set_imp(imp)

    def cal_margin(self,x,k=5):
        self.index.cal_margin(x,k)

    def set_lambda(self,l):
        self.index.set_lambda(l)
    def train(self, x,margin=0):
        self.pca.train(x)
        x = self.pca.apply(x)
        self.index.train(x,margin)
      
    def add(self,x):
        x = self.pca.apply(x)
        self.index.add(x)
    def search(self,x,k):
        x = self.pca.apply(x)
        return self.index.search(x,k)

class OPQAdaptive(Quantizer):
    def __init__(self, nbytes, d):
        super(OPQAdaptive, self).__init__()
        self.dim = d
        self.requires_train = True
        self.bits = nbytes * 8
        self.opq = faiss.OPQMatrix(d, nbytes,nbytes*8)
        self.index = faissAdaptive.IndexPQAdaptive(nbytes*8,nbytes,8)
        self.l = 0.5
    
    def set_imp(self,imp):
        self.imp = imp
    def set_lambda(self,l):
        self.l = l

    def sa_encode(self,x):
        x = self.opq.apply_py(x)
        return self.index.sa_encode(x)


    def cal_margin(self,x,k=5):
        negative_idx_imp = get_nearestneighbors_faiss(x[self.imp], x, k, "cuda", needs_exact=False)
        self.margin = np.unique(negative_idx_imp.flatten()).astype(np.int32)
        imp_idx = np.nonzero(self.imp)[0]
        self.margin = np.setdiff1d(self.margin,imp_idx,assume_unique=True)
        print("margin",self.margin.shape)

    def train(self, x,margin=0):
        self.opq.train(x)
        x = self.opq.apply_py(x)
        print(np.sum(self.imp),self.l)


        # imp_entropy = kl_estimate(self.imp)

        # self.cal_margin(x)
        levels = np.zeros(x.shape[0]).astype(np.int32)
        if(margin):
            print("k",margin)
            self.cal_margin(x,k=margin)

            imp_copy = np.copy(self.imp)
            imp_copy[self.margin] = 1
            top_level = allocate_budget(x,imp_copy,self.l)

            x_imp = x[self.imp]
            x_margin =x[self.margin]
            x_all = np.concatenate([x_imp,x_margin])
            margin_idx = np.zeros(x_all.shape[0]).astype(bool)
            margin_idx[x_imp.shape[0]:] = 1
            margin_p = self.margin.shape[0]/(np.sum(self.imp)+self.margin.shape[0])

            print(margin_p)

            middle_level = allocate_budget(x_all,margin_idx,margin_p,c=top_level)
            top_level = top_level-middle_level

            levels[self.margin] = 1
            levels[self.imp] = 2
            max_level = 3

            budget_per_level = np.array([256-top_level-middle_level,middle_level,top_level]).astype(np.int32)
            print("level",middle_level,top_level)

        else:
            levels[self.imp] = 1
            top_level = allocate_budget(x,self.imp,  self.l)
            max_level = 2
            budget_per_level = np.array([256-top_level,top_level]).astype(np.int32)
            print("level",top_level)

        # levels[self.margin] = 1
        # max_level = 2
        # imp_sum = np.sum(self.imp)
        # top_level = np.min([128,imp_sum//39])

        # other_sum = np.sum(self.margin)
        # middle_level = np.min([32,other_sum//39])

        self.index.preset_level(x.shape[0], faiss.swig_ptr(levels), max_level, faiss.swig_ptr(budget_per_level))
        f_p = np.zeros(x.shape[0])
        self.index.train_with_prob(x.shape[0],faiss.swig_ptr(x),faiss.swig_ptr(f_p),1)
        # imp_sum = np.sum(self.assign_level)
        # top_level = np.min([128,imp_sum//39])
        # budget_per_level = np.array([256-top_level,top_level]).astype(np.int32)
        # self.index.preset_level(x.shape[0], faiss.swig_ptr(levels), max_level, faiss.swig_ptr(budget_per_level))
        # max_level = 2
        # imp_sum = np.sum(self.assign_level)
        # top_level = np.min([128,imp_sum//39])
        # # top_level =128
        # print("top_level",top_level)
        # budget_per_level = np.array([256-top_level,top_level]).astype(np.int32)
        # self.index.preset_level(x.shape[0], faiss.swig_ptr(self.assign_level), max_level, faiss.swig_ptr(budget_per_level))
        # f_p = np.zeros(x.shape[0])
        # self.index.train_with_prob(x.shape[0],faiss.swig_ptr(x),faiss.swig_ptr(f_p),1)

    def quantize(self, x):
        if not self.index.is_trained:
            print("WARNING: training OPQ inside the quantize() function")
            self.train(x[:10000])
        print("Adding vectors")
        x = self.opq.apply_py(x)
        self.index.add(x)
        temp = self.index.reconstruct_n(0, x.shape[0])
        return self.opq.reverse_transform(temp)
        
    def add(self,x):
        if not self.index.is_trained:
            print("WARNING: training OPQ inside the add() function")
            self.train(x[:10000])
        print("Adding vectors")
        x = self.opq.apply_py(x)
        self.index.add(x)
    def search(self,x,k):
        x = self.opq.apply_py(x)
        return self.index.search(x,k)


class Binary(Quantizer):
    def __init__(self, d):
        super(Binary, self).__init__()
        self.bits = None
        self.asymmetric = False
        self.dim = d
        self.bits = (d + 7) // 8 * 8

    def quantize(self, x):
        assert x.ndim == 2
        return np.sign(x)
        # return to_binary(x)


class HNSWAdaptive():
    def __init__(self, d, m):
        self.dim = d
        self.m = m
        self.index = faissAdaptive.IndexHNSWFlatAdaptive(d,m)
        self.l = 0.5
        self.lid =None
        self.a1 = None
        self.a2 = None

    def train(self, x):
        return
    def set_lambda(self,l):
        self.l = l
    def set_lid(self,lid):
        self.lid = lid
    def set_weight(self,a1,a2):
        self.a1 = a1
        self.a2 = a2
    def set_imp(self,imp):
        self.imp = imp
    def add_with_prob(self,x,p):
        self.index.add_with_prob(x.shape[0],faissAdaptive.swig_ptr(x),faissAdaptive.swig_ptr(p))
    def add(self,x, k =5):
        if(not self.lid is None):
            lid = QuantileTransformer().fit_transform(self.lid.reshape(-1,1)).flatten()
        else:
            lid = QuantileTransformer().fit_transform(estimate_lid(x,k).reshape(-1,1)).flatten()
        r = np.random.uniform(low=0,high=1,size=x.shape[0])
        if(self.a1 is None):
            p = self.l*self.imp + 0.5*lid + 0.5*r
        else:
            p = self.l*self.imp + self.a1*lid + self.a2*r
        p = QuantileTransformer(n_quantiles=10000).fit_transform(p.reshape(-1,1)).flatten()
        self.add_with_prob(x,p)

    def search(self,x,k):
        return self.index.search(x,k)


class PCAHNSWAdaptive():
    def __init__(self, d, m,dint=64):
        self.index = HNSWAdaptive(dint,m)
        self.pca = faiss.PCAMatrix(d, dint)

    def train(self, x):
        self.pca.train(x)
        return
    def set_lambda(self,l):
        self.index.set_lambda(l)

    def set_imp(self,imp):
        self.index.set_imp(imp)
    def add_with_prob(self,x,p):
        self.index.add_with_prob(x,p)
    def add(self,x, k =5):
        x = self.pca.apply(x)
        self.index.add(x,k)
    def search(self,x,k):
        x = self.pca.apply(x)
        return self.index.search(x,k)


def getQuantizer(snippet, d):
    if snippet.startswith("zn_"):
        r2 = int(snippet.split("_")[1])
        return Zn(r2, d)
    elif snippet == "binary":
        return Binary(d)
    elif snippet == "none":
        return Identity(d)
    elif snippet.startswith("opq_"):
        nbytes = int(snippet.split("_")[1]) // 8
        return OPQ(nbytes, d)
    elif snippet.startswith("pqAdaptive_"):
        nbytes = int(snippet.split("_")[1]) // 8
        return PQAdaptive(nbytes, d)
    elif snippet.startswith("PCApqAdaptive_"):
        nbytes = int(snippet.split("_")[1]) // 8
        return PCAPQAdaptive(nbytes, d)
    elif snippet.startswith("apq_"):
        nbytes = int(snippet.split("_")[1]) // 8
        return faiss.index_factory(
            d, "PCA64,PQ%d" % (nbytes))
    elif snippet.startswith("pq_"):
        nbytes = int(snippet.split("_")[1]) // 8
        return faiss.IndexPQ(d,nbytes,8)
    elif snippet.startswith("opqAdaptive_"):
        nbytes = int(snippet.split("_")[1]) // 8
        return OPQAdaptive(nbytes, d)
    elif snippet.startswith("hnsw_"):
        m = int(snippet.split("_")[1])
        return faiss.IndexHNSWFlat(d,m)
    elif snippet.startswith("hnswAdaptive_"):
        m = int(snippet.split("_")[1])
        return HNSWAdaptive(d,m)
    elif snippet.startswith("PCAhnswAdaptive_"):
        m = int(snippet.split("_")[1])
        return PCAHNSWAdaptive(d,m)
    elif snippet.startswith("PCAhnsw_"):
        m = int(snippet.split("_")[1])
        return faiss.index_factory(
            d, "PCA64,HNSW%d" % (m))
    else:
        raise NotImplementedError("Quantizer not implemented")
