# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import division, print_function

try:
    import faiss
    hasfaiss = True
except:
    hasfaiss = False

import torch
import shutil
from os.path import join
from .quantizers import HNSWAdaptive, getQuantizer,PQAdaptive,OPQAdaptive
from .net import forward_pass
import numpy as np
import time


#######################################################
# Nearest neighbor search functions
#######################################################
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

    start = time.time()
    index.add(xb)
    _, I = index.search(xq, k)
    if verbose:
        print("  NN search (%s) done in %.2f s" % (
            device, time.time() - start))

    return I


def cdist2(A, B):
    return  (A.pow(2).sum(1, keepdim = True)
             - 2 * torch.mm(A, B.t())
             + B.pow(2).sum(1, keepdim = True).t())

def top_dist(A, B, k):
    return cdist2(A, B).topk(k, dim=1, largest=False, sorted=True)[1]

def get_nearestneighbors_torch(xq, xb, k, device, needs_exact=False, verbose=False):
    if verbose:
        print("Computing nearest neighbors (torch)")

    assert device in ["cpu", "cuda"]
    start = time.time()
    xb, xq = torch.from_numpy(xb), torch.from_numpy(xq)
    xb, xq = xb.to(device), xq.to(device)
    bs = 500
    I = torch.cat([top_dist(xq[i*bs:(i+1)*bs], xb, k)
                   for i in range(xq.size(0) // bs)], dim=0)
    if verbose:
        print("  NN search done in %.2f s" % (time.time() - start))
    I = I.cpu()
    return I.numpy()

if hasfaiss:
    get_nearestneighbors = get_nearestneighbors_faiss
else:
    get_nearestneighbors = get_nearestneighbors_torch




#######################################################
# Evaluation metrics
#######################################################


def sanitize(x):
    return np.ascontiguousarray(x, dtype='float32')


def evaluate(net, xq_all, xb, gt_all, quantizers, best_key, device=None,
             trainset=None,imp=None,all_skewness = [0.5],margin=0,imp_xb=None):
    net.eval()
    if device is None:
        device = next(net.parameters()).device.type
    all_recall = {}
    score = 0
    xbt = forward_pass(net, sanitize(xb), device=device)
    if trainset is not None:
        trainset = forward_pass(net, sanitize(trainset), device=device)
    for skewness in all_skewness:
        xq = xq_all[skewness]
        gt = gt_all[skewness]
        print("Process skwness {}".format(skewness))
        xqt = forward_pass(net, sanitize(xq), device=device)
        nq, d = xqt.shape
        res = {}
        for quantizer in quantizers:
            print(quantizer)
            qt = getQuantizer(quantizer, d)
            if isinstance(qt,PQAdaptive) or isinstance(qt,OPQAdaptive):
                assert not imp is None, "Training PQadaptive with Nan imp"
                qt.set_imp(imp)
                qt.set_lambda(skewness)
                qt.train(trainset,margin=margin)
            elif isinstance(qt,HNSWAdaptive):
                qt.set_imp(imp_xb)
                qt.set_lambda(skewness)
            else:
                qt.train(trainset)

            qt.add(xbt)
            _,I = qt.search(xqt,100)
            # xbtq = qt(xbt)
            # if not qt.asymmetric:
            #     xqtq = qt(xqt)
            #     I = get_nearestneighbors(xqtq, xbtq, 100, device)
            # else:
            #     I = get_nearestneighbors(xqt, xbtq, 100, device)
            if(isinstance(qt,faiss.IndexHNSWFlat) or isinstance(qt,HNSWAdaptive)):
                print("%s\t: " % (quantizer), end=' ')
            elif(isinstance(qt,faiss.IndexPQ)):
                print("%s\t: " % (quantizer), end=' ')
            else:
                print("%s\t nbit=%3d: " % (quantizer, qt.bits), end=' ')

            # compute 1-recall at ranks 1, 10, 100 (comparable with
            # fig 5, left of the paper)
            recalls = []
            for rank in 1, 10, 100:
                recall = (I[:, :rank] == gt[:, :1]).sum() / float(nq)
                key = '%s,rank=%d' % (quantizer, rank)
                if key == best_key:
                    score = recall
                recalls.append(recall)
                print('%.4f' % recall, end=" ")
            res[quantizer] = recalls
            print("")
            del qt
            all_recall[skewness] = res

    return all_recall, score



class ValidationFunction:

    def __init__(self, xq_all, xb,xt, gt_all, checkpoint_dir, validation_key,
                 quantizers=[],training = None,imp=None,all_skewness = [0.5],imp_xb = None):
        assert type(quantizers) == list
        self.xq_all = xq_all
        self.xb = xb
        self.gt_all = gt_all
        self.xt = xt
        self.training = training
        self.checkpoint_dir = checkpoint_dir
        self.best_key = validation_key
        self.best_score = 0
        self.quantizers = quantizers
        self.imp = imp
        self.all_skewness = all_skewness
        self.imp_xb = imp_xb

    def __call__(self, net, epoch, args, all_logs):
        """
        Evaluates the current state of the network without
        and with quantization and stores a checkpoint.
        """
        print("Valiation at epoch %d" % epoch)
        # also store current state of network + arguments
        if(self.training):
            res, score = evaluate(net, self.xq_all, self.xb, self.gt_all,
                              self.quantizers, self.best_key,trainset = self.xt,imp = self.imp,all_skewness = self.all_skewness,margin=args.margin_k,imp_xb=self.imp_xb)
        else:
            res, score = evaluate(net, self.xq, self.xb, self.ggt_all,
                              self.quantizers, self.best_key)
        all_logs[-1]['val'] = res
        if self.checkpoint_dir:
            fname = join(self.checkpoint_dir, "checkpoint.pth")
            print("storing", fname)
            torch.save({
                'state_dict': net.state_dict(),
                'epoch': epoch,
                'args': args,
                'logs': all_logs
            }, fname)
            if score > self.best_score:
                print("%s score improves (%g > %g), keeping as best"  % (
                    self.best_key, score, self.best_score))
                self.best_score = score
                shutil.copyfile(fname, fname + '.best')

        return res
