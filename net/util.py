from __future__ import division
import time
import argparse
import numpy as np
from torch import nn, optim
from net.lib.metrics import ValidationFunction, get_nearestneighbors, sanitize
from net.lib.net import Normalize, forward_pass
from net.lib.quantizers import getQuantizer
import torch.nn.functional as F
import torch
import itertools
import faissAdaptive
import faiss
import hypertools as hyp
import matplotlib.pyplot as plt

def get_nearestneighbors_imp(xq, xb,imp_idx, k, device, needs_exact=True, verbose=False,cross=False,need_dist=False):
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
        print("here")
    if device == 'cuda':
        index_imp = faiss.index_cpu_to_all_gpus(index_imp)
        index_other = faiss.index_cpu_to_all_gpus(index_other)

    xq_imp = xq[imp_idx]
    xq_other = xq[~imp_idx]
    xb_imp = xb[imp_idx]
    xb_other = xb[~imp_idx]

    I = np.empty((xq.shape[0],k))

    start = time.time()
    index_imp.add(xb_imp)
    index_other.add(xb_other)
    org_imp = np.where(imp_idx == True)[0]
    org_other = np.where(imp_idx == False)[0]

    if(cross):
        I_other_dist, I_other = index_imp.search(xq_other, k)
        I_imp_dist, I_imp = index_other.search(xq_imp, k)
        I_imp = org_other[I_imp]
        I_other = org_imp[I_other]
        
    else:
      I_imp_dist, I_imp = index_imp.search(xq_imp, k)
      I_other_dist, I_other = index_other.search(xq_other, k)
      I_other = org_other[I_other]
      I_imp = org_imp[I_imp]



    # project back to I
    I[imp_idx,:]= I_imp
    I[~imp_idx,:] = I_other
    if verbose:
        print("  NN search (%s) done in %.2f s" % (
            device, time.time() - start))

    I_dist = np.empty((xq.shape[0],k))
    if(need_dist):
        I_dist[~imp_idx,:] = I_other_dist
        I_dist[imp_idx,:] = I_imp_dist
        return I_dist,I
    else:
        return I
def repeat(l, r):
    return list(itertools.chain.from_iterable(itertools.repeat(x, r) for x in l))

def pairwise_NNs_inner_imp(x,imp,cross=True):
    """
    Pairwise nearest neighbors for L2-normalized vectors.
    Uses Torch rather than Faiss to remain on GPU.
    """
    # parwise dot products (= inverse distance)
    imp = torch.from_numpy(imp)
    imp = imp.to(x.device)
    x_imp = x[imp]
    x_other = x[~imp]
    imp_idx = torch.nonzero(imp)
    other_idx = torch.nonzero(~imp)
    if(cross):
        dots = torch.mm(x_imp, x_other.t())
        _,I_imp = torch.max(dots, 1)
        _,I_other = torch.max(dots.t(), 1)
        I_imp = other_idx[I_imp]
        I_other = imp_idx[I_other]
    else:
        I_imp = pairwise_NNs_inner(x_imp)
        I_other = pairwise_NNs_inner(x_other)
        I_imp = imp_idx[I_imp]
        I_other = other_idx[I_other]
    I = torch.zeros(x.shape[0]).long().to(x.device).view(-1,1)
    I[imp] = I_imp
    I[~imp] = I_other
    # n = x.shape[0]
    # dots.view(-1)[::(n+1)].fill_(-1)  # Trick to fill diagonal with -1
    # _, I = torch.max(dots, 1)  # max inner prod -> min distance
    return I.flatten()


def pairwise_NNs_inner(x):
    """
    Pairwise nearest neighbors for L2-normalized vectors.
    Uses Torch rather than Faiss to remain on GPU.
    """
    # parwise dot products (= inverse distance)
    dots = torch.mm(x, x.t())
    n = x.shape[0]
    dots.view(-1)[::(n+1)].fill_(-1)  # Trick to fill diagonal with -1
    _, I = torch.max(dots, 1)  # max inner prod -> min distance
    return I

def topk_NNs_inner(x,k):
    """
    Pairwise nearest neighbors for L2-normalized vectors.
    Uses Torch rather than Faiss to remain on GPU.
    """
    # parwise dot products (= inverse distance)
    dots = torch.mm(x, x.t())
    n = x.shape[0]
    dots.view(-1)[::(n+1)].fill_(-1)  # Trick to fill diagonal with -1
    _, I = torch.topk(dots,k, 1)  # max inner prod -> min distance
    return I


def triplet_optimize(xt, gt_nn,imp_idx, net, args, val_func,new_loss=True):
    """
    train a triplet loss on the training set xt (a numpy array)
    gt_nn:    ground-truth nearest neighbors in input space
    net:      network to optimize
    args:     various runtime arguments
    val_func: callback called periodically to evaluate the network
    """
    k = args.margin_k

    lr_schedule = [float(x.rstrip().lstrip()) for x in args.lr_schedule.split(",")]
    assert args.epochs % len(lr_schedule) == 0
    lr_schedule = repeat(lr_schedule, args.epochs // len(lr_schedule))
    print("Lr schedule", lr_schedule)

    N, kpos = gt_nn.shape
    qt = lambda x: x

    xt_var = torch.from_numpy(xt).to(args.device)

    # prepare optimizer
    if args.mode == 'grad_norm':
        print("using grad_norm")
        # weights = torch.nn.Parameter(torch.tensor([args.lambda_triplet,args.lambda_seperation,args.lambda_lid],device=args.device).float())
        scale = torch.tensor([args.lambda_triplet,args.lambda_uniform,args.lambda_seperation,args.lambda_lid],device=args.device)
        weights = torch.nn.Parameter(torch.tensor([1,1,1,1],device=args.device).float())
        weights.retain_grad()
        print(weights.is_leaf)
        optimizer = optim.SGD([{'params': net.parameters()},
                {'params': weights,'lr': 1e-4}], lr_schedule[0], momentum=args.momentum)
    else:
        optimizer = optim.SGD(net.parameters(), lr_schedule[0], momentum=args.momentum)
    pdist = nn.PairwiseDistance(2)
    all_logs = []
    for epoch in range(args.epochs):
        # Update learning rate
        args.lr = lr_schedule[epoch]

        for param_group in optimizer.param_groups:
            # print(param_group)
            param_group['lr'] = args.lr
            break

        t0 = time.time()

        # Sample positives for triplet
        rank_pos = np.random.choice(kpos, size=N)
        positive_idx = gt_nn[np.arange(N), rank_pos]


        # Sample negatives for triplet
        net.eval()
        print("  Forward pass")
        if(epoch == 0):
            xl_net = xt
        else:
            xl_net = forward_pass(net, xt, 1024)
        print("  Distances")
        I = get_nearestneighbors(xl_net, qt(xl_net), args.rank_negative, args.device, needs_exact=False)
        negative_idx = I[:, -1]

        I = get_nearestneighbors_imp(xl_net, qt(xl_net),imp_idx, k+3, args.device,cross=False, needs_exact=False)
        kth_neighbour =  I[:,k+2]
        range_kth_neighbour = I[:,1:k+1]
        nearhit = I[:,1]

        # k neighbour for imp data

        I = get_nearestneighbors_imp(xl_net, qt(xl_net),imp_idx, k, args.device,cross=True, needs_exact=False)
        negative_range_kth = I[:, :k]

          # Plot data:
        if(epoch % 10 == 0):
            np.random.seed(1234)
            sample_idx = np.random.choice(xt.shape[0],1000)
            imp_group = imp_idx[sample_idx]
            hyp.plot(xl_net[sample_idx], '.',group=imp_group,ndims=2,reduce="TSNE")
            plt.savefig("{}.png".format(epoch))
            plt.show()


        # training pass
        print("Train")
        net.train()
        avg_triplet, avg_uniform,avg_seperation, avg_loss,avg_sep_uniform,avg_lid = 0, 0, 0,0,0,0
        offending =offending_imp = idx_batch = 0

        # process dataset in a random order
        perm = np.random.permutation(N)

        t1 = time.time()

        for i0 in range(0, N, args.batch_size):
            i1 = min(i0 + args.batch_size, N)
            n = i1 - i0

            if(new_loss and idx_batch%args.update_freq == 0 and epoch != 0):
                net.eval()
                rank_pos = np.random.choice(kpos, size=N)
                positive_idx = gt_nn[np.arange(N), rank_pos]
                xl_net = forward_pass(net, xt, 1024)
                I = get_nearestneighbors(xl_net, qt(xl_net), args.rank_negative, args.device, needs_exact=False)
                negative_idx = I[:, -1]
 
                I = get_nearestneighbors_imp(xl_net, qt(xl_net),imp_idx, k+3, args.device,cross=False, needs_exact=False)
                kth_neighbour =  I[:,k+2]
                range_kth_neighbour = I[:,1:k+1]
                nearhit = I[:,1]


                I = get_nearestneighbors_imp(xl_net, qt(xl_net),imp_idx, k, args.device,cross=True, needs_exact=False)
                negative_range_kth = I[:, :k]
                net.train()




            data_idx = perm[i0:i1]


            # anchor, positives, negatives
            ins = xt_var[data_idx]
            pos = xt_var[positive_idx[data_idx]]
            neg = xt_var[negative_idx[data_idx]].view(-1, xt_var.shape[1])


            # do the forward pass (+ record gradients)
            ins, pos, neg = net(ins), net(pos), net(neg)
            pos, neg = qt(pos), qt(neg)
     
            if(new_loss):

                kth = xt_var[kth_neighbour[data_idx]]
                range_kth = xt_var[range_kth_neighbour[data_idx]].view(-1, xt_var.shape[1])

                # range_kth_negative = xt_var[range_kth_negative]
                kth = net(kth)
                kth = qt(kth)
                range_kth = net(range_kth)
                range_kth = qt(range_kth)
                range_kth_dist = torch.log(pdist(ins.repeat(1,k).view(-1,ins.shape[1]),range_kth)).view(-1,k).sum(dim=1)/(k)
                loss_lid = range_kth_dist.mean()


                t = 0.5
                nhit = xt_var[nearhit[data_idx]]
                nhit = qt(net(nhit))
                nmiss =  xt_var[negative_range_kth[data_idx]].view(-1, xt_var.shape[1])
                nmiss = qt(net(nmiss))
                hit_diss = pdist(ins,nhit)
                miss_dist = pdist(ins.repeat(1,k).view(-1,ins.shape[1]),nmiss).view(-1,k)
                loss_seperation =  torch.log(torch.exp(hit_diss/t)/(torch.exp(hit_diss/t) + torch.exp(miss_dist/t).sum(dim=1))).mean()
                # For reference
                threshold = pdist(ins,kth).detach()
                miss_dist = F.relu(threshold - miss_dist.detach()[:,0])
                offending_imp += torch.sum(miss_dist.data > 0).item()

            else:
                loss_lid = torch.tensor(0)
                loss_seperation =  torch.tensor(0)
                            # entropy loss
            k_batch = k
            I = topk_NNs_inner(ins,k_batch+1)
            I = I[:,1:]
            loss_uniform = -torch.log(pdist(ins.repeat(1,k_batch).view(-1,ins.shape[1]),ins[I].view(-1, ins.shape[1]))).view(-1,k_batch).sum(dim=1).mean()

            
            per_point_loss = pdist(ins, pos) - pdist(ins, neg)
            per_point_loss = F.relu(per_point_loss)
            loss_triplet = per_point_loss.mean()
            offending += torch.sum(per_point_loss.data > 0).item()
        

            # Grad norm
            if args.mode == 'grad_norm':
                if epoch == 0:
                    initial_task_loss = [loss_triplet.data.cpu(),loss_uniform.data.cpu(),loss_seperation.data.cpu(),loss_lid.data.cpu()]
                    initial_task_loss = torch.stack(initial_task_loss).numpy()
                task_loss = [loss_triplet,loss_uniform,loss_seperation,loss_lid]
                task_loss = torch.stack(task_loss)
                relu_weight = F.relu(weights)
                relu_weight = scale*relu_weight
                weighted_task_loss = torch.mul(relu_weight, task_loss)
                loss = torch.sum(weighted_task_loss)
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                W = net


                weights.grad.data = weights.grad.data * 0.0

                # get the gradient norms for each of the tasks
                # G^{(i)}_w(t) 
                norms = []
                for i in range(len(task_loss)):
                    # get the gradient of this task loss with respect to the shared parameters
                    gygw = torch.autograd.grad(task_loss[i], W.parameters(), retain_graph=True)
                    # compute the norm
                    norms.append(torch.norm(torch.mul(weights[i], gygw[0])))
                norms = torch.stack(norms)
                #print('G_w(t): {}'.format(norms))


                # compute the inverse training rate r_i(t) 
                # \curl{L}_i 
                if torch.cuda.is_available():
                    loss_ratio = task_loss.data.cpu().numpy() / initial_task_loss
                else:
                    loss_ratio = task_loss.data.numpy() / initial_task_loss
                # r_i(t)
                inverse_train_rate = loss_ratio / np.mean(loss_ratio)
                #print('r_i(t): {}'.format(inverse_train_rate))


                # compute the mean norm \tilde{G}_w(t) 
                if torch.cuda.is_available():
                    mean_norm = np.mean(norms.data.cpu().numpy())
                else:
                    mean_norm = np.mean(norms.data.numpy())
                #print('tilde G_w(t): {}'.format(mean_norm))


                # compute the GradNorm loss 
                # this term has to remain constant
                constant_term = torch.tensor(mean_norm * (inverse_train_rate ** args.alpha), requires_grad=False)
                if torch.cuda.is_available():
                    constant_term = constant_term.cuda()
                #print('Constant term: {}'.format(constant_term))
                # this is the GradNorm loss itself
                grad_norm_loss = torch.sum(torch.abs(norms - constant_term))
                #print('GradNorm loss {}'.format(grad_norm_loss))

                # compute the gradient for the weights
                weights.grad = torch.autograd.grad(grad_norm_loss, weights)[0]
            else:
                loss = args.lambda_triplet*loss_triplet + args.lambda_uniform * loss_uniform + args.lambda_seperation*loss_seperation + args.lambda_lid*loss_lid
                optimizer.zero_grad()
                loss.backward()

            # collect some stats
            avg_triplet += loss_triplet.data.item()
            avg_seperation += loss_seperation.data.item()
            avg_uniform += loss_uniform.data.item()
            avg_loss += loss.data.item()
            avg_lid += loss_lid.data.item()


            optimizer.step()

            idx_batch += 1

            if args.mode == 'grad_norm':
                normalize_coeff = 3 / torch.sum(weights.data, dim=0)
                weights.data = weights.data * normalize_coeff

        avg_triplet /= idx_batch
        avg_uniform /= idx_batch
        avg_loss /= idx_batch
        avg_seperation /= idx_batch
        avg_sep_uniform /= idx_batch
        avg_lid /= idx_batch

        logs = {
            'epoch': epoch,
            'loss_triplet': avg_triplet,
            'loss_uniform': avg_uniform,
            "loss_seperation": avg_seperation,
            'loss': avg_loss,
            'offending': offending,
            'offending_seperation':offending_imp,
            'lr': args.lr
        }
        all_logs.append(logs)

        t2 = time.time()
        # maybe perform a validation run
        if (epoch + 1) % args.val_freq == 0:
            logs['val'] = val_func(net, epoch, args, all_logs)


        t3 = time.time()

        # synthetic logging
        if(args.mode == 'grad_norm'):
            w_np = relu_weight.detach().cpu().numpy()
            print ('epoch %d, times: [hn %.2f s epoch %.2f s val %.2f s]'
               ' lr = %f'
               ' loss = %g = %g*%g + %g*%g +%g*%g + %g*%g, offending %d seperation %d' % (
            epoch, t1 - t0, t2 - t1, t3 - t2,
            args.lr,
            avg_loss, w_np[0], avg_triplet,w_np[1],avg_uniform,w_np[2],avg_seperation, w_np[3],avg_lid,offending,offending_imp
        ))

        else:
            print ('epoch %d, times: [hn %.2f s epoch %.2f s val %.2f s]'
                ' lr = %f'
                ' loss = %g = %g*%g + %g * %g +%g*%g + %g*%g, offending %d seperation %d' % (
                epoch, t1 - t0, t2 - t1, t3 - t2,
                args.lr,
                avg_loss, args.lambda_triplet,avg_triplet,args.lambda_uniform, avg_uniform,args.lambda_seperation,avg_seperation, args.lambda_lid,avg_lid,offending,offending_imp
            ))

        logs['times'] = (t1 - t0, t2 - t1, t3 - t2)

    return all_logs


def generate_result(l,xt,xq_all,xb,imp_idx,imp_xb,gt_all,all_skewness = [0.5],dir="./",args = None,new_loss=True):
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.checkpoint_dir = dir
    args.validation_quantizers = []
    args.validation_quantizers.extend(["hnsw_15"])
    args.validation_quantizers.extend(["hnswAdaptive_15"])
    args.validation_quantizers.extend(["opq_%d" % x for x in l])
    args.validation_quantizers.extend(["pqAdaptive_%d" % x for x in l])
    args.validation_quantizers.extend(["opqAdaptive_%d" % x for x in l])
    print(args)

    print ("load dataset %s" % args.database)
    # (xt, xb, xq, gt) = load_dataset(args.database, args.device, size=args.size_base, test=False)
    print(xt.shape)

    print ("keeping %d/%d training vectors" % (args.num_learn, xt.shape[0]))
    xt = sanitize(xt[:args.num_learn])

    print ("computing training ground truth")
    xt_gt = get_nearestneighbors(xt, xt, args.rank_positive+1, device=args.device)
    # remove the first example
    xt_gt =  xt_gt[np.arange(xt_gt.shape[0]),1:]


    print ("build network")

    dim = xb.shape[1]
    dint, dout = args.dint, args.dout

    net = nn.Sequential(
        nn.Linear(in_features=dim, out_features=dint, bias=True),
        nn.BatchNorm1d(dint),
        nn.ReLU(),
        nn.Linear(in_features=dint, out_features=dint, bias=True),
        nn.BatchNorm1d(dint),
        nn.ReLU(),
        nn.Linear(in_features=dint, out_features=dout, bias=True),
        Normalize()
    )

    if args.init_name != '':
        print ("loading state from %s" % args.init_name)
        ckpt = torch.load(args.init_name)
        net.load_state_dict(ckpt['state_dict'])
        start_epoch = ckpt['epoch']

    net.to(args.device)

    val = ValidationFunction(xq_all, xb,xt, gt_all, args.checkpoint_dir,
                            validation_key=args.save_best_criterion,
                            quantizers=args.validation_quantizers,training = True,imp=imp_idx,all_skewness = all_skewness,imp_xb=imp_xb)

    all_logs = triplet_optimize(xt, xt_gt,imp_idx, net, args, val,new_loss=new_loss)
    return all_logs




def generate_opq(l,xt,xq_all,xb,imp_idx,gt_all,all_skewness = [0.5]):
    all_recalls = {}
    for skewness in all_skewness:
        xq = xq_all[skewness]
        gt = gt_all[skewness]
        for l_sub in l:
            index = getQuantizer("opq_{}".format(l_sub), xt.shape[1])
            index.train(xt)
            index.add(xb)
            _,I_t = index.search(xq,100)
            recalls = []
            nq = xq.shape[0]
            for rank in 1, 10, 100:
                recall = (I_t[:, :rank] == gt[:, :1]).sum() / float(nq)
                recalls.append(recall)
                print('%.4f' % recall, end=" ")
            print("")
            all_recalls["opq_{}_{}".format(l_sub,skewness)] =  recalls
    return all_recalls

def generate_pq(l,xt,xq_all,xb,imp_idx,gt_all,all_skewness = [0.5]):
    all_recalls = {}
    for skewness in all_skewness:
        xq = xq_all[skewness]
        gt = gt_all[skewness]
        for l_sub in l:
            index = getQuantizer("pq_{}".format(l_sub), xt.shape[1])
            index.train(xt)
            index.add(xb)
            _,I_t = index.search(xq,100)
            recalls = []
            nq = xq.shape[0]
            for rank in 1, 10, 100:
                recall = (I_t[:, :rank] == gt[:, :1]).sum() / float(nq)
                recalls.append(recall)
                print('%.4f' % recall, end=" ")
            print("")
            all_recalls["pq_{}_{}".format(l_sub,skewness)] =  recalls
    return all_recalls

def generate_pca_pq(l,xt,xq_all,xb,imp_idx,gt_all,all_skewness = [0.5]):
    all_recalls = {}
    for skewness in all_skewness:
        xq = xq_all[skewness]
        gt = gt_all[skewness]
        for l_sub in l:
            index = getQuantizer("apq_{}".format(l_sub), xt.shape[1])
            index.train(xt)
            index.add(xb)
            _,I_t = index.search(xq,100)
            recalls = []
            nq = xq.shape[0]
            for rank in 1, 10, 100:
                recall = (I_t[:, :rank] == gt[:, :1]).sum() / float(nq)
                recalls.append(recall)
                print('%.4f' % recall, end=" ")
            print("")
            all_recalls["pq_{}_{}".format(l_sub,skewness)] =  recalls
    return all_recalls

def generate_pca_pq_adaptive(l,xt,xq_all,xb,imp_idx,gt_all,all_skewness = [0.5],k=10,return_index=False):
    all_recalls = {}
    if(return_index):
        all_index = {}
    for skewness in all_skewness:
        xq = xq_all[skewness]
        gt = gt_all[skewness]
        for l_sub in l:
            index = getQuantizer("PCApqAdaptive_{}".format(l_sub), xt.shape[1])
            index.set_imp(imp_idx)
            index.set_lambda(skewness)
            index.train(xt,margin=k)
            index.add(xb)
            _,I_t = index.search(xq,100)
            recalls = []
            nq = xq.shape[0]
            for rank in 1, 10, 100:
                recall = (I_t[:, :rank] == gt[:, :1]).sum() / float(nq)
                recalls.append(recall)
                print('%.4f' % recall, end=" ")
            print("")
            all_recalls["opqAdaptive_{}_{}".format(l_sub,skewness)] =  recalls
            if(return_index):
                all_index["opqAdaptive_{}_{}".format(l_sub,skewness)] = index
    if(return_index):
        return all_recalls,all_index
    return all_recalls



def generate_opq_adaptive(l,xt,xq_all,xb,imp_idx,gt_all,all_skewness = [0.5],k=10,return_index=False):
    all_recalls = {}
    if(return_index):
        all_index = {}
    for skewness in all_skewness:
        xq = xq_all[skewness]
        gt = gt_all[skewness]
        for l_sub in l:
            index = getQuantizer("opqAdaptive_{}".format(l_sub), xt.shape[1])
            index.set_imp(imp_idx)
            index.set_lambda(skewness)
            index.train(xt,margin=k)
            index.add(xb)
            _,I_t = index.search(xq,100)
            recalls = []
            nq = xq.shape[0]
            for rank in 1, 10, 100:
                recall = (I_t[:, :rank] == gt[:, :1]).sum() / float(nq)
                recalls.append(recall)
                print('%.4f' % recall, end=" ")
            print("")
            all_recalls["opqAdaptive_{}_{}".format(l_sub,skewness)] =  recalls
            if(return_index):
                all_index["opqAdaptive_{}_{}".format(l_sub,skewness)] = index
    if(return_index):
        return all_recalls,all_index
    return all_recalls

def generate_hnsw(l,xt,xq_all,xb,imp_idx,gt_all,all_skewness = [0.5],k=0):
    all_recalls = {}
    for skewness in all_skewness:
        xq = xq_all[skewness]
        gt = gt_all[skewness]
        for l_sub in l:
            index = getQuantizer("hnsw_{}".format(l_sub), xt.shape[1])
            index.add(xb)
            _,I_t = index.search(xq,100)
            recalls = []
            nq = xq.shape[0]
            for rank in 1, 10, 100:
                recall = (I_t[:, :rank] == gt[:, :1]).sum() / float(nq)
                recalls.append(recall)
                print('%.4f' % recall, end=" ")
            print("")
            all_recalls["hnsw_{}_{}".format(l_sub,skewness)] =  recalls
    return all_recalls


def generate_pca_hnsw(l,xt,xq_all,xb,imp_idx,gt_all,all_skewness = [0.5],k=0):
    all_recalls = {}
    for skewness in all_skewness:
        xq = xq_all[skewness]
        gt = gt_all[skewness]
        for l_sub in l:
            index = getQuantizer("PCAhnsw_{}".format(l_sub), xt.shape[1])
            index.train(xt)
            index.add(xb)
            _,I_t = index.search(xq,100)
            recalls = []
            nq = xq.shape[0]
            for rank in 1, 10, 100:
                recall = (I_t[:, :rank] == gt[:, :1]).sum() / float(nq)
                recalls.append(recall)
                print('%.4f' % recall, end=" ")
            print("")
            all_recalls["hnsw_{}_{}".format(l_sub,skewness)] =  recalls
    return all_recalls
def generate_PCA_hnswAdaptive(l,xt,xq_all,xb,imp_idx,gt_all,all_skewness = [0.5],k=0):
    all_recalls = {}
    for skewness in all_skewness:
        xq = xq_all[skewness]
        gt = gt_all[skewness]
        for l_sub in l:
            index = getQuantizer("PCAhnswAdaptive_{}".format(l_sub), xt.shape[1])
            index.train(xt)
            index.set_imp(imp_idx)
            index.set_lambda(skewness)
            index.add(xb)
            _,I_t = index.search(xq,100)
            recalls = []
            nq = xq.shape[0]
            for rank in 1, 10, 100:
                recall = (I_t[:, :rank] == gt[:, :1]).sum() / float(nq)
                recalls.append(recall)
                print('%.4f' % recall, end=" ")
            print("")
            all_recalls["hnsw_{}_{}".format(l_sub,skewness)] =  recalls
    return all_recalls



def generate_hnswAdaptive(l,xt,xq_all,xb,imp_idx,gt_all,all_skewness = [0.5],k=5):
    all_recalls = {}
    for skewness in all_skewness:
        xq = xq_all[skewness]
        gt = gt_all[skewness]
        for l_sub in l:
            index = getQuantizer("hnswAdaptive_{}".format(l_sub), xt.shape[1])
            index.set_imp(imp_idx)
            index.set_lambda(skewness)
            index.add(xb,k)
            _,I_t = index.search(xq,100)
            recalls = []
            nq = xq.shape[0]
            for rank in 1, 10, 100:
                recall = (I_t[:, :rank] == gt[:, :1]).sum() / float(nq)
                recalls.append(recall)
                print('%.4f' % recall, end=" ")
            print("")
            all_recalls["hnsw_{}_{}".format(l_sub,skewness)] =  recalls
            del index
    return all_recalls



def generate_hnswAdaptive_for_large(l,xt,xq_all,xb,imp_idx,gt_all,all_skewness = [0.5],k=5,lid = None):
    all_recalls = {}
    for skewness in all_skewness:
        xq = xq_all[skewness]
        gt = gt_all[skewness]
        for l_sub in l:
            index = getQuantizer("hnswAdaptive_{}".format(l_sub), xt.shape[1])
            index.set_imp(imp_idx)
            index.set_lid(lid)
            index.set_lambda(skewness)
            index.add(xb,k)
            _,I_t = index.search(xq,100)
            recalls = []
            nq = xq.shape[0]
            for rank in 1, 10, 100:
                recall = (I_t[:, :rank] == gt[:, :1]).sum() / float(nq)
                recalls.append(recall)
                print('%.4f' % recall, end=" ")
            print("")
            all_recalls["hnsw_{}_{}".format(l_sub,skewness)] =  recalls
            del index
    return all_recalls