# spectral_cluster.py
import torch
import torch.nn.functional as F
import numpy as np
import random
import time
import datetime
import os

import sklearn
from sklearn import metrics


from multi_kmeans_pp import MultiKMeans

from logger import Logger

from scipy.sparse.csgraph import laplacian as csgraph_laplacian

DEBUG = 0

def spectral_cluster(attn_maps,K=10,neighbor_mask=None,use_gpu=True,pre_labels=None):
    """
      Parameters
        attn_maps: Tensor (*,n_samples,n_samples)
            Attention map from Transfomrer as similarity matrix
        
        K: int
            Number of clusters, default: 10
        
        neighbor_mask: Tensor (n_samples,n_samples)
            Mask to reserve neighbors only

        pre_labels: Tensor (*,n_samples_pre)
            Label(Index of cluster) of data points of last module

      Returns
        labels:
            ['normal'] - Tensor (*,n_samples)
            ['debug'] - Tensor (len(K_trials),*,n_samples)
            Label(Index of cluster) of data points
    """

    batched = False
    if attn_maps.ndim == 3: # Batched data
        B,N,_ = attn_maps.shape
        batched = True
    else:
        B = 1
        N,_ = attn_maps.shape
    K_1 = K

    # 1. Generate similarity matrix -- only neighbor patches considered
    if neighbor_mask is None:
        if pre_labels is not None: # (*,2N)
            pre_mask = get_neighbor_mask_old(N*2,use_gpu=use_gpu) # (2N,2N) / (784,784)
            neighbor_mask = neighbor_mask_reduce(pre_mask,pre_labels,N,use_gpu=use_gpu) # (*,N,N)
        else:
            neighbor_mask = get_neighbor_mask_old(N,use_gpu=use_gpu) # (N,N)
    
    sim_mat = attn_maps*neighbor_mask # Reserve only neighbors (*,N,N)
    sim_mat = torch.softmax(sim_mat, dim=-1)
    sim_mat = 0.5 * (sim_mat + sim_mat.transpose(-2,-1)) # symmetrize (*,N,N)

    # 2. Compute degree matrix
    
    # 3. Laplacian Matrix and Normalized Laplacian Matrix
    normalized_laplacian_mat, diag_term = graph_laplacian(sim_mat) # (*,N,N), (*,N)
    
    # 4. Top K_1 eigen vector with respect to eigen values
    eig_values,eig_vectors = torch.linalg.eigh(normalized_laplacian_mat) # Eigen value decomposition of of a complex Hermitian or real symmetric matrix.
    # eigenvalues will always be real-valued, even when A is complex. It will also be ordered in ascending order.
    if batched:
        feat_mat = eig_vectors[:,:,:K_1] # (B,N,K_1)
    else:
        feat_mat = eig_vectors[:,:K_1] # (N,K_1)
    
    if diag_term is not None:
        feat_mat /= diag_term.unsqueeze(-1)

    # 5. KMeans Cluster
    if batched:
        kmeans = MultiKMeans(n_clusters=K,n_kmeans=B,max_iter=100)
        labels = kmeans.fit_predict(feat_mat) # (B,N)
        return labels # (B,N)
    else:
        kmeans = MultiKMeans(n_clusters=K,n_kmeans=1,max_iter=100)
        labels = kmeans.fit_predict(feat_mat.unsqueeze(0)) # (N,) -> (1,N)
        return labels[0] # (B,N) -> (N,)

def graph_laplacian(affinity:torch.Tensor,normed=True):
    # Borrowed from Sklearn - https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.laplacian.html
    batched = False
    if affinity.ndim == 3: # Batched data
        B,N,_ = affinity.shape
        batched = True
    else:
        B = 1
        N,_ = affinity.shape
    
    if batched:
        # https://pytorch.org/docs/stable/generated/torch.Tensor.fill_diagonal_.html
        torch.diagonal(affinity,dim1=-2,dim2=-1)[...] = 0 # (B,N)
        diag = affinity.sum(dim=-2) # (B,N)
        if normed:
            mask = (diag==0) # mask of isolated node (B,N)
            diag = torch.where(mask,1.,torch.sqrt(diag).to(torch.double)).to(diag.dtype) # (B,N)

            affinity /= diag.unsqueeze(-2) # Row
            affinity /= diag.unsqueeze(-1) # Col

            affinity *= -1
            # torch.diagonal(affinity,dim1=-2,dim2=-1)[...] = 1 - mask.float()
            torch.diagonal(affinity,dim1=-2,dim2=-1)[...] = 1 # (B,N)
        else:
            affinity *= -1
            torch.diagonal(affinity,dim1=-2,dim2=-1)[...] = diag
    else:
        # Non-batched
        affinity.fill_diagonal_(0) # (N,N) symmetric matrix
        diag = affinity.sum(dim=-2) # (N,)
        if normed:
            mask = (diag==0) # mask of isolated node
            diag = torch.where(mask,1.,torch.sqrt(diag).to(torch.double)).to(diag.dtype)
            
            affinity /= diag
            affinity /= diag[:,None]

            affinity *= -1
            # affinity.flatten()[::len(mask)+1] = 1 - mask.float()
            affinity.flatten()[::len(mask)+1] = 1
        else:
            affinity *= -1
            affinity.flatten()[::len(diag)+1] = diag
    
    return affinity,diag

def calinski_harabasz_score(X,labels,centroids=None):
    """
        Borrowed from https://github.com/scikit-learn/scikit-learn/blob/844b4be24/sklearn/metrics/cluster/_unsupervised.py#L251

        Implementation of https://scikit-learn.org/stable/modules/generated/sklearn.metrics.calinski_harabasz_score.html#sklearn.metrics.calinski_harabasz_score
    """
    assert X.ndim == 2
    N,_ = X.shape
    classes_,counuts_ = torch.unique(labels,sorted=True,return_counts=True)
    K = len(classes_)
    if DEBUG:
        print(f"[DEBUG] calinski_harabasz_score: K = {K}")
        print(f"[DEBUG] calinski_harabasz_score: counuts_ = {counuts_}")

    extra_disp, intra_disp = 0.0, 0.0
    center = torch.mean(X,dim=0)
    for q in range(K):
        cluster_q = X[labels==q]
        center_q = torch.mean(cluster_q,dim=0)
        if centroids is not None:
            center_q = centroids[q]
        extra_disp += len(cluster_q) * torch.sum((center_q-center)**2)
        intra_disp += torch.sum((cluster_q-center_q)**2)
    return (
        1.0
        if intra_disp == 0.0
        else (extra_disp*(N-K)) / (intra_disp*(K-1))
    )

def get_neighbor_mask_old(N,use_gpu=True):
    """
        neighbor: 8
    """
    P = int(N**(0.5))
    A = torch.zeros((N,N))
    ind = torch.arange(N)
    row = torch.div(ind,P,rounding_mode='floor')

    # Same row
    # ind + 1
    neigbor_ind = ind+1
    neighbor_row = torch.div(neigbor_ind,P,rounding_mode='floor')
    mask = (neigbor_ind<N) & (row==neighbor_row)
    A[ind[mask],neigbor_ind[mask]] = 1
    # ind - 1
    neigbor_ind = ind-1
    neighbor_row = torch.div(neigbor_ind,P,rounding_mode='floor')
    mask = (neigbor_ind>=0) & (row==neighbor_row)
    A[ind[mask],neigbor_ind[mask]] = 1
    # exit()

    # stride = [-(P+1),-P,-(P-1),-1]
    strides = [P-1,P,P+1]

    for s in strides:
        # ind + s
        neigbor_ind = ind+s
        neigbor_row = torch.div(neigbor_ind,P,rounding_mode='floor') - 1
        mask = (neigbor_ind<N) & (row==neigbor_row)
        A[ind[mask],neigbor_ind[mask]] = 1
        # ind - s
        neigbor_ind = ind-s
        neigbor_row = torch.div(neigbor_ind,P,rounding_mode='floor') + 1
        mask = (neigbor_ind>=0) & (row==neigbor_row)
        A[ind[mask],neigbor_ind[mask]] = 1

    if use_gpu:
        A = A.cuda()

    return A

def get_neighbor_mask(N,use_gpu=True):
    """
        neighbor: 4 (w/o diagonals)
    """
    P = int(N**(0.5))
    A = torch.zeros((N,N))
    ind = torch.arange(N)
    row = torch.div(ind,P,rounding_mode='floor')

    # Same row
    # ind + 1
    neigbor_ind = ind+1
    neighbor_row = torch.div(neigbor_ind,P,rounding_mode='floor')
    mask = (neigbor_ind<N) & (row==neighbor_row)
    A[ind[mask],neigbor_ind[mask]] = 1
    # ind - 1
    neigbor_ind = ind-1
    neighbor_row = torch.div(neigbor_ind,P,rounding_mode='floor')
    mask = (neigbor_ind>=0) & (row==neighbor_row)
    A[ind[mask],neigbor_ind[mask]] = 1
    # exit()

    # stride = [-(P+1),-P,-(P-1),-1]
    strides = [P]

    for s in strides:
        # ind + s
        neigbor_ind = ind+s
        neigbor_row = torch.div(neigbor_ind,P,rounding_mode='floor') - 1
        mask = (neigbor_ind<N) & (row==neigbor_row)
        A[ind[mask],neigbor_ind[mask]] = 1
        # ind - s
        neigbor_ind = ind-s
        neigbor_row = torch.div(neigbor_ind,P,rounding_mode='floor') + 1
        mask = (neigbor_ind>=0) & (row==neigbor_row)
        A[ind[mask],neigbor_ind[mask]] = 1

    if use_gpu:
        A = A.cuda()

    return A


def cluster_reduce(feats,labels,K,use_gpu=True):
    B,N,D = feats.shape # feats: (B,N,D)
    
    M = torch.zeros(B,K,N)
    B_ind = torch.arange(B).view(-1,1).expand(-1,N) # (B,N)
    N_ind = torch.arange(N).view(1,-1).expand(B,-1)  # (B,N)
    
    if use_gpu:
        M, B_ind, N_ind = M.cuda(), B_ind.cuda(), N_ind.cuda()
    
    M[B_ind,labels,N_ind] = 1
    M = torch.nn.functional.normalize(M, p=1, dim=-1)
    result = torch.bmm(M, feats)

    return result

def neighbor_mask_reduce(neighbor_mask,labels,K,use_gpu=True):
    B,N = labels.shape
    if neighbor_mask.ndim==2:
        neighbor_mask = neighbor_mask.contiguous().view(1,N,N).expand(B,-1,-1)
    
    M = torch.zeros(B,K,N)
    B_ind = torch.arange(B).view(-1,1).expand(-1,N) # (B,N)
    N_ind = torch.arange(N).view(1,-1).expand(B,-1)  # (B,N)
    if use_gpu:
        M, B_ind, N_ind = M.cuda(), B_ind.cuda(), N_ind.cuda()

    M[B_ind,labels,N_ind] = 1
    neighbor_mask = torch.bmm(M, neighbor_mask) # (B,K,N)
    neighbor_mask = torch.bmm(neighbor_mask,M.transpose(-2,-1)) # (B,K,K)
    #  Clear Diagonal
    neighbor_mask.flatten(1)[:, ::K + 1] = 0
    return (neighbor_mask > 0).float()


if __name__ == '__main__':
    seed = 99
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    
    # Logger
    time_info = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    log_dir = './log_sc/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        print(f"Create {log_dir}")
    log = Logger(log_dir+f'test_sc-{time_info}.log',level='debug')
    
    # Data preparation
    # Just for DEBUG
    B,N,D,K = 5,784,384,10
    # data = torch.Tensor([[0,1,0,0],
    #                      [2,1,0,0],
    #                      [0,0,3,0],
    #                      [1,2,0,0],
    #                      [0,1,1,1]])
    # data = torch.rand(B,N,D)

    file_dir = '/home/heyj/data/feature/train/n01910747/'
    file_list = os.listdir(file_dir)

    load_start_t = datetime.datetime.now()
    data = []
    for file_name in file_list[:100]: # Less images
        data.append(torch.load(os.path.join(file_dir,file_name))[1:])
    data = torch.stack(data).cuda() # torch.Size([3, 49, 384])
    # data = torch.load('/home/heyj/data/feature_50/train/n01910747/n0191074700000003.pth')[1:] # torch.Size([49, 384])
    load_t = (datetime.datetime.now() - load_start_t).total_seconds()
    print(f"data.shape: {data.shape} [{data.device}]")
    # print(torch.cuda.device_count())
    # exit(1)
    print(f"load {len(data)} images time: {load_t:.4f}s")

    # Test for sigma and K
    B,N,D = data.shape
    neighbor_mask = get_neighbor_mask(N)
    neighbor_mask = neighbor_mask.cuda()
    do_our = True 
    do_sklearn = False 

    #--------------------------------------------------------------------------------------------------------
    # Our spectral_cluster
    #--------------------------------------------------------------------------------------------------------
    if do_our:
        mini_batch_size = 16
        scores = []
        scores_skl = []
        configs = []
        sigma_trials = [31,40,50,75]
        gamma_trials = [0.0002,0.0003125,0.0005,0.0006,0.0008]
        K_trials = [10,15,20,25,28]

        log.logger.debug(f"\nOur spectral_cluster:")
        # for sigma in sigma_trials:
        for gamma in gamma_trials:
            # log.logger.debug(f"sigma:{sigma}")
            log.logger.debug(f"gamma:{gamma}")
            
            pred_labels = spectral_cluster(data,K,gamma=gamma,neighbor_mask=neighbor_mask,
                                            mode="debug",K_trials=K_trials) # (len(K_trials),B,N)
            for K_ind,K in enumerate(K_trials):
                mini_batch_indices = random.sample(range(B), mini_batch_size)
                # mini_batch_indices = [0] # DEBUG
                score = 0.0
                score_skl = 0.0
                for i in mini_batch_indices:
                    score += calinski_harabasz_score(data[i],pred_labels[K_ind,i])
                    score_skl += metrics.calinski_harabasz_score(
                                                data[i].cpu().numpy(),pred_labels[K_ind,i].cpu().numpy())
                    # print(type(score))
                    # print(type(score_skl))
                    # exit(1)
                score /= mini_batch_size
                score_skl /= mini_batch_size
                scores.append(score)
                scores_skl.append(score_skl)
                # configs.append(dict(sigma=sigma,K=K,labels=pred_labels[K_ind]))
                configs.append(dict(gamma=gamma,K=K))
                log.logger.debug(f" - K:{K}  score:{score:.4f}  score_skl:{score_skl:.4f}")
        
        # Print result
        max_ind = torch.argmax(torch.Tensor(scores))
        max_score = scores[max_ind]
        log.logger.debug(f"Max Score: {max_score}")
        log.logger.debug(f"Configurations: gamma:{configs[max_ind]['gamma']}  K:{configs[max_ind]['K']}")
    
    #--------------------------------------------------------------------------------------------------------
    # Sklearn's SpectralClustering
    #--------------------------------------------------------------------------------------------------------
    if do_sklearn:
        log.logger.debug(f"\nSklearn SpectralClustering:")
        scores_skl = []
        configs = []
        gamma_trials = [0.0003125,0.0005,0.0008]
        # sigma [100.0000,  70.7107,  50.0000,  31.6228,  25.0000]
        K_trials = [10,15,20]
        for gamma in gamma_trials:
            log.logger.debug(f"gamma:{gamma}")
            for K in K_trials:
                score_skl = 0.0
                for X in data:
                    X_ = X.cpu().numpy() # (784, 384)
                    y_pred = SpectralClustering(n_clusters=K, gamma=gamma).fit_predict(X_)
                    # score_skl += metrics.calinski_harabasz_score(X_,y_pred)
                    score_skl += calinski_harabasz_score(X,torch.from_numpy(y_pred))
                    exit(1)
                score_skl /= len(data)
                scores_skl.append(score_skl)
                configs.append(dict(gamma=gamma,K=K))
                log.logger.debug(f" - K:{K}  score_skl:{score_skl:.4f}")
        
        # Print result
        max_ind = torch.argmax(torch.Tensor(scores_skl))
        max_score = score_skl[max_ind]
        log.logger.debug(f"Max Score: {max_score}")
        log.logger.debug(f"Configurations: gamma:{configs[max_ind]['gamma']}  K:{configs[max_ind]['K']}")

    