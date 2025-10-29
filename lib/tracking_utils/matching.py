import cv2
import numpy as np
import scipy
import lap
from scipy.spatial.distance import cdist
import torch

from cython_bbox import bbox_overlaps as bbox_ious
from . import kalman_filter
import time

def merge_matches(m1, m2, shape):
    O,P,Q = shape
    m1 = np.asarray(m1)
    m2 = np.asarray(m2)

    M1 = scipy.sparse.coo_matrix((np.ones(len(m1)), (m1[:, 0], m1[:, 1])), shape=(O, P))
    M2 = scipy.sparse.coo_matrix((np.ones(len(m2)), (m2[:, 0], m2[:, 1])), shape=(P, Q))

    mask = M1*M2
    match = mask.nonzero()
    match = list(zip(match[0], match[1]))
    unmatched_O = tuple(set(range(O)) - set([i for i, j in match]))
    unmatched_Q = tuple(set(range(Q)) - set([j for i, j in match]))

    return match, unmatched_O, unmatched_Q


def _indices_to_matches(cost_matrix, indices, thresh):
    matched_cost = cost_matrix[tuple(zip(*indices))]
    matched_mask = (matched_cost <= thresh)

    matches = indices[matched_mask]
    unmatched_a = tuple(set(range(cost_matrix.shape[0])) - set(matches[:, 0]))
    unmatched_b = tuple(set(range(cost_matrix.shape[1])) - set(matches[:, 1]))

    return matches, unmatched_a, unmatched_b


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b


def ious(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=float)
    if ious.size == 0:
        return ious

    ious = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=float),
        np.ascontiguousarray(btlbrs, dtype=float)
    )

    return ious

def kld_ex(bboxes1, bboxes2, factor=3.):

    ious = np.zeros((bboxes1.shape[0], bboxes2.shape[0]), dtype=float)
    if ious.size == 0:
        return ious

    center1 = (bboxes1[:, None, :2] + bboxes1[:, None, 2:]) / 2  # [N, 1, 2]
    center2 = (bboxes2[None, :, :2] + bboxes2[None, :, 2:]) / 2  # [1, M, 2]
    
    # 计算中心偏移量
    whs = center1 - center2  # [N, M, 2]
    
    # 计算宽度和高度
    w1 = (bboxes1[:, None, 2] - bboxes1[:, None, 0])*factor + 1e-6  # [N, 1]
    h1 = (bboxes1[:, None, 3] - bboxes1[:, None, 1])*factor + 1e-6  # [N, 1]
    w2 = (bboxes2[None, :, 2] - bboxes2[None, :, 0])*factor + 1e-6  # [1, M]
    h2 = (bboxes2[None, :, 3] - bboxes2[None, :, 1])*factor + 1e-6  # [1, M]
    
    # 计算 KL 散度
    kl = (w2**2 / w1**2 + h2**2 / h1**2 
          + 4 * whs[..., 0]**2 / w1**2 + 4 * whs[..., 1]**2 / h1**2 
          + torch.log(w1**2 / w2**2) + torch.log(h1**2 / h2**2) - 2) / 2
    
    # 计算最终 KL 散度距离
    kld = 1 / (1 + kl)  # [N, M]
    
    return kld

def wd_ex(bboxes1, bboxes2, factor=3.):

    ious = np.zeros((bboxes1.shape[0], bboxes2.shape[0]), dtype=float)
    if ious.size == 0:
        return ious

    center1 = (bboxes1[:, None, :2] + bboxes1[:, None, 2:]) / 2
    center2 = (bboxes2[None, :, :2] + bboxes2[None, :, 2:]) / 2
    whs = center1 - center2

    center_distance = whs[..., 0] * whs[..., 0] + whs[..., 1] * whs[..., 1] + 1e-6  #

    w1 = (bboxes1[:, None, 2] - bboxes1[:, None, 0])*factor + 1e-6
    h1 = (bboxes1[:, None, 3] - bboxes1[:, None, 1])*factor + 1e-6
    w2 = (bboxes2[None, :, 2] - bboxes2[None, :, 0])*factor + 1e-6
    h2 = (bboxes2[None, :, 3] - bboxes2[None, :, 1])*factor + 1e-6

    wh_distance = ((w1 - w2) ** 2 + (h1 - h2) ** 2) / 4
    wasserstein = center_distance + wh_distance

    wd = 1/(1+wasserstein)

    return wd

def gaussian2D(shape, sigma=1):
    """生成高斯分布"""
    w, h = shape
    m, n = [(ss - 1.) / 2. for ss in shape]  # 计算高斯核的中心坐标
    y, x = np.ogrid[-m:m+1, -n:n+1]  # 创建坐标网格，范围从 -m 到 m 和 -n 到 n
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))  # 高斯函数
    h[h < np.finfo(h.dtype).eps * h.max()] = 0  # 将接近零的值置为零
    return h

def compute_cost_matrix(target1, target2):

    ious = np.zeros((len(target1), len(target2)), dtype=float)
    if ious.size == 0:
        return ious

    """计算两个目标集合之间的代价矩阵"""
    N, M = len(target1), len(target2)
    cost_matrix = np.zeros((N, M))

    # for tlx, tly, brx, bry in target1:
    #     if int(bry - tly)+1 == 0 or int(brx - tlx)+1 == 0:
    #         print("11111111111111")
    # for tlx, tly, brx, bry in target2:
    #     if int(bry - tly)+1 == 0 or int(brx - tlx)+1 == 0:
    #         print("222222222222222")

    # 提取每个目标的坐标和高斯分布
    gaussians1 = [gaussian2D((int(brx - tlx)+1, int(bry - tly)+1), sigma=max(brx - tlx + 1, bry - tly + 1) / 6) 
                  for tlx, tly, brx, bry in target1]
    
    gaussians2 = [gaussian2D((int(brx - tlx)+1, int(bry - tly)+1), sigma=max(brx - tlx + 1, bry - tly + 1) / 6) 
                  for tlx, tly, brx, bry in target2]
    

    # 计算代价矩阵
    for i in range(N):
        tlx1, tly1, brx1, bry1 = target1[i]
        gaussian1 = gaussians1[i]

        for j in range(M):
            tlx2, tly2, brx2, bry2 = target2[j]
            gaussian2 = gaussians2[j]

            # 计算两个目标的重叠区域
            x_overlap_start = max(tlx1, tlx2)
            x_overlap_end = min(brx1, brx2)
            y_overlap_start = max(tly1, tly2)
            y_overlap_end = min(bry1, bry2)

            # 如果没有重叠区域，跳过此对计算
            if x_overlap_start >= x_overlap_end or y_overlap_start >= y_overlap_end:
                cost_matrix[i, j] = 0
                continue

            # 计算重叠区域的相对索引
            x_start_idx1 = int(x_overlap_start - tlx1)
            x_end_idx1 = int(x_overlap_end - tlx1)
            y_start_idx1 = int(y_overlap_start - tly1)
            y_end_idx1 = int(y_overlap_end - tly1)

            x_start_idx2 = int(x_overlap_start - tlx2)
            x_end_idx2 = int(x_overlap_end - tlx2)
            y_start_idx2 = int(y_overlap_start - tly2)
            y_end_idx2 = int(y_overlap_end - tly2)

            # 获取重叠区域内的高斯值
            # gaussian1_overlap = gaussian1[y_start_idx1:y_end_idx1, x_start_idx1:x_end_idx1]
            # gaussian2_overlap = gaussian2[y_start_idx2:y_end_idx2, x_start_idx2:x_end_idx2]
            gaussian1_overlap = gaussian1[x_start_idx1:x_end_idx1, y_start_idx1:y_end_idx1]
            gaussian2_overlap = gaussian2[x_start_idx2:x_end_idx2, y_start_idx2:y_end_idx2]

            # 计算重叠区域内的点积并求和
            overlap_distance = np.sum(gaussian1_overlap * gaussian2_overlap)
            cost_matrix[i, j] = overlap_distance

    return cost_matrix

def iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = []
        btlbrs = []
        for track in atracks:
            atlwh = track.tlwh
            atlbrs.append(np.array([max(atlwh[0]-0.5*atlwh[2],0), max(atlwh[1]-0.5*atlwh[3],0), atlwh[0]+1.5*atlwh[2], atlwh[1]+1.5*atlwh[3]]))
        for track in btracks:
            btlwh = track.tlwh
            btlbrs.append(np.array([max(btlwh[0]-0.5*btlwh[2],0), max(btlwh[1]-0.5*btlwh[3],0), btlwh[0]+1.5*btlwh[2], btlwh[1]+1.5*btlwh[3]]))

    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix


def fuse_distance(atracks, btracks):
     
    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs_for_kld = [track.tlbr for track in atracks]
        btlbrs_for_kld = [track.tlbr for track in btracks]
        atlbrs = []
        btlbrs = []
        for track in atracks:
            atlwh = track.tlwh
            atlbrs.append(np.array([max(atlwh[0]-0.5*atlwh[2],0), max(atlwh[1]-0.5*atlwh[3],0), atlwh[0]+1.5*atlwh[2], atlwh[1]+1.5*atlwh[3]]))
        for track in btracks:
            btlwh = track.tlwh
            btlbrs.append(np.array([max(btlwh[0]-0.5*btlwh[2],0), max(btlwh[1]-0.5*btlwh[3],0), btlwh[0]+1.5*btlwh[2], btlwh[1]+1.5*btlwh[3]]))
    _ious = ious(atlbrs, btlbrs)
    cost_matrix1 = 1 - _ious
    
    atlbrs_for_kld = torch.tensor(np.array(atlbrs_for_kld))
    btlbrs_for_kld = torch.tensor(np.array(btlbrs_for_kld))
    _kld = kld_ex(atlbrs_for_kld, btlbrs_for_kld, factor=1.)
    cost_matrix2 = 1 - _kld
    
    cost_matrix = (cost_matrix1 + np.array(cost_matrix2)) / 2.

    return np.array(cost_matrix)


def v_iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlwh_to_tlbr(track.pred_bbox) for track in atracks]
        btlbrs = [track.tlwh_to_tlbr(track.pred_bbox) for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix

def embedding_distance(tracks, detections, metric='cosine'):
    """
    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param metric:
    :return: cost_matrix np.ndarray
    """

    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=float)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray([track.curr_feat for track in detections], dtype=float)
    #for i, track in enumerate(tracks):
        #cost_matrix[i, :] = np.maximum(0.0, cdist(track.smooth_feat.reshape(1,-1), det_features, metric))
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=float)
    cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))  # Nomalized features
    return cost_matrix


def gate_cost_matrix(kf, cost_matrix, tracks, detections, only_position=False):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position)
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
    return cost_matrix


def fuse_motion(kf, cost_matrix, tracks, detections, only_position=False, lambda_=0.98):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position, metric='maha')
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
        cost_matrix[row] = lambda_ * cost_matrix[row] + (1 - lambda_) * gating_distance
    return cost_matrix


def fuse_iou(cost_matrix, tracks, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    reid_sim = 1 - cost_matrix
    iou_dist = iou_distance(tracks, detections)
    iou_sim = 1 - iou_dist
    fuse_sim = reid_sim * (1 + iou_sim) / 2
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    #fuse_sim = fuse_sim * (1 + det_scores) / 2
    fuse_cost = 1 - fuse_sim
    return fuse_cost


def fuse_score(cost_matrix, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    fuse_cost = 1 - fuse_sim
    return fuse_cost