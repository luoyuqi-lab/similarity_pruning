import math
import os
import csv
import time
import numpy as np
from collections import Counter, defaultdict

def txt_loader(filepath):
    x = []
    with open(str(filepath), 'r') as f:
        lines = f.readlines()
        for line in lines:
            value = [float(s) for s in line.split()]
            x.append(value)
    return x

def part_contri(a, b):
    """
    :param a: time series 1
    :param b: time series 2
    :return: sum of Euclidean distance difference of every data point.
    """
    c = [abs(a[i]-b[i]) for i in range(len(a))]
    Eucli_dis = sum(c)
    return Eucli_dis

def cdtw(a, b, r, return_path=False):
    """ Compute the DTW distance between 2 time series with a global window constraint (max warping degree)
    :param a: the time series array 1, template, in x-axis direction
    :param b: the time series array 2, sample, in y-axis direction
    :param r: the size of Sakoe-Chiba warping band
    :return: the DTW distance cost_prev[k]
             path: the optimal dtw mapping path
             M: Warping matrix
             D: Distance matrix (by squared Euclidean distance)
    """
    M = []
    m = len(a)
    k = 0
    cost = [float('inf')] * (2 * r + 1)
    cost_prev = [float('inf')] * (2 * r + 1)
    for i in range(0, m):
        k = max(0, r - i)
        for j in range(max(0, i - r), min(m - 1, i + r) + 1):
            # Initialize the first cell
            if i == 0 and j == 0:
                cost[k] = (a[0] - b[0]) ** 2
                k += 1
                continue
            y = float('inf') if j - 1 < 0 or k - 1 < 0 else cost[k - 1]
            x = float('inf') if i < 1 or k > 2 * r - 1 else cost_prev[k + 1]
            z = float('inf') if i < 1 or j < 1 else cost_prev[k]
            cost[k] = min(x, y, z) + (a[i] - b[j]) ** 2
            k += 1
        # Move current array (cost matrix) to previous array
        cost_prev = cost
        if return_path:
            M.append(cost_prev.copy())
    # The DTW distance is in the last cell in the cost matrix of size O(m^2) or !!At the middle of our array!!
    if return_path:
        i = m - 1
        j = r
        rj = m - 1
        path = [[m - 1, m - 1]]
        k -= 1
        while i != 0 or rj != 0:
            # From [n,m] to [0,0] in cost matrix to find optimal path
            x = M[i][j - 1] if j - 1 >= 0 else float('inf')
            y = M[i - 1][j] if i - 1 >= 0 else float('inf')
            z = M[i - 1][j + 1] if i - 1 >= 0 and j + 1 <= 2 * r else float('inf')
            # Save the real location index of optimal warping path point a_i mapping with point b_j.
            if min(x, y, z) == y:
                path.append([i - 1, rj - 1])
                i = i - 1
                rj = rj - 1
            elif min(x, y, z) == x:
                path.append([i, rj - 1])
                j = j - 1
                rj = rj - 1
            else:
                path.append([i - 1, rj])
                i = i - 1
                j = j + 1
        return cost_prev[k], path
    else:
        return cost_prev[k - 1]


def assemble_extra_data(a, b, N, r):
    """
    :param a: Time series a
    :param b: Time series b
    :param r: warping window size in CDTW
    :param N: Divide time series into n parts.
    :return: pairwise subsequences of n (number of slicing) time series partitions.
    """
    x, y = [], []
    x1, y1 = [], []
    if len(a) == len(b):
        l = math.floor(len(a) / N)
        x.append(a[0:l])
        y.append(b[0:l])
        x1.append([0] + x[0] + a[l:l + r + 1])
        y1.append([0] + y[0] + b[l:l + r + 1])
        if N != 2:
            for i in range(1, N - 1):
                p = a[int(i * l): int((i + 1) * l)]
                x.append(p)
                p = b[int(0 + i * l): int((i + 1) * l)]
                y.append(p)
                x1.append([1] + a[max(0, i * l - (r + 1)):min(len(a)-1,(i + 1) * l + r + 1)])
                y1.append([1] + b[max(0, i * l - (r + 1)):min(len(a)-1,(i + 1) * l + r + 1)])
                i += 1
        x.append(a[(N - 1) * l:])
        y.append(b[(N - 1) * l:])
        x1.append([2] + a[(N - 1) * l - (r + 1): (N - 1) * l] + x[N - 1])
        y1.append([2] + b[(N - 1) * l - (r + 1): (N - 1) * l] + y[N - 1])
    else:
        print('Please align! Now we do not support different time stamps.')
    return x1, y1


def pdtw(a, b, r):
    """
    :param a: full template series
    :param b: full sample series
    :param r: max warping window / extra window size
    :return:
    """
    c = a[1:]
    d = b[1:]
    dis, p = cdtw(a[1:], b[1:], r, return_path=True)
    E = []
    #path = []
    for i in range(len(p)):
        Euclidean_d = (c[p[i][0]] - d[p[i][1]])
        if a[0] == 0 and p[i][0] > len(c) - r - 2:
            E.append(p[i])
            dis -= Euclidean_d * Euclidean_d
        elif a[0] == 1 and (p[i][0] < r + 1 or p[i][0] > len(c) - r - 2):
            E.append(p[i])
            dis -= Euclidean_d * Euclidean_d
        elif a[0] == 2 and p[i][0] < r + 1:
            E.append(p[i])
            dis -= Euclidean_d * Euclidean_d
        # else:
        #     path.append(p[i])
    return dis#path

def fill_envelope(series, w):
    # U = np.zeros_like(series)
    # L = np.zeros_like(series)
    U = [0] * len(series)
    L = [0] * len(series)
    for i in range(len(series)):
        start = max(i-w, 0)
        stop = min(i+w, len(series)-1)
        U[i] = max(series[start:stop+1])
        L[i] = min(series[start:stop+1])

    return U, L


def LB_Keogh(t, ub, lb, bsf):
    lb_dis = 0
    for i in range(len(t)):
        if lb_dis > bsf:
            return lb_dis
        ti = t[i]
        if ti > ub[i]:
            lb_dis += (ti - ub[i]) ** 2
        elif ti < lb[i]:
            lb_dis += (ti - lb[i]) ** 2

    return lb_dis

# Square Euclidean distance
def dist(a, b):
    return (a - b) ** 2

# Fast assumption of contribution (distance\similarity) of each segment
def segment_cont(a, b):
    return np.sum(np.abs(a - b))

def TS_segmentation(a, b, N, w):
    l = len(a) // N
    x1 = [a[max(0, i * l - (w + 1)):min((i + 1) * l, len(a))] for i in range(N)]
    y1 = [b[max(0, i * l - (w + 1)):min((i + 1) * l, len(b))] for i in range(N)]

    seg_cont_dis = [segment_cont(a[i * l:(i + 1) * l], b[i * l:(i + 1) * l]) for i in range(N)]
    sorted_indices = sorted(range(N), key=lambda k: seg_cont_dis[k], reverse=True)

    return x1, y1, sorted_indices

def LB_KP_EA(a,b,w,ub,lb,N,seg_num,bsf):
    #start = timeit.default_timer()
    if w >= 1 and len(a) >= 6:
        t0, te0, t1, t2 = a[0], a[-1], a[1], a[2]
        s0, se0, s1, s2 = b[0], b[-1], b[1], b[2]

        d01 = dist(t0, s1)
        d11 = dist(t1, s1)
        d10 = dist(t1, s0)

        if w == 1:
            first_3_dis = dist(t0, s0) + min(
                d11 + dist(t2, s2),
                min(
                    min(d01, d11) + dist(t1, s2),
                    min(d10, d11) + dist(t2, s1)
                )
            )
        else:
            first_3_dis = dist(t0, s0) + min(
                d11 + dist(t2, s2),
                min(
                    min(d01 + dist(t0, s2), min(d01, d11) + dist(t1, s2)),
                    min(d10 + dist(t2, s0), min(d10, d11) + dist(t2, s1))
                )
            )
        lb_dis = first_3_dis
        if lb_dis > bsf:
            return lb_dis

        t1, t2 = a[-2], a[-3]
        s1, s2 = b[-2], b[-3]

        d01 = dist(te0, s1)
        d11 = dist(t1, s1)
        d10 = dist(t1, se0)

        if w == 1:
            last_3_dis = dist(te0, se0) + min(
                d11 + dist(t2, s2),
                min(
                    min(d01, d11) + dist(t1, s2),
                    min(d10, d11) + dist(t2, s1)
                )
            )
        else:
            last_3_dis = dist(te0, se0) + min(
                d11 + dist(t2, s2),
                min(
                    min(d01 + dist(te0, s2), min(d01, d11) + dist(t1, s2)),
                    min(d10 + dist(t2, se0), min(d10, d11) + dist(t2, s1))
                )
            )
        lb_dis += last_3_dis
        if lb_dis > bsf:
            return lb_dis

    c, d, seg_con = TS_segmentation(a, b, N, w)
    l = math.floor(len(a) / N)

    lb_PK = lb_dis
    lb_K_temp = 0
    #startk = timeit.default_timer()
    for j in range(N):
        if seg_con[j] == 0:
            start = 3
            stop = l
        elif seg_con[j] == N - 1:
            start = (seg_con[j]) * l
            stop = len(a) - 3
        else:
            start = (seg_con[j]) * l
            stop = start + l
        # print("start:", start, "stop:", stop)
        if j < seg_num:
            for i in range(start, stop):
                if lb_PK + lb_K_temp > bsf:
                    return lb_PK + lb_K_temp
                ai = a[i]
                if ai > ub[i]:
                    lb_K_temp += (ai - ub[i]) ** 2
                elif ai < lb[i]:
                    lb_K_temp += (ai - lb[i]) ** 2
        else:
            for i in range(start, stop):
                if lb_PK + lb_K_temp > bsf:
                    return lb_PK + lb_K_temp
                ai = a[i]
                if ai > ub[i]:
                    lb_PK += (ai - ub[i]) ** 2
                elif ai < lb[i]:
                    lb_PK += (ai - lb[i]) ** 2
    #print("lb_PK:", lb_PK)
    #print("Keogh time:",(timeit.default_timer()-startk))
    for p in range(seg_num):
        e, f = c[seg_con[p]][1:], d[seg_con[p]][1:]
        m = len(e)
        if seg_con[p] == 0:
            lb_pdtw = - first_3_dis
        elif seg_con[p] == N - 1:
            lb_pdtw = - last_3_dis
        else:
            lb_pdtw = 0
        cost = [float('inf')] * (2 * w + 1)
        cost_prev = [float('inf')] * (2 * w + 1)
        additional_window_dis = float('inf')
        for i in range(0, m):
            k = max(0, w - i)
            for j in range(max(0, i - w), min(m - 1, i + w) + 1):
                # Initialize the first cell
                if i == 0 and j == 0:
                    cost[k] = (e[0] - f[0]) ** 2
                    k += 1
                    continue
                y = float('inf') if j - 1 < 0 or k - 1 < 0 else cost[k - 1]
                x = float('inf') if i < 1 or k > 2 * w - 1 else cost_prev[k + 1]
                z = float('inf') if i < 1 or j < 1 else cost_prev[k]
                cost[k] = min(x, y, z) + (e[i] - f[j]) ** 2
                k += 1
            # Move current array (cost matrix) to previous array
            #print("cost:", cost)
            if i == w+1:
                index_min_row = cost.index(min(cost))
                additional_window_dis = min(cost_prev[index_min_row-1], cost_prev[index_min_row], cost[index_min_row-1])
                #print("additional_window_dis:", additional_window_dis)
            cost_prev = cost
            if i > w+1:
                # print(min(cost))
                # print(additional_window_dis)
                lb_pdtw = min(cost) - additional_window_dis
                if lb_PK + lb_pdtw > bsf:
                    return lb_PK + lb_pdtw
        lb_PK = lb_PK + lb_pdtw
        #print("lb_Pdtw:", lb_pdtw)
    #print("PDTW time:", timeit.default_timer()-start)
    return lb_PK

def LB_New_plus(a, b, w):
    DM = []
    n = len(a) - 1
    # Calculate distance for the first and last elements
    lb_dis = (a[0] - b[0]) ** 2 + (a[-1] - b[-1]) ** 2
    # Calculate distance matrix and sum of min distances for the rest elements
    for i in range(1, n):
        min_distance = float('inf')
        for j in range(max(0, i - w), min(n, i + w + 1)):
            distance = (a[i] - b[j]) ** 2
            DM.append(distance)
            if distance < min_distance:
                min_distance = distance
        lb_dis += min_distance
    return lb_dis

def LB_New_plus_EA(a, b, w, bsf):
    DM = []
    n = len(a) - 1
    # Calculate distance for the first and last elements
    lb_dis = (a[0] - b[0]) ** 2 + (a[-1] - b[-1]) ** 2
    # Calculate distance matrix and sum of min distances for the rest elements
    for i in range(1, n):
        min_distance = float('inf')
        if lb_dis > bsf:
            return lb_dis
        for j in range(max(0, i - w), min(n, i + w + 1)):
            distance = (a[i] - b[j]) ** 2
            DM.append(distance)
            if distance < min_distance:
                min_distance = distance
        lb_dis += min_distance
    return lb_dis

def dist(a, b):
    return (a - b) ** 2

def LB_Petitjean(q, t, ut, lt, w, bsf):
    lb = 0
    istart = 0

    if w >= 1 and len(q) >= 6:
        q0, qe0, q1, q2 = q[0], q[-1], q[1], q[2]
        t0, te0, t1, t2 = t[0], t[-1], t[1], t[2]

        d01 = dist(q0, t1)
        d11 = dist(q1, t1)
        d10 = dist(q1, t0)

        if w == 1:
            lb = dist(q0, t0) + min(
                d11 + dist(q2, t2),
                min(
                    min(d01, d11) + dist(q1, t2),
                    min(d10, d11) + dist(q2, t1)
                )
            )
        else:
            lb = dist(q0, t0) + min(
                d11 + dist(q2, t2),
                min(
                    min(d01 + dist(q0, t2), min(d01, d11) + dist(q1, t2)),
                    min(d10 + dist(q2, t0), min(d10, d11) + dist(q2, t1))
                )
            )

        if lb > bsf:
            return lb

        q1, q2 = q[-2], q[-3]
        t1, t2 = t[-2], t[-3]

        d01 = dist(qe0, t1)
        d11 = dist(q1, t1)
        d10 = dist(q1, te0)

        if w == 1:
            lb += dist(qe0, te0) + min(
                d11 + dist(q2, t2),
                min(
                    min(d01, d11) + dist(q1, t2),
                    min(d10, d11) + dist(q2, t1)
                )
            )
        else:
            lb += dist(qe0, te0) + min(
                d11 + dist(q2, t2),
                min(
                    min(d01 + dist(qe0, t2), min(d01, d11) + dist(q1, t2)),
                    min(d10 + dist(q2, te0), min(d10, d11) + dist(q2, t1))
                )
            )

        if lb > bsf:
            return lb

        istart = 3

    #proj = [0]*len(q)
    proj = np.zeros(len(q))
    for i in range(istart, len(q) - istart):
        if lb > bsf:
            return lb
        qi = q[i]
        if qi > ut[i]:
            lb += dist(qi, ut[i])
            proj[i] = ut[i]
        elif qi < lt[i]:
            lb += dist(qi, lt[i])
            proj[i] = lt[i]
        else:
            proj[i] = qi

    up, lp = fill_envelope(proj, w)

    if lb > bsf:
        return lb

    for i in range(istart):
        proj[i] = q[i]
        proj[-i - 1] = q[-i - 1]
    # uq and lq are calculated in every template, ut and lt are calculated in every sample
    uq, lq= fill_envelope(q, w)
    for i in range(istart, len(t) - istart):
        if lb > bsf:
            return lb
        ti = t[i]
        if ti > up[i]:
            if up[i] > uq[i]:
                lb += dist(ti, uq[i]) - dist(up[i], uq[i])
            else:
                lb += dist(ti, up[i])
        elif ti < lp[i]:
            if lp[i] < lq[i]:
                lb += dist(ti, lq[i]) - dist(lp[i], lq[i])
            else:
                lb += dist(ti, lp[i])

    return lb

def LB_Petitjean_full(q, t, ut, lt, w):
    lb = 0
    istart = 0

    if w >= 1 and len(q) >= 6:
        q0, qe0, q1, q2 = q[0], q[-1], q[1], q[2]
        t0, te0, t1, t2 = t[0], t[-1], t[1], t[2]

        d01 = dist(q0, t1)
        d11 = dist(q1, t1)
        d10 = dist(q1, t0)

        if w == 1:
            lb = dist(q0, t0) + min(
                d11 + dist(q2, t2),
                min(
                    min(d01, d11) + dist(q1, t2),
                    min(d10, d11) + dist(q2, t1)
                )
            )
        else:
            lb = dist(q0, t0) + min(
                d11 + dist(q2, t2),
                min(
                    min(d01 + dist(q0, t2), min(d01, d11) + dist(q1, t2)),
                    min(d10 + dist(q2, t0), min(d10, d11) + dist(q2, t1))
                )
            )

        q1, q2 = q[-2], q[-3]
        t1, t2 = t[-2], t[-3]

        d01 = dist(qe0, t1)
        d11 = dist(q1, t1)
        d10 = dist(q1, te0)

        if w == 1:
            lb += dist(qe0, te0) + min(
                d11 + dist(q2, t2),
                min(
                    min(d01, d11) + dist(q1, t2),
                    min(d10, d11) + dist(q2, t1)
                )
            )
        else:
            lb += dist(qe0, te0) + min(
                d11 + dist(q2, t2),
                min(
                    min(d01 + dist(qe0, t2), min(d01, d11) + dist(q1, t2)),
                    min(d10 + dist(q2, te0), min(d10, d11) + dist(q2, t1))
                )
            )

        istart = 3

    proj = [0] * len(q)

    for i in range(istart, len(q) - istart):
        qi = q[i]
        if qi > ut[i]:
            lb += dist(qi, ut[i])
            proj[i] = ut[i]
        elif qi < lt[i]:
            lb += dist(qi, lt[i])
            proj[i] = lt[i]
        else:
            proj[i] = qi

    up, lp = fill_envelope(proj, w)

    for i in range(istart):
        proj[i] = q[i]
        proj[-i - 1] = q[-i - 1]
    # uq and lq are calculated in every template, ut and lt are calculated in every sample
    uq, lq= fill_envelope(q, w)
    for i in range(istart, len(t) - istart):
        ti = t[i]
        if ti > up[i]:
            if up[i] > uq[i]:
                lb += dist(ti, uq[i]) - dist(up[i], uq[i])
            else:
                lb += dist(ti, up[i])
        elif ti < lp[i]:
            if lp[i] < lq[i]:
                lb += dist(ti, lq[i]) - dist(lp[i], lq[i])
            else:
                lb += dist(ti, lp[i])

    return lb

def LB_Webb(q, t, ut, lt, lut, ult, window, bsf):
    lb = 0
    istart = 0

    if window >= 1 and len(q) >= 6:
        q0 = q[0]
        t0 = t[0]
        qe0 = q[-1]
        te0 = t[-1]
        q1 = q[1]
        t1 = t[1]
        t2 = t[2]
        q2 = q[2]

        d01 = dist(q0, t1)
        d11 = dist(q1, t1)
        d10 = dist(q1, t0)

        if window == 1:
            lb = dist(q0, t0) + min(
                d11 + dist(q2, t2),
                min(
                    min(d01, d11) + dist(q1, t2),
                    min(d10, d11) + dist(q2, t1)
                )
            )
        else:
            lb = dist(q0, t0) + min(
                d11 + dist(q2, t2),
                min(
                    min(d01 + dist(q0, t2), min(d01, d11) + dist(q1, t2)),
                    min(d10 + dist(q2, t0), min(d10, d11) + dist(q2, t1))
                )
            )

        if lb > bsf:
            return lb

        q1 = q[-2]
        t1 = t[-2]
        t2 = t[-3]
        q2 = q[-3]

        d01 = dist(qe0, t1)
        d11 = dist(q1, t1)
        d10 = dist(q1, te0)

        if window == 1:
            lb += dist(qe0, te0) + min(
                d11 + dist(q2, t2),
                min(
                    min(d01, d11) + dist(q1, t2),
                    min(d10, d11) + dist(q2, t1)
                )
            )
        else:
            lb += dist(qe0, te0) + min(
                d11 + dist(q2, t2),
                min(
                    min(d01 + dist(qe0, t2), min(d01, d11) + dist(q1, t2)),
                    min(d10 + dist(q2, te0), min(d10, d11) + dist(q2, t1))
                )
            )

        if lb > bsf:
            return lb

        istart = 3
    freeCountAbove = window
    freeCountBelow = window

    qEnd = len(q) - istart

    uq, lq = fill_envelope(q, window)
    uuq,luq = fill_envelope(uq, window)
    ulq,llq = fill_envelope(lq, window)
    for i in range(istart, qEnd):
        if lb > bsf:
            break
        qi = q[i]
        if qi > ut[i]:
            lb += dist(qi, ut[i])
            if ut[i] >= ulq[i]:
                freeCountBelow += 1
            else:
                freeCountBelow = 0
            freeCountAbove = 0
        elif qi < lt[i]:
            lb += dist(qi, lt[i])
            if lt[i] <= luq[i]:
                freeCountAbove += 1
            else:
                freeCountAbove = 0
            freeCountBelow = 0
        else:
            freeCountAbove += 1
            freeCountBelow += 1

        if i >= window + istart:
            j = i - window

            tj = t[j]
            uqj = uq[j]
            if tj > uqj:
                if freeCountAbove > 2 * window:
                    lb += dist(tj, uqj)
                else:
                    ultj = ult[j]
                    if tj > ultj and ultj >= uqj:
                        lb += dist(tj, uqj) - dist(ultj, uqj)
            else:
                lqj = lq[j]
                if tj < lqj:
                    if freeCountBelow > 2 * window:
                        lb += dist(tj, lqj)
                    else:
                        lutj = lut[j]
                        if tj < lutj and lutj <= lqj:
                            lb += dist(tj, lqj) - dist(lutj, lqj)

    for j in range(qEnd - window, qEnd):
        if lb > bsf:
            break

        tj = t[j]
        uqj = uq[j]
        if tj > uqj:
            if j >= qEnd - freeCountAbove + window:
                lb += dist(tj, uqj)
            else:
                ultj = ult[j]
                if tj > ultj and ultj >= uqj:
                    lb += dist(tj, uqj) - dist(ultj, uqj)
        else:
            lqj = lq[j]
            if tj < lqj:
                if j >= qEnd - freeCountBelow + window:
                    lb += dist(tj, lqj)
                else:
                    lutj = lut[j]
                    if tj < lutj and lutj <= lqj:
                        lb += dist(tj, lqj) - dist(lutj, lqj)

    return lb



def KNN(dist_list, label_list):
    """
    :param dist_list: list of distances
    :param label_list: list of labels
    :return: predicted label, new best-so-far distance
    """
    # 使用 Counter 统计每个标签的出现次数
    counter = Counter(label_list)
    max_count = max(counter.values())
    most_common_labels = [label for label, count in counter.items() if count == max_count]
    # 如果有多个标签出现次数相同，选择距离总和最小的标签
    if len(most_common_labels) > 1:
        label_distances = {label: sum(dist for label_, dist in zip(label_list, dist_list) if label_ == label) for label
                           in most_common_labels}
        predicted_label = min(label_distances, key=label_distances.get)
    else:
        predicted_label = most_common_labels[0]

    return predicted_label #new_bsf


###############################################
# Experiment utilities consolidated for reuse #
###############################################

def analyze_similarity_thresholds(dataset_name: str, w: int):
    """
    Analyze CDTW distances within same class vs different classes using LOOCV on TRAIN set.
    Returns a dict with stats including same-class median used as pruning baseline.
    """
    print(f"开始分析数据集: {dataset_name}")

    train_data = txt_loader(f'C:\\Users\\luoyu\\Desktop\\Univariate_arff\\{dataset_name}\\{dataset_name}_TRAIN.txt')

    class_data = defaultdict(list)
    for i, template in enumerate(train_data):
        label = int(template[0])
        time_series = template[1:]
        class_data[label].append((i, time_series))

    print(f"数据集 {dataset_name} 包含 {len(class_data)} 个类别")

    same_class_distances = []
    different_class_distances = []

    total_samples = len(train_data)
    processed = 0
    for test_idx in range(total_samples):
        test_template = train_data[test_idx]
        test_label = int(test_template[0])
        test_series = np.array(test_template[1:])

        for template_idx in range(total_samples):
            if template_idx == test_idx:
                continue
            template_template = train_data[template_idx]
            template_label = int(template_template[0])
            template_series = np.array(template_template[1:])

            try:
                distance = cdtw(template_series, test_series, w)
                if template_label == test_label:
                    same_class_distances.append(distance)
                else:
                    different_class_distances.append(distance)
            except Exception as e:
                print(f"计算CDTW距离时出错: {e}")
                continue

        processed += 1
        if processed % 10 == 0:
            print(f"已处理 {processed}/{total_samples} 个模板")

    results = {
        'dataset_name': dataset_name,
        'w': w,
        'same_class': {
            'count': len(same_class_distances),
            'mean': np.mean(same_class_distances) if same_class_distances else 0,
            'std': np.std(same_class_distances) if same_class_distances else 0,
            'min': np.min(same_class_distances) if same_class_distances else 0,
            'max': np.max(same_class_distances) if same_class_distances else 0,
            'median': np.median(same_class_distances) if same_class_distances else 0,
        },
        'different_class': {
            'count': len(different_class_distances),
            'mean': np.mean(different_class_distances) if different_class_distances else 0,
            'std': np.std(different_class_distances) if different_class_distances else 0,
            'min': np.min(different_class_distances) if different_class_distances else 0,
            'max': np.max(different_class_distances) if different_class_distances else 0,
            'median': np.median(different_class_distances) if different_class_distances else 0,
        },
    }

    if results['different_class']['mean'] > 0:
        results['distance_ratio'] = results['same_class']['mean'] / results['different_class']['mean']
    else:
        results['distance_ratio'] = float('inf')

    return results


def save_similarity_threshold_results(results: dict, csv_filename: str = "similarity_threshold_results.csv"):
    """Save threshold analysis results to CSV (appends)."""
    csv_dir = "C:\\Users\\luoyu\\Desktop\\KNN_DTW_tuning_result"
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)
    csv_path = os.path.join(csv_dir, csv_filename)

    file_exists = os.path.exists(csv_path)
    with open(csv_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow([
                'Dataset', 'W', 'Same_Class_Count', 'Same_Class_Mean', 'Same_Class_Std',
                'Same_Class_Min', 'Same_Class_Max', 'Same_Class_Median',
                'Different_Class_Count', 'Different_Class_Mean', 'Different_Class_Std',
                'Different_Class_Min', 'Different_Class_Max', 'Different_Class_Median',
                'Distance_Ratio', 'Effectiveness'
            ])
        effectiveness = "Effective" if results['distance_ratio'] < 1 else "Ineffective"
        writer.writerow([
            results['dataset_name'],
            results['w'],
            results['same_class']['count'],
            f"{results['same_class']['mean']:.6f}",
            f"{results['same_class']['std']:.6f}",
            f"{results['same_class']['min']:.6f}",
            f"{results['same_class']['max']:.6f}",
            f"{results['same_class']['median']:.6f}",
            results['different_class']['count'],
            f"{results['different_class']['mean']:.6f}",
            f"{results['different_class']['std']:.6f}",
            f"{results['different_class']['min']:.6f}",
            f"{results['different_class']['max']:.6f}",
            f"{results['different_class']['median']:.6f}",
            f"{results['distance_ratio']:.6f}",
            effectiveness,
        ])
    print(f"结果已保存到CSV文件: {csv_path}")


# Backward-compatible alias used by existing scripts
def save_results_to_csv(results: dict, csv_filename: str = "similarity_threshold_results.csv"):
    return save_similarity_threshold_results(results, csv_filename)


def load_similarity_statistics(dataset_name: str, w: int):
    """
    Load same-class median from CSV; if missing, compute via analyze_similarity_thresholds and save.
    Returns the median or None on failure.
    """
    csv_path = "C:\\Users\\luoyu\\Desktop\\KNN_DTW_tuning_result\\similarity_threshold_results.csv"

    if os.path.exists(csv_path):
        try:
            with open(csv_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    if row['Dataset'] == dataset_name:
                        return float(row['Same_Class_Median'])
        except Exception as e:
            print(f"读取相似度统计文件时出错: {e}")

    print(f"未找到数据集 {dataset_name} 的统计信息，正在自动生成...")
    try:
        data_path = f'C:\\Users\\luoyu\\Desktop\\Univariate_arff\\{dataset_name}\\{dataset_name}_TRAIN.txt'
        if not os.path.exists(data_path):
            print(f"错误: 数据集 {dataset_name} 的训练文件不存在: {data_path}")
            return None

        results = analyze_similarity_thresholds(dataset_name, w)
        save_similarity_threshold_results(results)
        print(f"成功生成数据集 {dataset_name} 的统计信息")
        return results['same_class']['median']
    except Exception as e:
        print(f"生成数据集 {dataset_name} 的统计信息时出错: {e}")
        return None


def KNN_DTW_original(dataset_name: str, w: int, k: int):
    """Plain KNN-DTW without pruning. Returns (accuracy, total_time)."""
    print(f"开始原始KNN-DTW分类: {dataset_name}, w={w}, k={k}")
    train_data = txt_loader(f'C:\\Users\\luoyu\\Desktop\\Univariate_arff\\{dataset_name}\\{dataset_name}_TRAIN.txt')
    test_data = txt_loader(f'C:\\Users\\luoyu\\Desktop\\Univariate_arff\\{dataset_name}\\{dataset_name}_TEST.txt')

    print(f"训练模板数量: {len(train_data)}")
    print(f"测试样本数量: {len(test_data)}")

    correct_predictions = 0
    start_time = time.time()

    for test_idx, test_sample in enumerate(test_data):
        test_label = int(test_sample[0])
        test_series = np.array(test_sample[1:])
        distances = []
        labels = []
        for template_sample in train_data:
            template_label = int(template_sample[0])
            template_series = np.array(template_sample[1:])
            distance = cdtw(template_series, test_series, w)
            distances.append(distance)
            labels.append(template_label)
        if len(distances) >= k:
            k_indices = np.argsort(distances)[:k]
            k_distances = [distances[i] for i in k_indices]
            k_labels = [labels[i] for i in k_indices]
            predicted_class = KNN(k_distances, k_labels)
        else:
            predicted_class = KNN(distances, labels)
        if predicted_class == test_label:
            correct_predictions += 1

    total_time = time.time() - start_time
    accuracy = correct_predictions / len(test_data)
    return accuracy, total_time


def KNN_DTW_with_pruning(dataset_name: str, w: int, k: int, same_class_median: float,
                          pruning_coefficient: float = 1.5, early_stop_count: int = 3):
    """
    KNN-DTW with early-stop pruning based on threshold = coefficient * same_class_median.
    Returns (accuracy, total_time, pruning_rate).
    """
    pruning_threshold = pruning_coefficient * same_class_median
    print(f"开始KNN-DTW分类 (带剪枝): {dataset_name}, w={w}, k={k}")
    print(f"同类别距离中位数: {same_class_median:.6f}")
    print(f"剪枝系数: {pruning_coefficient}")
    print(f"剪枝阈值: {pruning_threshold:.6f}")
    print(f"早期停止阈值: {early_stop_count} 个距离 < {pruning_threshold:.6f}")

    train_data = txt_loader(f'C:\\Users\\luoyu\\Desktop\\Univariate_arff\\{dataset_name}\\{dataset_name}_TRAIN.txt')
    test_data = txt_loader(f'C:\\Users\\luoyu\\Desktop\\Univariate_arff\\{dataset_name}\\{dataset_name}_TEST.txt')

    print(f"训练模板数量: {len(train_data)}")
    print(f"测试样本数量: {len(test_data)}")

    correct_predictions = 0
    pruned_count = 0
    start_time = time.time()

    for test_idx, test_sample in enumerate(test_data):
        test_label = int(test_sample[0])
        test_series = np.array(test_sample[1:])
        distances = []
        labels = []
        class_distance_counts = defaultdict(int)
        early_stop_triggered = False
        predicted_class = None

        for template_sample in train_data:
            template_label = int(template_sample[0])
            template_series = np.array(template_sample[1:])
            distance = cdtw(template_series, test_series, w)
            if distance < pruning_threshold:
                class_distance_counts[template_label] += 1
                if class_distance_counts[template_label] >= early_stop_count:
                    predicted_class = template_label
                    early_stop_triggered = True
                    pruned_count += 1
                    break
            distances.append(distance)
            labels.append(template_label)

        if not early_stop_triggered:
            if len(distances) >= k:
                k_indices = np.argsort(distances)[:k]
                k_distances = [distances[i] for i in k_indices]
                k_labels = [labels[i] for i in k_indices]
                predicted_class = KNN(k_distances, k_labels)
            else:
                predicted_class = KNN(distances, labels)

        if predicted_class == test_label:
            correct_predictions += 1

    total_time = time.time() - start_time
    accuracy = correct_predictions / len(test_data)
    pruning_rate = pruned_count / len(test_data)
    return accuracy, total_time, pruning_rate


def save_pruning_results(results: dict, csv_filename: str = "pruning_results.csv"):
    """Append pruning experiment results to CSV."""
    csv_dir = "C:\\Users\\luoyu\\Desktop\\KNN_DTW_tuning_result"
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)
    csv_path = os.path.join(csv_dir, csv_filename)
    file_exists = os.path.exists(csv_path)

    with open(csv_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow([
                'Dataset', 'W', 'K', 'Method', 'Same_Class_Median', 'Pruning_Coefficient', 'Early_Stop_Count',
                'Accuracy', 'Total_Time', 'Pruning_Rate', 'Speedup_Factor'
            ])
        writer.writerow([
            results['dataset_name'],
            results['w'],
            results['k'],
            results.get('method', 'Unknown'),
            f"{results['same_class_median']:.6f}" if 'same_class_median' in results else "N/A",
            results.get('pruning_coefficient', "N/A"),
            results.get('early_stop_count', "N/A"),
            f"{results['accuracy']:.6f}",
            f"{results['total_time']:.2f}",
            f"{results.get('pruning_rate', 0):.6f}",
            f"{results.get('speedup_factor', 1.0):.2f}",
        ])
    print(f"结果已保存到CSV文件: {csv_path}")
