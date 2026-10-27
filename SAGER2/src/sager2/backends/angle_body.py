import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice
@triton.jit
def _choose(A, B, C, pre, v, c, f0, WIDTH: tl.constexpr, PHI: tl.constexpr):
    phi = tl.full((), PHI, tl.float64)
    zero = tl.full((), 0, tl.float64)
    eps = tl.full((), 1.e-13, tl.float64)
    group_eps = tl.full((), 1.e-12, tl.float64)
    co, si = libdevice.cos(phi), libdevice.sin(phi)
    # For PHI <= pi/2 this bound proves the sign of each unit is fixed.
    bound = tl.abs(A) * (1 - co) + tl.abs(B) * si
    certified = tl.sum((tl.abs(pre) <= bound).to(tl.int32), 0) == 0
    r1 = tl.full((WIDTH,), PHI, tl.float64)
    r2 = tl.full((WIDTH,), PHI, tl.float64)
    root_count = tl.full((), 0, tl.int32)
    if not certified:
        radius = libdevice.hypot(A, B)
        ratio = tl.where(radius > 0, -C / tl.maximum(radius, 1.e-300), 0)
        alpha = libdevice.atan2(B, A)
        shift = libdevice.acos(tl.minimum(tl.maximum(ratio, -1), 1))
        x1, x2 = alpha + shift, alpha - shift
        x1 = libdevice.atan2(libdevice.sin(x1), libdevice.cos(x1))
        x2 = libdevice.atan2(libdevice.sin(x2), libdevice.cos(x2))
        valid = (radius > 0) & (tl.abs(C) <= radius)
        keep1 = valid & (x1 > -phi + eps) & (x1 < phi - eps)
        keep2 = valid & (x2 > -phi + eps) & (x2 < phi - eps)
        root_count = tl.sum(keep1.to(tl.int32) + keep2.to(tl.int32), 0)
        r1, r2 = tl.where(keep1, x1, phi), tl.where(keep2, x2, phi)
    best_phi = zero
    if root_count == 0:
        weighted = v * (pre > 0)
        P = tl.sum(A * weighted, 0)
        Q = tl.sum(B * weighted, 0)
        best_gain = zero
        left_gain = P * (co - 1) - Q * si
        take = left_gain > best_gain
        best_gain, best_phi = tl.where(take, left_gain, best_gain), tl.where(take, -phi, best_phi)
        right_gain = P * (co - 1) + Q * si
        take = right_gain > best_gain
        best_gain, best_phi = tl.where(take, right_gain, best_gain), tl.where(take, phi, best_phi)
        opt = libdevice.atan2(Q, P)
        opt_gain = P * (libdevice.cos(opt) - 1) + Q * libdevice.sin(opt)
        take = (tl.abs(opt) <= phi) & (opt_gain > best_gain)
        best_phi = tl.where(take, opt, best_phi)
    else:
        first = tl.minimum(tl.min(r1, 0), tl.min(r2, 0))
        mid = (-phi + first) / 2
        weighted = v * (C + A * libdevice.cos(mid) + B * libdevice.sin(mid) > 0)
        D = tl.sum(C * weighted, 0) + c
        P = tl.sum(A * weighted, 0)
        Q = tl.sum(B * weighted, 0)
        best_score = f0
        left = -phi
        loops = 0
        while (left < phi) & (loops <= 2 * WIDTH):
            right = tl.minimum(tl.minimum(tl.min(r1, 0), tl.min(r2, 0)), phi)
            a = D + P * libdevice.cos(left) + Q * libdevice.sin(left)
            take = a > best_score
            best_score, best_phi = tl.where(take, a, best_score), tl.where(take, left, best_phi)
            a = D + P * libdevice.cos(right) + Q * libdevice.sin(right)
            take = a > best_score
            best_score, best_phi = tl.where(take, a, best_score), tl.where(take, right, best_phi)
            opt = libdevice.atan2(Q, P)
            a = D + P * libdevice.cos(opt) + Q * libdevice.sin(opt)
            take = (opt >= left) & (opt <= right) & (a > best_score)
            best_score, best_phi = tl.where(take, a, best_score), tl.where(take, opt, best_phi)
            hit1 = (r1 < phi) & (tl.abs(r1 - right) < group_eps)
            hit2 = (r2 < phi) & (tl.abs(r2 - right) < group_eps)
            d1 = -A * libdevice.sin(r1) + B * libdevice.cos(r1)
            d2 = -A * libdevice.sin(r2) + B * libdevice.cos(r2)
            direction1 = tl.where(tl.abs(d1) <= group_eps, 0, tl.where(d1 > 0, 1, -1))
            direction2 = tl.where(tl.abs(d2) <= group_eps, 0, tl.where(d2 > 0, 1, -1))
            dv = (hit1 * direction1 + hit2 * direction2) * v
            D += tl.sum(dv * C, 0)
            P += tl.sum(dv * A, 0)
            Q += tl.sum(dv * B, 0)
            r1, r2 = tl.where(hit1, phi, r1), tl.where(hit2, phi, r2)
            left = right
            loops += 1
    return best_phi
