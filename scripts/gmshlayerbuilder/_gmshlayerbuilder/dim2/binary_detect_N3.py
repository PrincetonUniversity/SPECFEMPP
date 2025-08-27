import numpy as np

from _gmshlayerbuilder import lagrange

L = lagrange.build_lagrange_polys([-1, 0, 1])
Lp = lagrange.differentiate_polys(L)

maxfind_coefs_a = -Lp[:, 1]
maxfind_coefs_b = Lp[:, 0]


def _union_intersections(intersectlo, intersecthi, delta):
    """Helper: takes the union of two intersections.

    intersections are given by a list of points [a1,a2,...,an]
    with even n. The intervals are (a1,a2) union (a3,a4) union ... union (a[n-1],an)

    delta specifies tolerance (intervals smaller than this are truncated)
    """
    if len(intersecthi) == 0:
        return intersectlo
    if len(intersectlo) == 0:
        return intersecthi

    # union (of intersections; confusing, I know)
    join = []
    loind = 0
    hiind = 0
    Nlo = len(intersectlo)
    Nhi = len(intersecthi)
    while loind < Nlo or hiind < Nhi:
        # take next transition point
        if hiind == Nhi:
            t = intersectlo[loind]
            loind += 1
        elif loind == Nlo:
            t = intersecthi[hiind]
            hiind += 1
        elif intersectlo[loind] < intersecthi[hiind]:
            t = intersectlo[loind]
            loind += 1
        else:
            t = intersecthi[hiind]
            hiind += 1

        # ind % 2 == 0 --> t[ind] tranitions to intersecting
        # (we are currently in non-intersecting)

        if (loind % 2 == 0) and (hiind % 2 == 0):
            # we just left an intersecting region
            if len(join) % 2 != 0:
                join.append(t)
        else:
            # we are in an intersecting region (either stay or entered)
            if len(join) % 2 == 0:
                # (entered, so set this transition point)
                if len(join) > 0 and t - join[-1] < delta:
                    del join[-1]
                else:
                    join.append(t)
    return join


def _concat_intersections(intersectlo, intersecthi, delta):
    """Helper: takes the union of two intersections, but all points of
    intersectlo are guaranteed to be less than all points of intersecthi.
    """
    # empty on one side, trivial append
    if len(intersecthi) == 0:
        return intersectlo
    if len(intersectlo) == 0:
        return intersecthi

    if intersecthi[0] - intersectlo[-1] < delta:
        # combine intervals on the crossover
        return intersectlo[:-1] + intersecthi[1:]
    return intersectlo + intersecthi


def find_bbox(edge, t0, t1, compute_extrema: bool = True):
    # quadratics have 1 possible interior extreme value. should we ignore it?
    if compute_extrema:
        bd_params = np.clip(
            np.einsum("k,kd->d", maxfind_coefs_b, edge)
            / np.einsum("k,kd->d", maxfind_coefs_a, edge),
            t0,
            t1,
        )
        candidates = np.array([[t0, t1, bd_params[0]], [t0, t1, bd_params[1]]])
    else:
        candidates = np.array([[t0, t1], [t0, t1]])
    extrema = np.einsum("ji,dki,jd->dk", L, candidates[..., None] ** np.arange(3), edge)
    xmin, ymin = np.min(extrema, axis=1)
    xmax, ymax = np.max(extrema, axis=1)

    return xmin, ymin, xmax, ymax


def _find_intersection(
    edge1: np.ndarray,
    edge2: np.ndarray,
    param1lo: float,
    param1hi: float,
    param2lo: float,
    param2hi: float,
    eps1: float,
    eps2: float,
    delta: float,
    return_threshold: float,
) -> list[float] | bool:
    """Computes the points [s1,s2,...,sn], where
    param1lo <= s1 < ... < sn <= param1hi,
    marking the transition points between intersecting and not-intersecting intervals.
    [-1,s1] and [sn,1] are considered non-intersecting (n is always even). The
    sk are the parameters for edge1.

    If the sum of intervals is greater than return_threshold, returns True to exit.

    Two fuzzy parameters eps1 < eps2 are utilized.
    If edge1 is guaranteed to come within sqrt(eps2)
    of edge2, then the intersection is marked.
    If edge1 is guaranteed to stay outside sqrt(eps1)
    of edge2, then the intersection is not marked.
    If the distance squared on an interval
    is between eps1 and eps2, then there is a chance that an intersection is
    marked, dependent on which guarantee comes first. Since this algorithm
    uses subdivision, the minimum and maximum possible distance on an interval
    will converge to each other, so by ensuring eps1 < eps2,
    the subdivision process will eventually terminate.

    The argument delta specifies a parameter-space distance for which an interval
    is ignored. If sk - sk-1 < delta, then both points
    are removed.


    Args:
        edge1 (np.ndarray): _description_
        edge2 (np.ndarray): _description_
        param1lo (float): _description_
        param1hi (float): _description_
        param2lo (float): _description_
        param2hi (float): _description_
        eps1 (float): _description_
        eps2 (float): _description_
        delta (float): _description_
    """
    if param1hi - param1lo < delta:
        # if we can find a point on edge1 close enough to a point on edge2,
        # we can mark the entire segment
        midpoint1 = np.einsum(
            "ji,i,jd->d", L, ((param1lo + param1hi) / 2) ** np.arange(3), edge1
        )
        midpoint2 = np.einsum(
            "ji,i,jd->d", L, ((param2lo + param2hi) / 2) ** np.arange(3), edge2
        )
        if np.sum((midpoint1 - midpoint2) ** 2) < eps2:
            return [param1lo, param1hi]

    xmin1, ymin1, xmax1, ymax1 = find_bbox(
        edge1, param1lo, param1hi, compute_extrema=False
    )
    xmin2, ymin2, xmax2, ymax2 = find_bbox(
        edge2, param2lo, param2hi, compute_extrema=False
    )

    # min separations
    xsep12 = xmin1 - xmax2
    xsep21 = xmin2 - xmax1
    ysep12 = ymin1 - ymax2
    ysep21 = ymin2 - ymax1
    if xsep12 <= 0 and xsep21 <= 0:  # noqa: SIM108
        # boxes are intesecting in x: xsep=0
        xsepmin = 0
    else:
        # boxes not intersecting. One of xsep12 and xsep21 must be negative
        # (corresponding to the outer points; separation is not that one)
        xsepmin = xsep21 if xsep12 <= 0 else xsep12

    ysepmin = (
        0 if (ysep12 <= 0 and ysep21 <= 0) else (ysep21 if ysep12 <= 0 else ysep12)
    )

    if xsepmin**2 + ysepmin**2 > eps1:
        # minimum separation of bounding boxes past threshold.
        # no intersections possible
        return []

    # max separations
    xsepmax = -min(xsep12, xsep21)
    ysepmax = -min(ysep12, ysep21)
    if xsepmax**2 + ysepmax**2 < eps2:
        return [param1lo, param1hi]

    # not resolved enough; solve on subdivisions. Subdivide larger element
    if max(xmax1 - xmin1, ymax1 - ymin1) > max(xmax2 - xmin2, ymax2 - ymin2):
        # element 1 is larger
        param1mid = (param1hi + param1lo) / 2
        intersectlo = _find_intersection(
            edge1,
            edge2,
            param1lo,
            param1mid,
            param2lo,
            param2hi,
            eps1,
            eps2,
            delta=delta,
            return_threshold=return_threshold,
        )
        if isinstance(intersectlo, bool):
            return intersectlo
        intersecthi = _find_intersection(
            edge1,
            edge2,
            param1mid,
            param1hi,
            param2lo,
            param2hi,
            eps1,
            eps2,
            delta=delta,
            return_threshold=return_threshold,
        )
        if isinstance(intersecthi, bool):
            return intersecthi

        intersections = _concat_intersections(intersectlo, intersecthi, delta)
        length = sum(
            b - a for a, b in zip(intersections[::2], intersections[1::2], strict=True)
        )
        return True if length > return_threshold else intersections
    # element 2 is larger
    param2mid = (param2hi + param2lo) / 2
    intersectlo = _find_intersection(
        edge1,
        edge2,
        param1lo,
        param1hi,
        param2lo,
        param2mid,
        eps1,
        eps2,
        delta=delta,
        return_threshold=return_threshold,
    )
    if isinstance(intersectlo, bool):
        return intersectlo
    intersecthi = _find_intersection(
        edge1,
        edge2,
        param1lo,
        param1hi,
        param2mid,
        param2hi,
        eps1,
        eps2,
        delta=delta,
        return_threshold=return_threshold,
    )
    if isinstance(intersecthi, bool):
        return intersecthi
    intersections = _union_intersections(intersectlo, intersecthi, delta)
    length = sum(
        b - a for a, b in zip(intersections[::2], intersections[1::2], strict=True)
    )
    return True if length > return_threshold else intersections


def quadratic_beziers_intersect(
    edge1: np.ndarray,
    edge2: np.ndarray,
    eps1: float | None = None,
    eps2: float | None = None,
    delta: float = 1e-4,
    return_threshold: float = 4e-2,
) -> bool:
    """Computes whether or not edge1 and edge2 have an intersection of length
    at least `return_threshold` in parameter space.

    Two fuzzy parameters eps1 < eps2 are utilized.
    If edge1 is guaranteed to come within sqrt(eps2)
    of edge2, then the intersection is marked.
    If edge1 is guaranteed to stay outside sqrt(eps1)
    of edge2, then the intersection is not marked.
    If the distance squared on an interval
    is between eps1 and eps2, then there is a chance that an intersection is
    marked, dependent on which guarantee comes first. Since this algorithm
    uses subdivision, the minimum and maximum possible distance on an interval
    will converge to each other, so by ensuring eps1 < eps2,
    the subdivision process will eventually terminate.


    Args:
        edge1 (np.ndarray): 3x2 array of positions: [start, middle, end]
        edge2 (np.ndarray): 3x2 array of positions: [start, middle, end]
        eps1 (float | None): eps1 value, defaults to a quarter of eps2
            if eps2 is defined, or (1% of max|start - end|)^2 between the
            curves if not.
        eps2 (float | None): eps2 value, defaults to quadruple eps1.
        delta (float): parameter space argument for whether an interval
            should be ignored.
        return_threshold (float): the parameter space threshold parameter,
            defaults to 0.04
    """
    if eps1 is None:
        eps1 = (
            (
                1e-4
                * max(
                    sum((edge1[0, :] - edge1[2, :]) ** 2),
                    sum((edge2[0, :] - edge2[2, :]) ** 2),
                )
            )
            if eps2 is None
            else eps2 / 4
        )
    if eps2 is None:
        eps2 = eps1 * 4

    # subdivide at the extrema so that bounding box calculation can ignore it
    with np.errstate(divide="ignore", invalid="ignore"):
        extrema_points_1 = [
            -1,
            1,
            *np.nan_to_num(
                np.clip(
                    np.einsum("k,kd->d", maxfind_coefs_b, edge1)
                    / np.einsum("k,kd->d", maxfind_coefs_a, edge1),
                    -1,
                    1,
                ),
                nan=-1,
            ),
        ]

        extrema_points_2 = [
            -1,
            1,
            *np.nan_to_num(
                np.clip(
                    np.einsum("k,kd->d", maxfind_coefs_b, edge2)
                    / np.einsum("k,kd->d", maxfind_coefs_a, edge2),
                    -1,
                    1,
                ),
                nan=-1,
            ),
        ]

    extrema_points_1.sort()
    extrema_points_2.sort()

    intersects = []
    for isubdiv1 in range(3):
        param1lo = extrema_points_1[isubdiv1]
        param1hi = extrema_points_1[isubdiv1 + 1]
        tmp_intersects = []
        for isubdiv2 in range(3):
            segment_intersects = _find_intersection(
                edge1,
                edge2,
                param1lo,
                param1hi,
                extrema_points_2[isubdiv2],
                extrema_points_2[isubdiv2 + 1],
                eps1,
                eps2,
                delta,
                return_threshold,
            )
            if isinstance(segment_intersects, bool):
                return True
            tmp_intersects = _union_intersections(
                tmp_intersects,
                segment_intersects,
                return_threshold,
            )
            length = sum(
                b - a
                for a, b in zip(tmp_intersects[::2], tmp_intersects[1::2], strict=True)
            )
            if length > return_threshold:
                return True
        intersects = _concat_intersections(intersects, tmp_intersects, return_threshold)
        length = sum(
            b - a for a, b in zip(intersects[::2], intersects[1::2], strict=True)
        )
        if length > return_threshold:
            return True

    if len(intersects) == 0:
        return False
    if intersects[0] + 1 < delta:
        intersects[0] = -1
    if 1 - intersects[-1] < delta:
        intersects[-1] = 1

    filtered_intersects = [intersects[0]]
    for t in intersects[1:]:
        if t - filtered_intersects[-1] < delta:
            del filtered_intersects[-1]
        else:
            filtered_intersects.append(t)
    length = sum(b - a for a, b in zip(intersects[::2], intersects[1::2], strict=True))
    return length > return_threshold
