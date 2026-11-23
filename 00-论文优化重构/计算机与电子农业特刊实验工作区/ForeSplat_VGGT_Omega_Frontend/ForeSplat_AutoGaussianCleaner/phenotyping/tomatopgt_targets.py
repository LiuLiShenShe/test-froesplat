"""Recover scale across TomatoPGT scans of the same printed target board."""
from itertools import combinations

import cv2
import numpy as np
from scipy.optimize import least_squares


def view_features(ctx, name, sift):
    gray = cv2.cvtColor(np.asarray(ctx.image(name)), cv2.COLOR_RGB2GRAY)
    keys, descriptors = sift.detectAndCompute(gray, None)
    return np.float32([key.pt for key in keys]), descriptors


def transfer_pixels(reference, ctx, calibration):
    points = np.array([r['xyz_colmap_units'] for r in calibration['targets'].values()])
    reference_views = sorted(set.intersection(*(set(r['views']) for r in calibration['targets'].values())))
    sift = cv2.SIFT_create(nfeatures=1800, contrastThreshold=.015)
    matcher = cv2.BFMatcher()
    templates = []
    for i in np.unique(np.linspace(0, len(reference_views) - 1, 36, dtype=int)):
        name = reference_views[i]
        targets, z = reference.xyz_project(points, name)
        features, desc = view_features(reference, name, sift)
        if min(z) > 0 and desc is not None:
            templates.append((name, targets - .5, features, desc))
    data, cameras = ctx.cameras()
    records = {}
    labels = list(calibration['targets'])
    for name in sorted(cameras):
        features, desc = view_features(ctx, name, sift)
        if desc is None or len(desc) < 2:
            continue
        predictions = {label: [] for label in labels}
        for source, targets, source_keys, source_desc in templates:
            pairs = matcher.knnMatch(source_desc, desc, k=2)
            good = [m for m, n in pairs if m.distance < .72 * n.distance]
            if len(good) < 20:
                continue
            left = np.float32([source_keys[m.queryIdx] for m in good])
            right = np.float32([features[m.trainIdx] for m in good])
            H, mask = cv2.findHomography(left, right, cv2.USAC_MAGSAC, 2.)
            if H is None:
                continue
            inliers = mask.ravel() > 0
            if inliers.sum() < 20:
                continue
            hull = cv2.convexHull(left[inliers])
            xy = cv2.perspectiveTransform(targets[None], H)[0] + .5
            for label, original, transferred in zip(labels, targets, xy):
                if (cv2.pointPolygonTest(hull, tuple(original), True) >= -15 and np.all(transferred > 5)
                        and np.all(transferred < [data['width'] - 5, data['height'] - 5])):
                    predictions[label].append((transferred, source, int(inliers.sum())))
        accepted = {}
        for label, choices in predictions.items():
            if not choices:
                continue
            scores = [sum(np.linalg.norm(p[0] - q[0]) < 4 for q in choices) for p in choices]
            i = max(range(len(scores)), key=lambda i: (scores[i], choices[i][2]))
            xy, source, count = choices[i]
            if scores[i] >= 2:
                accepted[label] = dict(pixel=xy, reference_view=source, agreeing_templates=scores[i], inliers=count)
        if accepted:
            records[name] = accepted
    return records


def fit_points(ctx, records, calibration):
    data, cameras = ctx.cameras()
    K = np.asarray(data['K'])
    def triangulate(equations):
        h = np.linalg.svd(equations.reshape(-1, 4), full_matrices=False)[2][-1]
        return h[:3] / h[3]

    fitted = {}
    for label, record in calibration['targets'].items():
        names = sorted(n for n, found in records.items() if label in found)
        if len(names) < 8:
            return None
        poses = np.array([np.column_stack([cameras[n]['R'], cameras[n]['tvec']]) for n in names])
        centers = np.array([cameras[n]['center'] for n in names])
        extent = np.ptp([c['center'] for c in cameras.values()], axis=0).max()
        pairs = [pair for pair in combinations(range(len(names)), 2)
                 if np.linalg.norm(centers[pair[0]] - centers[pair[1]]) > .3 * extent]
        if not pairs:
            return None

        def project(xyz):
            camera_xyz = poses[:, :, :3] @ xyz + poses[:, :, 3]
            return camera_xyz[:, :2] / camera_xyz[:, 2:] * np.diag(K)[:2] + K[:2, 2]

        obs = np.array([records[n][label]['pixel'] for n in names])
        rays = (obs - K[:2, 2]) / np.diag(K)[:2]
        equations = rays[..., None] * poses[:, None, 2] - poses[:, :2]
        best = np.zeros(len(names), dtype=bool)
        for a, b in pairs:
            xyz = triangulate(equations[[a, b]])
            good = np.linalg.norm(project(xyz) - obs, axis=1) < 3.
            if good.sum() > best.sum():
                best = good
        if best.sum() < max(8, len(names) * .65):
            return None
        xyz = least_squares(lambda p: (project(p)[best] - obs[best]).ravel(),
                            triangulate(equations[best]), loss='soft_l1').x
        error = np.linalg.norm(project(xyz) - obs, axis=1)
        fitted[label] = dict(xyz_colmap_units=xyz, reference_xyz_cm=record['reference_xyz_cm'],
                             views=[name for name, ok in zip(names, best) if ok],
                             errors_undistorted_px=error, median_error_undistorted_px=np.median(error[best]),
                             positive_depth=bool(np.all((poses[:, 2, :3] @ xyz + poses[:, 2, 3])[best] > 0)))
    return fitted
