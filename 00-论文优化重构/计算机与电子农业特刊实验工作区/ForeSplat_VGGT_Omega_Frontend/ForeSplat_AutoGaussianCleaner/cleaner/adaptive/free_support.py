"""Automatic projected-depth support, with conservative AABB diagnostics.

Free-camera support is an auxiliary proxy. Actual RGB/mask/depth quality in
every observed camera still determines whether an edit is accepted.
"""
from scipy import ndimage
from ...evaluation.free_view_evaluation import generate,proxy


def free_support(scene,surface):
    if surface is None:return []
    mode=scene.config.get('free_background_support','aabb')
    if mode not in ['aabb','projected_surface']:raise ValueError('Unknown free-background support mode')
    views=[]
    for v in generate(scene,surface):
        mask,box=proxy(v,surface)
        outside_surface=ndimage.distance_transform_edt(~mask)>3
        outside_aabb=~box
        views.append(dict(view=v,mode=mode,surface_mask=mask,outside_surface=outside_surface,
            outside_aabb=outside_aabb,outside=outside_surface if mode=='projected_surface' else outside_aabb,
            normalizer=max(int(mask.sum() if mode=='projected_surface' else box.sum()),1)))
    return views
