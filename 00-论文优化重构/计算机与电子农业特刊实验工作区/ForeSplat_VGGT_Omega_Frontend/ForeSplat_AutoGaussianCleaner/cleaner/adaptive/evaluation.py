"""Cached common evaluation: unchanged protocol, additional plant-local contour safety."""
from pathlib import Path
import hashlib
import numpy as np
import torch
from scipy import ndimage
from ..foreground_protection import regions
from ..io import write_json
from ...evaluation.metrics import arrays
from ...evaluation.foreground_metrics import photometry, boundary_f1
from ...evaluation.artifact_metrics import artifact_metrics
from .selection import contour_metrics, view_quality_reasons, potential_dark_tradeoff
from .free_support import free_support
from .depth_recovery import metrics as depth_recovery_metrics, potential_tradeoff
from .foreground_support import support_masks, support_metrics


class Evaluator:
    def __init__(self, scene, output, baseline=None, lp=None, surface=None, original_model=None):
        self.scene, self.output, self.baseline, self.lp = scene, Path(output), baseline, lp
        self.reference = {}
        self.regions = {}
        self.initial_records = None
        self.visibility = {}
        self.force_full = False
        self.modified_original_ids = set()
        self.rejection_counts = {}
        self.photometric_cache = {}
        self.geometry_cache = {}
        self.contour_cache = {}
        self.cache_hits = dict(geometry=0,contour=0)
        self.free_views = free_support(scene,surface)
        self.reference_rgb = {}
        self.reference_model_identity = id(original_model) if original_model is not None else None
        self.scene.original_foreground_support = {}
        # Cameras and input images are immutable for a Scene. Reuse exactly
        # the same objects rather than rebuilding CUDA matrices per trial.
        self.camera_cache = {}
        original_get_view = scene.get_view
        def cached_view(index):
            if index not in self.camera_cache:
                self.camera_cache[index] = original_get_view(index)
            return self.camera_cache[index]
        scene.get_view = cached_view

    def evaluate(self, model, folder, indices=None, include_free=True):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        records = []
        indices = list(range(len(self.scene.views))) if indices is None else list(indices)
        for index in indices:
            c, target, mask, *_ = self.scene.get_view(index)
            if index not in self.regions:
                self.regions[index] = regions(mask, self.scene.config['boundary_relative_width'])
                # Default SSIM has a 7x7 window. Exact pixels in this band
                # determine all SSIM samples measured inside the foreground.
                self.regions[index]['photo_context'] = ndimage.maximum_filter(mask,size=7)>0
                self.regions[index]['contour_context'] = ndimage.distance_transform_edt(~mask)<=max(2,3*self.regions[index]['radius'])
            r = self.regions[index]
            with torch.no_grad():
                out = self.scene.render(c, model)
                if index not in self.visibility:
                    self.visibility[index] = np.packbits(out['radii'].detach().cpu().numpy() > 0)
                a = arrays(out)
                if not all(np.isfinite(value).all() for value in a.values()):
                    raise ValueError(f'Nonfinite native GGGS render: {c.image_name}')
            if self.baseline is None:
                old = a
                self.reference[index] = {k: a[k].copy() for k in ['depth', 'normal', 'alpha']}
                np.savez(folder/f'{index:04d}.npz', **self.reference[index])
            else:
                if index not in self.reference:
                    with np.load(Path(self.baseline)/f'{index:04d}.npz') as z:
                        self.reference[index] = {k: z[k].astype(np.float32) for k in ['depth', 'normal', 'alpha']}
                old = self.reference[index]
            photo, lpips_value = self.photometry_cached(index,a['rgb'],target,mask,r,a['alpha'])
            record = dict(index=index, name=c.image_name,
                **photo,
                **artifact_metrics(a['alpha'], a['rgb'], r),
                **self.contour_cached(index,a['alpha'],old['alpha'],mask,r),
                **self.geometry_cached(index,a,old,mask))
            record['fg_masked_lpips'] = lpips_value
            if self.scene.config.get('foreground_support_mode') == 'photo_relative_or_depth':
                if index not in self.scene.original_foreground_support:
                    if id(model) != self.reference_model_identity:
                        raise ValueError('Fixed foreground support must first be measured on the explicit untouched original model')
                    # The first evaluation always renders the untouched input;
                    # wrapper input/cache identity checks enforce that invariant.
                    _,_,_,depth,confidence = self.scene.get_view(index)
                    self.scene.original_foreground_support[index] = support_masks(a,target,mask,depth,confidence,self.scene.config)
                    self.reference_rgb[index] = a['rgb'].copy()
                    supported,geometry = self.scene.original_foreground_support[index]
                    np.savez_compressed(self.output/'comparison/initial'/f'{index:04d}_foreground_support.npz',
                        supported=supported,geometry=geometry)
                supported,geometry = self.scene.original_foreground_support[index]
                record.update(support_metrics(a,old,self.reference_rgb[index],target,mask,r,supported,geometry,self.scene.config))
            records.append(record)
            if len(records) % 32 == 0 or len(records) == len(indices):
                print(f'evaluation {folder.name} {len(records)}/{len(indices)}', flush=True)
        result = summarize(records, self.scene.config['evaluation_max_side'])
        if self.scene.config.get('foreground_support_mode') == 'photo_relative_or_depth':
            result['foreground_quality_protocol'] = 'Frozen original photo-relative-error or high-confidence depth support, conservative insufficient-support fallback, full-mask RGB/SSIM/LPIPS, and an additional per-pixel RMSE guard on excluded regions. Raw full-mask structure and geometry remain reported. Support is a proxy, not semantic geometry ground truth.'
        if include_free:
            result['free_background'] = self.free_background(model)
        write_json(folder/'metrics.json', result)
        if self.baseline is None:
            self.baseline = folder
        if self.initial_records is None:
            self.initial_records = {v['index']: v for v in records}
        return result

    def contour_cached(self,index,alpha,old,mask,r):
        digest=hashlib.blake2b(np.ascontiguousarray(alpha[r['contour_context']]).view(np.uint8),digest_size=32).digest()
        previous=self.contour_cache.get(index)
        if previous is not None and previous['digest']==digest:
            self.cache_hits['contour']+=1
            return previous['metrics'].copy()
        metrics=contour_metrics(alpha,old,mask,r['radius'])
        self.contour_cache[index]=dict(digest=digest,metrics=metrics.copy())
        return metrics

    def geometry_cached(self,index,a,old,mask):
        digest=hashlib.blake2b(digest_size=32)
        for key in ['alpha','depth','normal']:
            digest.update(np.ascontiguousarray(a[key][mask]).view(np.uint8))
        token=digest.digest();previous=self.geometry_cache.get(index)
        if previous is not None and previous['digest']==token:
            self.cache_hits['geometry']+=1
            return previous['metrics'].copy()
        bv=mask&(old['alpha']>.5)&(old['depth']>0)
        av=mask&(a['alpha']>.5)&(a['depth']>0);valid=bv&av
        unit=max(float(np.median(old['depth'][bv])),1e-9) if bv.any() else 1.
        delta=np.abs(a['depth'][valid]-old['depth'][valid])/unit
        norm_old,norm_new=np.linalg.norm(old['normal'],axis=2),np.linalg.norm(a['normal'],axis=2)
        nv=valid&(norm_old>.1)&(norm_new>.1)
        cos=(old['normal'][nv]*a['normal'][nv]).sum(1)/np.maximum(norm_old[nv]*norm_new[nv],1e-9)
        angles=np.degrees(np.arccos(np.clip(cos,-1,1)))
        metrics=dict(depth_relative_p95=float(np.quantile(delta,.95)) if len(delta) else 0.,
            depth_changed_fraction=float((delta>self.scene.config['quality']['depth_relative_p95']).mean()) if len(delta) else 0.,
            new_depth_invalid_fraction=float((bv&~av).sum()/max(bv.sum(),1)),
            normal_angle_p95_deg=float(np.quantile(angles,.95)) if len(angles) else 0.)
        if self.scene.config.get('certify_depth_recovery',False):
            metrics.update(depth_recovery_metrics(self.scene,index,a,old,mask,self.regions[index]))
        self.geometry_cache[index]=dict(digest=token,metrics=metrics.copy())
        return metrics

    def photometry_cached(self,index,rgb,target,mask,r,alpha):
        digest=hashlib.blake2b(np.ascontiguousarray(rgb[r['photo_context']]).view(np.uint8),digest_size=32).digest()
        old=self.photometric_cache.get(index)
        if old is not None and old['digest']==digest:
            # Reuse only RGB-derived metrics. Alpha and contours are measured
            # again even when rendered foreground colors are byte-identical.
            keys=['fg_mse','fg_psnr','fg_ssim','dark_foreground_mse','dark_foreground_pixels']
            photo={k:old['photo'][k] for k in keys}
            photo.update(fg_alpha=float(alpha[mask].mean()) if mask.any() else 1.,
                fg_coverage=float((alpha[mask]>.5).mean()) if mask.any() else 1.,
                boundary_coverage=float((alpha[r['boundary']]>.5).mean()) if r['boundary'].any() else 1.,
                skeleton_coverage=float((alpha[r['skeleton']]>.5).mean()) if r['skeleton'].any() else 1.,
                thin_coverage=float((alpha[r['thin']]>.5).mean()) if r['thin'].any() else 1.,
                boundary_f1=boundary_f1(alpha>.5,mask,r['radius']))
            dark=mask & (target @ np.array([.2126,.7152,.0722])<.12)
            photo['dark_foreground_alpha']=float(alpha[dark].mean()) if dark.any() else None
            return photo,old['lpips']
        photo=photometry(rgb,target,mask,r,alpha)
        value=None
        if self.lp is not None:
            x=torch.as_tensor((rgb*mask[...,None]).transpose(2,0,1)[None].copy(),device='cuda')*2-1
            y=torch.as_tensor((target*mask[...,None]).transpose(2,0,1)[None].copy(),device='cuda')*2-1
            with torch.no_grad():value=float(self.lp(x,y).mean())
        self.photometric_cache[index]=dict(digest=digest,photo=photo.copy(),lpips=value)
        return photo,value

    def free_background(self,model):
        from ..runtime import camera
        records=[]
        for i,support in enumerate(self.free_views):
            v,outside=support['view'],support['outside']
            c=camera(v['K'],v['E'],v['width'],v['height'],v['name'],i)
            with torch.no_grad():
                alpha=self.scene.render(c,model)['mask'].detach().cpu().numpy().squeeze()
            records.append(dict(name=v['name'],alpha_mass=float(alpha[outside].sum()),
                high_alpha_pixels=int((outside&(alpha>.5)).sum()),
                aabb_alpha_mass=float(alpha[support['outside_aabb']].sum()),
                surface_proxy_alpha_mass=float(alpha[support['outside_surface']].sum()),
                surface_proxy_foreground_alpha=float(alpha[support['surface_mask']].mean()) if support['surface_mask'].any() else None))
        return dict(views=records,alpha_mass=sum(v['alpha_mass'] for v in records),
            mode=self.scene.config.get('free_background_support','aabb'),
            aabb_alpha_mass=sum(v['aabb_alpha_mass'] for v in records),
            surface_proxy_alpha_mass=sum(v['surface_proxy_alpha_mass'] for v in records),
            reference='Automatically projected independent masked VGGT depth; projected surface and AABB are auxiliary proxies, not silhouette ground truth')

    def affected_views(self, original_ids):
        if self.force_full and (not self.modified_original_ids or
                any(int(i) in self.modified_original_ids for i in original_ids)):
            return list(range(len(self.scene.views)))
        ids = np.asarray(original_ids, np.int64)
        return [i for i, packed in sorted(self.visibility.items())
                if ((packed[ids//8] >> (7-ids%8)) & 1).any()]

    def evaluate_group(self, model, folder, original_group, current):
        affected = self.affected_views(original_group)
        if not affected:
            result={**current,'free_background':self.free_background(model)}
            write_json(Path(folder)/'metrics.json',result)
            return result, []
        updates = None
        if self.scene.config.get('rejection_preflight', False):
            maximum = self.scene.config.get('preflight_max_views', 8)
            ranked = sorted([i for i in affected if self.rejection_counts.get(i, 0)],
                key=lambda i: (-self.rejection_counts[i], i))[:maximum]
            for position in np.linspace(0, len(affected)-1, min(maximum, len(affected)), dtype=int):
                if len(ranked) >= maximum:
                    break
                index = affected[position]
                if index not in ranked:
                    ranked.append(index)
            probe = self.evaluate(model, Path(folder)/'preflight', ranked, include_free=False)
            reasons = self.view_failures(probe['views'], current)
            initial_failures = [v['index'] for v in probe['views'] if
                view_quality_reasons(v, self.initial_records[v['index']], self.scene.config)]
            defer_dark=(potential_dark_tradeoff(reasons,probe['views'],list(self.initial_records.values()),current['views'],self.scene.config) or
                potential_tradeoff(reasons,probe['views'],list(self.initial_records.values()),current['views'],self.scene.config))
            if reasons and initial_failures and not defer_dark:
                # The old splitter uses the first initial-reference failing
                # camera. Check every earlier affected camera so preflight
                # preserves that choice, not merely the final rejection.
                first = min(initial_failures)
                prefix = [i for i in affected if i < first and i not in ranked]
                if prefix:
                    extra = self.evaluate(model, Path(folder)/'canonical_prefix', prefix, include_free=False)
                    probe['views'] += extra['views']
                    ranked += prefix
                reasons = self.record_view_failures(probe['views'], current)
                records = {v['index']:v for v in current['views']}
                records.update({v['index']:v for v in probe['views']})
                result = summarize([records[i] for i in sorted(records)], self.scene.config['evaluation_max_side'])
                result.update(metrics_complete=False, preflight_rejection_reasons=reasons,
                    rerendered_views=ranked, affected_views=affected,
                    unmeasured_affected_views=sorted(set(affected)-set(ranked)),
                    canonical_first_rejection_preserved=True,
                    background_gain_scope='Partial measured training-view gain plus exact free-view evaluation; unmeasured training views retain current values; no complete global quality or training gain claimed')
                result['free_background'] = self.free_background(model)
                write_json(Path(folder)/'metrics.json', result)
                return result, affected
            remaining = [i for i in affected if i not in ranked]
            updates = self.evaluate(model, Path(folder)/'remaining', remaining, include_free=False) if remaining else probe
            if remaining:
                updates['views'] += probe['views']
            updates['free_background'] = self.free_background(model)
        if updates is None:
            updates = self.evaluate(model, folder, affected)
        self.record_view_failures(updates['views'], current)
        records = {v['index']:v for v in current['views']}
        records.update({v['index']:v for v in updates['views']})
        result = summarize([records[i] for i in sorted(records)], self.scene.config['evaluation_max_side'])
        result['rerendered_views'] = affected
        result['metrics_complete'] = True
        result['free_background'] = updates['free_background']
        result['unchanged_view_evidence'] = 'Only unchanged Gaussian members reuse original radii=0 footprint evidence, which is independent of other Gaussians/occlusion; groups containing modified members rerender all views'
        write_json(Path(folder)/'metrics.json',result)
        return result, affected

    def record_view_failures(self, records, current):
        reasons = self.view_failures(records, current)
        failed = {r.split(':',1)[0] for r in reasons}
        for record in records:
            if record['name'] in failed:
                index = record['index']
                self.rejection_counts[index] = self.rejection_counts.get(index, 0) + 1
        return reasons

    def view_failures(self, records, current):
        previous = {v['index']:v for v in current['views']}
        reasons = []
        ordered = sorted(records, key=lambda v:v['index'])
        for reference in [self.initial_records, previous]:
            for record in ordered:
                reasons += [r for r in view_quality_reasons(record, reference[record['index']], self.scene.config) if r not in reasons]
        return reasons

    def split_for_rejection(self, original_group, reasons):
        by_name = {v['name']:i for i,v in enumerate(self.scene.views)}
        ids = np.asarray(original_group,np.int64)
        for reason in reasons:
            name = reason.split(':',1)[0]
            if name not in by_name:continue
            packed = self.visibility[by_name[name]]
            visible = ((packed[ids//8] >> (7-ids%8)) & 1).astype(bool)
            if visible.any() and not visible.all():
                return [ids[~visible],ids[visible]]
        return []


def summarize(records, resolution):
    keys = [k for k,v in records[0].items() if isinstance(v,(int,float)) and k != 'index']
    mean = {k:float(np.mean([r[k] for r in records if r[k] is not None])) for k in keys}
    for k in ['fg_masked_lpips','dark_foreground_alpha','dark_foreground_mse']:
        values = [r[k] for r in records if r[k] is not None]
        mean[k] = float(np.mean(values)) if values else None
    for k in ['external_depth_mae','external_depth_p95']:
        values=[r[k] for r in records if r.get(k) is not None]
        if any(k in r for r in records):mean[k]=float(np.mean(values)) if values else None
    return dict(mean=mean, views=records, resolution_max_side=resolution,
        foreground_local_boundary_definition='Target silhouette band: distance<=3*boundary radius; distant cloud contours excluded from FG-local F1 only',
        global_boundary_f1_status='Reported unchanged for C0–C5 comparison',
        lpips_status='measured_foreground_masked_proxy' if mean['fg_masked_lpips'] is not None else 'unavailable')
