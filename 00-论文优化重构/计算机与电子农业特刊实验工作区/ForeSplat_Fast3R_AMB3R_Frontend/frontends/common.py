"""Lossless provenance, aspect-preserving inputs and versioned geometry bundles.

Pixel matrices map original pixel centers to model pixel centers (Pillow resize
convention), including half-pixel offsets. No exposure changes or mask erosion.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import csv
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / 'experiments' / 'KongQueZhuYu'

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def json_write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    def portable(x):
        if isinstance(x, Path): return str(x)
        if isinstance(x, np.ndarray): return portable(x.tolist())
        if isinstance(x, np.generic): return portable(x.item())
        if isinstance(x, float) and not math.isfinite(x): return None
        if isinstance(x, dict): return {k:portable(v) for k,v in x.items()}
        if isinstance(x, (list,tuple)): return [portable(v) for v in x]
        return x
    path.write_text(json.dumps(portable(value), indent=2, ensure_ascii=False, allow_nan=False))

def build_manifest(source, output):
    source = Path(source); rows = []; good_index = 0
    for rgb in sorted(source.glob('crop_[0-9][0-9][0-9][0-9].png')):
        mask = source / rgb.name.replace('crop_', 'mask_')
        row = dict(source_index=int(rgb.stem.split('_')[1]), frame_index='',
                   name=rgb.name, rgb_path=str(rgb), mask_path=str(mask),
                   width='', height='', mask_width='', mask_height='',
                   status='ok', error='', rgb_sha256=sha256(rgb),
                   mask_sha256=sha256(mask) if mask.exists() else '')
        try:
            if not mask.exists(): raise FileNotFoundError('corresponding mask missing')
            with Image.open(rgb) as im:
                row['width'], row['height'] = im.size; im.verify()
            with Image.open(mask) as im:
                row['mask_width'], row['mask_height'] = im.size; im.verify()
            if (row['width'], row['height']) != (row['mask_width'], row['mask_height']):
                raise ValueError('RGB/mask size mismatch')
            row['frame_index'] = good_index; good_index += 1
        except Exception as e:
            row['status'] = 'invalid_input'; row['error'] = str(e)
        rows.append(row)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    return rows

def load_inputs(manifest_path, long_edge=512, alignment=16, indices=None):
    import torch
    with open(manifest_path) as f:
        rows = [r for r in csv.DictReader(f) if r['status'] == 'ok']
    if indices is not None: rows = [rows[int(i)] for i in indices]
    rgbs=[]; masks=[]; valid=[]; transforms=[]
    for row in rows:
        with Image.open(row['rgb_path']) as im:
            im=im.convert('RGB'); w,h=im.size
            factor=long_edge/max(w,h); rw=max(1,round(w*factor)); rh=max(1,round(h*factor))
            pw=math.ceil(rw/alignment)*alignment; ph=math.ceil(rh/alignment)*alignment
            left=(pw-rw)//2; top=(ph-rh)//2
            resized=im.resize((rw,rh), Image.Resampling.BICUBIC)
            padded=Image.new('RGB',(pw,ph),(0,0,0)); padded.paste(resized,(left,top))
            rgb=np.asarray(padded)
        with Image.open(row['mask_path']) as im:
            m=(np.asarray(im.convert('L'))>0).astype(np.uint8)*255
            resized=Image.fromarray(m).resize((rw,rh),Image.Resampling.NEAREST)
            padded=Image.new('L',(pw,ph),0); padded.paste(resized,(left,top))
            mask=np.asarray(padded)>0
        v=np.zeros((ph,pw),bool);v[top:top+rh,left:left+rw]=True
        sx=rw/w;sy=rh/h
        a=np.array([[sx,0,left+(sx-1)/2],[0,sy,top+(sy-1)/2],[0,0,1]],dtype=np.float64)
        transforms.append(dict(name=row['name'], original_size_wh=[w,h],
            input_size_wh=[pw,ph], resize_size_wh=[rw,rh], resize_scale_xy=[sx,sy],
            padding_ltrb=[left,top,pw-rw-left,ph-rh-top],crop_ltrb=[0,0,0,0],
            pixel_transform=a.tolist(),pixel_transform_inverse=np.linalg.inv(a).tolist(),
            pixel_coordinate_convention='integer pixel centers; resize half-pixel convention',
            rgb_interpolation='Pillow bicubic',mask_interpolation='nearest',exposure='unchanged',mask_erosion=0))
        rgbs.append(rgb);masks.append(mask);valid.append(v)
    rgb=np.stack(rgbs)
    images=torch.from_numpy(rgb.copy()).permute(0,3,1,2).float()/127.5-1
    return dict(rows=rows,images=images,rgb=rgb,masks=np.stack(masks),
                valid_pixels=np.stack(valid),transforms=transforms)

def as_numpy(x):
    if hasattr(x,'detach'): x=x.detach().float().cpu().numpy()
    return np.asarray(x)

def write_ply(path, points, colors):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    points=np.asarray(points,dtype=np.float32).reshape(-1,3)
    colors=np.asarray(colors,dtype=np.uint8).reshape(-1,3)
    data=np.empty(len(points),dtype=[('x','<f4'),('y','<f4'),('z','<f4'),('red','u1'),('green','u1'),('blue','u1')])
    for i,k in enumerate(('x','y','z')): data[k]=points[:,i]
    for i,k in enumerate(('red','green','blue')): data[k]=colors[:,i]
    header=('ply\nformat binary_little_endian 1.0\nelement vertex %d\n'
            'property float x\nproperty float y\nproperty float z\n'
            'property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')%len(points)
    with open(path,'wb') as f: f.write(header.encode());data.tofile(f)

def export_bundle(out_dir, inputs, predictions, c2w, intrinsics, metadata, pose_status=None):
    """Export no averaged/fused geometry; each point retains a frame/pixel origin."""
    from scipy.ndimage import binary_erosion
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    for d in ('raw_predictions','pointmaps','confidence','cameras','rgb','mask','pointcloud','visualizations'):
        (out/d).mkdir(exist_ok=True)
    if (out/'bundle.json').exists(): raise FileExistsError(f'bundle exists: {out}')
    assert len(predictions)==len(inputs['rows'])
    c2w=as_numpy(c2w);intrinsics=as_numpy(intrinsics)
    camera_original=[];records=[];clouds={k:[] for k in ('raw','confidence','mask','mask_confidence','mask_confidence_q85')};colors={k:[] for k in clouds}
    for i,(row,pred) in enumerate(zip(inputs['rows'],predictions)):
        name=Path(row['name']).stem;p={k:as_numpy(v) for k,v in pred.items() if hasattr(v,'shape')}
        xyz=p['world_points'].reshape(*inputs['masks'][i].shape,3)
        conf=p['world_points_conf'].reshape(inputs['masks'][i].shape)
        m=inputs['masks'][i];v=inputs['valid_pixels'][i];rgb=inputs['rgb'][i]
        finite_geometry=np.isfinite(xyz).all(-1)&v
        finite=finite_geometry&np.isfinite(conf)
        good=finite&(conf>1.0)
        q=float(np.quantile(conf[finite],.85)) if finite.any() else float('nan')
        selections=dict(raw=finite_geometry,confidence=good,mask=finite_geometry&m,mask_confidence=good&m,mask_confidence_q85=good&m&(conf>=q))
        np.savez_compressed(out/'raw_predictions'/f'{name}.npz',**p)
        np.savez_compressed(out/'pointmaps'/f'{name}.npz',**{k:v for k,v in p.items() if 'point' in k or 'pts3d' in k or 'depth' in k})
        np.savez_compressed(out/'confidence'/f'{name}.npz',**{k:v for k,v in p.items() if 'conf' in k})
        Image.fromarray(rgb).save(out/'rgb'/f'{name}.png')
        Image.fromarray(m.astype(np.uint8)*255).save(out/'mask'/f'{name}.png')
        a=np.asarray(inputs['transforms'][i]['pixel_transform']);ko=np.linalg.inv(a)@intrinsics[i];camera_original.append(ko)
        json_write(out/'cameras'/f'{name}.json',dict(c2w=c2w[i],w2c=np.linalg.inv(c2w[i]),intrinsics_model=intrinsics[i],intrinsics_original=ko,pixel_transform=a,
            pose_status=pose_status[i] if pose_status else dict(status='model_prediction')))
        for key,sel in selections.items():
            clouds[key].append(xyz[sel]);colors[key].append(rgb[sel])
            if key in ('mask','mask_confidence'):
                write_ply(out/'pointcloud'/'global_per_frame'/f'{name}_{key}.ply',xyz[sel],rgb[sel])
        if 'local_points' in p:
            local=p['local_points'].reshape(xyz.shape);lc=p.get('local_confidence',conf).reshape(conf.shape)
            lf=np.isfinite(local).all(-1)&v;lg=lf&np.isfinite(lc)&(lc>1)
            for key,sel in dict(raw=lf,confidence=lg,mask=lf&m,mask_confidence=lg&m).items():
                write_ply(out/'pointcloud'/'local_per_frame'/f'{name}_{key}.ply',local[sel],rgb[sel])
        edge=m&~binary_erosion(m) # evaluation only; exported masks stay untouched
        def fraction(sel,den):return float(sel.sum()/max(den.sum(),1))
        fg=conf[m&finite];bg=conf[~m&finite]
        record=dict(frame_index=int(row['frame_index']),source_index=int(row['source_index']),name=row['name'],foreground_pixels=int(m.sum()),valid_foreground_points=int((m&good).sum()),
            foreground_coverage=fraction(m&good,m),edge_coverage=fraction(edge&good,edge),
            valid_image_geometry=fraction(good,v),fg_conf_mean=float(fg.mean()) if fg.size else None,
            bg_conf_mean=float(bg.mean()) if bg.size else None,edge_conf_mean=float(conf[edge&finite].mean()) if (edge&finite).any() else None,
            conf_q85=q,conf_q85_tie_fraction=float((conf[finite]==q).mean()) if finite.any() else None,
            conf_floor_one_fraction=float((conf[finite]==1).mean()) if finite.any() else None,
            mask_conf_q85_points=int(selections['mask_confidence_q85'].sum()))
        records.append(record)
    cloud_counts={}
    for key in clouds:
        pts=np.concatenate(clouds[key]);col=np.concatenate(colors[key]);cloud_counts[key]=len(pts)
        write_ply(out/'pointcloud'/f'global_{key}.ply',pts,col)
        if key=='mask_confidence':
            stride=max(1,math.ceil(len(pts)/250000));write_ply(out/'pointcloud'/'visualization_only.ply',pts[::stride],col[::stride])
    np.savez_compressed(out/'cameras'/'cameras.npz',c2w=c2w,w2c=np.linalg.inv(c2w),intrinsics_model=intrinsics,intrinsics_original=np.asarray(camera_original),names=np.array([r['name'] for r in inputs['rows']]))
    json_write(out/'pixel_transforms.json',inputs['transforms'])
    json_write(out/'pose_status.json',pose_status)
    with open(out/'per_frame_metrics.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    bundle=dict(schema='foresplat.frontend.v1',method=metadata.get('method'),frame_count=len(predictions),
        pose_convention='camera_to_world',world_frame=metadata.get('world_frame','first input camera reference'),
        scale=metadata.get('scale','ambiguous'),input_names=[r['name'] for r in inputs['rows']],
        prediction_files='raw_predictions/{frame_stem}.npz',pointmap_files='pointmaps/{frame_stem}.npz',
        camera_file='cameras/cameras.npz',pixel_transform_file='pixel_transforms.json',
        cloud_counts=cloud_counts,confidence_baseline='finite XYZ and confidence > 1.0; valid unpadded image pixels',
        confidence_quantile='finite XYZ and confidence > 1 AND per-frame confidence >= q85 over valid image pixels; ties may retain more than 15%',
        local_point_convention='individual camera frames; never concatenated as aligned global geometry',
        pointcloud_processing=metadata.get('pointcloud_processing','concatenation of original per-view predictions; no averaging, fusion or scale correction'),metadata=metadata)
    json_write(out/'bundle.json',bundle)
    return bundle

class FrontendBundle:
    """CPU reader for later 3DGS and mesh consumers; never launches training."""
    def __init__(self,path):
        self.path=Path(path);self.metadata=json.loads((self.path/'bundle.json').read_text())
        with np.load(self.path/'cameras'/'cameras.npz') as f:self.cameras=dict(f)
        self.pixel_transforms=json.loads((self.path/'pixel_transforms.json').read_text())
        self.pose_status=json.loads((self.path/'pose_status.json').read_text())
        with open(self.path/'per_frame_metrics.csv') as f:self.frame_records=list(csv.DictReader(f))
    def frames(self):
        for i,name in enumerate(self.metadata['input_names']):
            stem=Path(name).stem
            with np.load(self.path/'raw_predictions'/f'{stem}.npz') as f:pred=dict(f)
            local_file=self.path/'local_unmapped_predictions'/f'{stem}.npz'
            local_provenance=None
            if local_file.exists():
                # Rejected cluster-world maps and poses must never replace the
                # failed global registration. Only independent camera-local
                # head outputs are available as this optional diagnostic.
                with np.load(local_file) as f:
                    for key in ('local_points','local_confidence'):
                        if key in f and key not in pred:pred[key]=f[key]
                local_provenance=dict(file=str(local_file),global_registration=False,
                    convention='independent camera-local head; rejected cluster world geometry is not merged')
            with Image.open(self.path/'rgb'/f'{stem}.png') as image:rgb=np.asarray(image)
            with Image.open(self.path/'mask'/f'{stem}.png') as image:mask=np.asarray(image)>0
            record=self.frame_records[i]
            yield dict(name=name,bundle_index=i,frame_index=int(record['frame_index']),
                source_index=int(record['source_index']),predictions=pred,rgb=rgb,mask=mask,
                c2w=self.cameras['c2w'][i],w2c=self.cameras['w2c'][i],
                intrinsics_model=self.cameras['intrinsics_model'][i],
                intrinsics_original=self.cameras['intrinsics_original'][i],
                pixel_transform=self.pixel_transforms[i],
                local_prediction_provenance=local_provenance,
                pose_status=self.pose_status[i] if self.pose_status else None)

if __name__=='__main__':
    rows=build_manifest('/data/fj/F2DMAS/03-SAM/KongQueZhuYu',EXPERIMENT/'manifest.csv')
    print(f'manifest: {len(rows)} rows, {sum(r["status"]=="ok" for r in rows)} usable')
