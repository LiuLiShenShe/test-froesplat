from common import *
from scipy import ndimage
import csv
def main():
    d=ROOT/'task0_gaussian_cleaning/best_cleaned';assert (d/'scene_208/render_complete.json').exists() and (d/'historical_208/render_complete.json').exists();index=json.loads((OLD/'surface_inputs/e10_gggs_surface/render_index.json').read_text());marker=json.loads((ROOT/'task1_p3_local_repair/pot_marker/marker_region.json').read_text());center=(np.array(marker['bbox_min'])+np.array(marker['bbox_max']))/2;rows=[]
    for item in index:
        c=item['camera'];name=c['name'];sid=item['source_id'];E=np.asarray(c['world_view_transform']).T;pc=center@E[:3,:3].T+E[:3,3];uv=pc[:2]/pc[2]*[c['Fx'],c['Fy']]+[c['Cx'],c['Cy']];x,y=np.rint(uv).astype(int)
        with np.load(SEL/'P2_GGGS_Cleaning/surface_inputs/rendered'/f'{name}.npz') as z:fg=z['foreground_mask'].copy();gt=z['gt_rgb'].copy();p2=z['render_raw'].copy()
        labels,n=ndimage.label(fg);marker_mask=np.zeros_like(fg);component=0
        if 0<=x<c['width'] and 0<=y<c['height']:
            roi=labels[max(0,y-30):min(c['height'],y+31),max(0,x-30):min(c['width'],x+31)];possible=np.unique(roi);possible=possible[possible>0];sizes=np.bincount(labels.ravel());possible=possible[sizes[possible]<.05*fg.sum()]
            if len(possible):component=int(max(possible,key=lambda k:(roi==k).sum()));marker_mask=labels==component
        plantpot=fg&~marker_mask;edge=plantpot&~ndimage.binary_erosion(plantpot,iterations=5)
        images={'P2':p2}
        for label,folder in [('R5_historical','historical_208'),('Best_PlantPot','scene_208')]:
            with np.load(d/folder/'rendered'/f'{name}.npz') as z:images[label]=z['render_raw'].copy()
        for label,rgb in images.items():
            err=(rgb.clip(0,1)-gt)**2;rows.append(dict(experiment=label,source_id=sid,marker_mask_pixels=int(marker_mask.sum()),marker_component_identified=component,plant_pot_fg_pixels=int(plantpot.sum()),plant_pot_fg_psnr=float(-10*np.log10(err[plantpot].mean())),plant_pot_boundary_psnr=float(-10*np.log10(err[edge].mean())),full_historical_fg_psnr=float(-10*np.log10(err[fg].mean()))))
        print('semantic region comparison',sid,int(marker_mask.sum()),flush=True)
    with (ROOT/'comparisons/photometric_semantic_regions.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    write(ROOT/'comparisons/photometric_semantic_regions_summary.json',dict(summaries={label:{k:float(np.mean([r[k] for r in rows if r['experiment']==label])) for k in ['plant_pot_fg_psnr','plant_pot_boundary_psnr','full_historical_fg_psnr','marker_mask_pixels']} for label in images},mask_rule='Existing RAP component around projected actual marker center, search30px, component<5% foreground; main plant/pot component is never excluded. Occluded/merged uncertain markers remain in scoring. No SAM rerun.',interpretation='Supplementary same Plant+Pot regions separate semantic marker exclusion from leaf loss. Historical full208 metrics retain original RAP without changes.'))
if __name__=='__main__':main();sys.stdout.flush();os._exit(0)
