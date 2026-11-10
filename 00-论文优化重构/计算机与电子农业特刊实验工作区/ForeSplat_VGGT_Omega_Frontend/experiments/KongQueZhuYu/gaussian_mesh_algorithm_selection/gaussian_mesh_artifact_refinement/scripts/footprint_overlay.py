from common import *
from plyfile import PlyData
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from PIL import Image

def main():
    d=ROOT/'task0_gaussian_cleaning/diagnosis';v=PlyData.read(str(SOURCE))['vertex'].data
    xyz=np.column_stack([v[k] for k in ['x','y','z']]);q=np.column_stack([v[f'rot_{k}'] for k in range(4)]).astype(np.float64);q/=np.linalg.norm(q,axis=1)[:,None];w,x,y,z=q.T
    R=np.stack([1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y),2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x),2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)],1).reshape(-1,3,3)
    sigma=np.hypot(np.exp(np.column_stack([v[f'scale_{k}'] for k in range(3)]).astype(np.float64)),v['filter_3D'][:,None]);cov=np.einsum('nij,nj,nkj->nik',R,sigma*sigma,R)
    protocol=json.loads((SEL/'comparison/shared_free_views.json').read_text());name='free_0053_-15';c=next(c for c in protocol['views'] if c['name']==name);E=np.asarray(c['world_view_transform']).T;pc=xyz@E[:3,:3].T+E[:3,3];positive=pc[:,2]>1e-5
    with np.load(d/'free_contribution_views'/f'{name}.npz') as f:g=f['contribution'][:,2]
    ix=np.argsort(-g);ix=ix[(g[ix]>.01)&positive[ix]][:250];p=pc[ix];J=np.zeros((len(ix),2,3));J[:,0,0]=c['Fx']/p[:,2];J[:,1,1]=c['Fy']/p[:,2];J[:,0,2]=-c['Fx']*p[:,0]/p[:,2]**2;J[:,1,2]=-c['Fy']*p[:,1]/p[:,2]**2
    cv=np.einsum('ij,njk,lk->nil',E[:3,:3],cov[ix],E[:3,:3]);uvcov=np.einsum('nij,njk,nlk->nil',J,cv,J);uv=np.column_stack([p[:,0]/p[:,2]*c['Fx']+c['Cx'],p[:,1]/p[:,2]*c['Fy']+c['Cy']]);eigen,basis=np.linalg.eigh(uvcov)
    removed=np.load(ROOT/'task0_gaussian_cleaning/best_cleaned/removed_original_gaussian_ids.npy');gone=np.isin(ix,removed)
    save_npz_fast(d/'projected_ellipse_diagnostic.npz',original_ids=ix,centers_uv=uv,covariance_uv=uvcov,filtered_sigma=sigma[ix],native_black_alpha_contribution=g[ix],deleted=gone)
    fig,ax=plt.subplots(figsize=(7,12));ax.imshow(Image.open(d/'P2_preview/free'/name/'RGB.png'))
    for j in range(len(ix)):
        angle=np.degrees(np.arctan2(basis[j,1,1],basis[j,0,1]));ellipse=Ellipse(uv[j],4*np.sqrt(max(eigen[j,1],0)),4*np.sqrt(max(eigen[j,0],0)),angle=angle,fill=False,lw=.5,alpha=.45,color='red' if gone[j] else 'cyan');ax.add_patch(ellipse)
    ax.set_xlim(0,c['width']);ax.set_ylim(c['height'],0);ax.set_title('Top 250 native black-ROI contributors; red=removed\nFiltered covariance 2-sigma projection diagnostic');fig.tight_layout();fig.savefig(d/'projected_ellipse_overlay.png',dpi=150);plt.close(fig)
    write(d/'PROJECTED_ELLIPSE_SCOPE.json',dict(camera=name,top250_native_contribution_original_ids=ix.tolist(),method='First-order J R covariance R^T J^T from actual filtered GGGS scales/rotations; 2-sigma ellipse. Does not reproduce rasterizer screen clamp or pixel low-pass, and is explicitly a geometry diagnostic. Actual visibility/contribution comes from independently validated native rasterizer Jacobian, not this overlay.'))
if __name__=='__main__':main()
