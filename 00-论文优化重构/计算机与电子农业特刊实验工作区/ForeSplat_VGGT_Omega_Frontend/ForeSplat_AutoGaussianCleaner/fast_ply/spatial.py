"""Sparse exact voxel keys; bounded candidate queries without a global KNN tree."""
from itertools import product
import numpy as np

OFFSETS=list(product((-1,0,1),repeat=3))
BITS=21
MASK=(1<<BITS)-1


def sample(a,maximum=200000):
    # Deterministic row sample for robust thresholds. No N-sized quantile copy.
    step=max(1,int(np.ceil(len(a)/maximum)))
    return a[::step]


def quantile(a,q):
    return np.quantile(sample(a),q)


def build(features,cell_scale,block_rows):
    xyz=features['xyz']; n=len(xyz)
    extent=np.linalg.norm(np.diff(quantile(xyz,[.01,.99]),axis=0))
    origin=xyz.min(0).astype('f8')
    # Packed keys are exact with one empty boundary cell at each end. An extreme
    # extent can enlarge cells, reported to the caller; no coordinate wraparound.
    span=np.ptp(xyz.astype('f8'),axis=0).max()
    h=max(float(quantile(features['effective_median'],.5))*cell_scale,
          float(extent)/4096, float(span)/(MASK-3),np.finfo('f4').tiny)
    keys=np.empty(n,'u8')
    for start in range(0,n,block_rows):
        cell=np.floor((xyz[start:start+block_rows].astype('f8')-origin)/h).astype('u8')+1
        keys[start:start+len(cell)]=(cell[:,0]<<(2*BITS)) | (cell[:,1]<<BITS) | cell[:,2]
    unique,inverse,counts=np.unique(keys,return_inverse=True,return_counts=True)
    sums_b=np.bincount(inverse,weights=features['brightness'])
    # Geometric scale mean limits domination by huge splats in a cell.
    log_s=np.log(np.maximum(features['effective_max'],np.finfo('f4').tiny))
    sums_s=np.bincount(inverse,weights=log_s)
    del inverse
    return dict(keys=keys,unique=unique,counts=counts,sums_b=sums_b,sums_s=sums_s,
                cell_size=h,extent=float(extent))


def query(index,features,ids,block_rows=32768):
    n=len(ids)
    support=np.zeros(n,'i4'); cells=np.zeros(n,'u1'); opposite=np.zeros(n,'u1')
    neighbor_b=np.full(n,np.nan,'f4');neighbor_s=np.full(n,np.nan,'f4')
    unique=index['unique'];counts=index['counts'];keys=index['keys']
    for start in range(0,n,block_rows):
        ii=ids[start:start+block_rows];size=len(ii); key=keys[ii]
        total=np.zeros(size,'i8'); bsum=np.zeros(size,'f8');ssum=np.zeros(size,'f8')
        occupied=np.zeros((27,size),bool)
        for j,(dx,dy,dz) in enumerate(OFFSETS):
            delta=dx*(1<<(2*BITS))+dy*(1<<BITS)+dz
            kk=key+np.uint64(delta) if delta>=0 else key-np.uint64(-delta)
            loc=np.searchsorted(unique,kk)
            clipped=np.minimum(loc,len(unique)-1)
            valid=(loc<len(unique)) & (unique[clipped]==kk)
            positions=np.flatnonzero(valid); at=clipped[valid]
            total[positions]+=counts[at]
            bsum[positions]+=index['sums_b'][at];ssum[positions]+=index['sums_s'][at]
            occupied[j,positions]=True
        available=total>1
        neighbor_b[start:start+size][available]=((bsum[available]-features['brightness'][ii[available]])/(total[available]-1)).astype('f4')
        logs=np.log(np.maximum(features['effective_max'][ii],np.finfo('f4').tiny))
        neighbor_s[start:start+size][available]=np.exp((ssum[available]-logs[available])/(total[available]-1)).astype('f4')
        # Self does not count as support or an occupied neighboring direction.
        support[start:start+size]=(total-1).astype('i4')
        occupied[13]=False
        cells[start:start+size]=occupied.sum(0)
        opposite[start:start+size]=(occupied[:13] & occupied[:13:-1]).sum(0)
    return dict(support=support,neighbor_brightness=neighbor_b,neighbor_scale=neighbor_s,
                occupied_neighbors=cells,opposite_pairs=opposite)
