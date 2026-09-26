"""Residuo telúrico fino en 8100–8400 Å: amplitud α de la estructura de H2O de la
curva de transmisión que queda en el espectro corregido. α=0 corrección perfecta,
α=+1 sin corregir, α<0 sobrecorrección. Nula: la misma curva desplazada ±3–15 Å."""
import numpy as np
from astropy.io import fits
from scipy.ndimage import median_filter
def spec(path, col='flux'):
    d=fits.open(path)['SPECTRUM'].data if 'SPECTRUM' in [h.name for h in fits.open(path)] else fits.open(path)[1].data
    return np.asarray(d['wave_A'],float), np.asarray(d[col],float), np.asarray(d[[c for c in d.names if 'err' in c.lower()][0]],float)
def hp(y,k=31):
    y=np.where(np.isfinite(y),y,np.nan); base=median_filter(np.nan_to_num(y,nan=np.nanmedian(y)),size=k,mode='nearest'); return y-base
def alpha(w,f,T,lo=8100,hi=8400,excl=((8170,8210),),shift=0.0):
    Ts=np.interp(w, w+shift, T)
    ok=np.isfinite(f)&(f>0)&np.isfinite(Ts)&(Ts>0)
    lf=np.full_like(f,np.nan); lf[ok]=np.log(f[ok]); lt=np.log(np.clip(Ts,1e-3,None))
    r=hp(lf); t=hp(lt)
    m=(w>lo)&(w<hi)&np.isfinite(r)&np.isfinite(t)
    for a,b in excl: m&=~((w>a)&(w<b))
    A=np.c_[t[m],np.ones(m.sum())]; c,*_=np.linalg.lstsq(A,r[m],rcond=None)
    return c[0], np.std(t[m])
def informe(nombre,w,f,T):
    a0,amp=alpha(w,f,T)
    nul=[alpha(w,f,T,shift=s)[0] for s in np.r_[-15:-2.9:1.5, 3:15.1:1.5]]
    sd=np.std(nul)
    a_all,_=alpha(w,f,T,excl=())
    depthNa=np.interp([8183.3,8194.8],w,1-T)
    print(f"{nombre:34s} α = {a0:+.3f} ± {sd:.3f} (nula, n={len(nul)})   con Na I dentro: {a_all:+.3f}   "
          f"profundidad telúrica en Na I: {100*depthNa[0]:.1f}/{100*depthNa[1]:.1f} % → residuo ≈ {100*a0*depthNa.mean():+.2f} %")
