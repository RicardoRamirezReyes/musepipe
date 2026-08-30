#!/usr/bin/env python
"""Cuanto sesga la `apcorr` de C1/C2 ignorar que ROXs 42B es una binaria.

`apcorr = F(<=25 px)/F(box3)` del MODELO. Si se ajusta UNA PSF a lo que son DOS
estrellas, la forma sale mas ancha de lo que es una fuente puntual, y esa forma
sesgada es la que se usa para corregir apertura. Aqui se comparan las dos formas:

  * forma de 1 componente  -> lo que C1 obtiene hoy
  * forma del ajuste de 2  -> la PSF de una fuente puntual, sin el compañero dentro

El control es ROXs 12 b: sin binaria, las dos formas deben coincidir.

La geometria de la secundaria NO se ajusta: se fija con la astrometria publicada
(Keck/NIRC2 **2022.621**, rho = 51 +- 2 mas, PA = 148 +- 3 deg; Inglis et al. 2026
da Omega = 150.0 +- 0.6 con i = 91.0 +- 0.4), que es **la misma epoca** que estos
datos (2022.66). A 25.42 mas/px eso son **2.006 px** en azimut de pixel -122 deg,
o sea offset (dy, dx) = (-1.701, -1.063). Asi el ajuste gana **un solo** grado de
libertad: la razon de flujos.

Medido el 2026-08-30 (`docs/2026-08-30_binaria_42b_sesga_la_apcorr.md`), 45 bins
de 100 A: en ROXs 42B b f = 0.108 +- 0.033 y **se dobla del azul (0.056) al rojo
(0.115)**, con un sesgo de la `apcorr` del **+6.9 %**; en el control f = 0.023 +-
0.020 sin esa tendencia y el sesgo es +1.5 %, que es el suelo del metodo.
"""
import json, numpy as np
from astropy.io import fits
from scipy.optimize import least_squares

DY, DX = -1.701, -1.063
R_AJUSTE = 9.0

def moffat(yy,xx,y0,x0,alpha,beta,q,th):
    c,s=np.cos(th),np.sin(th); Y,X=yy-y0,xx-x0
    u=(X*c+Y*s); v=(-X*s+Y*c)/max(q,1e-3)
    return (1+(u*u+v*v)/alpha**2)**(-beta)

def ajusta(img,py,px,dos):
    ny,nx=img.shape; yy,xx=np.indices((ny,nx),float)
    m=(np.hypot(yy-py,xx-px)<=R_AJUSTE)&np.isfinite(img)
    Y,X,D=yy[m],xx[m],img[m]; esc=np.nanmax(D)
    def resid(p):
        y0,x0,A,al,be,q,th,bg=p[:8]; f=p[8] if dos else 0.0
        mod=A*moffat(Y,X,y0,x0,al,be,q,th)
        if dos: mod=mod+A*f*moffat(Y,X,y0+DY,x0+DX,al,be,q,th)
        return (mod+bg-D)/esc
    p0=[py,px,esc,2.0,2.5,0.9,0.0,0.0]+([0.3] if dos else [])
    lo=[py-3,px-3,0,0.3,1.05,0.3,-np.pi,-abs(esc)]+([0.0] if dos else [])
    hi=[py+3,px+3,10*esc,20,12,1.0,np.pi,abs(esc)]+([1.0] if dos else [])
    r=least_squares(resid,p0,bounds=(lo,hi),max_nfev=4000)
    return r.x

def apcorr_de(al,be,q,th, n=401, R=25.0):
    """F(<=25 px)/F(box3) de una Moffat: exactamente lo que calcula C2."""
    g=np.linspace(-R,R,n); Y,X=np.meshgrid(g,g,indexing="ij")
    P=moffat(Y,X,0,0,al,be,q,th); da=(g[1]-g[0])**2
    dentro=(np.hypot(Y,X)<=R)
    F25=(P[dentro]).sum()*da
    caja=(np.abs(Y)<=1.5)&(np.abs(X)<=1.5)
    Fbox=(P[caja]).sum()*da
    return F25/Fbox

for run,etiq in (("ROXs42Bb_realigned","ROXs 42B b — BINARIA"),
                 ("ROXs12b_realigned","ROXs 12 b — CONTROL")):
    qc=json.load(open(f"runs/{run}/stages/stage01c_qc.json")); py,px=qc["primary"]["pos_yx"]
    with fits.open(f"runs/{run}/stages/stage02_xcorr_cube_stack.fits",memmap=True) as h:
        wave=np.asarray(h["WAVELENGTH"].data,float); cubo=h["CUBES"].data
        cubo=cubo[0] if cubo.ndim==4 else cubo
        print(f"\n=== {etiq} ===")
        print(f"{'lambda':>10} {'f':>6} {'apcorr 1c':>10} {'apcorr 2c':>10} {'sesgo':>8} "
              f"{'alpha 1c':>9} {'alpha 2c':>9} {'q 1c':>6} {'q 2c':>6}")
        ses=[]; fs_all=[]
        for l0 in np.arange(4800,9300,100):
            sel=(wave>=l0)&(wave<l0+500)
            if sel.sum()<4: continue
            img=np.nanmedian(np.asarray(cubo[sel],float),axis=0)
            p1=ajusta(img,py,px,False); p2=ajusta(img,py,px,True)
            a1=apcorr_de(p1[3],p1[4],p1[5],p1[6]); a2=apcorr_de(p2[3],p2[4],p2[5],p2[6])
            s=100*(a1/a2-1); ses.append(s)
            fs_all.append((l0+50,p2[8],a1,a2,s))
        import csv
        with open(f"/tmp/binaria_{run}.csv","w",newline="") as fh:
            w=csv.writer(fh); w.writerow(["lambda_A","f","apcorr_1c","apcorr_2c","sesgo_pct"])
            w.writerows(fs_all)
        ff=np.array([r[1] for r in fs_all])
        print(f"  bins={len(fs_all)}  f mediana={np.median(ff):.3f} disp={ff.std():.3f} "
              f"| f(azul<6000)={np.median([r[1] for r in fs_all if r[0]<6000]):.3f} "
              f"f(rojo>8000)={np.median([r[1] for r in fs_all if r[0]>8000]):.3f}")
        ses=np.array(ses)
        print(f"  sesgo de la apcorr por ignorar la segunda componente: "
              f"mediana {np.median(ses):+.1f}%  rango [{ses.min():+.1f}, {ses.max():+.1f}]%")
