J/A+A/551/A107  X-shooter spectra of 24 young stellar objects    (Manara+, 2013)
================================================================================
X-shooter spectroscopy of young stellar objects.
II. Impact of chromospheric emission on accretion rate estimates.
    Manara C.F., Testi L., Rigliaco E., Alcala J.M., Natta A., Stelzer B.,
    Biazzo K., Covino E., Covino S., Cupani G., D'Elia V., Randich S.
   <Astron. Astrophys., 551, A107 (2013)>
   =2013A&A...551A.107M
================================================================================
ADC_Keywords: Stars, pre-main sequence ; Spectroscopy ; Equivalent widths
Keywords: stars: pre-main sequence - stars: low-mass - stars: activity

Abstract:
    The lack of knowledge of photospheric parameters and the level of
    chromospheric activity in young low-mass pre-main sequence stars
    introduces uncertainties when measuring mass accretion rates in
    accreting (Class II) young stellar objects. A detailed investigation
    of the effect of chromospheric emission on the estimates of mass
    accretion rate in young low-mass stars is still missing. This can be
    undertaken using samples of young diskless (Class III) K and M-type
    stars.

    Our is to measure the chromospheric activity of Class III pre main
    sequence stars to determine its effect on the estimates of the
    accretion luminosity (Lacc) and mass accretion rate (dMacc/dt) in
    young stellar objects with disks.

    Using VLT/X-shooter spectra, we analyzed a sample of 24 nonaccreting
    young stellar objects of spectral type between K5 and M9.5. We
    identified the main emission lines normally used as tracers of
    accretion in Class II objects, and we determined their fluxes in order
    to estimate the contribution of the chromospheric activity to the line
    luminosity.

    We have used the relationships between line luminosity and accretion
    luminosity derived in the literature for Class II objects to evaluate
    the impact of chromospheric activity on the accretion rate
    measurements. We find that the typical chromospheric activity would
    bias the derived accretion luminosity by L_acc,noise_<10^-3^L_{sun}_, with
    a strong dependence on the Teff of the objects. The noise on Macc
    depends on stellar mass and age, and the typical values of
    log(M_acc,noise_) range between ~-9.2 to -11.6M_{sun}_/yr. Values of
    L_acc_<10^-3^L_{sun}_ obtained in accreting low-mass pre main sequence
    stars through line luminosity should be treated with caution because
    the line emission may be dominated by the contribution of
    chromospheric activity.

Description:
    Reduced, flux- and wavelength-calibrated, telluric corrected 1-d
    spectra of all the 24 objects considered in this work. Each object's
    spectrum is divided in the three arms of X-Shooter, namely the UVB
    arm, the VIS arm, and the NIR arm. The spectra are given in fits
    format and are not corrected for extinction, but the objects are known
    to have low Av (<0.2mag).

File Summary:
--------------------------------------------------------------------------------
 FileName   Lrecl  Records   Explanations
--------------------------------------------------------------------------------
ReadMe         80        .   This file
table1.dat    143       24   Known parameters from the literature
table3.dat     38       24   Stellar parameters derived for the objects
                              in our sample
table4.dat     60       24   Spectral types obtained using the method based
                              on the spectral indices described in Sect. 3
                              and in Appendix B
tableb1.dat   110       12   Near-IR spectral indices
table6.dat    195       24   Fluxes and equivalent widths of Balmer lines
table7.dat    160       24   Fluxes and equivalent widths of He and Ca lines
refs.dat       69       29   References
spectra.dat    69       24   X-shooter spectra of ClassIII YSOs
sp/*            .       72   Individual UVB, Visual and near-IR spectra in FITS
--------------------------------------------------------------------------------

See also:
 J/A+A/548/A56 : X-shooter spectra of 12 young stellar objects  (Rigliaco+, 2012)
 J/A+A/561/A2  : 36 accreting YSOs emission lines (Alcala+, 2014)

Byte-by-byte Description of file: table1.dat
--------------------------------------------------------------------------------
   Bytes Format Units   Label     Explanations
--------------------------------------------------------------------------------
   1- 10  A10   ---     Name      Source name
  12- 35  A24   ---     OName     Other names
  37- 38  I2    h       RAh       Right ascension (J2000)
  40- 41  I2    min     RAm       Right ascension (J2000)
  43- 47  F5.2  s       RAs       Right ascension (J2000)
      49  A1    ---     DE-       Declination sign (J2000)
  50- 51  I2    deg     DEd       Declination (J2000)
  53- 54  I2    arcmin  DEm       Declination (J2000)
  56- 60  F5.2  arcsec  DEs       Declination (J2000)
  62- 72  A11   ---     Region    Region: Lup III, TW Hya, or {sigma} Ori
  74- 76  I3    pc      Dist      Distance (1)
  78- 81  A4    ---     Sp.l      MK spectral type from literature
  83- 87  F5.2  mag     Umag      ?=- U magnitude
  89- 93  F5.2  mag     Bmag      ?=- B magnitude
  95- 99  F5.2  mag     Vmag      ?=- V magnitude
 101-105  F5.2  mag     Rmag      ?=- R magnitude
 107-111  F5.2  mag     Imag      ?=- I magnitude
 113-117  F5.2  mag     Jmag      ?=- J magnitude
 119-123  F5.2  mag     Hmag      ?=- H magnitude
 125-129  F5.2  mag     Kmag      ?=- K magnitude
 131-143  A13   ---     Ref       References, in refs.dat file
--------------------------------------------------------------------------------
Note (1): Distances to TW Hya objects are obtained by Weinberger et al.
    (2013ApJ...762..118W), by Torres et al. (2008, Handbook of Star Forming
    Regions, 2, 757), and by Mamajek (2005ApJ...634.1385M),
    to sigma Ori by Brown et al. (1994, Cat. J/A+A/289/101), and
    to Lupus III by Comeron (2008, Handbook of Star Forming Regions, 42, 295).
--------------------------------------------------------------------------------

Byte-by-byte Description of file: table3.dat
--------------------------------------------------------------------------------
   Bytes Format Units   Label     Explanations
--------------------------------------------------------------------------------
   1- 10  A10   ---     Name      Source name
  13- 16  A4    ---     SpT       MK spectral type (2)
  18- 21  I4    K       Teff      Effective temperature (2)
  23- 27  F5.2  [Lsun]  log(L*)   stellar luminositiy
  29- 32  F4.2  Msun    M*        Stellar mass
  34- 38  F5.2  [Lsun]  <Lacc>    Average accretion luminosity
                                  <log(L_acc,noise_)>  (see sec. 6)
--------------------------------------------------------------------------------
Note (2): The spectral type-Teff relation is adopted from Luhman et al.
   (2003, Cat. J/ApJ/593/1093) for M-type objects and from Kenyon & Hartmann
   (1995, Cat. J/ApJS/101/117) for K-type objects.
--------------------------------------------------------------------------------

Byte-by-byte Description of file: table4.dat
--------------------------------------------------------------------------------
   Bytes Format Units  Label    Explanations
--------------------------------------------------------------------------------
   1- 10  A10   ---    Name     Source name
  12- 15  A4    ---    SpT      MK spectral type derived from this work,
                                   as explained in Sect. 3.1
  17- 20  A4    ---    Sp.V     MK spectral type obtained using the spectral
                                   indices in the VIS part of the spectrum,
                                   as explained in Sect. 3.2
  22- 25  A4    ---    Sp.1     MK spectral type obtained using H2O-K2 index (3)
  27- 30  A4    ---    Sp.5     MK spectral type obtained using H2O index (3)
  32- 35  A4    ---    Sp.8     MK spectral type obtained using sH2Oj index (3)
  37- 40  A4    ---    Sp.11    MK spectral type obtained using sH2Ok index (3)
  42- 45  A4    ---    Sp.9     MK spectral type obtained using sH2OH1 index (3)
  47- 50  A4    ---    Sp.2     MK spectral type obtained using Ij index (3)
  52- 55  A4    ---    Sp.3     MK spectral type obtained using IH index (3)
  57- 60  A4    ---    Sp.12    Spectral type obtained using HP index (3)
--------------------------------------------------------------------------------
Note (3): All the columns refer to the results obtained using NIR spectral
  indices, as explained in Appendix B. SpType are reported only in the range of
  validity of each index. Indexes are defined in tableb1.dat.
--------------------------------------------------------------------------------

Byte-by-byte Description of file: tableb1.dat
--------------------------------------------------------------------------------
   Bytes Format Units   Label     Explanations
--------------------------------------------------------------------------------
   1-  6  A6    ---     Index     Index name
   9- 15  A7    ---     Range     Range of validity
  17- 39  A23   ---     Num       Numerator [nm]
  41- 69  A29   ---     Den       Denominator [nm]
  71-110  A40   ---     Ref       Reference
--------------------------------------------------------------------------------

Byte-by-byte Description of file: table6.dat
--------------------------------------------------------------------------------
   Bytes Format Units       Label     Explanations
--------------------------------------------------------------------------------
   1- 10  A10   ---         Name      Source name
  12- 17  F6.3 10-17W/m2    F(Ha)     H{alpha} flux (10^-14^erg/cm^2^/s)
  19- 23  F5.3 10-17W/m2  e_F(Ha)     rms uncertainty on F(Ha)
      25  A1    ---       l_F(Hb)     Limit flag on F(Hb)
  26- 32  F7.4 10-17W/m2    F(Hb)     H{beta} flux (10^-14^erg/cm^2^/s)
  34- 38  F5.3 10-17W/m2  e_F(Hb)     ? rms uncertainty on F(Hb)
      39  A1    ---       l_F(Hg)     Limit flag on F(Hg)
  40- 46  F7.4 10-17W/m2    F(Hg)     H{gamma} flux (10^-14^erg/cm^2^/s)
  48- 53  F6.4 10-17W/m2  e_F(Hg)     ? rms uncertainty on F(Hg)
      54  A1    ---       l_F(Hd)     Limit flag on F(Hd)
  55- 61  F7.4 10-17W/m2    F(Hd)     H{delta} flux (10^-14^erg/cm^2^/s)
  63- 68  F6.4 10-17W/m2  e_F(Hd)     ? rms uncertainty on F(Hd)
      69  A1    ---       l_F(H8)     Limit flag on F(H8)
  70- 76  F7.4 10-17W/m2    F(H8)     H8 flux (10^-14^erg/cm^2^/s)
  78- 83  F6.4 10-17W/m2  e_F(H8)     ? rms uncertainty on F(H8)
      84  A1    ---       l_F(H9)     Limit flag on F(H9)
  85- 90  F6.4 10-17W/m2    F(H9)     H9 flux (10^-14^erg/cm^2^/s)
  92- 97  F6.4 10-17W/m2  e_F(H9)     ? rms uncertainty on F(H9)
      98  A1    ---       l_F(H10)    Limit flag on F(H10)
  99-106  F8.6 10-17W/m2    F(H10)    H10 flux (10^-14^erg/cm^2^/s)
 108-113  F6.4 10-17W/m2  e_F(H10)    ? rms uncertainty on F(H10)
     114  A1    ---       l_F(H11)    Limit flag on F(H11)
 115-122  F8.6 10-17W/m2    F(H11)    H11 flux (10^-14^erg/cm^2^/s)
 124-129  F6.4 10-17W/m2  e_F(H11)    ? rms uncertainty on F(H11)
 131-135  F5.1  km/s        Ha.1      H{alpha}10% width
 137-140  F4.2  ---         Bj        Balmer jump ratio, intened as F360/F400,
                                      (fluxes at ~360nm and ~400nm)
 142-147  F6.2  0.1nm       W(Ha)     ?=- H{alpha} equivalent width
 149-154  F6.2  0.1nm       W(Hb)     ?=- H{beta} equivalent width
 156-161  F6.2  0.1nm       W(Hg)     ?=- H{gamma} equivalent width
 163-168  F6.2  0.1nm       W(Hd)     ?=- H{delta} equivalent width
 170-175  F6.2  0.1nm       W(H8)     ?=- H8 equivalent width
 177-182  F6.2  0.1nm       W(H9)     ?=- H9 equivalent width
 184-189  F6.2  0.1nm       W(H10)    ?=- H10 equivalent width
 191-195  F5.2  0.1nm       W(H11)    ?=- H11 equivalent width
--------------------------------------------------------------------------------

Byte-by-byte Description of file: table7.dat
--------------------------------------------------------------------------------
   Bytes Format Units     Label     Explanations
--------------------------------------------------------------------------------
   1- 10  A10   ---       Name      Source name
      12  A1    ---     l_F(5876)    Limit flag on F(5876)
  13- 20  E8.3  mW/m2     F(5876)    HeI {lambda}587.6 flux
  22- 29  E8.3  mW/m2   e_F(5876)    ? rms uncertainty on F(5876)
      31  A1    ---     l_F(10836)   Limit flag on HeI2
  32- 39  E8.3  mW/m2     F(10836)   HeI {lambda}1083.6 flux
  41- 48  E8.3  mW/m2   e_F(10836)   ? rms uncertainty on F(10836)
      50  A1    ---     l_F(3934)    Limit flag on F(3934)
  51- 58  E8.3  mW/m2     F(3934)    CaII {lambda}393.4 flux
  60- 67  E8.3  mW/m2   e_F(3934)    ? rms uncertainty on F(3934)
      69  A1    ---     l_F(8499)    Limit flag on F(8499)
  70- 77  E8.3  mW/m2     F(8499)    CaII {lambda}849.9 flux
  79- 86  E8.3  mW/m2   e_F(8499)    ? rms uncertainty on F(8499)
      88  A1    ---     l_F(8542)    Limit flag on F(8542)
  89- 96  E8.3  mW/m2     F(8542)    CaII {lambda}854.2 flux
  98-105  E8.3  mW/m2   e_F(8542)    ? rms uncertainty on F(8542)
     107  A1    ---     l_F(8662)    Limit flag on F(8662)
 108-115  E8.3  mW/m2     F(8662)    CaII {lambda}866.2 flux
 117-124  E8.3  mW/m2   e_F(8662)    ? rms uncertainty on F(8662)
 126-130  F5.2  0.1nm     W(5876)    ?=- HeI {lambda}587.6 equivalent width
 132-136  F5.2  0.1nm     W(10836)   ?=- HeI {lambda}1083.6 equivalent width
 138-143  F6.2  0.1nm     W(3934)    ?=- CaII {lambda}393.4 equivalent width
 145-149  F5.2  0.1nm     W(8499)    ?=- CaII {lambda}849.9 equivalent width
 151-155  F5.2  0.1nm     W(8542)    ?=- CaII {lambda}854.2 equivalent width
 157-160  F4.2  0.1nm     W(8662)    ?=- CaII {lambda}866.2 equivalent width
--------------------------------------------------------------------------------

Byte-by-byte Description of file: refs.dat
--------------------------------------------------------------------------------
   Bytes Format Units   Label     Explanations
--------------------------------------------------------------------------------
   1-  2  I2    ---     Ref       Reference number
   4- 22  A19   ---     BibCode   BibCode
  24- 42  A19   ---     Aut       Author's name
  45- 69  A25   ---     Com       Comments
--------------------------------------------------------------------------------

Byte-by-byte Description of file: spectra.dat
--------------------------------------------------------------------------------
   Bytes Format Units   Label     Explanations
--------------------------------------------------------------------------------
   1- 10  A10   ---     Name      Object name
  12- 15  A4    ---     SpT       Spectral type of the object, this paper
  18- 29  A12   ---     UVB       UVB (330-550nm) spectrum in subdirectory sp
  32- 43  A12   ---     VIS       VIS (550-1000nm) spectrum in subdirectory sp
  46- 57  A12   ---     NIR       NIR (1000-2400nm) spectrum in subdirectory sp
  59- 61  F3.1  arcsec  SlU       Width of the UVB slit
  63- 65  F3.1  arcsec  SlV       Width of the VIS slit
  67- 69  F3.1  arcsec  SlN       Width of the NIR slit
--------------------------------------------------------------------------------

Acknowledgements:
    Carlo Felice Manara, cmanara(at)eso.org

References:
    Rigliaco et al., Paper I    2012A&A...548A..56R, Cat. J/A+A/548/A56
    Stelzer et al.,  Paper III  2013A&A...558A.141S
    Alcala et al.,   Paper IV   2014A&A...561A...2A, Cat. J/A+A/561/A2

================================================================================
(End)     Carlo Felice Manara [ESO], Patricia Vannier [CDS]          08-Mar-2014
