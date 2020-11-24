func pboot 
/* DOCUMENT pboot

PIONIER data reduction with the bootstrap method.  Each OI FITS file contains
the bootstrap along the wavelength dimension, one after another.  Thus the
arrays in the OI_WAVE, OI_VIS, OI_VIS2, OI_T3 extensions will have have
a length nwave * nboot, where nwave is the number of wavelengths and
nboot the number of bootstraps. 

MAIN METHODS

pbootReduceDirectory - Reduce PIONIER scans and produce raw visibilities
pbootCalibrateDirectory - Calibrate raw visibilities 

SEE ALSO

pndrs
 
*/
{
  info, pboot;
}

/* include pndrs silently. Packages that start to talk alone are a pain
in the donkey */

_builtin_write = write;
func write(x, .., format=) { }
#include "pndrsPatch.i"
write = _builtin_write;
/*
_builtin_window = window;
func window(n, display=, dpi=, wait=, private=, hcp=, dump=,
    legends=, style=, width=,height=,rgb=, parent=, xpos=, ypos=)
{
  _builtin_window, n, display="", hcp=pr1(n), dpi=dpi, wait=wait, 
    private=private,
    dump=dump, legends=legends, style=style, width=width, height=height,
    rgb=rgb, parent=parent, xpos=xpos, ypos=ypos; 
}*/

func catch(x) { return 0; }

func pbootSignalComputeBispNorm(bisp, bispnorm, clo, &t3Amp, &t3AmpErr, gui=,square=)
{
  if (!is_void(gui))
  {
    gui = gui + 2;
  }
  bisp_re = ( bisp * exp( -2.i*pi * clo(,[1],) / 360.0 ) ).re;
  pndrsSignalComputeAmp2Ratio, bisp_re, bispnorm, t3Amp, t3AmpErr,
     gui=gui, square=square;
}


func pbootScanComputeClosurePhases(coherData, imgLog, norm2, pos, &oiT3, &clo, &cloErr, gui=)
/* DOCUMENT pndrsScanComputeClosurePhases(coherData, imgLog, &oiT3, &clo, &cloErr, gui=)

   DESCRIPTION
   Perform an estimate of the closure-phase with the following method:
   - filter the data
   - compute the bispectrum in the direct space (coherData should be complex)
   - average the bispectrum over the nopd
   - bootstrap the bispectrum over the nscan
   - take the phase
   - average/rms over the nboot

   PARAMETERS
   - coherData: should contain complex scans

   SEE ALSO
 */
{
  yocoLogInfo,"pbootScanComputeClosurePhases()";
  local i, data, opd, nbase, bispectrum, map, ntel, dx, polId, staId, l;
  local nscan, boot, pol, npol, baseId, filt, predWidth, lbd0, sig0, lbdB;
  local t3Amp, t3AmpErr, clo, cloErr;
  
  extern pbootIndexBootstrap;

  nscan = dimsof(*coherData.regdata)(-1);
  pbootIndexBootstrap = pbootGenerateIndexBootstrap(nscan=nscan, nboot=pbootNboot);

  pndrsSignalComputeAmp2Ratio = pbootSignalComputeAmp2Ratio;
  pndrsSignalComputePhase = pbootSignalComputePhase;
  oiFitsSetDataArray = pbootSetDataArray;
  
  
  /* Get the data and the opds */
  pndrsGetData, coherData, imgLog, data, opd, map, oLog;
  nbase = dimsof(data)(0);
  nstep = dimsof(data)(3);
  ntel  = max(max(map.t1,map.t2));
  pol   = yocoListClean(map.pol)(*);
  npol  = numberof(pol);
  nlbd  = dimsof(data)(2);

  /* Default Wave */
  pndrsGetDefaultWave, oLog, lbd0, lbdB, sig0;

  /* Filter the data */
  // if (strmatch( pndrsGetBand(oLog(1)), "K") ) filt = pndrsFilterKClo;
  // if (strmatch( pndrsGetBand(oLog(1)), "H") ) filt = pndrsFilterHClo;
  // pndrsSignalSquareFiltering, data, opd, filt, dataf;

  /* Compute the predicted width of the pic based on
     the tau0 and scanning speed */
  predWidth = pndrsGetPicWidth(oLog, checkUTs=1);
  
  /* Filter around the expected frequency with expected width.
     Force the filter the very low frequencies. FIXME: this could
     probably be improved with a Gaussian-cut filter */
  pndrsSignalFourierTransform, data, opd, ft, freq;
  ft *= ((indgen(nstep) > 15) & (indgen(nstep) < (nstep/2-5)))(-,) &
    ( abs(abs(freq)(-,)-sig0) < predWidth(,-,-,) );

  /* Come back in direct space */
  pndrsSignalFourierBack, ft, freq, dataf;
  
  /* Loop over all the possible 3T closure phase,
     hypothesis is that the map has t1<t2. Closure phases
     are in the order 123, 124, 134, 234 */
  for (p=1 ; p<=npol ;p++)
    for( i=1 ; i<=ntel-2 ; i++)
      for( j=i+1 ; j<=ntel-1 ; j++)
        for( k=j+1 ; k<=ntel ; k++)
          {
            grow, baseId, 
              [[where(map.t1==i & map.t2==j & map.pol==pol(p))(1),
                where(map.t1==j & map.t2==k & map.pol==pol(p))(1),
                where(map.t1==i & map.t2==k & map.pol==pol(p))(1)]];
            grow, staId, [ [i,j,k] ];
            grow, polId, p;
          }

  /* Compute the bispectrum (x,y,nopd,nscan,closure) */
  d = dimsof(data);
  bispectrum = dataf(,,,baseId(1,)) * dataf(,,,baseId(2,)) * conj( dataf(,,,baseId(3,)) );

  /* Average over the opds. Now dimension is
     [lbd, scan, bispectrum]   */
  bispectrum  = bispectrum(,avg,,);

  /* Compute the phase average and errors */
  pndrsSignalComputePhase, bispectrum,
    clo, cloErr, gui=gui;

  /* Compute the expected amplitude of the bispectrum */
  opd = opd - opd(avg,-,) - pos(-,);
  env = exp(- (pi*opd(-,)/lbd0^2*lbdB)^2 / 5.0);
  norm2env = env^2 * norm2;
  bispnorm = sqrt(abs(norm2env(,,,baseId(1,)) * norm2env(,,,baseId(2,)) *
                      norm2env(,,,baseId(3,))));
  bispnorm = bispnorm(,avg,,);

  /* Use a coherent estimator for the bispectrum */
  pbootSignalComputeBispNorm, bispectrum, bispnorm, clo, t3Amp, t3AmpErr;
  
  /* build the oiT3 */
  oiT3 = [];
  nclo = dimsof(baseId)(0);
  log  = oiFitsGetOiLog(coherData, imgLog);
  oiT3 = array( oiFitsGetOiStruct("oiT3", -1), nclo);
  oiT3.hdr.logId   = log.logId;
  oiT3.hdr.dateObs = strtok(log.dateObs,"T")(1);
  oiT3.mjd         = (*coherData.time)(avg);
  oiT3.intTime     = (*coherData.exptime)(sum);
  oiT3.staIndex    = staId;

  /* Fill the data */
  flag = char(cloErr>30.0);
  oiFitsSetDataArray, oiT3,,t3Amp, t3AmpErr, clo, cloErr, flag;

  /* Build the default insName */
  oiT3.hdr.insName = pndrsDefaultInsName(log, pol(polId));

  /* Add QC parameters */
  yocoLogInfo," add QC parameters";
  id = oiFitsGetId(coherData.hdr.logId, imgLog.logId);
  iss = totxt( pndrsPionierToIss(imgLog(id), staId) )(sum,);
  pndrsSetLogInfoArray, imgLog, id, "qcPhi%sAvg", iss, averagePhase(clo,1);
  pndrsSetLogInfoArray, imgLog, id, "qcPhi%sErr", iss, average(cloErr,1);

  /* Plot the gui */
  if (gui && pndrsBatchPlotLevel) {
    sta  = (pndrsGetLogInfo(oLog,"issStation%i",staId)+["-","-",""])(sum,);
    if (allof(sta=="--"))
      sta  = (totxt(staId)+["-","-",""])(sum,);
    main = swrite(format="%s - %.4f", oLog.target, (*coherData.time)(avg));
    
    window,gui;
    pndrsPlotAddTitles,sta,main,"Complex bispectrum for all scans (first, middle, last channels)",
      "Re{bispectrum}", "Im{bispectrum}";

    winkill,gui+1;
    yocoNmCreate,gui+1,nclo,dx=0.06,dy=0.01;
    yocoPlotPlpMulti,clo,tosys=indgen(nclo),symbol=0,dy=cloErr;
    yocoPlotPlgMulti,clo,tosys=indgen(nclo);
    yocoNmLimits,0.5,nlbd+0.5,max(-180,min(clo-5)),min(180,max(clo+5));
    pndrsPlotAddTitles,sta,main,"Final closure phases","spec. channel","closures";
  }
  

  yocoLogTrace,"pbootScanComputeClosurePhases done";
  return 1;
}


func pbootGetUVCoordIss(imgLog, t1, t2, &uCoord, &vCoord, &sta1, &sta2)
/* DOCUMENT pndrsGetUVCoordIss(imgLog, t1, t2, &uCoord, &vCoord, &sta1, &sta2)

   DESCRIPTION
   Return the UV coordinates from the ISS for the PIONIER beams t1 and t2.
   It makes use of the function pndrsPionierToIss to convert the PIONIER
   beams into ISS beams.

   PARAMETERS
   - imgLog
   - t1, t2: PIONIER beams (numbering of the structure "map" and of PIONIER itself)
   - uCoord, vCoord:  returned corrdinates in meters
   - sta1, sta2: corresponding station names.
 */
{
  local A, B, lenISS, angISS, sign, base;
  sta1 = sta2 = uCoord= vCoord = [];
  
  /* Init */
  if (numberof(imgLog)>1) error;

  /* Convert t1 and t2 in ISS numbers in header */
  t1 = pndrsPionierToIss(imgLog, t1);
  t2 = pndrsPionierToIss(imgLog, t2);

  /* Loop on the telescope pairs since impossible to vector's */
  for (i=1;i<=numberof(t1);i++) 
  {
  
    /* The ISS sign are always define from lowest to highest ISS numbers,
       therefore we will swap the sign of the uv-plan for our
       baselines defined the other way */
    A = min(t1(i),t2(i));
    B = max(t1(i),t2(i));

    /* Read angle and length of baselines, deal
       with the case when one beam is 0 (3T case).
       Return Baseline at start and end in a
       complex way. */
    Base0 = (A == 0 || B == 0)? 0+0i: pndrsGetIssBaseFromTel(imgLog(1), A, B);

    /* Version 1: Phase sign seems to be wrong: binaries observed with PIONIER,
       reduced with this software and fitted with LITpro give 180deg
       inversion in position.
       
       Version 2+: revers the UV-plane. */
    sign   = [-1,1](1 + (t1(i) < t2(i)));

    /* This formula has been checked with aspro2 and gives similar
       results */
    grow, uCoord, sign * Base0.im;
    grow, vCoord, sign * Base0.re;
  
  }

  return 1;
}


func pndrsFixMissingIss(&oiLog, &oiVis, &oiVis2, &oiT3)
{
  missingLst  = oiLog.lst == 0;
  if (anyof(missingLst))
  {
    /* LST determination with sub-second accuracy */

    vltlong = -70.4042;
    mjd = oiLog.mjdObs;
    d = mjd - 51544.5;
    t = d / 36525.0; // Julian century
    lst = 280.46061837 + 360.98564736629 * d;
    lst += (0.000387933 - t / 38710000 ) * t * t; // secular correction
    lst += vltlong;
    lst = (lst % 360) / 360 * 86400; // lst in seconds in pndrs 

    hasLst = where(! missingLst);
    missesLst = where(missingLst);
    // there is a difference of a few seconds (roundup??)
    lst += (oiLog.lst(hasLst) - lst(hasLst))(avg);
    noLst = where(oiLog.mjdObs == 0);
    oiLog.lst(missesLst) = lst(missesLst); 
  
  }
  
  /* Fill the UV plan of oiVis2 */
  if (is_array(oiVis2)) 
  {
    for (i = 1; i <= numberof(oiVis2); ++i) 
    {
      local uCoord, vCoord;
      oLog = oiFitsGetOiLog(oiVis2(i), oiLog);
      pbootGetUVCoordIss, oLog, oiVis2(i).staIndex(1,), oiVis2(i).staIndex(2,), uCoord, vCoord;
      write, format="%8.1f%8.1f%8.1f%8.1f  %s\n", oiVis2(i).uCoord, oiVis2(i).vCoord, uCoord, vCoord, oLog.obsName;
      if (oiVis2(i).uCoord == 0)
      {
        oiVis2(i).uCoord   = uCoord;
        oiVis2(i).vCoord   = vCoord; 
      }
    }
  }

  /* Fill the UV plan of oiVis2 */
  if (is_array(oiVis)) 
  {
    for (i = 1; i <= numberof(oiVis); ++i) 
    {
      local uCoord, vCoord;
      if (oiVis(i).uCoord == 0)
      {
        oLog = oiFitsGetOiLog(oiVis(i), oiLog);
        pbootGetUVCoordIss, oLog, oiVis(i).staIndex(1,), oiVis(i).staIndex(2,), uCoord, vCoord;
        oiVis(i).uCoord   = uCoord;
        oiVis(i).vCoord   = vCoord;
      }
    }
  }

    /* Fill the UV plan of oiT3 */
  if (is_array(oiT3)) 
  {
    for (i = 1; i <= numberof(oiT3); ++i)
    {
      local u1Coord, v1Coord, u2Coord, v2Coord;
      if (oiT3(i).u1Coord == 0)
      {
        oLog = oiFitsGetOiLog(oiT3(i), oiLog);
        
        pbootGetUVCoordIss, oLog(1), oiT3(i).staIndex(1,), oiT3(i).staIndex(2,), u1Coord, v1Coord;
        pbootGetUVCoordIss, oLog(1), oiT3(i).staIndex(2,), oiT3(i).staIndex(3,), u2Coord, v2Coord;
        oiT3(i).u1Coord = u1Coord;
        oiT3(i).v1Coord = v1Coord;
        oiT3(i).u2Coord = u2Coord;
        oiT3(i).v2Coord = v2Coord;
      }
    }
  }

}

/* to allow redefining pndrs functions locally, yet calling the original
   ones when necessary */ 

func pbootSavePndrsFunction(name)
{
  if (!symbol_exists("pbootPndrs" + name))
  {
    symbol_set, "pbootPndrs" + name, symbol_def("pndrs" + name);
  }
}
func pbootSaveOiFitsFunction(name)
{
  if (!symbol_exists("pbootOiFits" + name))
  {
    symbol_set, "pbootOiFits" + name, symbol_def("oiFits" + name);
  }
}

pbootSavePndrsFunction,  "SavePdf";
pbootSavePndrsFunction,  "ComputeSingleOiData";
pbootSavePndrsFunction,  "ScanComputeAmp2PerScan";
pbootSavePndrsFunction,  "ScanComputeAmp2PerScanAbcd";
pbootSavePndrsFunction,  "ScanComputePolarDiffPhases";
pbootSavePndrsFunction,  "ScanComputeClosurePhases";
pbootSavePndrsFunction,  "SignalComputeAmp2Ratio";
pbootSavePndrsFunction,  "SignalComputePhase";
pbootSavePndrsFunction,  "PlotTfForAllSetups";
pbootSaveOiFitsFunction, "SetDataArray";
pbootSaveOiFitsFunction, "WriteFile";
pbootSaveOiFitsFunction, "CleanDataFromDummy";
pbootSaveOiFitsFunction, "GetBaseLength";
pbootSaveOiFitsFunction, "ApplyTf";

/*************************** Helper routines *****************************/

func pbootArraysLike(in1, &out1, in2, &out2, in3, &out3, in4, &out4,
  in5, &out5, in6, &out6, in7, &out7, in8, &out8, in9, &out9, in10, &out10)
{

  out1 = pbootArrayLike(in1);
  out2 = pbootArrayLike(in2);
  out3 = pbootArrayLike(in3);
  out4 = pbootArrayLike(in4);
  out5 = pbootArrayLike(in5);
  out6 = pbootArrayLike(in6);
  tout7 = pbootArrayLike(in7);
  out8 = pbootArrayLike(in8);
  out9 = pbootArrayLike(in9);
  out10 = pbootArrayLike(in10);

}

func pbootArrayLike(in1)
/* DOCUMENT pbootArrayLike(in1)

Generate a new array of same OIFITS type and length as in1.

*/

{
 
  local out1;
  
  if (!is_array(in1))
  {
    return [];
  }

  oiFitsCopyArrays, in1, out1;
   
  for (i = 1; i <= numberof(in1); ++i)
  {
    if (oiFitsIsOiWave(in1) && structof(in1(i).effWave) == pointer)
    {
      out1(i).effWave = & (*in1(i).effWave);
      out1(i).effBand = & (*in1(i).effBand);
    }
    if (oiFitsIsOiVis2(in1) && structof(in1(i).vis2Data) == pointer)
    {
      ndata = dimsof(*out1(i).flag);
      out1(i).vis2Data = & array(1.0, ndata);
      out1(i).vis2Err = & array(1e+8, ndata);
      out1(i).flag = & array(char, ndata);
    }
    if (oiFitsIsOiT3(in1) && structof(in1(i).t3Amp) == pointer)
    {
      ndata = dimsof(*out1(i).flag);
      out1(i).t3Amp = & array(1.0, ndata);
      out1(i).t3AmpErr = & array(1e+8, ndata);
      out1(i).t3Phi = & array(0.0, ndata);
      out1(i).t3PhiErr = & array(1e+8, ndata);
      out1(i).flag = & array(char, ndata);
    }
    if (oiFitsIsOiVis(in1) && structof(in1(i).visAmp) == pointer)
    {
      ndata = dimsof(*out(i).flag);
      out1(i).visAmp = & array(1.0, ndata);
      out1(i).visAmpErr = & array(1e+8, ndata);
      out1(i).visPhi = & array(0.0, ndata);
      out1(i).visPhiErr = & array(1e+8, ndata);
      out1(i).flag = & array(char, ndata);
    }
  }
   
  return out1;

}

func unique(x, tol=)
{

  if (is_void(tol))
  {
    tol = 1e-10;
  }
  if (!is_array(x))
  {
    return x;
  }

  if (numberof(x) == 1)
    return x;
  x = x(sort(x));

  if (noneof(typeof(x) == ["float", "double", "complex"]))
  {
    w = where(x(2:) != x(:-1))
  } else
  {
    w = where(abs(x(2:) - x(:-1)) > tol * abs(x(2:)));
  }
  y = [x(1)];
  if (numberof(w))
    grow, y, x(1 + w);
  return y;

}




/************************* Bootstrap manipulation **************************/


struct pbootCalibratorBootstrap_t 
/* DOCUMENT pbootCalibratorBootstrap_t

Structure containing bootstraps of calibrators.  They are given
as an array of dimension # of calibrators x # of bootstraps referring
to the log Id's of calibrator observations. 

*/
{
  string setup;
  pointer logId;
  pointer origLogId;
}

func pbootGenerateIndexBootstrap(nscan=, nboot=)
/* DOCUMENT pbootGenerateIndexBootstrap(nscan=, nboot=)

Does nboot sorts of nscan values with repeats in the set {1, ..., nscan}. 
The result has dimension nscan x nboot. By default nboot is 500 and nscan 
is 100. 

*/
{

  yocoLogInfo, "pbootGenerateIndexBootstrap()";

  if (is_void(nboot)) 
  {
    nboot = 5000;
    yocoLogInfo, "Default number of bootstraps is 500";
  }
  if (is_void(nscan))
  {
    yocoLogInfo, "Default number of scans is 100"; 
    nscan = 100;
  }

  boot = 1 + int(nscan * random(nscan, nboot));
  boot(,1) = indgen(nscan);
  return boot;

  yocoLogTrace, "pbootGenerateIndexBootstrap done";

}

func pbootSetBootstrap(oiLog, oiWave, oiArray, &out1, in1, &out2, in2, &out3, in3, 
  &out4, in4, &out5, in5, &out6, in6, &out7, in7, boot=, calBoot=)
/* DOCUMENT pbootSetBootstrap(oiLog, oiWave, &out1, in1, .., boot=, calBoot=)

Set bootstrap number boot in OI FITS structure out1 (and out2, out3, ..) 
given by the standard OI FITS in1 (and in2, in3, ...)

*/
{
  pbootSetBootstrapHelper, oiLog, oiWave, oiArray, out1, in1, boot=boot, calBoot=calBoot;
  pbootSetBootstrapHelper, oiLog, oiWave, oiArray, out2, in2, boot=boot, calBoot=calBoot;
  pbootSetBootstrapHelper, oiLog, oiWave, oiArray, out3, in3, boot=boot, calBoot=calBoot;
  pbootSetBootstrapHelper, oiLog, oiWave, oiArray, out4, in4, boot=boot, calBoot=calBoot;
  pbootSetBootstrapHelper, oiLog, oiWave, oiArray, out5, in5, boot=boot, calBoot=calBoot;
  pbootSetBootstrapHelper, oiLog, oiWave, oiArray, out6, in6, boot=boot, calBoot=calBoot;
  pbootSetBootstrapHelper, oiLog, oiWave, oiArray, out7, in7, boot=boot, calBoot=calBoot;
  pbootSetBootstrapHelper, oiLog, oiWave, oiArray, out8, in8, boot=boot, calBoot=calBoot;
}

func pbootSetBootstrapHelper(oiLog, oiWave, oiArray, &out1, in1, boot=, calBoot=)
{

  ndata = numberof(in1);
  if (!ndata)
  {
    return;
  }

  outBase = oiFitsGetBaseName(out1, oiArray);
  inBase = oiFitsGetBaseName(in1, oiArray);
  outLogId = out1.hdr.logId;
  inLogId = in1.hdr.logId;

  for (i = 1; i <= ndata; ++i)
  {
    local nwave, nboot;
    pbootGetBootstrapSize, in1(i), oiWave, nwave, nboot;
    rg = (1 + nwave * (boot - 1)):(nwave + nwave * (boot - 1)); 
    j = where(inBase(i) == outBase & inLogId(i) == outLogId)(1);
    if (oiFitsIsOiVis2(in1))
    {
      (*out1(j).vis2Data)(rg) = *in1(i).vis2Data;
      (*out1(j).vis2Err)(rg) = *in1(i).vis2Err;
      (*out1(j).flag)(rg) = *in1(i).flag;
      continue;
    }
    if (oiFitsIsOiT3(in1))
    {
      (*out1(j).t3Amp)(rg) = *in1(i).t3Amp;
      (*out1(j).t3AmpErr)(rg) = *in1(i).t3AmpErr;
      (*out1(j).t3Phi)(rg) = *in1(i).t3Phi;
      (*out1(j).t3PhiErr)(rg) = *in1(i).t3PhiErr;
      (*out1(j).flag)(rg) = *in1(i).flag;
      continue;
    }
    if (oiFitsIsOiVis(in1))
    {
      (*out1(j).visAmp)(rg) = *in1(i).visAmp;
      (*out1(j).visAmpErr)(rg) = *in1(i).visAmpErr;
      (*out1(j).visPhi)(rg) = *in1(i).visPhi;
      (*out1(j).visPhiErr)(rg) = *in1(i).visPhiErr;
      (*out1(j).flag)(rg) = *in1(i).flag;
      continue;
    }
    if (oiFitsIsOiWave(in1))
    {
      (*out1(j).effWave)(rg) = *in1(i).effWave;
      (*out1(j).effBand)(rg) = *in1(i).effBand;
    }
  }

}

func pbootKeepBootstrap(oiLog, oiWave, oiArray, oiDiam, oiCalBoot,
  in1, &out1, &out1cal, in2, &out2, &out2cal, in3, &out3, &out3cal,
  in4, &out4, &out4cal, in5, &out5, &out5cal, in6, &out6, &out6cal, 
  in7, &out7, &out7cal,
  boot=)
/* pbootKeepBootstrap(oiLog, oiWave, oiArray, oiDiam, oiCalBoot,
  in1, &out1, &out1cal, in2, &out2, &out2cal, ...)

Keep a given bootstrap of input array "in1"  given by bootstrap
index "boot".  There are two outputs: "out1" and "out1cal".  In "out1" 
the calibrators have not been bootstrapped (just the raw scans) so the
data have exactly the same MJD as in the original data.  In "out1cal"
calibrators have been bootstrapped so some calibrator observations
are not present, and some calibrator observations are repeated.

*/
{

  local oCalBoot;

  /* extract bootstrap calibrators */

  oDiam = oiDiam(,boot);
  oiFitsCopyArrays, oiCalBoot, oCalBoot;
  for (i = 1; i <= numberof(oiCalBoot); ++i)
  {
    logId = *oiCalBoot(i).logId;
    oCalBoot(i).logId = is_array(logId)? & logId(,boot): pointer(0);
    oCalBoot(i).origLogId = & (*oiCalBoot(i).origLogId);
  }

  /* do it for all arguments */
  pbootKeepBootstrapHelper, oiLog, oiWave, oiArray, oDiam, oCalBoot, in1, out1, out1cal, boot=boot;
  pbootKeepBootstrapHelper, oiLog, oiWave, oiArray, oDiam, oCalBoot, in2, out2, out2cal, boot=boot;
  pbootKeepBootstrapHelper, oiLog, oiWave, oiArray, oDiam, oCalBoot, in3, out3, out3cal, boot=boot;
  pbootKeepBootstrapHelper, oiLog, oiWave, oiArray, oDiam, oCalBoot, in4, out4, out4cal, boot=boot;
  pbootKeepBootstrapHelper, oiLog, oiWave, oiArray, oDiam, oCalBoot, in5, out5, out5cal, boot=boot;
  pbootKeepBootstrapHelper, oiLog, oiWave, oiArray, oDiam, oCalBoot, in6, out6, out6cal, boot=boot;
  pbootKeepBootstrapHelper, oiLog, oiWave, oiArray, oDiam, oCalBoot, in7, out7, out7cal, boot=boot;

}

func pbootKeepBootstrapHelper(oiLog, oiWave, oiArray, oDiam, oCalBoot, 
  in1, &out1, &out1cal, boot=)
{
  local oCalBoot;

  oiFitsCopyArrays, in1, out1;

  /* First step: isolate the bootstrap data as encoded along the 
     wavelength direction.  */
 
  nData = numberof(in1);
 
  for (i = 1; i <= nData; ++i)
  {
   
    local nwave, nboot;

    data = in1(i);

    pbootGetBootstrapSize, data, oiWave, nwave, nboot;
    rg = (1 + nwave * (boot - 1)):(nwave + nwave * (boot - 1)); 
   
    if (oiFitsIsOiWave(data))
    {
      out1(i).effWave = &(*data.effWave)(rg);
      out1(i).effBand = &(*data.effBand)(rg);
      continue;
    }
    
    if (oiFitsIsOiVis(data))
    {
      out1(i).visAmp = & (*data.visAmp)(rg);
      out1(i).visAmpErr = & (*data.visAmpErr)(rg);
      out1(i).visPhi = & (*data.visPhi)(rg);
      out1(i).visPhiErr = & (*data.visPhiErr)(rg);
      out1(i).flag = & (*data.flag)(rg);
      continue;
    }

    if (oiFitsIsOiVis2(data))
    {
      out1(i).vis2Data = & (*data.vis2Data)(rg);
      out1(i).vis2Err = & (*data.vis2Err)(rg);
      out1(i).flag = & (*data.flag)(rg);
      continue;
    }

    if (oiFitsIsOiT3(data))
    {
      out1(i).t3Amp = & (*data.t3Amp)(rg);
      out1(i).t3AmpErr = & (*data.t3AmpErr)(rg);
      out1(i).t3Phi = & (*data.t3Phi)(rg);
      out1(i).t3PhiErr = & (*data.t3PhiErr)(rg);
      out1(i).flag = & (*data.flag)(rg);
      continue;
    }

  } /* for (i = 1; i <= nData; ++i) */

  /* Second step: bootstrap the calibrators for each setup.  If a 
     calibrator is not found (e.g. one of the baselines has been
     trashed due to bad data), it is replaced by another with large
     error bars. */

  oiFitsCopyArrays, out1, out1cal;

  if (oiFitsIsOiVis(out1) || oiFitsIsOiT3(out1) || oiFitsIsOiVis2(out1))
  {

    /* loop on individual setups and baselines (or triplets) */

    iscal = oiFitsGetIsCal(out1, oDiam);
    setups = pbootGetSetupBase(out1, oiLog, oiArray);
    setupList = unique(setups);
    nSetup = numberof(setupList);

    for (iSetup = 1; iSetup <= nSetup; ++iSetup)
    {

      /* calibrator index for this setup and baseline (or triplet) */
      id = where(iscal & setups == setupList(iSetup));
      if (!numberof(id))
      {
        continue;
      }

      /* the calibrator boostrap is for setups (not baseline wise),
         it gives a log id, we do a lookup into oData(id).hdr.logId.
       */   
      thisSetup =  oiFitsDefaultSetup(out1(id(1)), oiLog);
      oCal = out1(id);
      logId = oCal.hdr.logId;
      calBoot = oCalBoot(where(oCalBoot.setup == thisSetup)(1));

        /* Lookup current calibrator ids into original calibrator ids
           for this setup. It is mandatory because some ids may be missing
           for this baseline/triplet/observable. */
      calOrigLogId = *calBoot.origLogId;
      calBootLogId = *calBoot.logId; 
      origId = oiFitsGetId(logId, calOrigLogId);
      calBootLogId = calBootLogId(origId);
      
        /* look at the corresponding bootstrapped ids */
      bootId = oiFitsGetId(calBootLogId, logId);

        /* Boot may be zero if a calibrator is not found for this baseline.
         replace it by another one */
      zero = where(bootId == 0);
      if (numberof(zero))
      {
        bootId(zero) = 1;
      }

        /* finally do the bootstrap */
      out1cal(id) = oCal(*)(bootId);

        /* ... and set large error bars to not found (and spuriously replaced)
          calibrators so that it won't bias anything in the fits/calculations
          /etc. */
      for (j = 1; j <= numberof(zero); ++j)
      {
          
        if (oiFitsIsOiVis(out1cal))
        {
          out1cal(id)(j).visAmpErr = & array(1e+12, nwave);
          out1cal(id)(j).visAmpErr = & array(1e+12, nwave);
          continue;
        }
        if (oiFitsIsOiVis2(out1cal))
        {
          out1cal(id)(j).vis2Err = & array(1e+12, nwave);
          continue;
        }
        if (oiFitsIsOiT3(out1cal))
        {
          out1cal(id)(j).t3AmpErr = & array(1e+12, nwave);
          out1cal(id)(j).t3PhiErr = & array(1e+12, nwave);
          continue;
        }
        
      } /* for (j = 1; j <= numberof(zero); ++j) */
    
    } /* for (iSetup = 1; iSetup <= nSetup; ++iSetup) */

  } /* if (oiFitsIsOiVis(in1) || oiFitsIsOiT3(in1) || oiFitsIsOiVis2(in1)) */


}

              

func pbootGenerateCalibratorBootstrap(oiLog, oiVis, oiVis2, oiT3, oiDiam, 
  &oDiam, &oCalBoot, nboot=, funcSetup=)
/* DOCUMENT pbootGenerateCalibratorBootstrap(oiLog, oiVis, oiVis2, oiT3, oiDiam, 
  &oDiam, &oCalBoot, nboot=, funcSetup=)

Performs two things:
* For each bootstrap, the calibrators' diameters are taken randomly.
* For each instrumetal setup and each boostrap, calibrator observations are 
  chosen randomly (with repeats) from the actual list of calibrator 
  observations. 

*/ 
{

  yocoLogInfo, "pbootGenerateCalibratorBootstrap()";

  oCalBoot = [];

  if (is_void(nboot))
  {
    nboot = 5000;
  }

  visSetup = is_array(oiVis)? oiFitsDefaultSetup(oiVis, oiLog): [];
  vis2Setup = is_array(oiVis2)? oiFitsDefaultSetup(oiVis2, oiLog): [];
  t3Setup = is_array(oiT3)? oiFitsDefaultSetup(oiT3, oiLog): [];
  setups = unique(grow(visSetup, vis2Setup, t3Setup));
  nSetup = numberof(setups);

  /* For each instrumental setup, bootstrap the calibrator observations. */
 
  for (iSetup = 1; iSetup <= nSetup; ++iSetup)
  {

    local oVis, oVis2, oT3;

    setup = setups(iSetup);
    oiFitsCopyArrays, oiVis, oVis, oiVis2, oVis2, oiT3, oT3;
    oiFitsKeepSetup, oiLog, oVis, oVis2, oT3, setupList=[setup], 
      funcSetup=funcSetup;  
    
    id = []
    pData = [&oVis, &oVis2, &oT3];
    for (iData = 1; iData <= 3; ++iData)
    {
      data = *pData(iData);
      if (is_array(data)) 
      {
        cal = data(where(oiFitsGetIsCal(data, oiDiam)));
        grow, id, (is_array(cal)? cal.hdr.logId: []);
      }
    }
   
    if (numberof(id))
    {
      id = unique(id);
      boot = id(pbootGenerateIndexBootstrap(nscan=numberof(id), nboot=nboot));
      boot = boot(sort(boot,1));
    }
    else
    {
      id = [];
      boot = [];
    }
    cbt = pbootCalibratorBootstrap_t(setup=setup, logId=&boot, origLogId=&id);
    grow, oCalBoot, cbt;
  }

  /* For each boostrap, the diameter values are taken randomly */

  oDiam = array(oiDiam, nboot);
  err = gaussdev(dimsof(oDiam)) * oiDiam.diamErr;
  maxerr = .999 * oiDiam.diam;
  oDiam.diam += min(maxerr, max(-maxerr, err));

  yocoLogTrace, "pbootGenerateCalibratorBootstrap() done";

  
}

func pbootGetBootstrapSize(oiData, oiWave, &nwave, &nboot)
{

  if (is_array(oiData)) oiData = oiData(1);
  wave = *oiFitsGetOiWave(oiData, oiWave).effWave;
  
  if (structof(oiData) == struct_oiVis2_p) 
  {
    ndata = dimsof(*oiData.vis2Data)(2);
  }
  else if (structof(oiData) == struct_oiT3_p) 
  {
    ndata = dimsof(*oiData.t3Phi)(2);
  }
  else if (structof(oiData) == struct_oiVis_p)
  {
    ndata = dimsof(*oiData.visAmp)(2);
  }
  else
  {
    ndata = 0;
  }
  
  ndata = max(ndata, numberof(wave)); 
  
  nwave = numberof(unique(wave));
  nboot = ndata / nwave;

  return nboot;

}



/***************** pndrsPlot.i **********************************************/

func pbootSavePdf(win, org, name, autoRotation=)
{
  if (!pndrsBatchPlotLevel) return 1;
  /* no plots for V^2... */
  if (strpart(name, 1:3) == "env"
      || strpart(name, 1:3) == "psd"
      || strpart(name, 1:4) == "vis2"
      || strpart(name, 1:4) == "amp2"
      || strpart(name, 1:5) == "t3amp"
     )
  {
    return 1;
  }
  window, win;
    /* Add the counter if the call is done
         with two arguments */
  if (is_array(name) ) {
     pndrsPdfCounter++;
     name = org +swrite(format="_%03d_",pndrsPdfCounter) + name;
  } else {
     name = org;
  }
  if (strpart(name, -3:0) != ".pdf") {
    name += ".pdf";
  }
  pdf, name;

  return 1;
}

/*************************** pndrsSignal.i ***********************************/



func pbootSignalComputePhase(bisp, &phases, &phasesErr, gui=, square=)
/* DOCUMENT pbootSignalComputePhase(bisp, &phases, &phasesErr, 
            square=, gui=)
 *
 * Given bispectra "bisp" for each scan, compute "phases" for each scan 
 * selection given in "boot" and phase errors "phasesErr" using the 
 * dispersion across bootstraps.
 *
 */
{

  yocoLogInfo, "pbootSignalComputePhase()";
  
  extern pbootIndexBootstrap;
  boot = pbootIndexBootstrap;

  local bisp0;
  nlbd  = dimsof(bisp)(2);
  idm = nlbd / 2 + 1;
  nscan = dimsof(bisp)(3);
  nwin  = dimsof(bisp)(4);
  nboot = dimsof(boot)(3);
  if (is_void(square)) square = 1; 

  /* Init arrays with meaningless values */
  phases    = array(float(0.0), nlbd, nboot, nwin);
  phasesErr = array(float(1000.0), nlbd, nboot, nwin);
  
  /* Loop on baseline */
  for ( w=1 ; w<=nwin ;w++) 
  {

    /* Keep only valid scans (non zero), assuming all wavelength
       have the same selections */
    id = where(abs(bisp)(1,,w));
    nid = numberof(id);
    
    /* If not enough scans, reject the data */
    if (float(nid) / nscan < 0.25 && nid < 10) {
      yocoLogInfo,swrite(format="Base %i accepted scans: %3d (%d)    -  t3=rejected",w,nid,nscan);
      continue;
    }

    /* Compute the closure for each bootstrap, error is dispersion */
    bispb = bisp(,,w)(,boot)(,avg,);
    phases(,,w)    = oiFitsArg(bispb) * 180 / pi;
    phasesErr(,,w) = oiFitsArg(bispb * conj(bispb))(,rms)(,-) * 180 / pi;
    
    /* Verbose */
    yocoLogInfo,swrite(format="Base %i accepted scans: %3d (%d)    "+
                       "-  phi=%+7.2fdeg+-%5.1f  -  phi=%+7.2fdeg+-%5.1f",
                       w,nid,nscan,
                       phases(1,1,w),phasesErr(1,1,w),
                       phases(0,1,w),phasesErr(0,1,w));
    
  } /* end loop on baseline */ 

  /* Eventually plot */
  if (is_array(gui))
  {
    print, winkill;
    winkill,gui;
    winkill,gui+1;
    yocoNmCreate,gui,2,nwin/2,dx=0.01,dy=0.01;
    for (i=1;i<=nwin;i++) 
    {
      if (nlbd > 1) {

        yocoPlotPlpMulti, bisp(0,,i).im, bisp(0,,i).re, fill=0, tosys=i, color="blue";
        yocoPlotPlpMulti, bisp(1,,i).im, bisp(1,,i).re, fill=0, tosys=i, color="red";
      }
      yocoPlotPlpMulti, bisp(idm,,i).im, bisp(idm,,i).re, fill=0, tosys=i;
      limits, square=square;

      Max = max(abs(limits()(1:4)));
      tmp = 2.* Max * exp( 1.i*( phases(0,1,i) + phasesErr(0,1,i)*[-1,0,1] )*pi/180 )(-,) * [0,1];
      yocoPlotPlgMulti, tmp.im, tmp.re;
      tmp = 2.* Max * exp( 1.i*( phases(1,1,i) + phasesErr(1,1,i)*[-1,0,1] )*pi/180 )(-,) * [0,1];
      yocoPlotPlgMulti, tmp.im, tmp.re, color="red";
      limits,-Max,Max,-Max,square=1;
      gridxy,2,2;
    }
    yocoNmCreate,gui+1;
  }
  
  yocoLogTrace, "pbootSignalComputePhase done";

  return 1;

}



func pbootSignalComputeAmp2Ratio(amp2, norm2, &vis2, &vis2Err, gui=, square=)
/* DOCUMENT pbootSignalComputeAmp2Ratio(amp2, norm2, &vis2, &vis2Err,
 *                                                     gui=, square=)
 *
 * Given square amplitude and square normalization for each scan, compute
 * square visibility amplitudes "vis2" for each scan selection given in 
 * "boot" and errors "vis2Err" using the dispersion across bootstraps.
 *
 */
{

  yocoLogInfo, "pbootSignalComputeAmp2Ratio()";
  
  extern pbootIndexBootstrap;
  boot = pbootIndexBootstrap;
   
  nlbd  = dimsof(amp2)(2);
  idm   = nlbd / 2;
  nscan = dimsof(amp2)(3);
  nwin  = dimsof(amp2)(4);
  nboot = dimsof(boot)(3);
  if (is_void(square)) square = 1;

  /* Init arrays with meaningless values */
  vis2    = array(float(1.0),  nlbd, nboot, nwin);
  vis2Err = array(float(100.0), nlbd, nboot, nwin);
  
  
  /* Loop on baseline */
  for ( w=1 ; w<=nwin ;w++) {
    amp2b  = amp2(,,w);
    norm2b = norm2(,,w);
    valid = amp2b(1,) != 0 & norm2b(1,) != 0;
    id = where(valid);
    invalidid = where(!valid);
    nid = numberof(id);
    if (numberof(invalidid))
    {
      amp2b(,invalidid) = 1e-50; 
      norm2b(,invalidid) = 1e-52;
    }

    /* If not enough scans, reject the data */
    if (float(nid) / nscan < 0.25 && nid < 10) {
      yocoLogInfo,swrite(format="Base %i accepted scans: %3d (%d)    -  v2=rejected",w,nid,nscan);
      continue;
    }

    /* Average over the boot and compute the vis2 */
    vis2b  = amp2b(,boot)(,sum,) / norm2b(,boot)(,sum,);
    vis2Err(,,w) = vis2b(,rms);
    vis2(,,w)    = vis2b;

    /* Verbose */
    yocoLogInfo,swrite(format="Base %i accepted scans: %3d (%d)    "+
                       "-  v2=%5.1f%%+-%4.1f  -  v2=%5.1f%%+-%4.1f",
                       w,nid,nscan,
                       vis2(1,1,w)*100, vis2Err(1,1,w)*100,
                       vis2(0,1,w)*100, vis2Err(0,1,w)*100);
  }
  /* end loop on base */
  
  /* Eventually plot */
  if ( is_array(gui) )
  {
    print, winkill;
    winkill,gui;
    yocoNmCreate,gui,2,nwin/2,dx=0.01,dy=0.01;
    for (i=1;i<=nwin;i++) 
    {
      if (nlbd > 1) {
        yocoPlotPlpMulti, amp2(0,,i), norm2(0,,i), fill=0, tosys=i,color="blue";
        yocoPlotPlpMulti, amp2(1,,i), norm2(1,,i), fill=0, tosys=i,color="red";
      }
      yocoPlotPlpMulti, amp2(idm,,i), norm2(idm,,i), fill=0, tosys=i;
      if (square) limits, square=1; else limits;
      Max = max(abs(limits()(1:4)));
      tmp = span(0,2.*Max,2) * (vis2(0,i)!=0);
      yocoPlotPlgMulti, tmp * (vis2(0,1,i) + vis2Err(0,1,i)*[-1,0,1])(-,), tmp;
      yocoPlotPlgMulti, tmp * (vis2(1,1,i) + vis2Err(1,1,i)*[-1,0,1])(-,), tmp, color="red";
      limits,-Max/5,Max,-Max/5,square=1;
      gridxy,2,2;
    }
  }

  yocoLogTrace, "pbootSignalComputeAmp2Ratio done";

  return 1;
}



          


/************************** pndrsBatch.i *************************************/


func pbootComputeSingleOiData(&oiVis2, &oiT3, &oiVis, &oiWave,
                        &oiLog, &oiArray, &oiTarget,
                        mode=, outputFile=,
                        inspect=, dpi=,
                        outputOiDataFile=,
                        snrThreshold=,
                        inputCatalogFile=,
                        checkPhase=,
                        inputScienceFile=,
                        inputDarkFile=,
                        inputMatrixFile=,
                        inputSpecCalFile=,
                        inputScriptFile=)
{

  pndrsScanComputePolarDiffPhases = pbootScanComputePolarDiffPhases;
  pndrsScanComputeClosurePhases = pbootScanComputeClosurePhases;
  pndrsScanComputeAmp2PerScan = pbootScanComputeAmp2PerScan;
  pndrsScanComputeAmp2PerScanAbcd = pbootScanComputeAmp2PerScanAbcd;
  oiFitsCalibrateDiam = pbootCalibrateDiam;
  oiFitsCalibrateDiamOiT3 = pbootsCalibrateDiamOiT3;

  pbootIndexBootstrap = pbootGenerateIndexBootstrap(nboot=nboot, nscan=nscan);

  ok = pbootPndrsComputeSingleOiData(oiVis2, oiT3, oiVis, oiWave, 
        oiLog, oiArra, oiTarget,
        mode=mode, outputFile=outputFile, inspect=0, 
        inputScienceFile=inputScienceFile, inputDarkFile=inputDarkFile,
        inputMatrixFile=inputMatrixFile, inputSpecCalFile=inputSpecCalFile,
        inputScriptFile=inputScriptFile, inputCatalogFile=inputCatalogFile,
        snrThreshold=snrThreshold, outputOiDataFile=outputOiDataFile,
        checkPhase=checkPhase);

  return ok;

}


/******************** pndrsProcess.i *****************************************/

func pbootScanComputePolarDiffPhases(imgData, imgLog, &oiVis, &phases, &phasesErr, gui=)
{
  yocoLogInfo, "pbootScanComputePolarDiffPhases()";

  extern pbootIndexBootstrap;
  pndrsSignalComputePhase = pbootSignalComputePhase;
  oiFitsSetDataArray = pbootSetDataArray;
  ok = pbootPndrsScanComputePolarDiffPhases(imgData, imgLog, oiVis, 
              phases, phasesErr);
  
  return ok;

  yocoLogTrace, "pbootScanComputePolarDiffPhases done";

}

func pbootScanComputeAmp2DirectSpace(coherData, imgLog, filtIn, filtOut, norm, &oiVis2, &amp2, &amp2Err, gui=)
{

  error, "Direct space bootstrap not implemented";

}

func pbootScanComputeAmp2PerScan(coherData, darkData, &imgLog, norm2, pos, 
  &oiVis2, &amp2, &amp2Err, gui=)
{
  
  yocoLogInfo, "pbootScanComputeAmp2PerScan()";

  extern pbootIndexBootstrap;

  pndrsSignalComputeAmp2Ratio = pbootSignalComputeAmp2Ratio;
  oiFitsSetDataArray = pbootSetDataArray;

  ok = pbootPndrsScanComputeAmp2PerScan(coherData, darkData, imgLog, norm2, pos,
          oiVis2, amp2, amp2Err);

  if (ok && gui)
  {
    yocoLogWarning, "Plots for bootstrapped V^2 not implemented.";
  }

  yocoLogTrace, "pbootScanComputeAmp2PerScan done";

  return ok;
 
}

func pbootScanComputeAmp2PerScanAbcd(coherData, &imgLog, norm2, pos, 
  &oiVis2, &amp2, &amp2Err, gui=)
{
  
  yocoLogInfo, "pbootScanComputeAmp2PerScanAbcd()";

  extern pbootIndexBootstrap;
  

  pndrsSignalComputeAmp2Ratio = pbootSignalComputeAmp2Ratio;
  oiFitsSetDataArray = pbootSetDataArray;

  if (ok && gui)
  {
    print, winkill;
    yocoNmCreate, gui;
    yocoNmCreate, gui+1;
  }
  ok = pbootPndrsScanComputeAmp2PerScanAbcd(coherData, imgLog, norm2, pos,
          oiVis2, amp2, amp2Err);

  if (ok && gui)
  {
    print, winkill;
    yocoNmCreate, gui;
    yocoNmCreate, gui+1;
    yocoLogWarning, "Plots for bootstrapped V^2 not implemented.";
  }

  yocoLogTrace, "pbootScanComputeAmp2PerScanAbcd done";

  return ok;
 
}





/******************** OiFitsUtils.i *****************************************/

func pbootFitsReduceDim(x)
{
  if (dimsof(x)(0) > 2)
  {
    return x(*,);
  }
  return x;
}

func pbootSetDataArray(&oiData, i, amp, ampErr, phi, phiErr, flag, nowrap=)
{
  
  amp = pbootFitsReduceDim(amp);
  ampErr = pbootFitsReduceDim(ampErr);
  phi = pbootFitsReduceDim(phi);
  phiErr = pbootFitsReduceDim(phiErr);
  flag = pbootFitsReduceDim(flag); 
  return pbootOiFitsSetDataArray(oiData, i, amp, ampErr, phi, phiErr, flag, 
                         nowrap=nowrap);

}



func pbootCleanDataFromDummy(&oiVis2, &oiT3, &oiVis, maxVis2Err=, maxT3PhiErr=,
  remove=)
{
  yocoLogInfo,"pbootCleanDataFromDummy()";
  
  local amp, ampErr, phi, phiErr, flag;

  oiFitsGetBaseLength = pbootGetBaseLength;
  ok = pbootOiFitsCleanDataFromDummy(oiVis2, oiT3, oiVis, 
    maxVis2Err=maxVis2Err, maxT3PhiErr=maxT3PhiErr);

  yocoLogTrace, "pbootCleanDataFromDummy done";
 
  return ok;

}

func pbootGetBaseLength(oiData)
/* Avoid 0 baseline and elimination of data point */
{

  return pbootOiFitsGetBaseLength(oiData) + 1e-10;

}
 



func pbootGetSetupBase(oiData, oiLog, oiArray, funcSetup=, perPointing=)
/* DOCUMENT  pbootGetSetupBase(oiData, oiLog, oiArray)

Returns the instrument setup + baseline/triplet as a string.

*/
{
  if (is_void(perPointing))
  {
    perPointing = 1;
  }
  setup = pbootGetSetup(oiData, oiLog, perPointing=perPointing);
  setup += " - " + oiFitsGetBaseName(oiData, oiArray);
  return setup;

}


func pbootWriteFile(outputFile, oiTarget, oiWave, oiArray, oiVis2, oiVis, oiT3, oiLog, overwrite=, funcLog=)
/* DOCUMENT pbootWriteFile(outputFile, oiTarget, oiWave, oiArray, 
        oiVis2, oiVis, oiT3, oiLog, overwrite=, funcLog=)

Same as oiFitsWriteFile but takes into account that bootstrapped data
might be present.  

SEE ALSO

oiFitsWriteFile

*/
{
  
  yocoLogInfo, "pbootWriteFile ()";
  
  local oTarget, oWave, oArray, oVis2, oVis, oT3, oLog;

  oiFitsCopyArrays, oiTarget, oTarget, oiWave, oWave, oiArray, oArray,
                    oiVis2, oVis2, oiVis, oVis, oiT3, oT3, oiLog, oLog;
  oiFitsCleanUnused, oTarget, oWave, oArray, oVis2, oVis, oT3, oLog;

  for (iWave = 1; iWave <= numberof(oWave); ++iWave)
  {
    local nboot, nwave;
    w = oWave(iWave);
    insName = w.hdr.insName; 
    vis2 = oiVis2(where(oiVis2.hdr.insName == insName))(1);
    pbootGetBootstrapSize, vis2, w, nwave, nboot;
    if (numberof(*w.effWave) == nwave) // include boostraps
    {
      l = (*w.effWave)(,-:1:nboot)(*);
      dl = (*w.effBand)(,-:1:nboot)(*);
      oWave(iWave) = struct_oiWavelength_p(hdr=w.hdr, effWave=&l, effBand=&dl);
    }
  }
  
  pbootOiFitsWriteFile, outputFile, oTarget, oWave, oArray,
      oVis2, oVis, oT3, oLog, overwrite=overwrite, funcLog=funcLog;

  yocoLogTrace, "pbootWriteFile() done";

}

func pbootApplyTf(oiDataRaw, oiDataTfp, oiArray, oiLog, 
                   &oiDataCal, &oiDataTfe,
                   funcSetup=, tdelta=, tfMode=, errMode=, onTime=,
                   minAmpErr=, minPhiErr=, param=)
/* DOCUMENT pbootApplyTf(oiDataRaw, oiDataTfp, oiArray, oiLog,
                          &oiDataCal, &oiDataTfe,
                          funcSetup=, tdelta=, tfMode=, errMode=, onTime=,
                          minAmpErr=, minPhiErr=, param=)

Convert 'oiDataRaw' into 'oiDataCal' by interpolating the transfer-function
points 'oiDataTfp'. This is done setup-by-setup, to ensure the oiDataRaw are
calibrated by the TF estimation obtained with same setup.

INPUT

- oiDataRaw, oiDataTfp: raw oiData and TF data points
- oiArray, oiLog:

OUTPUT

- &oiDataCal: calibrated oiData
- &oiDataTfe: interpolation of the Transfer-function over the
 period of time spaned by the data set. Usefull for plot and
 check purposes. If onTime=1, then this value are only provided
 for the same mjdtime as oiDataRaw.

KEY WORDS

- tfMode=0: interp. closest TF-points
      1: smooth-length     (>1pt)
      2: average           (>1pt)  (default)
      3: linear fit        (>2pt)
      4: quadratic fit     (>3pt)
      5: 1+b.cos2(alt+az)  (>3pt)
      6: 1+a cos(alt) + b*cos(alt) + ... (>5 pt)
      If the number of points is not enought, then
      the TF is averaged only.

 5: need the following functions to be defined:
    oiFitsGetAltAz(oiData, oiLog)
      
- tdelta= time-step for computing the output oiDataTfe.
 This has no influence on the calibrated oiDataCal since
 oiDataTfe is only computed for plot/check purpose.
 
- funcSetup= (optional) temporary override 'oiFitsDefaultSetup'
 see the help of 'oiFitsDefaultSetup' for more information

- minAmpErr= (optional) the relative error of Amp is forced
 to be larger (or equal) than this value. Default is 0.1  (10%).

- minPhiErr= (optional) same as previous but for the absolute
 error on the phase (default is 0.1deg).

FIXME:

- better handle the correlated errors !
- maximum accuracy of vis2Tf is fixed to +/-1% (AMBER perfo) !

SEE ALSO

pboot

 */
{
  yocoLogInfo, "pbootApplyTf (" + string(oiFitsStructRoot(oiDataRaw)) + ")";
  
  /* Local variable and clean outputs */
  local ampRaw, dampRaw, phiRaw, dphiRaw, flagRaw,
        ampTfo, dampTfo, phiTfo, dphiTfo, flagTfe,
        ampTfe, dampTfe, phiTfe, dphiTfe, flagTfe;
  
  /* Check arguments */
  if (is_void(funcSetup)) funcSetup = oiFitsDefaultSetup;
  if (is_void(tdelta)) tdelta = 1./288.;
  if (is_void(tfMode)) tfMode = 2;
  if (is_void(minAmpErr)) minAmpErr = 0.02;
  if (is_void(minPhiErr)) minPhiErr = 0.1;
  if (!oiFitsIsOiData(oiDataRaw)) return yocoError("oiDataRaw not valid");
  if (!oiFitsIsOiData(oiDataTfp)) return yocoError("oiDataTfp not valid");
  if (!oiFitsIsOiArray(oiArray)) return yocoError("oiArray not valid");

  /* Prepare output and sort */
  oiDataCal = oiDataTfe = [];
  oiDataRaw = oiDataRaw(sort(oiDataRaw.mjd));
  oiDataTfp = oiDataTfp(sort(oiDataTfp.mjd));

  
  /* Found the different setup, add the baseline so discriminate them */
  setupTfp = pbootGetSetupBase(oiDataTfp, oiLog, oiArray, funcSetup=funcSetup);
  setupRaw = pbootGetSetupBase(oiDataRaw, oiLog, oiArray, funcSetup=funcSetup);
  setups = unique(setupRaw);
  nSetup = numberof(setups);
  print, setups;
  /* --- Loop on the setup, to calibrate them individually */
  for (iSetup = 1; iSetup <= nSetup; iSetup++) {

    /* Found the TF and Obs estimations for this setup */    
    thisSetup = setups(iSetup);
    oDataTfp = oiDataTfp(where(setupTfp == thisSetup));
    oDataRaw = oiDataRaw(where(setupRaw == thisSetup));
    nTfp = numberof(oDataTfp);

    /* Some verbose information in case the setup is uncalibratable
       but contains some real information */
    if (nTfp == 0) {
      yocoLogWarning,"No TF estimation for setup " + thisSetup;
      mjd = pr1(oDataRaw.mjd(avg));
      id = pr1(oDataRaw.targetId);
      yocoLogInfo, "Setup has <mjd>=" + mjd + " and target IDs " + id;
      continue;
    } else {
      yocoLogTest, "Found TF-points for setup "+ thisSetup;
    }

    /* Prepare the output of calibrated data and tf-estimation data */
    oDataTfe = oDataRaw(*);
    oDataTfi = oDataRaw(*);

    /* Enlarge oDataTfe, larger than the time spanned by the data, so that the
       display is smooth */
    if (!onTime) 
    {
      nTfe = long(abs(oDataRaw.mjd(ptp)) / tdelta) + 2;
      grow, oDataTfe, array(oDataTfe(1), nTfe);
      mjdmin = min(oDataRaw.mjd) - tdelta;
      mjdmax = max(oDataRaw.mjd) + tdelta;
      oDataTfe(-nTfe+1:0).mjd = span(mjdmin, mjdmax, nTfe);
    }


    /* We consider a maximum accuracy on the TF measurement */
    oiFitsGetData, oDataTfp, ampTfp, dampTfp, phiTfp, dphiTfp, flagTfp, 1;    
    dampTfp = max(dampTfp, abs(minAmpErr * ampTfp));
    dphiTfp = max(dphiTfp, minPhiErr);
    oiFitsSetDataArray, oDataTfp, ,ampTfp, dampTfp, phiTfp, dphiTfp, flagTfp;

    /* Compute and interpolate the TF with an interpolation function */
    tfFunc = pbootGetTfFunc(tfMode, npt=nTfp, ncal=nTfp / 5);
    tfFunc, oDataTfp, oDataTfi, oiLog, errMode=errMode, param=param;
    tfFunc, oDataTfp, oDataTfe, oiLog, errMode=errMode, param=param;

    /* Get the TF values */
    oiFitsGetData, oDataRaw, ampRaw, dampRaw, phiRaw, dphiRaw, flagRaw, 1;
    oiFitsGetData, oDataTfi, ampTfi, dampTfi, phiTfi, dphiTfi, flagTfe, 1;
    oiFitsGetData, oDataTfe, ampTfe, dampTfe, phiTfe, dphiTfe, flagTfe, 1;
    
    /* Avoid null values in the TF */
    ampTfi  += 1e-10;
    dampTfi += 1e-10;
    ampRaw  += 1e-10;
    
    /* We consider a maximum accuracy on the calibration */
    dampTfe = max(dampTfe, abs(minAmpErr * ampTfe));
    dampTfi = max(dampTfi, abs(minAmpErr * ampTfi));
    dphiTfe = max(dphiTfe, minPhiErr);
    dphiTfi = max(dphiTfi, minPhiErr);
    
    /* Now apply the TF and propagate it to Amplitude data */
    ampCal  = ampRaw / ampTfi;
    dampCal = abs(dampRaw / ampRaw, dampTfi / ampTfi) * abs(ampCal);

    /* Now apply the TF and propagate it to Phase data */
    phiCal  = phiRaw - phiTfi;
    dphiCal = abs(dphiRaw, dphiTfi);

    /* FIXME: flag is not handled */
    flagCal = array(char(0), dimsof(ampCal));
    flagTfe = array(char(0), dimsof(ampTfe));
    
    /* Put the data back */
    oDataCal = oDataTfi(*);
    oiFitsSetDataArray, oDataCal, , ampCal, dampCal, phiCal, dphiCal, flagCal;
    oiFitsSetDataArray, oDataTfe, , ampTfe, dampTfe, phiTfe, dphiTfe, flagTfe;

    /* Ack for EXOZODI */
    if (oiFitsStrHasMembers(oDataTfi, "vis2ErrSys") ) {
      statTi = oiFitsGetStructData(oDataTfi, "vis2ErrSys", 1);
      statCal = statTi / ampTfi * ampCal;
      oiFitsSetStructDataArray, oDataCal, ,"vis2ErrSys", statCal;
    }

    /* Grow the output arrays */
    grow, oiDataCal, oDataCal;
    grow, oiDataTfe, oDataTfe;
    
  } /* --- End loop on setup */

  /* Sort by time */
  oiDataCal = oiDataCal(sort(oiDataCal.mjd));
  oiDataTfe = oiDataTfe(sort(oiDataTfe.mjd));

  yocoLogTrace, "pbootApplyTf done";
  return 1;

}


func pbootCalibrateNight(oiVis2, oiT3, oiVis, 
                         oiVis2Calboot, oiT3Calboot, oiVisCalBoot,
                         oiWave, oiArray, oiTarget, oiLog, oiDiam,
                          &oiVis2Cal, &oiVisCal, &oiT3Cal,
                          &oiVis2Tfp,&oiVisTfp,&oiT3Tfp,
                          &oiVis2Tfe,&oiVisTfe,&oiT3Tfe,
                          vis2TfMode=, t3TfMode=, visTfMode=,
                          vis2TfErrMode=,t3TfErrMode=,visTfErrMode=,
                          vis2TfParam=,t3TfParam=,visTfParam=,
                          overwrite=, fileRoot=)
/* DOCUMENT pbootCalibrateNight(oiVis2, oiT3, oiVis, 
                         oiVis2Calboot, oiT3Calboot, oiVisCalBoot,
                          oiWave, oiArray, oiTarget, oiLog, oiDiam,
                          &oiVis2Cal, &oiVisCal, &oiT3Cal,
                          &oiVis2Tfe, &oiVisTfe, &oiT3Tfe,
                          vis2TfMode=, t3TfMode=, visTfMode=,
                          overwrite=, fileRoot=)

Calibrate from the transfer function of the night by computing the TF
on the calibrators and removing it from all data (from both science and cal)

Execute the following functions on oiVis2, oiT3, and oiVis:
- oiFitsExtractTf    : extract the TF-estimations
- oiFitsApplyTf      : interpolate TF-estimations and calibrate the data
                      (opt: tfMode)

The calibrated oiData are stores into proper OIFITS files, one
per 'insName' setup during the night, using oiFitsWriteFiles
(opt: fileRoot, overwrite).

Note that points obtained on the calibration stars are also calibrated
and written into the file. So that: if the calibration star is unresolved,
its vis2 is ~1, but if the calibration star is resolved its vis2 is <1.

PARAMETERS
- vis2TfMode, t3TfMode, visTfMode : see oiFitsApplyTf

- fileRoot, overwrite : files where to write the calibrated data
 this file is eventually overwriten if overwrite=1

SEE ALSO

pboot
oiFitsCalibrateNight

*/
{
  yocoLogInfo,"pbootCalibrateNight()";
  yocoLogInfo," dimsof(oiVis2)="+pr1(dimsof(oiVis2));
  yocoLogInfo," dimsof(oiVis) ="+pr1(dimsof(oiVis));
  yocoLogInfo," dimsof(oiT3)  ="+pr1(dimsof(oiT3));

  /* Init and reset outputs */
  oiT3Tfp = oiT3Tfa = oiT3Cal = oiT3Tfe =[];
  oiVis2Tfp = oiVis2Tfa = oiVis2Cal = oiVis2Tfe = [];
  oiVisTfp = oiVisTfa = oiVisCal = oiVisTfe = [];

  /* Check parameter */
  if( !oiFitsIsOiArray(oiArray) )   return yocoError("oiArray not valid");
  if( !oiFitsIsOiWave(oiWave) )     return yocoError("oiWave not valid");
  if( !oiFitsIsOiTarget(oiTarget) ) return yocoError("oiTarget not valid");
  if( !oiFitsIsOiLog(oiLog) )       return yocoError("oiLog not valid");
  if( !oiFitsIsOiDiam(oiDiam) )     return yocoError("oiDiam not valid");

  /* Default parameters */
  if ( is_void(fileRoot))  fileRoot=0;
  if ( is_void(overwrite)) overwrite=1;     // overwrite
  
  /* Extract the TF estimation over the night,
     Group them if possible, and calibrate the night */
  if (is_array(oiT3)) {
    if( !oiFitsIsOiT3(oiT3) ) return yocoError("oiT3 not valid");
    oiFitsExtractTf, oiT3Calboot, oiWave, oiDiam, oiT3Tfp;
    pbootApplyTf, oiT3, oiT3Tfp, oiArray, oiLog, oiT3Cal, oiT3Tfe,
      tfMode=t3TfMode, errMode=t3TfErrMode, param=t3TfParam;
  }
  
  /* Extract the TF estimation over the night */
  if (is_array(oiVis2)) {
    if( !oiFitsIsOiVis2(oiVis2) ) return yocoError("oiVis2 not valid");
    oiFitsExtractTf, oiVis2Calboot, oiWave, oiDiam, oiVis2Tfp;
    pbootApplyTf, oiVis2, oiVis2Tfp, oiArray, oiLog, oiVis2Cal, oiVis2Tfe,
      tfMode=vis2TfMode, errMode=vis2TfErrMode, param=vis2TfParam;
  }

  /* Extract the TF estimation over the night */
  if (is_array(oiVis)) {
    if( !oiFitsIsOiVis(oiVis) ) return yocoError("oiVis not valid");
    oiFitsExtractTf, oiVisCalboot, oiWave, oiDiam, oiVisTfp;
    pbootApplyTf, oiVis, oiVisTfp, oiArray, oiLog, oiVisCal, oiVisTfe,
      tfMode=visTfMode, errMode=visTfErrMode, param=visTfParam;
  }
  
  /* Write the calibrated OIFITS */
  if ( fileRoot!=0 ) {
    oiFitsWriteFiles, fileRoot, oiTarget, oiWave, oiArray, oiVis2Cal,
      oiVisCal, oiT3Cal, oiLog, overwrite=overwrite;
  }
  
  yocoLogTrace,"pbootCalibrateNight() done";
  return 1;

}


/****************************** a few fixes ********************************/



func pbootListId(id, ref)
/* DOCUMENT pbootListId(id, ref)

Like original (yocoListId) but doesn't blow for large data sets

*/
{
    local out, i;

    if (is_void(ref) || is_void(id))
        error, "yocoListId takes exactly 2 parameters";

    out = array(0, dimsof(id));
    for (i = 1; i <= numberof(id); ++i)
    {
      idx = where(ref == id(i));
      if (numberof(idx))
      {
        out(i) = idx(0);
      }
    }
    /* print, "yocolistid: ", numberof(ref), numberof(id), out; */
    return out;
}

func pbootGetSetup(oiData, oiLog, perPointing=, funcSetup=)
/* pbootGetSetup(oiData, oiLog)

Unique set up ID includes Julian Day Number to help reduce multi-night
data.
 
 */
{

  date = int(oiData.mjd + 0.5);
  setup = swrite(format="%s_MJD=%i", pndrsGetSetup(oiData, oiLog), date);
  
  /* dirty hack */
  if (perPointing)
  {
    obsname = oiLog.obsName(oiData.hdr.logId);
    sel = strword(obsname, "_", 2)(1:2,);
    pointing = strpart(obsname, sel);
    setup += "_" + pointing; 
  }

  return setup
}




/*************** alt-az polarisation + smoothing  length *********************/


func pbootGetTfFunc(tfMode, npt=, ncal=)
/* DOCUMENT pbootGetTfFunc(tfMode, npt=, ncal=)

Return the TF interpolation function. tfMode gives what is wanted, but
if the number of calibrator observations (npt) or calibrators (ncal) is
not enought, revert to a simpler TF model.

SEE ALSO

pboot

*/
{

  if (is_func(tfMode))
  {
    return tfMode;
  }
  if (is_void(tfMode))
  {
    tfMode = 1;
  }

  /* As a security measure, we ask for at least 2/3 calibrators
     for alt-az correction */
 
  if (tfMode == 6 || tfMode == "altazfree")
  {
    if (npt >= 5 && ncal >= 3)
    {
      return pbootTfSmoothAltAzFit;
    }
    tfMode = 1;
  }

  if (tfMode == 5 || tfMode == "altaz")
  {
    if (npt >= 3 && ncal >= 2)
    {
      return oiFitsTfAltAzFit;
    }
    tfMode = 2;
  }
  
  if (tfMode == 4 || tfMode == "quadratic")
  {
    if (npt >= 4)
    {
      return oiFitsTfQuadraticFit;
    }
    tfMode = 3;
  }

  if (tfMode == 3 || tfMode == "linear")
  {
    if (npt >= 3)
    {
      return oiFitsTfLinearFit;
    }
    tfMode = 2;
  }

  if (tfMode == 0 || tfMode == "interp") 
  {
    if (npt >= 2)
    {
      return oiFitsTfInterp;
    }
    tfMode = 2;
  }

  if (tfMode == 1 || tfMode == "smooth")
  {
    if (npt >= 2)
    {
      return oiFitsTfSmoothLength;
    }
    tfMode = 2;
  }

  return oiFitsTfAverage;
  
}




/************************ Top-level ******************************************/

func pbootReduceDirectory(inputDir, mode=, overwrite=, overwrite_cal=,
 inspect=, silent=, inputScriptFile=, nboot=)
/* DOCUMENT pbootReduceDirectory(inputDir, mode=, overwrite=, 
  inspect=, silent=, inputScriptFile=, nboot=)

Reduce PIONIER scans and produce OI FITS files. 

SEE ALSO

pboot
pbootCalibrateDirectory
 
 */
{
  yocoLogInfo, "pbootReduceDirectory()";

  /* bootstrap */ 
  local pbootIndexBootstrap, pbootNboot;
  if (is_void(nboot))
  {
    nboot = 5000;
  }
  pbootNboot = nboot;

  if (silent) { 
    pndrsBatchMakeSilentGraphics,1;
  }

  
  /* Overwrite some pndrs functions */
  pndrsComputeSingleOiData = pbootComputeSingleOiData;
  pndrsSavePdf = pbootSavePdf; 
  oiFitsDefaultSetup = pbootGetSetup;
  yocoListId = pbootListId;
  oiFitsWriteFile = pbootWriteFile;

  pndrsComputeAllMatrix, inputDir=inputDir, overwrite=args.owcal;
  pndrsComputeAllSpecCal, inputDir=inputDir, overwrite=args.owval;

  ok = pndrsComputeAllOiData(inputDir=inputDir, mode=mode, inspect=inspect,
    overwrite=overwrite, inputScriptFile=inputScriptFile);
  
  yocoLogTrace, "pbootReduceDirectory() done";

  return ok;

}


func pbootCalibrateDirectory(inputDir, catalogsDir=, scriptFile=,rmPdf=,
  maxbootplot=, vis2TfMode=, t3TfMode=, visTfMode=, Xaxis=, 
  plot_calibrated=, bootmin=, bootmax=)
/* DOCUMENT pbootCalibrateDirectory(inputDir, catalogsDir=, 
   scriptFile=,rmPdf=, nboot_plot=, vis2TfMode=, t3TfMode=, visTfMode=, Xaxis=,
    bootmin=, bootmax=)

Calibrate the OI FITS data.
- PDF files with the transfer function plot
- an OIFITS file with all calibrated data (science and calibrators)
- several OIFITS, one per science target, with calibrated data.

PARAMETERS

- inputDir: directory where the raw data are.

- catalogsDir: directory where catalogs (calibrators) can be found.

- rmPdf: whether to remove existing PDF from the calibrated data
directory. True by default.

- vis2TfMode, t3TfMode, visTfmode: interpolation modes for the
TF estimation at science points.  By defaults it's smoothing length
if I remember well.  Values can be: "smooth", "altaz", "average",
"linear", "quadratic", "altazfree", "interp". 

- maxbootplot: maximum of bootstraps to plot. 5 by default.

- scriptFile is the (optional) name of the yorick script
that will be executed by the function just before calling
oiFitsCalibrateNight. It can be used to make some filtering,
grouping, removing part of the night... or any advance
operation on the structures oiVis2, oiT3, oiVis, oiArray and
oiTarget. If scriptFile is not defined, the function will load
"pndrsScript.i" if it exists.

- Xaxis: x-axis for the TF plots, by default ["altaz", "mjd"], i.e.
both dependence in alt + az and time are plotted.
   
SEE ALSO

pboot
pbootReduceDirectory

*/
{

  yocoLogInfo,"pbootCalibrateDirectory()";
  local oiTarget, oiWave, oiArray, oiVis2, oiVis, oiT3, oiLog,
        nboot, oiDiamBoot, oiCalBoot; 

  /* overwrite some pndrs functions for this one to work well */ 
  pndrsSavePdf = pbootSavePdf; 
  oiFitsDefaultSetup = pbootGetSetup;
  yocoListId = pbootListId;

  /* Only plot the first bootstraps. By default, plot data against
     time and alt+az. */
  if (is_void(maxbootplot))
  {
    maxbootplot = 5;
  }
  if (is_void(Xaxis))
  {
    Xaxis = ["mjd", "altaz"];
  }

  /* Default */
  maxVis2Err  = 0.25;
  maxT3PhiErr = 20.0;
  Avg0 = [[1.641,1.71233],[2.04,2.06]];
  
  /* Check the argument and go into this directory */
  if ( !pndrsCheckDirectory(inputDir,2) ) {
     yocoError,"Check argument of pndrsCalibrateAllOiData";
    return 0;
  }
  
  /* Find a root for the files that are created. Take the part of the
     input directory name that is before _ (if any) */
  strRoot0 = yocoStrSplit(yocoFileSplitName(strpart(inputDir,:-1)),"_")(1);
  strRoot  = inputDir+"/"+strRoot0;

  /* Default for file with calibration stars */
  calibFile = strRoot+"_oiDiam.fits";
  
  /* Set a logging file */
  yocoLogSetFile, strRoot+"_log.txt", overwrite=1;

  /* Load all the OIDATA files. Actually would be better
     to use the DPR.CATG parameters but they are not defined
     by pndrsComputeOiData.
     
     IMPORTANT: wavelengths should never be sorted, or bootstrapping will
     be messed up. Hence clean=0. Partial cleaning can be perform, but careful
     to what the cleaning functions do!
      */
  oiFitsLoadFiles,inputDir+"/P*oidata.fits", oiTarget, oiWave, oiArray,
    oiVis2, oiVis, oiT3, oiLog,
    shell=1, readMode=-1, clean=0;
  oiFitsClean, oiTarget, oiWave, oiArray, oiVis2, oiVis, oiT3, oiLog;

  /* Check if observations have been loaded */
  if ( !is_array(oiTarget) ) {
    yocoLogWarning,"No reduced data (oidata.fits).";
    yocoLogSetFile;
    return 1;
  }

  /* Some cleaning */
  pbootCleanDataFromDummy, oiVis2, oiT3, oiVis,
    maxVis2Err=maxVis2Err, maxT3PhiErr=maxT3PhiErr, remove=0;

  /* This is a hack to deal with the fact that the star names may have changed:
     update the names in oiTarget */
  oiTarget.target = pndrsCheckTargetInList(oiTarget.target);
  
  /* This is a hack to deal with the fact that the star names may have changed:
     Open the calibFile, update names and re-write it... */
  if ( open(calibFile, "r", 1) )
  {
    oiFitsLoadOiDiam, calibFile, oiDiam;
    oiDiam.target = pndrsCheckTargetInList(oiDiam.target);
    oiFitsWriteOiDiam, calibFile, oiDiam, overwrite=1;
  }

  /* This is a hack to fix missing LST values (when communication with
     the VLTI control software is lost) */
  pndrsFixMissingIss, oiLog, oiVis, oiVis2, oiT3;

  /* Search for the diameters */
  oiFitsLoadOiDiamFromCatalogs, oiTarget, oiDiam, oiDiamFile=calibFile, 
    overwrite=0, catalogsDir=catalogsDir;

  /* Eventually remove PDF */
  if (rmPdf==1) {
    yocoLogInfo,"Remove files from previous calibration (except oiDiam.fits)";
    system, "rm -rf *_SCI_* *_CAL_* *_TF_* *ummary*txt > /dev/null 2>&1";
  }

  /* Write a full summary of the night */
  oiFitsListObs, oiVis2, oiTarget, oiArray, oiLog, oiDiam,
    dtime=1e-6, file=strRoot+"_summaryFull.txt",filename=1;
  
  /* Write a short summary of the night */
  oiFitsListObs, oiVis2, oiTarget, oiArray, oiLog, oiDiam,
    file=strRoot+"_summaryShort.txt",filename=1;
  
  /* Write a very short summary of the night */
  oiFitsListObs, oiVis2, oiTarget, oiArray, oiLog, oiDiam,
    dtime=1.0, file=strRoot+"_summaryList.txt",filename=1;

  /* Include script file */
  if ( yocoTypeIsStringScalar(scriptFile) && yocoTypeIsFile(scriptFile) ) {
    yocoLogInfo,"Include script file:",scriptFile(*)(1);
    include,scriptFile,3;
  } else if ( yocoTypeIsFile(strRoot+"_pndrsScript.i") ) {
    yocoLogInfo,"Include script file:",strRoot+"_pndrsScript.i";
    include,strRoot+"_pndrsScript.i",3;
  } else if ( yocoTypeIsFile(inputDir+"pndrsScript.i") ) {
    yocoLogInfo,"Include script file:",inputDir+"pndrsScript.i";
    include,inputDir+"pndrsScript.i",3;
  } else {
    yocoLogInfo,"No script to load";
  }
  
  
  pbootGetBootstrapSize, oiVis2, oiWave, [], nboot;

  /* Generate the calibrator boostrap */
 
  pbootGenerateCalibratorBootstrap, oiLog, oiVis, oiVis2, oiT3, oiDiam,
     oiDiamBoot, oiCalBoot, nboot=nboot;
 
  /* Generate new arrays to contain calibrated and transfer function
     points */
  
  pbootArraysLike, oiVis, oiVisCal, oiVis2, oiVis2Cal, oiT3, oiT3Cal;
  
  /* Calibrate all boostraps one after another */


  for (iboot = 1; iboot <= nboot; ++iboot)
  {
     
    local oWave, oVis, oVis2, oT3, oVisCalboot, oVis2Calboot, oT3Calboot,
          oVis2Cal, oVisCal, oT3Cal, oVis2Tfp, oVisTfp, oT3Tfp,
          oVis2Tfe, oVisTfe, oT3Tfe;
    
    yocoLogInfo, "calibrating bootstrap #" + pr1(iboot);

    /* Extract bootstrap # iboot so that OIFITS structures have 
       the format expected by the oiFitsUtil.i routines.  For each OI structure
       two outputs are given: (1) the actual observations restricted to
       a given scan bootstrap; (2) a set where calibrators have been 
       bootstrapped. (2) is used to compute/plot the TF, but finally
       calibrated visibilities are evaluated at (1). */

    pbootKeepBootstrap, oiLog, oiWave, oiArray, oiDiamBoot, oiCalBoot, 
      oiVis, oVis, oVisCalboot, oiVis2, oVis2, oVis2Calboot, 
      oiT3, oT3, oT3Calboot, oiWave, oWave, [], boot=iboot;
    oDiam = oiDiamBoot(,iboot);

    /* Calibrate the data setup by setup */
    pbootCalibrateNight, oVis2, oT3, oVis, 
       oVis2Calboot, oT3Calboot, oVisCalboot, 
       oWave, oiArray, oiTarget, oiLog, oDiam,
       oVis2Cal, oVisCal, oT3Cal, oVis2Tfp, oVisTfp, oT3Tfp,
       oVis2Tfe, oVisTfe, oT3Tfe, 
       vis2TfMode=vis2TfMode, t3TfMode=t3TfMode, visTfMode=visTfMode,
       vis2TfParam=vis2TfParam, t3TfParam=t3TfParam, visTfParam=visTfParam,
       vis2TfErrMode=vis2TfErrMode, t3TfErrMode=t3TfErrMode,
       visTfErrMode=visTfErrMode;

    /* Some plots & OI_FITS to allow to assess the data quality of
       the bootstrap.  (Note that 1st bootstrap is the standard reduction.)
       Don't do it for 500 bootstraps though... */

    if (iboot <= maxbootplot)
    {

      /* Plot the TF on a setup basis. */

      for (iXaxis = 1; iXaxis <= numberof(Xaxis); ++iXaxis)
      {
        ax = Xaxis(iXaxis);
        pbootPlotTfForAllSetups, oWave, oiTarget, oiArray, oiLog, 
          oVis2Calboot, oT3Calboot, oVisCalboot, 
          oVis2Cal, oT3Cal, oVisCal, 
          oVis2Tfe, oT3Tfe, oVisTfe, 
          oVis2Tfp, oT3Tfp, oVisTfp, 
          strRoot=swrite(format="%s_%s%s_boot%03i_", strRoot, "TF", ax, iboot),
          X=ax;
      };

      /* Write the TF estimates to a fits file */
      
      f = swrite(format="%s_TFESTIMATE_boot%03i_oidata.fits", strRoot, iboot);
      oiFitsWriteFiles, f, oiTarget, oWave, oiArray,
        oVis2Tfp, oVisTfp, oT3Tfp, oiLog, overwrite=1;
    
    }

    /* Now put the bootstrap back to the full OIFITS */

    pbootSetBootstrap, oiLog, oiWave, oiArray, 
      oiVis2Cal, oVis2Cal, oiVisCal, oVisCal, oiT3Cal, oT3Cal, boot=iboot;
 
  }

  /* Clean data from dummies */
  pbootCleanDataFromDummy, oiVis2Cal, oiT3Cal, oiVisCal,
    maxVis2Err=maxVis2Err, maxT3PhiErr=maxT3PhiErr;
    
  /* Write the calibrated data */
  if (is_array(oiVis2Cal) || is_array(oiT3Cal))
  {
    print, "ALL_oiDataCalib" , numberof(oiWave);
    oiFitsWriteFiles,strRoot+"_ALL_oiDataCalib.fits",
      oiTarget,oiWave,oiArray,oiVis2Cal,oiVisCal,oiT3Cal,oiLog,overwrite=1;
  }
  else
  {
    yocoLogInfo,"Cannot write oiDataCalib.fits: no calibrated science stars";
  }

  /* Write a summary with all science stars that
     have been properly calibrated */
  if ( is_array(oiVis2Cal) &&
       is_array((id = where( oiFitsGetIsCal(oiVis2Cal, oiDiam)==0 ))) )
  {
    oiFitsListObs, oiVis2Cal(id), oiTarget, oiArray, oiLog, oiDiam,
        dtime=1, file=strRoot+"_summaryScience.txt",date=1;
  }
  else
  {
    yocoLogInfo,"Cannot write summaryScience.txt: no calibrated science stars";
  }
  
  /* Loop on all stars to write all of them in a different file */
  allStars = oiTarget;
  for ( i=1 ; i<=numberof(allStars) ; i++ ) 
  {
    
    local oTarget, oVis2, oVis, oT3, oArray, oWave, oLog, oWave;
 
    /* Get the name */
    name = allStars(i).target;
    yocoLogInfo,"Make a summary for target: "+name+" ("+pr1(i)+" over "+pr1(numberof(allStars))+")";

    /* This is most probably the internal source, so no need to
       produce the PDF for this one */
    if (name=="INTERNAL") continue;

    /* Copy arrays and keep only this target */
    oiFitsCopyArrays, oiTarget, oTarget, oiVis2Cal, oVis2, oiVisCal,
      oVis, oiT3Cal, oT3, oiArray, oArray, oiWave, oWave, oiLog, oLog;
    oiFitsKeepTargets, oTarget, oVis2, oVis, oT3, trgList=name;

    /* Clean the arrays and check if something remains */
    oiFitsCleanFlag, oVis2, oVis, oT3;
    if (is_void(oT3) || is_void(oVis2)) {
      yocoLogInfo,"No valid data for this target... skip.";
      continue;
    }

    /* Write the file */
    name = ( oiFitsGetIsCal(allStars(i), oiDiam) ? "CAL_" : "SCI_" )+name;
    oiFitsWriteFiles,strRoot+"_"+name+"_oiDataCalib.fits",oTarget,oWave,oArray,oVis2,,oT3,oLog,overwrite=1;
 
  }
    
  /* Change permission of all these newly created files */
  system,"chmod ug+w "+strRoot+"_* > /dev/null 2>&1";

  yocoLogInfo,"pbootCalibrateDirectory done";
  yocoLogSetFile;

  return 1;

}

func pbootGetSetupPlot(oiData, oiLog, funcSetup=)
{
  return pbootGetSetup(oiData, oiLog, perPointing=1);
}

func pbootPlotTfForAllSetups(oiWave, oiTarget, oiArray, oiLog,
                             oiVis2, oiT3, oiVis,
                             oiVis2Cal, oiT3Cal, oiVisCal,
                             oiVis2Tfe, oiT3Tfe, oiVisTfe,
                             oiVis2Tfp, oiT3Tfp, oiVisTfp,
                             strRoot=, X=)
{
  yocoLogInfo,"pbootPlotTfForAllSetups()";

  /* pndrsPlotTfForAllSetups overwrite extern variables */

  local oWave,
        oVis2, oVis2Cal, oVis2Tfe, oVis2Tfp,
        oT3, oT3Cal, oT3Tfe, oT3Tfp;

  local oiFitsGetSetup;

  oiFitsGetSetup = pbootGetSetupPlot;
  oiFitsDefaultSetup = pbootGetSetupPlot;

 
  pbootPndrsPlotTfForAllSetups(oiWave, oiTarget, oiArray, oiLog,
                             oiVis2, oiT3, oiVis,
                             oiVis2Cal, oiT3Cal, oiVisCal,
                             oiVis2Tfe, oiT3Tfe, oiVisTfe,
                             oiVis2Tfp, oiT3Tfp, oiVisTfp,
                             strRoot=strRoot, X=X)

  yocoLogTrace,"pbootPlotTfForAllSetups done.";
  return 1;

}

/************************* Calib *********************************************/

func pbootCalibrateDiam(&oiVis2,oiWave,oiDiam)
/* DOCUMENT oiFitsCalibrateDiam(&oiVis2,oiWave,oiDiam)

   DESCRIPTION
   oiVis2 are calibrated from the target diameters
   stored in the structure oiDiam.

   PARAMETERS
   - oiVis2 (input/output), oiWave, oiDiam:
 */
{
  local i,y,w,l,diam,vis2, vis2err, phi, phiErr;
  yocoLogTrace,"oiFitsCalibrateDiam()";

  /* check inputs */
  if( !oiFitsIsOiVis2(oiVis2) ) return yocoError("oiVis2 not valid");
  if( !oiFitsIsOiWave(oiWave) ) return yocoError("oiWave not valid");
  

  for (i=1;i<=numberof(oiVis2);i++) {

    /* Get the diameter and its error */
    diam    = oiFitsGetOiDiam(oiVis2(i), oiDiam).diam;
    diamErr = oiFitsGetOiDiam(oiVis2(i), oiDiam).diamErr;

    /* if not a known diameter, skip */
    if (diam<=0) continue;
    
    /* get the data */
    oiFitsGetData,oiVis2(i),vis2,vis2Err;
    w = oiFitsGetLambda(oiVis2(i),oiWave) * 1e-6;

    nboot = numberof(vis2) / numberof(w);
    w = w(:,-:1:nboot)(*);

    /* calibrating factors */
    vis2m = abs( yocoMathAiry( abs( oiVis2(i).uCoord, oiVis2(i).vCoord) * diam * yocoAstroSI.mas / w ) )^2.0;
    ya = abs( yocoMathAiry( abs( oiVis2(i).uCoord, oiVis2(i).vCoord) * (diam+diamErr) * yocoAstroSI.mas / w ) )^2.0;
    yb = abs( yocoMathAiry( abs( oiVis2(i).uCoord, oiVis2(i).vCoord) * (diam-diamErr) * yocoAstroSI.mas / w ) )^2.0;
    vis2mErr = abs(ya-yb) / 2.0;

    /* Calibrate */
    vis2c  = vis2 / vis2m;
    
    /* Propagate errors (add variances)*/
    vis2  += 1e-10;
    vis2m += 1e-10;
    vis2cErr  = abs( vis2Err/vis2*vis2c, vis2mErr/vis2m*vis2c );
    
    /* Put the data back */
    _ofSSDs, oiVis2, i, "vis2Data", vis2c;
    _ofSSDs, oiVis2, i, "vis2Err",  vis2cErr;

    /* Put statistical error if structure is defined */
    oiFitsStrReadMembers, oiVis2(i), name;
    if (anyof(name=="vis2ErrSys")) {
      yocoLogInfo,"Fill vis2ErrSys";
      _ofSSDs, oiVis2, i, "vis2ErrSys",  vis2mErr/vis2m*vis2c;
    }
  }
  
  yocoLogTrace,"pbootCalibrateDiam done";
  return 1;
}

func pbootCalibrateDiamOiT3(&oiT3,oiWave,oiDiam)
{
  local i,y,w,l,diam,vis2, vis2err, phi, phiErr, amp, ampErr;
  yocoLogInfo,"oiFitsCalibrateDiamOiT3()";

  /* check inputs */
  if( !oiFitsIsOiT3(oiT3) ) return yocoError("oiT3 not valid");
  if( !oiFitsIsOiWave(oiWave) ) return yocoError("oiWave not valid");

  for (i=1;i<=numberof(oiT3);i++) {

    /* Get the diameter and its error */
    diam    = oiFitsGetOiDiam(oiT3(i), oiDiam).diam;
    diamErr = oiFitsGetOiDiam(oiT3(i), oiDiam).diamErr;

    /* if not a known diameter, skip */
    if (diam<=0) continue;
    
    /* get the data */
    oiFitsGetData,oiT3(i),amp,ampErr,phi,phiErr;
    w = oiFitsGetLambda(oiT3(i),oiWave) * 1e-6;

    nboot = numberof(vis2) / numberof(w);
    w = w(:,-:1:nboot)(*);


    /* calibrating factors */
    vis1 = yocoMathAiry( abs( oiT3(i).u1Coord, oiT3(i).v1Coord) * diam * yocoAstroSI.mas / w );
    vis2 = yocoMathAiry( abs( oiT3(i).u2Coord, oiT3(i).v2Coord) * diam * yocoAstroSI.mas / w );
    vis3 = yocoMathAiry( abs(-oiT3(i).u1Coord -oiT3(i).u2Coord,
                             -oiT3(i).v1Coord -oiT3(i).v2Coord) * diam * yocoAstroSI.mas / w );
    phical = oiFitsArg(complex(vis1 * vis2 * vis3));
    ampcal = abs( vis1 * vis2 * vis3 );
    
    /* Calibrate */
    phic  = oiFitsArg( exp(1.i * (phi/180*pi - phical)) )/pi*180;
    ampc  = amp / ampcal;

    /* Propagate errors (add variances)...
       FIXME: to be done */
    // amp  += 1e-10;
    // ampc += 1e-10;
    // vis2cErr  = abs( ampErr/vis2*vis2c, ampErr/vis2m*vis2c );
    
    /* Put the data back */
    _ofSSDs, oiT3, i, "t3Phi", phic;
    _ofSSDs, oiT3, i, "t3Amp", ampc;
  }
  
  yocoLogTrace,"pbootCalibrateDiamOiT3 done";
  return 1;
}


/************************ Final settings *************************************/
 
/* Version */

if (strpart(pndrsVersion, -3:) != "boot")
{
  pndrsVersion += "_boot";
}

