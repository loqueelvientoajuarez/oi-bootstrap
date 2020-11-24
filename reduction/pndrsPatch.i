#include "pndrs.i"

func pndrsGetPrism(oiLog)
{
  /* Note that we changed from OPTI3 to OPTI2 in 57263*/
  prism = ( ( 1*(oiLog.opti3Name=="FREE") +
              1*(oiLog.opti3Name=="WOLL") +
              2*(oiLog.opti3Name=="SMALL") +
              3*(oiLog.opti3Name=="LARGE") + 
              4*(oiLog.opti3Name=="GRISM") +
              4*(oiLog.opti3Name=="GRI+WOL") ) * (oiLog.mjdObs<=57263) +
            ( 1*(oiLog.opti2Name=="FREE") +
              1*(oiLog.opti3Name=="WOLL") +
              4*(oiLog.opti2Name=="GRISM") +
              4*(oiLog.opti2Name=="GRI+WOL") ) * (oiLog.mjdObs>57263) );
  
  return ["UNKNOWN","FREE","SMALL","LARGE","GRISM"](1+prism);
}


func pndrsPatchMissingLst(&oiLog)
{

  yocoLogInfo, "pndrsPatchMissingLst()";

  missingLst  = oiLog.lst == 0;
  if (noneof(missingLst))
  {
    return 0;
  }

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
  if (numberof(hasLst))
  {
    lst += (oiLog.lst(hasLst) - lst(hasLst))(avg);
  }
  noLst = where(oiLog.mjdObs == 0);
  oiLog.lst(missesLst) = lst(missesLst); 

  return numberof(missesLst);

}

func pndrsPatchGetBestIssEntry(missingIss, i)
{
  test1 = missingIss(:i-1);
  ok1 = where(!test1);
  if (numberof(ok1))
  {
    return max(ok1);
  }
  
  test2 = missingIss(:i+1);
  ok2 = where(!test2);
  if (numberof(ok2))
  {
    return i + min(ok2);
  }
  
  return -1;
   
}


func pndrsPatchMissingIss(&oiLog)
{
  yocoLogInfo, "pndrsPatchMissingIss()";

  missingIss = oiLog.issStation1 == " "; 
  nfixes = 0;
  if (noneof(missingIss))
  {
    return nfixes;
  }

  for (i = 1; i <= numberof(oiLog); ++i)
  {
    obsname = oiLog(i).obsName;
    if (missingIss(i) && noneof(obsname == ["dailyAlignment", "NiobateCheck"]))
    {
      j = pndrsPatchGetBestIssEntry(missingIss, i);
      if (j != -1)
      {
        nfixes += 1;
        oiLog(i).issStation1 = oiLog(j).issStation1;
        oiLog(i).issStation2 = oiLog(j).issStation2;
        oiLog(i).issStation3 = oiLog(j).issStation3;
        oiLog(i).issStation4 = oiLog(j).issStation4;
        oiLog(i).issTelName1 = oiLog(j).issTelName1;
        oiLog(i).issTelName2 = oiLog(j).issTelName2;
        oiLog(i).issTelName3 = oiLog(j).issTelName3;
        oiLog(i).issTelName4 = oiLog(j).issTelName4;
        oiLog(i).issInput1 = oiLog(j).issInput1;
        oiLog(i).issInput2 = oiLog(j).issInput2;
        oiLog(i).issInput3 = oiLog(j).issInput3;
        oiLog(i).issInput4 = oiLog(j).issInput4;
        oiLog(i).issT1x = oiLog(j).issT1x;
        oiLog(i).issT2x = oiLog(j).issT2x;
        oiLog(i).issT3x = oiLog(j).issT3x;
        oiLog(i).issT4x = oiLog(j).issT4x;
        oiLog(i).issT1y = oiLog(j).issT1y;
        oiLog(i).issT2y = oiLog(j).issT2y;
        oiLog(i).issT3y = oiLog(j).issT3y;
        oiLog(i).issT4y = oiLog(j).issT4y;
        oiLog(i).issT1z = oiLog(j).issT1z;
        oiLog(i).issT2z = oiLog(j).issT2z;
        oiLog(i).issT3z = oiLog(j).issT3z;
        oiLog(i).issT4z = oiLog(j).issT4z;
      }
    }
  }
  
  return nfixes;

}

if (is_void(_pndrsPatchReadLog))
{
  _pndrsPatchReadLog = pndrsReadLog;
}
func pndrsReadLog(inputDir, &oiLog, overwrite=, save=)
{
  if (is_void(save))
  {
    save = 1;
  }
  ok = _pndrsPatchReadLog(inputDir, oiLog, overwrite=overwrite, save=save);
  nfixes = pndrsPatchMissingLst(oiLog);
  nfixes += pndrsPatchMissingIss(oiLog);
  if (save)
  {
    local logName;
    here = get_cwd(".");
    cd, inputDir;
    pndrsGetLogName, inputDir, logName;
    chmode = (strpart(rdline(popen("ls -l -d .",0)),3:3)!="w");
    if (chmode) system,"chmod  u+w . > /dev/null 2>&1";
    oiFitsWriteOiLog, logName, oiLog, overwrite=1; 
    if (chmode) system,"chmod  u-w . > /dev/null 2>&1";
    cd, here;
  }
  return ok;
}


