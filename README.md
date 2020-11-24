Yorick extension to the VLTI/PIONIER `pndrs` pipeline to obtain full determination of correlated errors using the bootstrap method and python utilities to analyse the resulting data.

Used by Lachaume et al. (2019), MNRAS, 484, 2656.  Raw FITS files freely
obtainable from the ESO database are not included.

A fully functional `pndrs` install is needed.

Reduction of PIONIER data (`reduction/`)
* Yorick libraries
    * `pbootPatch.i`: patches needed to have pndrs work fine with bootstraps
    * `pbottBatch.i`: set of methods to do the bootstrap reduction
* Executables
    * `pndrsReduce`: the original pipeline with slight modifications 
(mostly enabling a smooth batch mode)
    * `pbootReduce`: the bootstrap variation 

Calibration of the visibilities and data analysis (`calibration+analysis/`)
* `oifits.py`: module to read/write OIFITS file  
* `oilog.py`: make a log of existing raw OIFITS files
* `oical.py`: calibrated visibilities for each bootstrap (it can be done with `pndrs` but without all the nice plots)
* `bootdiamfit.py`: perform a diameter fit on each bootstrap
* other files are experimental
