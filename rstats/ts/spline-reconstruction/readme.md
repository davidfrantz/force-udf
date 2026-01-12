# Time series reconstruction for forest using splines

&copy;
Copyright 2024, David Klehr

## Run with

- program: ``force-higher-level``
- submodule: ``TSA``
- DATE_RANGE: ``xxxx-07-01 yyyy-06-31``
    * xxxx = three years before your target year
    * yyyy = one year after your target year
    * e.g. for target year 2022: ``2019-07-01 2023-06-31``
- UDF type: ``RSTATS_TYPE = PIXEL``
- required parameters:``none``
- required R libraries: ``none``

## References

- Bolton, D.K., Gray, J.M., Melaas, E.K., Moon, M., Eklundh, L., Friedl, M.A., 2020. **Continental-scale land surface phenology from harmonized Landsat 8 and Sentinel-2 imagery**. *Remote Sensing of Environment 240*, 111685. https://doi.org/10.1016/j.rse.2020.111685.