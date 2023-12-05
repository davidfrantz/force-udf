#load libraries
lib <- "../../rlib"

.libPaths(c(.libPaths(), lib))

sfLibrary(zoo, lib.loc = lib)
sfLibrary(sandwich, lib.loc = lib)
sfLibrary(strucchangeRcpp, lib.loc = lib)
sfLibrary(bfast, lib.loc = lib)


# dates:     vector with dates     of input data (class: Date)
# sensors:   vector with sensors   of input data (class: character)
# bandnames: vector with bandnames of input data (class: character)
force_rstats_init <- function(dates, sensors, bandnames){
	return(c("Breakdate1", "Breakdate2", "Breakdate3", "NR", "RMSD1", 
	         "RMSD2", "RMSD3", "Trend1", "Trend2", "Trend3",  "Trend4"))
}
  
# inarray:   2D-array with dim = c(length(dates), length(bandnames))
#            No-Data values are encoded as NA.  (class: Integer)
# dates:     vector with dates     of input data (class: Date)
# sensors:   vector with sensors   of input data (class: character)
# bandnames: vector with bandnames of input data (class: character)
# nproc:     number of CPUs the UDF may use. Always 1 for pixel functions (class: Integer)

force_rstats_pixel <- function(inarray, dates, sensors, bandnames, nproc){
  
  # check for missing and zero values
  if (all(inarray == 0) | all (is.na(inarray))){
    
    return(rep(-99999, 11))
  } 
  
  # extract year and doy from date information of inarray
  years <- dates |>
    format("%Y") |>
    as.numeric()
  
  doy <- dates |>
    format("%j") |>
    as.numeric()
  
  # create year identifier for the time series
  x <- years + doy/365

  # check for non-unique values and create mean value if two observations per time from different sensors appear
  y <- split(inarray[,1], x) |> sapply(mean, na.rm = TRUE)
  
  # check the uniqueness of x, otherwise zoo() will return an error
  x <- unique(x)

  # create time series
  ts <- zoo(y, x, frequency = 365) |> 
    as.ts()

  #save(list = ls(), file = "workspace.RData", envir = environment())  

  # apply bfast with 3 breaks
  bf = bfastlite(ts, breaks = 3)

  # get trend, breaks and break dates
  breakpoints_table <- strucchangeRcpp::breakpoints(bf[["breakpoints"]])
  breakpoints <- breakpoints_table$breakpoints
  break_dates <- x[breakpoints]
  
  # if breakpoints are determined, retrieve magnitude, rmsd, and number of breakpoints
  if (any(!is.na(breakpoints))){
  mag <- magnitude(bf$breakpoints)
  rmsd <- as.numeric(unlist(mag$Mag[,4]))
  nr <- as.numeric(length(breakpoints))
  
  # calculate trend 
  trendtable <- data.frame(coef(bf$breakpoints))
  trendtable$trend_recl <- with(trendtable, ifelse(trend < 0, 1, 2))
  trend <- as.numeric(trendtable[,9])
  
  # otherwise return 0 
  }else {
    break_dates <- rep(0,3)
    rmsd <- rep(0,3)
    nr <- 0
    trend <- rep(0,4)  
  }

    return(c(break_dates[1],  break_dates[2],  break_dates[3],nr, rmsd[1], rmsd[2], rmsd[3], trend[1], trend[2], trend[3], trend[4]))

}
