# global header, e.g., for loading libraries
# library(...)

# dates:     vector with dates     of input data (class: Date)
# sensors:   vector with sensors   of input data (class: character)
# bandnames: vector with bandnames of input data (class: character)
force_rstats_init <- function(dates, sensors, bandnames){

    return(c("cumulative", "minimum", "variation"))
}


# inarray:   2D-array with dim = c(length(dates), length(bandnames))
#            No-Data values are encoded as NA.  (class: Integer)
# dates:     vector with dates     of input data (class: Date)
# sensors:   vector with sensors   of input data (class: character)
# bandnames: vector with bandnames of input data (class: character)
# nproc:     number of CPUs the UDF may use. Always 1 for pixel functions (class: Integer)
force_rstats_pixel <- function(inarray, dates, sensors, bandnames, nproc){

  s <- sum(inarray[,1], na.rm = TRUE) / 1e2
  m <- min(inarray[,1], na.rm = TRUE)
  v <- sd(inarray[,1],  na.rm = TRUE) / mean(inarray[,1],  na.rm = TRUE) * 1e4

  return(c(s, m, v))
}
