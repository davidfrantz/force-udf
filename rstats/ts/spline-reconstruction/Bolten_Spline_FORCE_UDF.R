force_rstats_init <- function(dates, sensors, bandnames){

  # Year which should be reconstructed
  year_to_interpolate <- 2023
  # Days to predict in this year and the intervall: 60 to 330 (1st March to 26th November)
  DOYs_to_predict <- seq(60,330,by =10)
  dates_to_predict <- as.Date(paste(year_to_interpolate, DOYs_to_predict), format = "%Y %j")
  
  band_names <- paste(substr(as.character(dates_to_predict),1,4),substr(as.character(dates_to_predict),6,7),substr(as.character(dates_to_predict),9,10), sep ="")
  return(band_names)
}

force_rstats_pixel <- function(inarray, dates, sensors, bandnames, nproc){
  
  # Year which should be interpolated (same like above)
  year_to_interpolate <- 2023
  # Days to predict in this yearand the intervall: 60 to 330 (1st March to 26th November)
  DOYs_to_predict <- seq(60,330,by =10) 
  dates_to_predict <- as.Date(paste(year_to_interpolate, DOYs_to_predict), format = "%Y %j")

  # spline variables
  # smoothing factor for the spline reconstruction
  smooth_fac <- 0.5
  # Bolton's variable of maximum weight to assing for the predessesor years
  # the year of reconstruction has a wheight of 1
  max_weight <- 0.2
  
  # cumulate the DOY to the year of interpolation
  # start year 2015 (example), because of Sentinel 2 launch date, for e.g. Landsat adjust to your time span
  start_year <- 2015
  DOYs_to_predict <- (year_to_interpolate - start_year) * 365 + DOYs_to_predict
  
  tryCatch({
    # grap FORCE no-data case
    if (all(is.na(inarray[,1]))){
      return(rep(-9999,length(DOYs_to_predict)))
    }
  
    # calculate cumulative DOYs for the input data 
    DOYs <- as.numeric(format(dates, "%j"))
    years <- as.numeric(substr(as.character(dates),1,4))
    cumulative_DOYs <- (years - start_year) * 365 + DOYs
    
    # join the data to a dataframe
    df <- data.frame(x=cumulative_DOYs,y=inarray[,1])
    
    # ------- 1.1 calcualte Mean Function --------------
    euc.dist <- function(x1, x2) sqrt(sum((x1 - x2) ^ 2))
    
    # define Start and endpoints for the three spline reconstuctions
    DOY_borders_year <- c((year_to_interpolate-start_year)*365 - 180, (year_to_interpolate-start_year+1)*365+180)
    DOY_borders_b    <- c((year_to_interpolate-start_year-1)*365 - 180, (year_to_interpolate-start_year)*365+180)
    DOY_borders_bb   <- c((year_to_interpolate-start_year-2)*365 - 180, (year_to_interpolate-start_year-1)*365+180)
    
    # create dataframes for reconstuction
    data_year <- na.exclude(df[df$x %in% seq(DOY_borders_year[1],DOY_borders_year[2]),c(1,2)])
    data_b   <- na.exclude(df[df$x %in% seq(DOY_borders_b[1] ,DOY_borders_b[2]),c(1,2)])
    data_bb  <- na.exclude(df[df$x %in% seq(DOY_borders_bb[1],DOY_borders_bb[2]),c(1,2)])
    
    # calculate spline model for year of reconstruction and predict
    DOYs_target_year <- seq(DOY_borders_year[1],DOY_borders_year[2])
    tryCatch({
      eval( spline_model_year <<- smooth.spline(data_year$x,data_year$y, spar =smooth_fac) )
      eval( predict_year <<- predict(spline_model_year, x = DOYs_target_year)$y )
    }, error = function(err) {
      return(rep(-9999,length(DOYs_to_predict)))
    })
    
    #calculate d_max
    mean_year <- mean(na.exclude(data_year$y))
    mean_prediction <- rep(mean_year,length(DOYs_target_year))
    d_max = euc.dist(mean_prediction, predict_year) / 10000
    
    
    # --------- 1.2 spline for precessor years ------------
    # one year before
    # predict with DOYs of year of reconstruction, for difference calculation 
    # between the two spline reconstructions
    tryCatch({
      eval( spline_model_b <<- smooth.spline(data_b$x+365,data_b$y, spar =smooth_fac) )
      eval( predict_b <<- predict(spline_model_b, x = DOYs_target_year)$y )
    }, error = function(err) {
      return(rep(-9999,length(DOYs_to_predict)))
    })
    d_b = euc.dist(predict_year, predict_b) / 10000
    
    # two years before
    # predict with DOYs of year of reconstruction, for difference calculation 
    # between the two spline reconstructions
    tryCatch({
      eval( spline_model_bb <<- smooth.spline(data_bb$x+(365*2),data_bb$y, spar =smooth_fac) )
      eval( predict_bb <<- predict(spline_model_bb, x = DOYs_target_year)$y )
    }, error = function(err) {
      return(rep(-9999,length(DOYs_to_predict)))
    })
    d_bb = euc.dist(predict_year, predict_bb) / 10000
    
    # ---------- 1.3 Calculate weights -------------------
    # one year before
    if (d_max != 0) {
      weight_b = max_weight*(1-(d_b/d_max))
      if (weight_b < 0){
        weight_b = 0
      }
    } else {weight_b = 0}
    
    # two years before
    if (d_max != 0) {
      weight_bb = max_weight*(1-(d_bb/d_max))
      if (weight_bb < 0){
        weight_bb = 0
      }
    } else {weight_bb = 0}
    
    #----------- 1.4 apply weights and calculate weighted spline --------------
    # combine the time series to one year and assign weights to the new data points
    combined_x <- c(data_year$x , (data_b$x+365)[weight_b>0] , (data_bb$x + (365*2))[weight_bb>0])
    combined_y <- c(data_year$y , data_b$y[weight_b>0]       , data_bb$y[weight_bb>0])
    vec_weights <- c(rep(1,length(data_year$x)),
                     rep(weight_b,length(data_b$x))[weight_b>0],
                     rep(weight_bb,length(data_bb$x))[weight_bb>0])
    
    # calculate weighted spline
    tryCatch({
      eval( spline_model_combined <<- smooth.spline(x=combined_x, y=combined_y, w = vec_weights , spar =smooth_fac) )
      eval( predict_combined <<- predict(spline_model_combined, x = DOYs_to_predict)$y )
    }, error = function(err) {
      return(rep(-9999,length(DOYs_to_predict)))
    })
    return(predict_combined)
  
  }, error = function(err) {
    return(rep(-9999,length(DOYs_to_predict)))
  })
}