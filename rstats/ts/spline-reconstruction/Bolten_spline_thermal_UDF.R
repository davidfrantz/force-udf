force_rstats_init <- function(dates, sensors, bandnames){
  
  # Days to predict in the target year
  DOYs_to_predict <- c(seq(10,100,by =10), seq(120,260, by = as.integer(140/7)) , seq(270,360,by =10))
  DOYs_to_predict<- sprintf("%03d", DOYs_to_predict)
  return(c(paste("BLU_",as.character(years[1]),DOYs_to_predict, sep =''),
           paste("GRN_",as.character(years[2]),DOYs_to_predict, sep =''),
           paste("RED_",as.character(years[3]),DOYs_to_predict, sep =''),
           paste("RE1_",as.character(years[4]),DOYs_to_predict, sep =''),
           paste("RE2_",as.character(years[5]),DOYs_to_predict, sep =''),
           paste("RE3_",as.character(years[6]),DOYs_to_predict, sep =''),
           paste("BNIR_",as.character(years[7]),DOYs_to_predict, sep =''),
           paste("NIR_",as.character(years[8]),DOYs_to_predict, sep =''),
           paste("SW1_",as.character(years[9]),DOYs_to_predict, sep =''),
           paste("SW2_",as.character(years[10]),DOYs_to_predict, sep ='')
           ))
  
}

force_rstats_pixel <- function(inarray, dates, sensors, bandnames, nproc){
  # ==========================
  # global variables
  # ==========================
  # Days to predict in the target year
  DOYs_to_predict <- c(seq(10,100,by =10), seq(120,260, by = as.integer(140/7)) , seq(270,360,by =10))  # Days 10 to 360 of every year
  
  # spline variables
  # smoothing factor for the spline reconstruction
  smooth_fac <- 0.5
  # Bolton's variable of maximum weight to assing for the predessesor years
  # the year of reconstruction has a wheight of 1
  max_weight <- 0.2
  
  tryCatch({
    # grap FORCE no-data case
    if (all(is.na(inarray[,1]))){
      return(rep(-9999, length(DOYs_to_predict) * (dim(inarray)[2]-1) ) )
    }
    year_to_interpolate <- tail(as.numeric(substr(as.character(dates),1,4)),n=1)
    # cumulate the DOYs to the year of interpolation
    DOYs_to_predict <- (year_to_interpolate - 2015) * 365 + DOYs_to_predict 
    #define borders for the three spline interpolations
    DOY_borders_year <- c((year_to_interpolate-2015)*365 - 180, (year_to_interpolate-2015+1)*365+180)
    DOY_borders_b    <- c((year_to_interpolate-2015-1)*365 - 180, (year_to_interpolate-2015)*365+180)
    DOY_borders_bb   <- c((year_to_interpolate-2015-2)*365 - 180, (year_to_interpolate-2015-1)*365+180)
    
    # euclidean distance function
    euc.dist <- function(x1, x2) sqrt(sum((x1 - x2) ^ 2))
  
    # calculate cumulative DOYs for the input data
    Thermal_DOYs <- as.numeric(inarray[,11])
    DOYs <- as.numeric(format(dates, "%j"))
    years <- as.numeric(substr(as.character(dates),1,4))
    cumulative_DOYs <- (years - 2015) * 365 + DOYs
    cumulative_Thermal_DOYs <- (years - 2015) * 365 + Thermal_DOYs

    output_array <- c()
    for (band_num in seq(1,dim(inarray)[2]-1)){ # -1 because in the last array the Thermal time is stored
      # ==========================
      # data from this year
      # ==========================
      data_year <-  c(inarray[,band_num][cumulative_DOYs %in% seq((year_to_interpolate-2015)*365+180,(year_to_interpolate-2015+1)*365)],# data of second half of the year
                      inarray[,band_num][cumulative_DOYs %in% seq((year_to_interpolate-2015)*365    ,(year_to_interpolate-2015+1)*365)],# data of this year
                      inarray[,band_num][cumulative_DOYs %in% seq((year_to_interpolate-2015)*365    ,(year_to_interpolate-2015  )*365 +180) ])# data of the first half this year
      Thermal_Time <- c(cumulative_Thermal_DOYs[cumulative_DOYs %in% seq((year_to_interpolate-2015)*365+180,(year_to_interpolate-2015+1)*365)]-365,# dates of second half of the year , subtracted with 365 to match year before
                      cumulative_Thermal_DOYs[cumulative_DOYs %in% seq((year_to_interpolate-2015)*365    ,(year_to_interpolate-2015+1)*365)],# dates of this year
                      cumulative_Thermal_DOYs[cumulative_DOYs %in% seq((year_to_interpolate-2015)*365    ,(year_to_interpolate-2015  )*365+180 )]+365)# dates of the first half this year, added with 365 to match year after
      Thermal_Time <- Thermal_Time[!is.na(data_year)]
      data_year <- na.exclude(data_year)
      
      
      # ==========================
      # data from year before
      # ==========================
      data_b <-  c(inarray[,band_num][cumulative_DOYs     %in% seq((year_to_interpolate-2015-1)*365+180,(year_to_interpolate-2015)*365)],# data of second half of the year
                   inarray[,band_num][cumulative_DOYs     %in% seq((year_to_interpolate-2015-1)*365    ,(year_to_interpolate-2015)*365)],# data of this year
                   inarray[,band_num][cumulative_DOYs     %in% seq((year_to_interpolate-2015-1)*365    ,(year_to_interpolate-2015-1  )*365 +180)])# data of the first half this year
      Thermal_Time_b <- c(cumulative_Thermal_DOYs[cumulative_DOYs %in% seq((year_to_interpolate-2015-1)*365+180,(year_to_interpolate-2015)*365)]-365,# dates of second half of the year , subtracted with 365 to match year before
                   cumulative_Thermal_DOYs[cumulative_DOYs %in% seq((year_to_interpolate-2015-1)*365    ,(year_to_interpolate-2015)*365)],# dates of this year
                   cumulative_Thermal_DOYs[cumulative_DOYs %in% seq((year_to_interpolate-2015-1)*365    ,(year_to_interpolate-2015-1)*365+180 )]+365)# dates of the first half this year, added with 365 to match year after
      Thermal_Time_b <- Thermal_Time_b[!is.na(data_b)]
      data_b <- na.exclude(data_b)
      
      # ==========================
      # data from two years before
      # ==========================
      data_bb <-  c(inarray[,band_num][cumulative_DOYs     %in% seq((year_to_interpolate-2015-2)*365+180,(year_to_interpolate-2015-1)*365)],# data of second half of the year
                    inarray[,band_num][cumulative_DOYs     %in% seq((year_to_interpolate-2015-2)*365    ,(year_to_interpolate-2015-1)*365)],# data of this year
                    inarray[,band_num][cumulative_DOYs     %in% seq((year_to_interpolate-2015-2)*365    ,(year_to_interpolate-2015-2  )*365 +180)])# data of the first half this year
      Thermal_Time_bb <- c(cumulative_Thermal_DOYs[cumulative_DOYs %in% seq((year_to_interpolate-2015-2)*365+180,(year_to_interpolate-2015-1)*365)]-365,# dates of second half of the year , subtracted with 365 to match year before
                          cumulative_Thermal_DOYs[cumulative_DOYs %in% seq((year_to_interpolate-2015-2)*365    ,(year_to_interpolate-2015-1)*365)],# dates of this year
                          cumulative_Thermal_DOYs[cumulative_DOYs %in% seq((year_to_interpolate-2015-2)*365    ,(year_to_interpolate-2015-2)*365+180 )]+365)# dates of the first half this year, added with 365 to match year after
      Thermal_Time_bb <- Thermal_Time_bb[!is.na(data_bb)]
      data_bb <- na.exclude(data_bb)
      
      # calculate spline model and predict
      DOYs_whole_timespan <- seq(DOY_borders_year[1]+183,DOY_borders_year[2]-183)# is now only the predicted year
      predict_year <- predict(smooth.spline(x=Thermal_Time, y=data_year, spar =smooth_fac), x = DOYs_whole_timespan)$y
      
      # ------- 1.1 calcualte Mean Function --------------
      #calculate d_max
      mean_year <- mean(na.exclude(data_year))
      mean_prediction <- rep(mean_year,length(DOYs_whole_timespan))
      d_max = euc.dist(mean_prediction, predict_year) / 10000
  
      #--------- 1.2 spline for precessor years ------------
      predict_b  <- predict(smooth.spline(Thermal_Time_b+365     ,data_b , spar =smooth_fac), x = DOYs_whole_timespan)$y
      d_b = euc.dist(predict_year, predict_b) / 10000
  
      predict_bb <- predict(smooth.spline(Thermal_Time_bb+(365*2),data_bb, spar =smooth_fac), x = DOYs_whole_timespan)$y
      d_bb = euc.dist(predict_year, predict_bb) / 10000
      
      #---------- 1.3 Calculate weights -------------------
      if (d_max != 0) {
        weight_b = max_weight*(1-(d_b/d_max))
        if (weight_b < 0){
          weight_b = 0
        }
      } else {weight_b = 0}
  
      if (d_max != 0) {
        weight_bb = max_weight*(1-(d_bb/d_max))
        if (weight_bb < 0){
          weight_bb = 0
        }
      } else {weight_bb = 0}
      
      #----------- 1.4 apply weights and calculate wheigthed spline --------------
      combined_x <- c(Thermal_Time, (Thermal_Time_b+365)[weight_b>0] , (Thermal_Time_bb + (365*2))[weight_bb>0])
      combined_y <- c(data_year   , data_b[weight_b>0]               , data_bb[weight_bb>0])
      vec_weights <- c(rep(1,length(Thermal_Time)),
                       rep(weight_b,length(Thermal_Time_b))[weight_b>0],
                       rep(weight_bb,length(Thermal_Time_bb))[weight_bb>0])
      # calculate weighted spline
      predict_combined <- predict(smooth.spline(x=combined_x, y=combined_y, w = vec_weights , spar =smooth_fac),x = DOYs_to_predict)$y
      output_array <- c(output_array,as.integer(predict_combined))
    }
    return(output_array)
    
  }, error = function(err) {
    return(rep(-9999,length(DOYs_to_predict) * (dim(inarray)[2]-1) ))
  })
}