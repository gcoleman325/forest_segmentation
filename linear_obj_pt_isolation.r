## import packages
library(TreeLS)
library(nabor)
library(dplyr)
library(data.table)
library(terra)
library(rlas)
library(raster)
library(lidR)
library(e1071)
library(ggplot2)
library(MASS)
library(sf)

# prep for batch process
inDir <- [your folder w files to process]
files <- list.files(pattern = ".las", inDir)

# dataframe to save values
df <- data.frame()
titles <- c()

# disgustingly nested for loops to change shp_line() params
th1 = 4
for (j in 1:3){
  th1 <- th1 + 4
  th2 <- 0
  k <- 0
  for (h in 1:3){
    # adjusts param th2 (verticality)
    th2 <- th2 + 0.05
    k <- 0
    
    for (m in 1:3){
      # adjusts param k (number of neighbors used)
      k <- k + 8
      
      # vector to save values for set of scans
      batch <- c()
      for (i in 1:length(files)){
        
        # read in file
        file <- files[i]
        file_name <- substring(file, 1, -4)
        tls <- readTLS(paste0(inDir, "\\", file))
        
        # process n print
        tls@data <- subset(tls@data, select = -c(20))
        tls <- filter_poi(tls, Z<0.5)
        tls <- segment_shapes(tls, shp_hline(th1 = th1, th2 = th2, k = k))
        tls@data <- tls@data[Shape == T]
        numPts <- nrow(tls@data)
        batch <- append(batch, numPts)
      }
      
      # adds batch to data frame and title to titles
      df <- rbind(df, batch)
      titles <- cbind(paste0(th1, "/", th2, "/", k))
      
      # adds comment for reference
      print(paste0("finished processing using th1=", as.character(th1), ", th2=", as.character(th2),", and k=",as.character(k)))
    }
  }
}
df <- cbind(df, titles)
print(df)
write.csv(df, "D:\\cc_2go\\no_trees\\numPtsLinear.csv")
