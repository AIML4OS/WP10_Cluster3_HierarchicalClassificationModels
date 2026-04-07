library(readxl)


library(data.table)

# load data ------------------------------------------------------------------------------------------------------
# 
# as example data we use the dictionary from the Statistics Austrias classification data base
dat <- fread("https://www.statistik.at/kdb/downloads/csv/prod/OENACE2025_DE_CAL.txt",
             colClasses = c("integer","character","character","character","character","character"),
             col.names = c("level","edv-code","NACE25","Text","X","1D"),
             encoding = "Latin-1")
dat[,level:=NULL]
dat[,`edv-code`:=NULL]
dat[,X:=NULL]
dat[,'1D':=NULL]

dat[,ID:=1:nrow(dat)]
#shuffle input
dat <- dat[sample(.N)]

message("Number of unique codes: ",length(unique(dat$NACE25)))

# add additional variable ------------------------------------------------------

# education (factor variable with 5 levels)
dat[,edu:=sample(1:5,.N,replace = T, prob = c(0.1,0.3,0.3,0.2,0.1))]

# citizenship (factor, 2 levels)
dat[,citizen:=sample(1:2,.N,replace = T, prob = c(0.8,0.2))]


dat[,count:=.N,by=NACE25]

dat[,Code:=gsub("[^0-9]", "", NACE25)]

message("Rows input data: ", nrow(dat))

