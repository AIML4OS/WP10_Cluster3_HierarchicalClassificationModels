# load data and set fixed set of train and validation split to be used with all hierarchical model approaches
# test split will always be mountSTAT::mountO(folder="B/Datenaustausch/Klartextvercodungen/Erhebungen/MZ"),"Für Monitoring/mz_nace_2024Q4_Teil1.csv")
library(data.table)

dat[,(target):=lapply(.SD,factor),.SDcols=target]

dat <- dat[!is.na(Code)]
dat[,Text_clean:=tolower(Text)]
umlaut <- c("ä","ö","ü","ä","ö","ü","&","ß","ä","ü","ß","ö","","ü","ß",
            "ä","ä","","ö","ü","ä","ü")
kodierung <- c("ã„","ã–","ãœ","ã¤","ã¶","ã¼","&amp;","ãÿ","„","ç¬","ãÿ",
               "ã", 'â“',"š","á","ž","çï","&amp","”","Ã¼","Ã¤","\u0081")
dat[,Text_clean:=stri_replace_all_regex(Text_clean, kodierung, umlaut, vectorize_all = FALSE)]

dat[,Text_clean:=simplify_text(Text_clean)]
dat <- dat[Text_clean!="",]
dat[is.na(Text_clean),.N]

dat[,Text_clean:=substr(Text_clean,1,90)] # cut off after 90 characters

#split into train/valid/test data set
set.seed(555)
index.train <- sample(1:nrow(dat),floor(nrow(dat)*0.60))
index.valid <- setdiff(1:nrow(dat),index.train)

set.seed(123)
index.test <- sample(index.valid,floor(length(index.valid)*0.5))
index.valid <- setdiff(index.valid,index.test)

dat_eval <- dat[index.test,]
