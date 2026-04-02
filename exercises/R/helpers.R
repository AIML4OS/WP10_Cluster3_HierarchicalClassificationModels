#helper functions NACE
library(data.table)
library(stringi)
library(stringr)
library(stopwords)

simplify_text <- function(datcol){
  enterprise_type <- list(
    GMBH_AT = paste("(österreich|austria)",c("gmbh($|\\s|\\s& co(\\. (og|kg)|))","ges(\\.|)m(\\.|)b(\\.|)h(\\.|)($|\\s|\\s& co(\\. (og|kg)|))",
                                             "Gesellschaft m(\\.|)b(\\.|)h(\\.|)($|\\s|\\s& co(\\. (og|kg)|))",
                                             "gesellschaft mit beschränkter haftung($|\\s)")),
    GMBH=c("gmbh($|\\s|\\s& co(\\. (og|kg)|))","ges(\\.|)m(\\.|)b(\\.|)h(\\.|)($|\\s|\\s& co(\\. (og|kg)|))",
           "Gesellschaft m(\\.|)b(\\.|)h(\\.|)($|\\s|\\s& co(\\. (og|kg)|))",
           "gesellschaft mit beschränkter haftung($|\\s)","GmbH\\.($|\\s)","g\\.m\\.b\\.h\\.($|\\s)"),
    AT = c("austria$","österreich$"),
    FRANCHISE = c("franchise GmbH", "franchise kg", "franchise og",
                  "Franchise (&|und).*GmbH($|\\s)","franchise gmbh & co kg","franchise betriebs- und beratungs-gmbh","franchise-verband",
                  "franchise holding gmbh & co. kg","franchise at gmbh","franchise system gmbh"),
    AG = c("Aktiengesellschaft$","AG($|\\s)","Holding AG($|\\s)","Aktiengesell- schaft($|\\s)"),
    OG = "OG($|\\s)",
    KG = c("KG($|\\s)","Kommanditgesellschaft($|\\s)"),
    ZT = c("ZT($|\\s)","ziviltechniker($|\\s)"),
    OTHER= c("Speditionsgesellschaft m\\.b\\.H\\.($|\\s)",
             "Handelsgesellschaft m\\.b\\.H\\.($|\\s)",
             "handelsgesmbh\\.($|\\s)","baugmbh($|\\s)","baugmbh","handelsgesmbh($|\\s)",
             "Baugesellschaft m\\.b\\.H\\.($|\\s)",
             "Baugesellschaft m\\.b\\.H\\. u\\. Co\\. KG\\.",
             "BetriebsgmbH($|\\s)","betriebsges\\. m\\.b\\.h\\.",
             "e\\.U\\.($|\\s)",
             "e\\.u\\. Inh\\..*($|\\s)",
             "Österreichische Warenhandels-Aktiengesellschaft($|\\s)",
             "\\sInhaber.*e(\\.|)U(\\.|)($|\\s)",
             "\\sinh\\..*($|\\s)")
  )
  
  enterprise_type <-tolower( unlist(enterprise_type))
  
  datcol <- tolower(datcol)
  datcol <- stri_replace_all_regex(datcol,enterprise_type,"",vectorize_all = F)
  
  patterns <- c("[:|;|\\*|~|\\?|=|\\<|\\>|%|\\.|\\$|€|_|µ|\\^|°|\\||,|§|\\(|\\)|\\-|&]")
  replacements <- c(""  )
  
  #alle umlaute zu vokalen
  umlaut <- c("Ä","Ö","Ü","ä","ö","ü","ß")
  kodierung <- c("a","o","u","a","o","u","ss")
  datcol <- stri_replace_all_regex(datcol, umlaut, kodierung, vectorize_all = FALSE)
  
  #stopwords removal
  stopwords <- stopwords(language = "de", source =  "snowball")
  stopwords <- stopwords[!stopwords%in%c("nicht")]
  stopwords <- paste0("\\b",stopwords,"\\b")
  stopwords <- stri_replace_all_regex(stopwords,umlaut,kodierung,vectorise_all = FALSE)
  datcol <- stri_replace_all_regex(datcol, stopwords, "", vectorize_all = FALSE)
  
  
  datcol <- stri_replace_all_regex(datcol, patterns, replacements, vectorize_all = FALSE)
  datcol <- trimws(stri_replace_all_regex(datcol,"\\s+"," "))
  return(tolower(datcol))
}

# tokenizer
setupEmbedding2 <- function(dat, string_col = "STRING_CLEAN2", ngram = 3, roll = FALSE, 
                            keep_spaces = TRUE, space_token = TRUE, tok=NULL,
                            maxlen=NULL, left_pad=FALSE){
  
  dat <- copy(dat)
  
  if(!is.data.table(dat)){
    dat <- as.data.table(dat)
  }
  
  if(keep_spaces == FALSE){
    dat[,c(string_col):=gsub("\\s","",get(string_col))]
  }
  
  dat[,ID:=1:nrow(dat)]
  x <- dat[,.(Word=unlist(str_split(get(string_col),pattern=" "))),by=.(ID)]
  x[,Position:=1:.N,by=.(ID)]
  x_gram <- x[,.(Word_ngram=split_word(unlist(.BY),ngram = ngram, roll = roll)),by=.(Word)]
  
  if(keep_spaces == TRUE & space_token == TRUE){
    x_gram <- rbind(x_gram,
                    data.table(Word="_space_",Word_ngram="_space_"))
  }
  
  x_gram[,Word_order:=1:.N,by=.(Word)]
  
  if(keep_spaces == TRUE & space_token == TRUE){
    x[,filter_length:=.N>1,by=.(ID)]
    x_space <- x[filter_length == TRUE,.(Position = (Position+.5)[-.N]), by = .(ID)]
    x_space[,Word:="_space_"]
    x <- rbind(x, x_space,use.names = TRUE, fill = TRUE)
  }
  
  x <- x[x_gram,on=.(Word), nomatch = NULL, allow.cartesian = TRUE]
  setorder(x,ID,Position,Word_order)
  
  if(is.null(tok)){
    x[order(Word_ngram),TOKEN:=.GRP,by=.(Word_ngram)]
    num_words <- x[,uniqueN(Word_ngram)] + 1
    tok <- unique(x[,.(Word_ngram,TOKEN)])
  }else{
    x[tok,TOKEN:=TOKEN,on=.(Word_ngram)]
    num_words <- tok[,uniqueN(TOKEN)] + 1
  }
  
  x[,help_position:=1:.N,by=.(ID)]
  x[help_position==1 & is.na(TOKEN),TOKEN:=0]
  x <- x[!is.na(TOKEN)]
  
  out_idx <- dcast(x,ID~help_position,value.var = "TOKEN", fill=0)
  out_idx[,ID:=NULL]
  out_idx <- as.matrix(out_idx)
  
  #shorten emb to maxlen
  if(!is.null(maxlen)){
    if(maxlen<dim(out_idx)[2]){
      out_idx <- out_idx[,1:maxlen]
    }else if(maxlen>dim(out_idx)[2]){
      pad <- matrix(0,ncol=maxlen-ncol(out_idx),nrow=nrow(out_idx))
      out_idx <- cbind(out_idx,pad)
    }
  }
  
  if(left_pad){
    out_idx <- t(apply(out_idx,1,left_padding))
  }
  
  tok=rbind(tok,data.table("Word_ngram"="_pad_","TOKEN"=0)) #add padding token
  return(list(out_idx, num_words=num_words, tok = tok))
}


split_word <- function(x_word, ngram=3, roll = FALSE){
  
  if(nchar(x_word,keepNA=FALSE,allowNA = TRUE)<=ngram){
    return(x_word)
  }
  
  x_word <- unlist(str_split(x_word, pattern=""))
  
  if(roll == FALSE){
    subset_indices <- seq(1,length(x_word),by=ngram)
    x_word <- sapply(subset_indices,function(start, x_word, ngram){
      out <- x_word[start:min(length(x_word),(start+ngram-1))]
      paste(out,collapse="")
    },x_word=x_word,ngram=ngram)
  }else{
    x_word <- paste(x_word, collapse = " ")
    x_word <- tokenize_ngrams(x_word,n=ngram,ngram_delim="", simplify = TRUE)
    x_word <- unlist(x_word)
  }
  
  # x_word <- paste(x_word,collapse=" ")
  return(x_word)
  
}


#moves padding from right to left
left_padding <- function(vec){
  pos_zero <- which(vec==0)[1]
  if(is.na(pos_zero)){
    return(vec)
  }else{
    num_zero <- length(vec)-which(vec==0)[1]+1
    temp <- vec[1:pos_zero-1]
    out <- c(rep(0,num_zero),temp)
    return(out)
  }
}

# help function to add cols of 0s to matrix
add0Cols <- function(mat,ncol){
  
  fill0 <- ncol-ncol(mat)
  if(fill0>0){
    mat <- cbind(mat,matrix(0,ncol=fill0,nrow=nrow(mat)))
  }else if(fill0<0){
    mat <- mat[,1:ncol,drop=FALSE]
  }
  return(mat)
}

# transform token matrix into one hot encode feature matrix
setupOneHot <- function(x, id_train = NULL, col_position = NULL){
  
  # build one hot encoding matrix from tokenized inputs
  # one hot encode 
  n_rec <- nrow(x)
  # if(!is.null(id_train)){
  #  n_rec <- length(id_train) 
  # }
  
  x <- as.data.table(x)
  m_vars <- copy(colnames(x))
  x[,ID:=.I]
  x <- melt(x, id.vars="ID", measure.vars = m_vars, value.name = "Token")
  x <- x[Token!=0]
  
  if(!is.null(col_position)){
    x[col_position,IDF:=IDF,on=.(Token)]
  }else{
    if(!is.null(id_train)){
      x[ID %in% id_train,IDF:=log(n_rec/uniqueN(ID)), by=.(Token)] # for training use only training subset
    }else{
      x[,IDF:=log(n_rec/uniqueN(ID)), by=.(Token)] # for training use only training subset
    }
    x[,IDF:=IDF[!is.na(IDF)][1],by=.(Token)]
  }
  x[is.na(IDF),IDF:=log(n_rec/1)]
  x[,TF:=as.numeric(.N),by=.(ID,Token)]
  x[,TF:=TF/max(TF),by=.(ID)]
  x[,TFIDF:=TF * IDF]
  x <- unique(x, by=c("Token","ID"))
  # n_rows <- x[,uniqueN(ID)]
  if(!is.null(col_position)){
    x[col_position,Token_position:=Token_position, on=.(Token)]
    x <- x[!is.na(Token_position)]
  }else{
    x[,Token_position:=.GRP,by=.(Token)] # relabel tokens
    col_position <- unique(x[,.(Token, Token_position, IDF)])
  }
  n_cols <- max(col_position[["Token_position"]]) 
  # gc(x)
  x <- Matrix::sparseMatrix(i=x[["ID"]], j=x[["Token_position"]],x=x[["TFIDF"]],dims = c(n_rec,n_cols))
  
  return(list(x, col_position))
  
}


#moves padding from right to left
left_padding <- function(vec){
  pos_zero <- which(vec==0)[1]
  if(is.na(pos_zero)){
    return(vec)
  }else{
    num_zero <- length(vec)-which(vec==0)[1]+1
    temp <- vec[1:pos_zero-1]
    out <- c(rep(0,num_zero),temp)
    return(out)
  }
}

#evaluation metrics
multiclass_precision_recall <- function(y_true, y_pred) {
  
  # Ensure factors with same levels
  classes <- sort(unique(c(y_true, y_pred)))
  y_true <- factor(y_true, levels = classes)
  y_pred <- factor(y_pred, levels = classes)
  
  cm <- table(y_true, y_pred)
  
  precision <- numeric(length(classes))
  recall <- numeric(length(classes))
  
  for (i in seq_along(classes)) {
    TP <- cm[i, i]
    FP <- sum(cm[, i]) - TP
    FN <- sum(cm[i, ]) - TP
    
    precision[i] <- ifelse((TP + FP) == 0, NA, TP / (TP + FP))
    recall[i]    <- ifelse((TP + FN) == 0, NA, TP / (TP + FN))
  }
  
  names(precision) <- classes
  names(recall) <- classes
  
  # Macro averages (ignore NA)
  macro_precision <- mean(precision, na.rm = TRUE)
  macro_recall    <- mean(recall, na.rm = TRUE)
  
  # Micro averages
  TP_total <- sum(diag(cm))
  FP_total <- sum(colSums(cm) - diag(cm))
  FN_total <- sum(rowSums(cm) - diag(cm))
  
  micro_precision <- TP_total / (TP_total + FP_total)
  micro_recall    <- TP_total / (TP_total + FN_total)
  
  list(
    confusion_matrix = cm,
    precision_per_class = precision,
    recall_per_class = recall,
    macro_precision = macro_precision,
    macro_recall = macro_recall,
    micro_precision = micro_precision,
    micro_recall = micro_recall
  )
}
