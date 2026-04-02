train_hlevel <- function(batch_size=1000,
                         top_k="sparse_categorical_accuracy",
                         patience=5,
                         max_epochs=15,
                         model,
                         index.train,
                         index.valid,
                         callbacks,
                         preds #matrix mit predictions, ncol=#Code im vorherigem level
                         ){

  num_steps <- ceiling(length(index.train)/batch_size)
  eval_model <- c(loss = 1000, accuracy=0,top_5_categorical_accuracy = 0, top_10_categorical_accuracy = 0)
  current_best <- 0#eval_model[topk]
  current_best_m <- 1
  patience <- 5
  for(m in 1:max_epochs){
    message("Epoch ",m,"/",max_epochs)
    
    num_steps <- ceiling(length(index.train)/batch_size)
    
    all_samples <- index.train
    all_samples <- sample(all_samples, size=length(all_samples))
    
    pb <- progress_bar$new(
      format = "Step :current/:total [:bar] :percent in :elapsed",
      total = num_steps,
      clear = FALSE,  # Keeps the bar visible after completion
      width = 60
    )
    
    for(step in 1:num_steps){
      
      batch <- all_samples[1:min(length(all_samples),batch_size)]
      all_samples <- all_samples[-c(1:length(batch))]
      
      x.train <- list(OneHot = as.matrix(cbind(x[batch,],x2[batch,])),
                      Transformer_3Gram=x_1[batch,],
                      Transformer_5Gram=x_2[batch,],
                      hier_pred=as.matrix(preds[batch,]))
      y <- as.array(as.integer(dat[batch,][["NACE_target"]]))
      
      
      
      weights.batch <- NULL
      if(useWeights == TRUE){
        weights.batch <- matrix(as.numeric(dat[batch,]$count), nrow=length(batch))
      }
      
      model$train_on_batch(
          x=x.train, 
          y=y
        )
      
      pb$tick()
      rm(x.train);gc()
    }
    
    # evaluate on batch and take averages
    all_samples <- index.valid
    out_eval <- list()
    num_steps <- ceiling(length(index.valid)/batch_size)
    for(step in 1:num_steps){
      batch <- all_samples[1:min(length(all_samples),batch_size)]
      all_samples <- all_samples[-c(1:length(batch))]
      
      x.valid <- list(OneHot = as.matrix(cbind(x[batch,],x2[batch,])),
                      Transformer_3Gram=x_1[batch,],
                      Transformer_5Gram=x_2[batch,],
                      hier_pred=as.matrix(preds[batch,]))
      y <- as.array(as.integer(dat[batch,][["NACE_target"]]))

      
      # pred_model <- predict_on_batch(model, x.valid)
      # pred_model <- cbind(data.table(Target=dat[batch,][["ISCO_target"]]),pred_model)
      # setnames( pred_model , paste0("V",1:(ncol(pred_model)-1)),
      #           as.character(y_lookup[order(ISCO_target)]$V1))
      # out_pred <- c(out_pred, list(pred_model))
      
      eval_model <- model|>evaluate(
                             x=x.valid, 
                             y=y,
                             callbacks = callbacks)
      eval_model <- as.data.table(as.list(eval_model))
      eval_model[,N:=length(batch)]
      
      out_eval <- c(out_eval, list(eval_model))
      rm(x.valid);gc()
    }
    out_eval <- rbindlist(out_eval)
    avg_cols <- colnames(out_eval)
    avg_cols <- avg_cols[avg_cols !="N"]
    eval_model <- out_eval[,sapply(.SD,function(z,N){weighted.mean(z,N)},N=N),.SDcols=c(avg_cols)]
    print(eval_model)
    
    if(eval_model[[topk]]>current_best){
      best_model <- model
      current_best <- eval_model[[topk]]
      current_best_m <- m
      #keras::save_model_hdf5(model, filepath = file.path(path_save,"model.hdf5"))
      #keras::save_model_hdf5(model, filepath = file.path(deployPath,"model.hdf5"))
      
      #write(yaml_params,file=file.path(path_save,"parameter_config.yaml"))
      #write(yaml_params,file=file.path(deployPath,"parameter_config.yaml"))
      
    }
    if(m-current_best_m>(patience-1)){
      break
    }
    gc();
  }
  return(best_model)
}


predict_hlevel <- function(batch_size=1000,
                           model_h0,
                           preds=NULL,
                           index,
                           x_1, #transformer 1st gram
                           x_2, #transformer 2nd gram
                           x, #onehot 1st gram
                           x2 #onehot 2nd gram
                           ){
  all_samples <- index
  
  out_eval <- list()
  num_steps <- ceiling(length(all_samples)/batch_size)
  dat_pred <- list()
  for(step in 1:num_steps){
    
    if(step %% 25 == 0){
      print(step)
    }
    
    batch <- all_samples[1:min(length(all_samples),batch_size)]
    all_samples <- all_samples[-c(1:length(batch))]
    
    x.test <- list(OneHot = as.matrix(cbind(x[batch,],x2[batch,])),
                   Transformer_3Gram=x_1[batch,],
                   Transformer_5Gram=x_2[batch,]
    )
    
    if(!is.null(preds)){
      x.test <- c(list(hier_pred=as.matrix(preds[batch,])),
                  x.test)
    }
    
    y_pred <- model_h0$predict_on_batch(x.test)
    dat_pred <- c(dat_pred,list(as.data.table(y_pred)))
    remove(y_pred);gc()
  }
  
  dat_pred <- rbindlist(dat_pred)
  return(dat_pred)
}


predict_hier_model <- function(dat, #test data input to predict codes for
                               model_h0, 
                               model_h1,
                               model_h2,
                               model_h3,
                               translator_list, 
                               FLAGS,
                               text_clean_colname="Text_clean",
                               roll=FALSE,
                               keep_spaces=TRUE,
                               batch_size=1000,
                               tok_1,
                               tok_2,
                               params,
                               top_k=10){

  # Setup INPUTS
  if(is.character(tok_1)){
    tok_1 <- readRDS(tok_1)
  }
  x_emb_1 <- setupEmbedding2(dat,ngram=FLAGS$nGram1,string_col = text_clean_colname, 
                             roll = roll, keep_spaces = keep_spaces,
                             tok=tok_1)
  
  x_1 <- x_emb_1[[1]]
  num_words_1 <- x_emb_1$num_words
  
  if(ncol(x_1)>params$shape_ngram1){
    x_1 <- x_1[,1:params$shape_ngram1]
  }else{
    x_1 <- add0Cols(x_1,ncol=params$shape_ngram1)
  }
  
  
  if(is.character(tok_2)){
    tok_2 <- readRDS(tok_2)
  }
  x_emb_2 <- setupEmbedding2(dat,ngram=FLAGS$nGram2,string_col = "Text_clean", 
                             roll = roll, keep_spaces = keep_spaces,
                             tok = tok_2)
  x_2 <- x_emb_2[[1]]
  num_words_2 <- x_emb_2$num_words
  
  if(ncol(x_2)>params$shape_ngram2){
    x_2 <- x_2[,1:params$shape_ngram2]
  }else{
    x_2 <- add0Cols(x_2,ncol=params$shape_ngram2)
  }
  
  # onehot input
  x <- setupOneHot(x_1, col_position = params$OneHot$x1_info)
  x2 <- setupOneHot(x_2,, col_position = params$OneHot$x2_info)
  
  x <- x[[1]]
  x2 <- x2[[1]]
  
  #predict h0 level
  if(is.character(model_h0)){
    model_h0 <- load_model_hdf5(model_h0)
  }
  
  preds_h0 <- predict_hlevel(batch_size=batch_size,model_h0=model_h0,
                             index=1:nrow(dat),
                             x_1=x_1, #transformer 1st gram
                             x_2=x_2, #transformer 2nd gram
                             x=x, #onehot 1st gram
                             x2=x2) #onehot 2nd gram
  
  #predict h1 level
  if(is.character(model_h1)){
    model_h1 <- load_model_hdf5(model_h1)
  }
  preds_h1 <- predict_hlevel(batch_size=batch_size,
                             model_h0=model_h1,
                             index=1:nrow(dat),
                             preds=preds_h0,
                             x_1=x_1, #transformer 1st gram
                             x_2=x_2, #transformer 2nd gram
                             x=x, #onehot 1st gram
                             x2=x2)
  
  
  #predict h2 level
  if(is.character(model_h2)){
    model_h2 <- load_model_hdf5(model_h2)
  }
  preds_h2 <- predict_hlevel(batch_size=batch_size,model_h0=model_h2,
                             index=1:nrow(dat),
                             preds=preds_h1,
                             x_1=x_1, #transformer 1st gram
                             x_2=x_2, #transformer 2nd gram
                             x=x, #onehot 1st gram
                             x2=x2)
  
  
  #predict h3 level
  if(is.character(model_h3)){
    model_h3 <- load_model_hdf5(model_h3)
  }
  preds_h3 <- predict_hlevel(batch_size=1000,model_h0=model_h3,
                             index=1:nrow(dat),
                             preds=preds_h2,
                             x_1=x_1, #transformer 1st gram
                             x_2=x_2, #transformer 2nd gram
                             x=x, #onehot 1st gram
                             x2=x2)
  
  
  #concat and translate all predictions
  preds_all <- list(preds_h0,preds_h1,
                    preds_h2,preds_h3)
  
  preds_max <- lapply(preds_all,function(x){
    apply(x,1,which.max)-1
  })
  
  preds_max <- data.table(do.call(cbind,preds_max))

  
  # Apply the function to each column with the corresponding translator
  if(is.character(translator_list)){
    translator_list <- readRDS(translator_list)
  }
  
  if(is.null(top_k)|top_k==1){
    translated_dt <- as.data.table(mapply(translate_column, preds_max, translator_list[1:4], SIMPLIFY = FALSE))
    setnames(translated_dt, c("h0","h1","h2","h3"))
    translated_dt[,h0:=substr(h0,1,1)] #remove whitespace
    
    preds_full <- apply(translated_dt, 1, paste, collapse = "")
    preds_full <- substr(preds_full,2,6)
    
    return(preds_full)
  }else{
    return(topk_hier_pred(translator_list,
                           preds_h1,
                           preds_h2,
                           preds_h3,
                           top_k,
                           dat))
  }
}


#hier werden die topk predictions gefunden, indem die höchsten probabilities per level
# jeweils mit den anderen levels multipliziert werden
# es werden pro level nur die probabilities >1/n_hier_classes verwendet
# dadurch können auch weniger als topk predictions ausgegeben werden
expand_elements <- function(h1, h2, h3, y_lookups, top_k = 10) {
  y_lookups <- y_lookups[[1]] #wird als list(list()) über Map gepassed

  combinations <<- as.data.table(expand.grid(h1[[2]], h2[[2]], h3[[2]]))
  setDT(combinations)
  combinations$product <<- apply(combinations, 1, prod)
  setnames(combinations,c("Var1","Var2","Var3"),c("p1","p2","p3"))
  
  expanded_classes <- as.data.table(expand.grid(h1[[1]], h2[[1]], h3[[1]], stringsAsFactors = FALSE))
  
  grid1 <<- cbind(combinations,expanded_classes)
  
  grid1[,pred1:=translate_column(Var1,y_lookups[[2]])]
  grid1[,pred2:=translate_column(Var2,y_lookups[[3]])]
  grid1[,pred3:=translate_column(Var3,y_lookups[[4]])]
  
  
  grid1[,code:=paste0(pred1,pred2,pred3,sep="")]
  
  #filter out invalid codes
  grid1[,valid:=1]
  grid1[!code %in% y_lookups[[5]],valid:=0]
  
  grid1 <- grid1[valid==1,]
  
  grid1[,topk:=0]
  grid1[order(-product)[1:top_k], topk := 1]
  
  top_indices <- order(-grid1$product)[1:top_k]
  
  # Assign ranks 1 highest to lowest based on the sorted order
  grid1[top_indices, topk := rank(-product, ties.method = "first")]
  
  grid1 <- grid1[topk %in% 1:top_k,]
  
  topk_probs <- grid1[order(product,decreasing = T),product]
  topk_preds <- grid1[order(product,decreasing = T),code]
  
  return(list(topk_preds,topk_probs))
}


# wir brauchen hier nur hier 1-3 (nicht hier 0), weil hier 0 und hier 1 in Kombination eindeutig sind
topk_hier_pred <- function(y_lookups, #string oder lookuplist (translator_list)
                           preds_h1,
                           preds_h2,
                           preds_h3,
                           top_k,
                           dat){

  if(is.character(y_lookups)){
    y_lookups <- readRDS(y_lookups)
  }
  
  #finden die Codes (bzw teilcodes) die eine höhere Wahrscheinlichkeit als Zufälligkeit haben (=1/Anzahl mögl Codes)
  h1_classes <- apply(preds_h1,1,function(x)return(list(which(x>(1/ncol(preds_h1)))-1,
                                                        x[x>(1/ncol(preds_h1))])))
  
  h2_classes <- apply(preds_h2,1,function(x)return(list(which(x>(1/ncol(preds_h2)))-1,
                                                        x[x>(1/ncol(preds_h2))])))                  
  
  h3_classes <- apply(preds_h3,1,function(x)return(list(which(x>(1/ncol(preds_h3)))-1,
                                                        x[x>(1/ncol(preds_h3))]))) 
  
  # print(length(h1_classes))
  # print(length(h2_classes))
  # print(length(h3_classes))
  
  # Apply probability multiplications to all elements in lists ->find top pred classes
  results <- Map(expand_elements, h1_classes, h2_classes, h3_classes,
                 MoreArgs = list(top_k = top_k, y_lookups=list(y_lookups)))
  
  res <- rbindlist(Map(function(row, lst) {
    row_dt <- as.data.table(row)
    # Repeat row for each element in the vectors
    row_dt[rep(1, length(lst[[1]]))][, `:=`(Class = lst[[1]], Probability = lst[[2]])]
    
  }, split(dat, seq_len(nrow(dat))), results))
  
  return(res)
}

translate_column <- function(col, translator) {
  merge(data.table(NACE_target = col), translator, by = "NACE_target", all.x = TRUE,sort=F)[[2]]
}

