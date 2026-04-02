# stacked model: one model per level/digit

library(progress)
library(data.table)
source("train_hlevel.R")

target <- "Code"

# parameter settings for tokenization
keep_spaces <- TRUE
roll <- FALSE

if(!exists("dat")|!exists("index.train")|!exists("index.valid")|!exists("dat_eval")){
  source("load_data_NACE.R")
  source("train_test_dat.R")
}

hparams <- fread("data/best_params.csv")
params <- list()
# Hyperparameter flags ---------------------------------------------------
# FLAGS are used for Parametertuning: https://tensorflow.rstudio.com/guides/tfruns/tuning
FLAGS <- flags(
  
  # transformer flags
  flag_numeric("dropout1", hparams$dropout1),
  flag_numeric("dropout2", hparams$dropout2),
  flag_numeric("num_heads", hparams$num_heads),
  flag_numeric("num_transformer_blocks",hparams$num_trans_blocks),
  flag_numeric("ff_dim",hparams$hidden_layer_dim),
  flag_numeric("maxlen",50),
  flag_numeric("embed_dim",hparams$embed_dim),
  flag_numeric("dense_dim",hparams$dense_dim),
  flag_numeric("nGram1",3),
  flag_numeric("nGram2",5),
  
  # one hot layer
  flag_numeric("dropoutOnehot",0.3),
  flag_integer("unitsOnehot",250),
  
  # second to last relu layer
  flag_numeric("dropoutrelu",.1), # dropout for checkbox layer 
  flag_integer("unitsrelu",450), # untis after checkbox layer
  
  # Pretrained Embeddings
  flag_numeric("dropoutPretrained",0.3),
  flag_integer("unitsPretrained",250),
  
  # model training parameter
  flag_numeric("epochs",20),
  flag_numeric("patience",3),
  flag_string("monitor","val_top_10_categorical_accuracy"),
  flag_numeric("batchsize",32),
  flag_numeric("min_delta",0.003),
  flag_numeric("learningrate",0.001),
  flag_numeric("regularization",0.001),
  
  flag_string("Target","Code")
)



dat[,h1:=substr(Code,1,2)]
dat[,h2:=substr(Code,3,4)]
dat[,h3:=substr(Code,5,5)]

unique_level1 <- unique(dat[,substr(Code,1,2)])
unique_level2 <- unique(dat[,substr(Code,1,4)])
unique_level3 <- unique(dat[,Code])

dat <- dat[,h0:=substr(NACE25,1,2)]
dat <- dat[!is.na(Text),]
# ------------------------------------------------------------------------------------------------------
# Setup INPUTS
# tokenizer first nGram
x_emb_1 <- setupEmbedding2(dat,ngram=FLAGS$nGram1,string_col = "Text_clean", roll = roll, keep_spaces = keep_spaces)
x_1 <- x_emb_1[[1]]
num_words_1 <- x_emb_1[[2]]
tok_1 <- x_emb_1[[3]]

if(nrow(x_1) != nrow(dat)){
  stop("Tokenization failed")
}

# tokenization 2nd nGram
x_emb_2 <- setupEmbedding2(dat,ngram=FLAGS$nGram2,string_col = "Text_clean", roll = roll, keep_spaces = keep_spaces)
x_2 <- x_emb_2[[1]]
num_words_2 <- x_emb_2[[2]]
tok_2 <- x_emb_2[[3]]

if(nrow(x_2) != nrow(dat)){
  stop("Tokenization failed")
}

dim(x_1) #39
dim(x_2) #31

dat[,NACE_target:=as.numeric(factor(get(FLAGS$Target)))-1]
y_lookup <- unique(dat[,.(get(FLAGS$Target),NACE_target)])

n_classes <- length(unique(dat$NACE_target))

params$shape_ngram1 <- ncol(x_1)
params$shape_ngram2 <- ncol(x_2)

# build one hot encoding matrix from tokenized inputs
# one hot encode 
x <- setupOneHot(x_1)
x2 <- setupOneHot(x_2)

x1_info <- x[[2]]
x2_info <- x2[[2]]

params$OneHot$x1_info <- x[[2]]
params$OneHot$x2_info <- x2[[2]]

x <- x[[1]]
x2 <- x2[[1]]

dat[, ID_Text := as.integer(factor(Text, levels = unique(Text)))]

dat[,count:=.N,by=Code]

# define neural network architecture H0 ---------------------------------------------------
target <- "h0"

dat[,NACE_target:=as.numeric(factor(get(target)))-1]
y_lookuph0 <- unique(dat[,.(get(target),NACE_target)])


# one hot input
input_length <- ncol(x) + ncol(x2)
input_onehot <- layer_input(shape=input_length, name="OneHot")
model_onehot <- input_onehot %>%
  layer_dropout(FLAGS$dropoutOnehot)%>%
  layer_dense(FLAGS$unitsOnehot,activation = "relu")%>%
  layer_batch_normalization()


# first ngram embedding followed by transformer
input_length <- ncol(x_1)
input_tr3 <- layer_input(shape=input_length, name="Transformer_3Gram")

input_pretrained <- NULL


# first ngram embedding followed by transformer
model_tr3 <- build_model_k3(
  input_tr3,
  num_heads = FLAGS$num_heads,
  ff_dim = FLAGS$ff_dim, # hidden layer size in transformer layer
  num_transformer_blocks = FLAGS$num_transformer_blocks,
  dense_dim = FLAGS$dense_dim,
  dropout1 = FLAGS$dropout1,
  dropout2 = FLAGS$dropout2,
  maxlen = min(input_length,FLAGS$maxlen),
  embed_dim = FLAGS$embed_dim,
  num_words = num_words_1,
  pretrained_inputs=input_pretrained
)

# 2nd ngram embedding followed by transformer
input_length <- ncol(x_2)
input_tr5 <- layer_input(shape=input_length, name="Transformer_5Gram")
model_tr5 <- build_model_k3(
  input_tr5,
  num_heads = FLAGS$num_heads,
  ff_dim = FLAGS$ff_dim, # hidden layer size in transformer layer
  num_transformer_blocks = FLAGS$num_transformer_blocks,
  dense_dim = FLAGS$dense_dim,
  dropout1 = FLAGS$dropout1,
  dropout2 = FLAGS$dropout2,
  maxlen = min(input_length,FLAGS$maxlen),
  embed_dim = FLAGS$embed_dim,
  num_words = num_words_2,
  pretrained_inputs=input_pretrained
)




# concatenate
model_list <- list(model_onehot,  model_tr3, model_tr5) 
inputsAll <- c(input_onehot, input_tr3,input_tr5) #c(input_checkbox, list(input_tr3),list(input_tr5))

model <- layer_concatenate(model_list)

# relu layers
model <- model%>%
  layer_dropout(FLAGS$dropoutrelu)%>%
  layer_dense(FLAGS$unitsrelu,activation = "relu")

# final layer
output <- model%>%
  layer_batch_normalization() %>%
  layer_dense(uniqueN(dat[[target]]),activation="softmax",
              kernel_regularizer = regularizer_l2(l=FLAGS$regularization))

# compile model and set compiler flags
model <- keras_model(
  inputs=inputsAll,
  outputs=output
)
model$compile(
  optimizer = optimizer_adam(learning_rate = 0.001),
  loss = "sparse_categorical_crossentropy",
  metrics = list("sparse_categorical_accuracy")
)



summary(model)


# training model  ---------------------------------------------------


callbacks <- list(keras3::callback_early_stopping(monitor = "val_loss", 
                                                 min_delta = 0.01, 
                                                 patience = 10, 
                                                 verbose = 1, 
                                                 restore_best_weights=TRUE))

# train on batch to minimize memory usage
batch_size <- 1000
num_steps <- ceiling(length(index.train)/batch_size)
eval_model <- c(loss = 1000, accuracy=0,top_5_categorical_accuracy = 0, top_10_categorical_accuracy = 0)
max_epochs <- 10
topk <- "sparse_categorical_accuracy"
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
                    Transformer_5Gram=x_2[batch,])
    y <- as.array(as.integer(dat[batch,][["NACE_target"]]))
    
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
                    Transformer_5Gram=x_2[batch,])
    y <- as.array(as.integer(dat[batch,][["NACE_target"]]))
    
    
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
  
  if(eval_model[[topk]]>current_best){
    best_model <- model
    
    current_best <- eval_model[[topk]]
    current_best_m <- m
    
  }
  if(m-current_best_m>(patience-1)){
    break
  }
  gc();
}


model_h0 <- copy(best_model)

# model h1 ----------------------------------------------------------------
target <- "h1"

dat[,NACE_target:=as.numeric(factor(get(target)))-1]
y_lookuph1 <- unique(dat[,.(get(target),NACE_target)])

# new layer: inputs are predicted probabilities from previous layers
input_length <- uniqueN(dat$h0)
input_h0 <- layer_input(shape=input_length, name="hier_pred")
model_h0_pred <- input_h0 %>%
  layer_dropout(0.2)%>%
  layer_dense(250,activation = "relu")%>%
  layer_batch_normalization()


# concatenate
model_list <- list(model_onehot,  model_tr3, model_tr5,model_h0_pred) 
inputsAll <- c(input_onehot, input_tr3,input_tr5,input_h0) 

model <- layer_concatenate(model_list)

# relu layers
model <- model%>%
  layer_dropout(FLAGS$dropoutrelu)%>%
  layer_dense(FLAGS$unitsrelu,activation = "relu")

# final layer
output <- model%>%
  layer_batch_normalization() %>%
  layer_dense(uniqueN(dat[[target]]),activation="softmax",
              kernel_regularizer = regularizer_l2(l=FLAGS$regularization))

# compile model and set compiler flags
model <- keras_model(
  inputs=inputsAll,
  outputs=output
)
model$compile(
  optimizer = optimizer_adam(learning_rate = 0.001),
  loss = "sparse_categorical_crossentropy",
  metrics = list("sparse_categorical_accuracy")
)



summary(model)

# training model  ---------------------------------------------------


callbacks <- list(keras3::callback_early_stopping(monitor = "val_loss", 
                                                 min_delta = 0.01, 
                                                 patience = 10, 
                                                 verbose = 1, 
                                                 restore_best_weights=TRUE))

preds_h0 <- predict_hlevel(batch_size=1000,
                           model_h0=model_h0,
                           index=c(index.train,index.valid),
                           preds=NULL,
                           x_1, #transformer 1st gram
                           x_2, #transformer 2nd gram
                           x, #onehot 1st gram
                           x2 #obehot 2nd gram
                           )

useWeights <- FALSE
#train next level, with prediction probability matrix of previous level as input
model_h1 <- train_hlevel(batch_size=1000,
                         top_k="sparse_categorical_accuracy",
                         patience=5,
                         max_epochs=10,
                         index.train=index.train,
                         index.valid=index.valid,
                         preds=preds_h0,
                         model=model,
                         callbacks=callbacks)


# model h2 ----------------------------------------------------------------

target <- "h2"

dat[,NACE_target:=as.numeric(factor(get(target)))-1]
y_lookuph2 <- unique(dat[,.(get(target),NACE_target)])

# new layer: inputs are predicted probabilities from previous layers
input_length <- uniqueN(dat$h1)
input_h1 <- layer_input(shape=input_length, name="hier_pred")
model_h1_pred <- input_h1 %>%
  layer_dropout(0.2)%>%
  layer_dense(250,activation = "relu")%>%
  layer_batch_normalization()


# concatenate
model_list <- list(model_onehot,  model_tr3, model_tr5,model_h1_pred) 
inputsAll <- c(input_onehot, input_tr3,input_tr5,input_h1) 

model <- layer_concatenate(model_list)

# relu layers
model <- model%>%
  layer_dropout(FLAGS$dropoutrelu)%>%
  layer_dense(FLAGS$unitsrelu,activation = "relu")

# final layer
output <- model%>%
  layer_batch_normalization() %>%
  layer_dense(uniqueN(dat[[target]]),activation="softmax",
              kernel_regularizer = regularizer_l2(l=FLAGS$regularization))

# compile model and set compiler flags
model <- keras_model(
  inputs=inputsAll,
  outputs=output
)
model$compile(
  optimizer = optimizer_adam(learning_rate = 0.001),
  loss = "sparse_categorical_crossentropy",
  metrics = list("sparse_categorical_accuracy")
)



summary(model)

# training model  ---------------------------------------------------


callbacks <- list(keras3::callback_early_stopping(monitor = "val_loss", 
                                                 min_delta = 0.01, 
                                                 patience = 10, 
                                                 verbose = 1, 
                                                 restore_best_weights=TRUE))

preds_h1 <- predict_hlevel(batch_size=1000,
                           model_h0=model_h1,
                           index=c(index.train,index.valid),
                           preds=preds_h0,
                           x_1, #transformer 1st gram
                           x_2, #transformer 2nd gram
                           x, #onehot 1st gram
                           x2 #obehot 2nd gram
                           )

model_h2 <- train_hlevel(batch_size=1000,
                         top_k="sparse_categorical_accuracy",
                         patience=3,
                         max_epochs=7,
                         index.train=index.train,
                         index.valid=index.valid,
                         preds=preds_h1,
                         model=model,
                         callbacks=callbacks)



# model h3 ----------------------------------------------------------------
target <- "h3"

dat[,NACE_target:=as.numeric(factor(get(target)))-1]
y_lookuph3 <- unique(dat[,.(get(target),NACE_target)])

# new layer: inputs are predicted probabilities from previous layers
input_length <- uniqueN(dat$h2)
input_h2 <- layer_input(shape=input_length, name="hier_pred")
model_h2_pred <- input_h2 %>%
  layer_dropout(0.2)%>%
  layer_dense(250,activation = "relu")%>%
  layer_batch_normalization()


# concatenate
model_list <- list(model_onehot,  model_tr3, model_tr5,model_h2_pred) 
inputsAll <- c(input_onehot, input_tr3,input_tr5,input_h2) 

model <- layer_concatenate(model_list)

# relu layers
model <- model%>%
  layer_dropout(FLAGS$dropoutrelu)%>%
  layer_dense(FLAGS$unitsrelu,activation = "relu")

# final layer
output <- model%>%
  layer_batch_normalization() %>%
  layer_dense(uniqueN(dat[[target]]),activation="softmax",
              kernel_regularizer = regularizer_l2(l=FLAGS$regularization))

# compile model and set compiler flags
model <- keras_model(
  inputs=inputsAll,
  outputs=output
)
model$compile(
  optimizer = optimizer_adam(learning_rate = 0.001),
  loss = "sparse_categorical_crossentropy",
  metrics = list("sparse_categorical_accuracy")
)



summary(model)

# training model  ---------------------------------------------------


callbacks <- list(keras3::callback_early_stopping(monitor = "val_loss", 
                                                 min_delta = 0.01, 
                                                 patience = 10, 
                                                 verbose = 1, 
                                                 restore_best_weights=TRUE))

preds_h2 <- predict_hlevel(batch_size=1000,
                           model_h0=model_h2,
                           index=c(index.train,index.valid),
                           preds=preds_h1,
                           x_1, #transformer 1st gram
                           x_2, #transformer 2nd gram
                           x, #onehot 1st gram
                           x2 #obehot 2nd gram
                           )

model_h3 <- train_hlevel(batch_size=1000,
                         top_k="sparse_categorical_accuracy",
                         patience=3,
                         max_epochs=5,
                         index.train=index.train,
                         index.valid=index.valid,
                         preds=preds_h2,
                         model=model,
                         callbacks=callbacks)



preds_h3 <- predict_hlevel(batch_size=1000,model_h0=model_h3,index=index.test,
                           preds=preds_h2,
                           x_1, #transformer 1st gram
                           x_2, #transformer 2nd gram
                           x, #onehot 1st gram
                           x2 #obehot 2nd gram
                           )


translator_list <- list(y_lookuph0,
                        y_lookuph1,
                        y_lookuph2,
                        y_lookuph3,
                        unique(dat$Code))

# Evaluation --------------------------------------------------------------

top_k <- 10
if(is.null(tok_k)){
  preds <- predict_hier_model(dat_eval, #test data input to predict codes for
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
                                    tok_1=tok_1, 
                                    tok_2=tok_2,
                                    params=params,
                                    top_k = NULL)
  
  dat_eval[,preds:=preds]
  
  acc <- mean(dat_eval$Code==dat_eval$preds)
  eval_m <- multiclass_precision_recall(dat_eval$Code,dat_eval$preds)
  
  # hier accuracy @1
  y_pred <- dat_eval$Code
  y_true <- dat_eval$preds
  
  y_pred_mat <- matrix(c(substr(y_pred,1,2),substr(y_pred,3,4),substr(y_pred,5,5)),ncol = 3)
  y_true_mat <- matrix(c(substr(y_true,1,2),substr(y_true,3,4),substr(y_true,5,5)),ncol = 3)
  
  hier_acc <- hier_accuracy(y_pred_mat,y_true_mat)
  
  res_stacked <- data.table(metric=c("accuarcy","precision","recall","hier_accuracy"),
                            value=c(acc,eval_m$macro_precision,eval_m$macro_recall,hier_acc),
                            type="stacked")
  
  openxlsx::write.xlsx(res_stacked,
                       paste0("data/results_stacked_model",Sys.Date(),".xlsx"))
  
}else{
  preds_top10 <- predict_hier_model(dat_eval, #test data input to predict codes for
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
                                    tok_1=tok_1, 
                                    tok_2=tok_2,
                                    params=params,
                                    top_k = top_k)
  
  preds_top10[,POS:=1:.N,by=ID]
  
  # accuracy hier model:
  acc <- preds_top10[POS==1, .(acc_hier = any(Class == Code)), by = ID][, mean(acc_hier)]
  
  #top3 mit hier model: 
  top3 <- preds_top10[POS<=3, .(acc_hier = any(Class == Code)), by = ID][, mean(acc_hier)]
  
  #top5 mit hier model: 
  top5 <- preds_top10[POS<=5, .(acc_hier = any(Class == Code)), by = ID][, mean(acc_hier)]
  
  #top10 mit hier model:
  top10 <- preds_top10[POS<=10, .(acc_hier = any(Class == Code)), by = ID][, mean(acc_hier)]
  
  # hier accuracy @1
  y_pred <- preds_top10[POS==1,Class]
  y_true <- preds_top10[POS==1,Code]
  
  y_pred_mat <- matrix(c(substr(y_pred,1,2),substr(y_pred,3,4),substr(y_pred,5,5)),ncol = 3)
  y_true_mat <- matrix(c(substr(y_true,1,2),substr(y_true,3,4),substr(y_true,5,5)),ncol = 3)
  
  hier_acc <- hier_accuracy(y_pred_mat,y_true_mat)
  
  res_stacked <- data.table(metric=c("accuarcy","top3_accuracy","top5_accuracy","top10_accuracy","hier_accuracy"),
                            value=c(acc,top3,top5,top10,hier_acc),
                            type="stacked")
  
  openxlsx::write.xlsx(res_stacked,
                       paste0("data/results_stacked_model",Sys.Date(),".xlsx"))
}




