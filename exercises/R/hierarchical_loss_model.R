library(progress)
library(keras3)
library(tensorflow)


source("hierarchical_loss.R")

hparams <- fread("data/best_params.csv")
params <- list()

roll <- FALSE #TRUE
keep_spaces <- TRUE
use_hier_loss <- TRUE #TRUE when hierarchical loss should be used during trainign

target <- "Code"

if(!exists("dat")|!exists("index.train")|!exists("index.valid")){
  source("load_data_NACE.R")
  source("train_test_dat.R")
}

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
  flag_numeric("embed_dim",128),#params$embed_dim),
  flag_numeric("dense_dim",128),#params$dense_dim),
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



# ------------------------------------------------------------------------------------------------------
# Setup INPUTS
x_emb_1 <- setupEmbedding2(dat,ngram=FLAGS$nGram1,string_col = "Text_clean", roll = roll, keep_spaces = keep_spaces)
x_1 <- x_emb_1[[1]]
num_words_1 <- x_emb_1[[2]]
tok_1 <- x_emb_1[[3]]

if(nrow(x_1) != nrow(dat)){
  stop("Tokenization failed")
}

x_emb_2 <- setupEmbedding2(dat,ngram=FLAGS$nGram2,string_col = "Text_clean", roll = roll, keep_spaces = keep_spaces)
x_2 <- x_emb_2[[1]]
num_words_2 <- x_emb_2[[2]]
tok_2 <- x_emb_2[[3]]

if(nrow(x_2) != nrow(dat)){
  stop("Tokenization failed")
}


dim(x_1) #45
dim(x_2) #34

dat[,NACE_target:=as.numeric(factor(get(FLAGS$Target)))-1]
y_lookup <- unique(dat[,.(get(FLAGS$Target),NACE_target)])


n_classes <- length(unique(dat$NACE_target))
# ------------------------------------------------------------------------------------------------------


length(index.train)
length(index.valid)
nrow(dat)




# build one hot encoding matrix from tokenized inputs
# one hot encode 
x <- setupOneHot(x_1)
x2 <- setupOneHot(x_2)

params$OneHot$x1_info <- x[[2]]
params$OneHot$x2_info <- x2[[2]]


x <- x[[1]]
x2 <- x2[[1]]


dat[, ID_Text := as.integer(factor(Text, levels = unique(Text)))]

gc()


# define eval metrics --------------------------------------------------------------------
metric_top_5_categorical_accuracy <-
  custom_metric("top_5_categorical_accuracy", function(y_true, y_pred) {
    metric_sparse_top_k_categorical_accuracy(y_true, y_pred, k = 5)
  })

metric_top_10_categorical_accuracy <-
  custom_metric("top_10_categorical_accuracy", function(y_true, y_pred) {
    metric_sparse_top_k_categorical_accuracy(y_true, y_pred, k = 10)
  })

# create matrix with penalties
nace_codes <- as.character(sort(unique(y_lookup$V1))) #have to be sorted in increasing oder
n_codes <- length(nace_codes)


dist_matrix <- matrix(0, nrow = n_codes, ncol = n_codes)

for (i in 1:n_codes) {
  for (j in 1:n_codes) {
    dist_matrix[i, j] <- hierarchy_distance(nace_codes[i], nace_codes[j], max_depth = 5)
  }
}
rownames(dist_matrix) <- colnames(dist_matrix) <- nace_codes
dist_matrix <- dist_matrix / max(dist_matrix)

# penalty matrix must be a tensor for loss function
hierarchy_matrix_tensor <- tf$constant(as.matrix(dist_matrix), dtype = tf$float32)


# define neural network architecture  ---------------------------------------------------
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
  pretrained_inputs=NULL
)

# second ngram embedding followed by transformer
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
  pretrained_inputs=NULL
)

params$shape_ngram1 <- ncol(x_1)
params$shape_ngram2 <- ncol(x_2)
                            
                            
# concatenate model parts
model_list <- list(model_onehot,  model_tr3, model_tr5) 
inputsAll <- c(input_onehot, input_tr3,input_tr5) 
model <- layer_concatenate(model_list)

# relu layers
model <- model%>%
  layer_dropout(FLAGS$dropoutrelu)%>%
  layer_dense(FLAGS$unitsrelu,activation = "relu")

# final layer
output <- model%>%
  layer_batch_normalization() %>%
  layer_dense(uniqueN(dat[["NACE_target"]]),activation="softmax",
              kernel_regularizer = regularizer_l2(l=FLAGS$regularization))

# compile model and set compiler flags
model <- keras_model(
  inputs=inputsAll,
  outputs=output
)
model$compile(
  optimizer = optimizer_adam(learning_rate = FLAGS$learningrate),
  loss=ifelse(use_hier_loss,custom_combined_loss,"sparse_categorical_crossentropy"), 
  metrics = list("sparse_categorical_accuracy", metric_top_10_categorical_accuracy, metric_top_5_categorical_accuracy
                         ,weighted_accuracy
  )
)

summary(model)

# training model  ---------------------------------------------------


callbacks <- list(keras3::callback_early_stopping(monitor = "val_loss", 
                                                 min_delta = 0.01, 
                                                 patience = 10, 
                                                 verbose = 1, 
                                                 restore_best_weights=TRUE))

######
# train on batch to minimize memory usage
batch_size <- 1000
num_steps <- ceiling(length(index.train)/batch_size)
eval_model <- c(loss = 1000, accuracy=0,top_5_categorical_accuracy = 0, top_10_categorical_accuracy = 0)
max_epochs <- FLAGS$epochs
topk <- "top_5_categorical_accuracy"
current_best <- eval_model[topk]
current_best_m <- 1
patience <- 10
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
    
    y <-  as.array(as.integer(dat[batch,][["NACE_target"]]))

    model$train_on_batch(
      x=x.train, 
      y=y)
    
    
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
    y=as.array(as.integer(dat[batch,][["NACE_target"]]))
    
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

model <- best_model


# eval on test data set ------------------------------------------------------
x_emb_1 <- setupEmbedding2(dat_eval,ngram=3,string_col = "Text_clean", 
                           roll = roll, keep_spaces = keep_spaces,
                           tok=tok_1)

x_1 <- x_emb_1[[1]]
num_words_1 <- x_emb_1$num_words

if(ncol(x_1)>params$shape_ngram1){
  x_1 <- x_1[,1:params$shape_ngram1]
}else{
  x_1 <- add0Cols(x_1,ncol=params$shape_ngram1)
}



x_emb_2 <- setupEmbedding2(dat_eval,ngram=FLAGS$nGram2,string_col = "Text_clean", 
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

x.test <- list(OneHot = as.matrix(cbind(x,x2)),
                Transformer_3Gram=x_1,
                Transformer_5Gram=x_2)

preds <- model$predict_on_batch(x.test)
out <- cbind(dat_eval[,.SD,.SDcols=c("Text_clean","Code","ID")],as.data.table(preds))
setnames(out,paste0("V",1:ncol(preds)),as.character(sort(unique(y_lookup$V1))))
out <- melt(out,id.vars=c("ID","Text_clean","Code"),
                 variable.factor = FALSE,
                 value.name="Probability",variable.name = "Class")

setorderv(out,c("ID","Probability"),order = c(1,-1))
out[,POSITION:=1:.N,by=c("ID")]
preds_top10 <- out[POSITION<=10,]

# accuracy hier model:
acc <- preds_top10[POSITION==1, .(acc_hier = any(Class == Code)), by = ID][, mean(acc_hier)]

#top3 mit hier model: 
top3 <- preds_top10[POSITION<=3, .(acc_hier = any(Class == Code)), by = ID][, mean(acc_hier)]

#top5 mit hier model: 
top5 <- preds_top10[POSITION<=5, .(acc_hier = any(Class == Code)), by = ID][, mean(acc_hier)]

#top10 mit hier model:
top10 <- preds_top10[POSITION<=10, .(acc_hier = any(Class == Code)), by = ID][, mean(acc_hier)]

# hier accuracy @1
y_pred <- preds_top10[POSITION==1,Class]
y_true <- preds_top10[POSITION==1,Code]

y_pred_mat <- matrix(c(substr(y_pred,1,2),substr(y_pred,3,4),substr(y_pred,5,5)),ncol = 3)
y_true_mat <- matrix(c(substr(y_true,1,2),substr(y_true,3,4),substr(y_true,5,5)),ncol = 3)

hier_acc <- hier_accuracy(y_pred_mat,y_true_mat)

result_hier_loss <- data.table(metric=c("accuarcy","top3_accuracy","top5_accuracy","top10_accuracy","hier_accuracy"),
                               value=c(acc,top3,top5,top10,hier_acc),
                               type="hierarchical_loss")
openxlsx::write.xlsx(result_hier_loss,
                     paste0("data/results_hierarchical_loss",Sys.Date(),".xlsx"))


