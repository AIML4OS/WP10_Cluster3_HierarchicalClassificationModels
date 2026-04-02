library(keras3)
library(tensorflow)

#keras3 version -----------------------------------------------------------------
PositionalEmbedding <- new_layer_class(
  classname = "PositionalEmbedding",
  register = TRUE,
  
  initialize = function(maxlen, embed_dim, ...) {
    super()$`__init__`(...)
    self$maxlen <- as.integer(maxlen)
    self$embed_dim <- as.integer(embed_dim)
    self$pos_emb <- layer_embedding(
      input_dim = self$maxlen,
      output_dim = self$embed_dim
    )
  },
  
  call = function(x) {
    batch_size <- tf$shape(x)[1]
    seq_len <- tf$shape(x)[2]
    
    positions <- tf$range(start = 0L, limit = seq_len, delta = 1L)
    pos_embeddings <- self$pos_emb(positions)  # (seq_len, embed_dim)
    
    # Add batch dimension
    pos_embeddings <- tf$expand_dims(pos_embeddings, axis = 0L)
    
    # Dynamically tile to match batch size
    pos_embeddings <- tf$tile(
      pos_embeddings,
      multiples = tf$stack(c(batch_size, 1L, 1L))
    )
    
    pos_embeddings
  }
)

TransformerBlock <- new_layer_class(
  classname = "TransformerBlock",
  register = TRUE,
  
  initialize = function(embed_dim, num_heads, ff_dim, rate = 0.1, ...) {
    super()$`__init__`(...)
    
    self$att <- layer_multi_head_attention(
      num_heads = num_heads,
      key_dim   = embed_dim,
      dropout   = rate
    )
    
    self$ffn_dense1 <- layer_dense(units=as.integer(ff_dim), activation = "relu")
    self$ffn_dense2 <- layer_dense(units=as.integer(embed_dim))
    
    self$norm1 <- layer_layer_normalization(epsilon = 1e-6)
    self$norm2 <- layer_layer_normalization(epsilon = 1e-6)
    
    self$drop1 <- layer_dropout(rate=rate)
    self$drop2 <- layer_dropout(rate=rate)
  },
  
  call = function(x, training = FALSE) {
    # Multi-head attention + residual + norm
    attn_out <- self$att(x, x)
    attn_out <- self$drop1(attn_out, training = training)
    x1 <- self$norm1(x + attn_out)
    
    # Feed-forward + residual + norm
    ffn_out <- self$ffn_dense1(x1)
    ffn_out <- self$ffn_dense2(ffn_out)
    ffn_out <- self$drop2(ffn_out, training = training)
    x2 <- self$norm2(x1 + ffn_out)
    
    x2
  }
)


build_model_k3 <- function(
    inputs,
    embed_dim,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    maxlen,
    dense_dim,
    num_words,
    dropout1 = 0,
    dropout2 = 0,
    pretrained_inputs = NULL
) {
  
  # Token embedding
  x <- inputs |>
    layer_embedding(
      input_dim  = num_words,
      output_dim = embed_dim,
      mask_zero = TRUE
    )

  
  pos_embed <- PositionalEmbedding(maxlen=maxlen, embed_dim=embed_dim)(inputs)
  
  # Add embeddings
  x <- layer_add(list(x, pos_embed))

  
  # Transformer blocks
  for (i in seq_len(num_transformer_blocks)) {
    x <- TransformerBlock(embed_dim=embed_dim, 
                          num_heads=num_heads, 
                          ff_dim=ff_dim, 
                          rate = dropout1)(x)
  }
  
  
  x <- x |>
    layer_global_max_pooling_1d() |>
    layer_dense(dense_dim, activation = "relu") |>
    layer_dropout(dropout2)
  
  
  return(x)
}

