#definiere Matrix mit penalties je nachdem wie weit ein Code vom anderen entfernt ist
shared_prefix_length <- function(code1, code2) {
  n <- min(nchar(code1), nchar(code2))
  for (i in seq_len(n)) {
    if (substr(code1, 1, i) != substr(code2, 1, i)) {
      return(i - 1)
    }
  }
  return(n)
}

hierarchy_distance <- function(code1, code2, max_depth = 5) {
  shared <- shared_prefix_length(code1, code2)
  return(max_depth - shared)
}


# hierarische loss funktion
custom_hierarchy_loss <- function(y_true, y_pred) {
  
  # #y_true müssen integer sein
  y_true_labels <- tf$cast(y_true, tf$int32)
  
  #distances for all true labels to the predicted labels -> pick the correct distance row= vector of penalties
  distances <- tf$gather(hierarchy_matrix_tensor, y_true_labels)
  # y_pred=softmax output (probablitly vec), distances=penalties
  loss <- tf$reduce_sum(y_pred * distances, axis = -1L)
  #tf$print("loss:", loss)
  # mean über batch
  tf$reduce_mean(loss)
}


custom_combined_loss <- function(y_true, y_pred) {
  # Hierarchical part
  y_true_labels <- tf$cast(y_true, tf$int32)
  #tf$print(">>> shape y_true_lables:",y_true_labels$shape)
  distances <- tf$gather(hierarchy_matrix_tensor, y_true_labels)
  hierarchy_loss <- tf$reduce_sum(y_pred * distances, axis = -1L)
  
  # Crossentropy part
  crossentropy_loss <- tf$keras$losses$sparse_categorical_crossentropy(y_true, y_pred,from_logits=FALSE)
  
  # Combine both (e.g., weight 0.7 hierarchy + 0.3 CE)
  tf$reduce_mean(0.7 * hierarchy_loss + 0.3 * crossentropy_loss)
}

#hierarchisch gewichtete accuracy
weighted_accuracy <- custom_metric("weighted_accuarcy", function(y_true, y_pred) {
  y_true_labels <- tf$cast(tf$squeeze(y_true), tf$int32)
  #tf$print("y_true_labels shape:", tf$shape(y_true_labels))
  
  y_pred_labels <- tf$argmax(y_pred, axis = -1L)
  y_pred_labels <- tf$cast(y_pred_labels, tf$int32)
  
  #tf$print("y_pred_labels shape:", tf$shape(y_pred_labels))
  
  # Distanz für jedes Paar True/Predicted aus der Tensor-Matrix holen
  distances <- tf$gather_nd(hierarchy_matrix_tensor, tf$stack(list(y_true_labels, y_pred_labels), axis = 1L))
  
  # Weighted accuracy: 1 - dist (weil dist in [0,1] normalisiert ist)
  weighted_acc <- 1 - distances
  
  tf$reduce_mean(weighted_acc)
})

