hier_accuracy <- function(pred_matrix, true_matrix) {
  # Ensure input is a matrix
  stopifnot(is.matrix(pred_matrix), is.matrix(true_matrix))
  
  # Ensure dimensions match
  stopifnot(all(dim(pred_matrix) == dim(true_matrix)))
  
  # Number of levels (columns)
  num_levels <- ncol(true_matrix)
  
  # Compute accuracy per row
  acc_values <- sapply(1:nrow(true_matrix), function(i) {
    pred_seq <- pred_matrix[i, ]
    true_seq <- true_matrix[i, ]
    
    # Find first incorrect level
    correct_levels <- sum(cumprod(pred_seq == true_seq))  # Counts matching until first mismatch
    
    # Compute hierarchical accuracy
    correct_levels / num_levels
  })
  
  # Return mean accuracy
  mean(acc_values)
}