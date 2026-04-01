setwd("R/")

# load helper functions
source("helpers.R")
# load NACE dictionary as toy data set
source("load_data_NACE.R")
# load function(s) for hierarchical evaluation
source("eval.R")
# model definitions
source("helper_transformer.R")

# define target variable to classify
target <- "Code"

# clean data and split into train/test/valid set
source("train_test_dat.R")


# Flat Model --------------------------------------------------------------
source("flat_model.R")
result_flat


# Stacked Model -----------------------------------------------------------
source("stacked_model.R")
res_stacked #TODO

# Multiple Outputs Model --------------------------------------------------
source("multiple_outputs_model.R")
result_multiple_outs

# Hierarchical Loss -------------------------------------------------------
source("hierarchical_loss_model.R")
result_hier_loss



rbind(result_flat,
      res_stacked,
      result_multiple_out,
      result_hier_loss)