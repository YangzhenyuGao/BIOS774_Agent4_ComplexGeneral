# Report R version, library paths, and whether packages a multiome/pathology analysis might call are present.
cat("R", as.character(getRversion()), "\n"); cat("libPaths:", paste(.libPaths(), collapse=" | "), "\n")
pk <- c("Seurat","Signac","SeuratObject","Matrix","glmnet","ranger","survival","data.table","ggplot2","BradleyTerry2","pROC","boot","irlba","uwot","hdf5r","GenomicRanges","EnsDb.Hsapiens.v86","rmarkdown","knitr")
for (p in pk) { v <- tryCatch(as.character(packageVersion(p)), error=function(e) "MISSING"); cat(sprintf("RPKG %-22s %s\n", p, v)) }
ok <- tryCatch({ suppressPackageStartupMessages(library(Matrix)); m <- rsparsematrix(100, 50, 0.1); s <- svd(as.matrix(m)); TRUE }, error=function(e) FALSE)
cat("R_LINALG_OK", ok, "\n")
