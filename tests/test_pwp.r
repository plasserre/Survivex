library(survival)
data(cgd)

fit <- coxph(Surv(tstart, tstop, status) ~ treat + age + cluster(id), 
             data = cgd, method = "efron")

# Extract score residuals
resid_score <- residuals(fit, type="score")

cat("\n=== SCORE RESIDUALS INFO ===\n")
cat("Dimensions:", dim(resid_score), "\n")
cat("\nFirst 10 score residuals:\n")
print(head(resid_score, 10))

# Sum scores by subject (first 10 subjects)
cat("\n=== SCORE SUMS BY SUBJECT (first 10) ===\n")
for(i in 1:min(10, length(unique(cgd$id)))) {
  subj_rows <- which(cgd$id == i)
  if(length(subj_rows) > 0) {
    subj_scores <- resid_score[subj_rows, , drop=FALSE]
    total_score <- colSums(subj_scores)
    cat(sprintf("Subject %d (%d obs): treat=%.6f, age=%.6f\n", 
                i, length(subj_rows), total_score[1], total_score[2]))
  }
}

# Compute B matrix manually
score_by_cluster <- matrix(0, nrow=length(unique(cgd$id)), ncol=2)
colnames(score_by_cluster) <- c("treat", "age")

for(i in 1:nrow(cgd)) {
  cluster_idx <- which(sort(unique(cgd$id)) == cgd$id[i])
  score_by_cluster[cluster_idx,] <- score_by_cluster[cluster_idx,] + resid_score[i,]
}

B <- t(score_by_cluster) %*% score_by_cluster

cat("\n=== B MATRIX (from R score residuals) ===\n")
print(B)

cat("\n=== CHECK: Max absolute score per cluster ===\n")
max_scores <- apply(abs(score_by_cluster), 2, max)
cat("Max treat score:", max_scores[1], "\n")
cat("Max age score:", max_scores[2], "\n")