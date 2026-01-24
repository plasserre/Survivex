# Validation script for recurrent event models
# Compares SurviveX AG, PWP-TT, PWP-GT against R's survival::coxph

library(survival)

# Read data
df <- read.csv("tests/recurrent_event_data.csv")
cat("Data loaded:", nrow(df), "rows,", length(unique(df$id)), "subjects\n")
cat("Total events:", sum(df$event), "\n\n")

# ============================================================
# 1. Andersen-Gill Model (Breslow)
# No stratification, counting process format, clustered by subject
# ============================================================
cat(paste(rep("=", 60), collapse=""), "\n")
cat("1. ANDERSEN-GILL MODEL (Breslow ties)\n")
cat(paste(rep("=", 60), collapse=""), "\n")

ag_breslow <- coxph(Surv(start, stop, event) ~ x1 + x2,
                    data = df,
                    method = "breslow",
                    cluster = id)
cat("\nCoefficients:\n")
print(coef(ag_breslow), digits=15)
cat("\nRobust SE:\n")
print(sqrt(diag(vcov(ag_breslow))), digits=15)
cat("\nNaive SE:\n")
print(sqrt(diag(ag_breslow$naive.var)), digits=15)
cat("\nLog-likelihood:\n")
print(ag_breslow$loglik[2], digits=15)
cat("\n")

# ============================================================
# 2. Andersen-Gill Model (Efron)
# ============================================================
cat(paste(rep("=", 60), collapse=""), "\n")
cat("2. ANDERSEN-GILL MODEL (Efron ties)\n")
cat(paste(rep("=", 60), collapse=""), "\n")

ag_efron <- coxph(Surv(start, stop, event) ~ x1 + x2,
                  data = df,
                  method = "efron",
                  cluster = id)
cat("\nCoefficients:\n")
print(coef(ag_efron), digits=15)
cat("\nRobust SE:\n")
print(sqrt(diag(vcov(ag_efron))), digits=15)
cat("\nNaive SE:\n")
print(sqrt(diag(ag_efron$naive.var)), digits=15)
cat("\nLog-likelihood:\n")
print(ag_efron$loglik[2], digits=15)
cat("\n")

# ============================================================
# 3. PWP Total Time Model (Efron)
# Stratified by event number, total time scale
# ============================================================
cat(paste(rep("=", 60), collapse=""), "\n")
cat("3. PWP TOTAL TIME MODEL (Efron ties)\n")
cat(paste(rep("=", 60), collapse=""), "\n")

pwp_tt <- coxph(Surv(start, stop, event) ~ x1 + x2 + strata(enum),
                data = df,
                method = "efron",
                cluster = id)
cat("\nCoefficients:\n")
print(coef(pwp_tt), digits=15)
cat("\nRobust SE:\n")
print(sqrt(diag(vcov(pwp_tt))), digits=15)
cat("\nNaive SE:\n")
print(sqrt(diag(pwp_tt$naive.var)), digits=15)
cat("\nLog-likelihood:\n")
print(pwp_tt$loglik[2], digits=15)
cat("\n")

# ============================================================
# 4. PWP Total Time Model (Breslow)
# ============================================================
cat(paste(rep("=", 60), collapse=""), "\n")
cat("4. PWP TOTAL TIME MODEL (Breslow ties)\n")
cat(paste(rep("=", 60), collapse=""), "\n")

pwp_tt_breslow <- coxph(Surv(start, stop, event) ~ x1 + x2 + strata(enum),
                         data = df,
                         method = "breslow",
                         cluster = id)
cat("\nCoefficients:\n")
print(coef(pwp_tt_breslow), digits=15)
cat("\nRobust SE:\n")
print(sqrt(diag(vcov(pwp_tt_breslow))), digits=15)
cat("\nNaive SE:\n")
print(sqrt(diag(pwp_tt_breslow$naive.var)), digits=15)
cat("\nLog-likelihood:\n")
print(pwp_tt_breslow$loglik[2], digits=15)
cat("\n")

# ============================================================
# 5. PWP Gap Time Model (Efron)
# Stratified by event number, gap time scale
# ============================================================
cat(paste(rep("=", 60), collapse=""), "\n")
cat("5. PWP GAP TIME MODEL (Efron ties)\n")
cat(paste(rep("=", 60), collapse=""), "\n")

pwp_gt <- coxph(Surv(gap_time, event) ~ x1 + x2 + strata(enum),
                data = df,
                method = "efron",
                cluster = id)
cat("\nCoefficients:\n")
print(coef(pwp_gt), digits=15)
cat("\nRobust SE:\n")
print(sqrt(diag(vcov(pwp_gt))), digits=15)
cat("\nNaive SE:\n")
print(sqrt(diag(pwp_gt$naive.var)), digits=15)
cat("\nLog-likelihood:\n")
print(pwp_gt$loglik[2], digits=15)
cat("\n")

# ============================================================
# 6. PWP Gap Time Model (Breslow)
# ============================================================
cat(paste(rep("=", 60), collapse=""), "\n")
cat("6. PWP GAP TIME MODEL (Breslow ties)\n")
cat(paste(rep("=", 60), collapse=""), "\n")

pwp_gt_breslow <- coxph(Surv(gap_time, event) ~ x1 + x2 + strata(enum),
                         data = df,
                         method = "breslow",
                         cluster = id)
cat("\nCoefficients:\n")
print(coef(pwp_gt_breslow), digits=15)
cat("\nRobust SE:\n")
print(sqrt(diag(vcov(pwp_gt_breslow))), digits=15)
cat("\nNaive SE:\n")
print(sqrt(diag(pwp_gt_breslow$naive.var)), digits=15)
cat("\nLog-likelihood:\n")
print(pwp_gt_breslow$loglik[2], digits=15)
cat("\n")

# ============================================================
# Save reference values as CSV for Python comparison
# ============================================================
results <- data.frame(
  model = c("ag_breslow", "ag_breslow", "ag_efron", "ag_efron",
            "pwp_tt_efron", "pwp_tt_efron", "pwp_tt_breslow", "pwp_tt_breslow",
            "pwp_gt_efron", "pwp_gt_efron", "pwp_gt_breslow", "pwp_gt_breslow"),
  variable = rep(c("x1", "x2"), 6),
  coef = c(coef(ag_breslow), coef(ag_efron),
           coef(pwp_tt), coef(pwp_tt_breslow),
           coef(pwp_gt), coef(pwp_gt_breslow)),
  robust_se = c(sqrt(diag(vcov(ag_breslow))), sqrt(diag(vcov(ag_efron))),
                sqrt(diag(vcov(pwp_tt))), sqrt(diag(vcov(pwp_tt_breslow))),
                sqrt(diag(vcov(pwp_gt))), sqrt(diag(vcov(pwp_gt_breslow)))),
  naive_se = c(sqrt(diag(ag_breslow$naive.var)), sqrt(diag(ag_efron$naive.var)),
               sqrt(diag(pwp_tt$naive.var)), sqrt(diag(pwp_tt_breslow$naive.var)),
               sqrt(diag(pwp_gt$naive.var)), sqrt(diag(pwp_gt_breslow$naive.var))),
  loglik = c(rep(ag_breslow$loglik[2], 2), rep(ag_efron$loglik[2], 2),
             rep(pwp_tt$loglik[2], 2), rep(pwp_tt_breslow$loglik[2], 2),
             rep(pwp_gt$loglik[2], 2), rep(pwp_gt_breslow$loglik[2], 2))
)

write.csv(results, "tests/recurrent_event_R_reference.csv", row.names = FALSE)
cat("\nReference values saved to tests/recurrent_event_R_reference.csv\n")
