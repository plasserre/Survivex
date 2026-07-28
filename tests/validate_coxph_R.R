# Gold-standard Cox PH reference values from R's `survival` package.
#
# Driven by tests/validate_coxph_vs_R.py, which exports byte-identical data
# to tests/data/*.csv before this runs. This script fits:
#   1. CoxPH Breslow & Efron               (Rossi, Lung)
#   2. Stratified CoxPH Breslow            (Rossi/wexp, Lung/sex)
#   3. CoxPH + Ridge (L2)                  (Rossi, penalized::penalized)
# and writes tidy (case, variable, coef) rows to tests/coxph_R_reference.csv.
#
# coxph is Newton-Raphson on the partial likelihood -- the SAME algorithm
# survivex uses -- so agreement is expected at machine precision, not merely
# "close". Tight control (eps=1e-12) puts R at the true stationary point so
# the comparison is solver-vs-solver, not tolerance-vs-tolerance.

suppressMessages(library(survival))

ctrl <- coxph.control(eps = 1e-12, iter.max = 200)
rows <- list()

add_rows <- function(case, beta) {
  data.frame(
    case = case,
    variable = names(beta),
    coef = as.numeric(beta),
    stringsAsFactors = FALSE
  )
}

rossi <- read.csv("tests/data/rossi.csv")
lung <- read.csv("tests/data/lung.csv")

# ---- 1. Plain CoxPH: Breslow & Efron ------------------------------------
cat(strrep("=", 60), "\n")
cat("1-2. CoxPH Breslow & Efron\n")
cat(strrep("=", 60), "\n")

fit_plain <- function(case, formula, data, method) {
  fit <- coxph(formula, data = data, ties = method, control = ctrl)
  cat("\n", case, " (", method, ")\n", sep = "")
  print(coef(fit), digits = 15)
  add_rows(case, coef(fit))
}

rossi_formula <- Surv(time, event) ~ fin + age + prio
lung_formula <- Surv(time, event) ~ age + sex + ph_karno

rows[["rossi_breslow"]] <- fit_plain("rossi_breslow", rossi_formula, rossi, "breslow")
rows[["rossi_efron"]]   <- fit_plain("rossi_efron",   rossi_formula, rossi, "efron")
rows[["lung_breslow"]]  <- fit_plain("lung_breslow",  lung_formula,  lung,  "breslow")
rows[["lung_efron"]]    <- fit_plain("lung_efron",    lung_formula,  lung,  "efron")

# ---- 2. Stratified CoxPH: Breslow ---------------------------------------
cat("\n", strrep("=", 60), "\n", sep = "")
cat("3. Stratified CoxPH (Breslow)\n")
cat(strrep("=", 60), "\n")

rossi_strat <- coxph(Surv(time, event) ~ fin + age + prio + strata(wexp),
                     data = rossi, ties = "breslow", control = ctrl)
cat("\nrossi_strat_wexp_breslow\n")
print(coef(rossi_strat), digits = 15)
rows[["rossi_strat"]] <- add_rows("rossi_strat_wexp_breslow", coef(rossi_strat))

lung_strat <- coxph(Surv(time, event) ~ age + ph_karno + strata(sex),
                    data = lung, ties = "breslow", control = ctrl)
cat("\nlung_strat_sex_breslow\n")
print(coef(lung_strat), digits = 15)
rows[["lung_strat"]] <- add_rows("lung_strat_sex_breslow", coef(lung_strat))

# ---- 3. CoxPH + Ridge: penalized::penalized -----------------------------
# penalized's Cox uses Breslow ties and an L2 objective whose gradient is
# lambda2*b (i.e. penalty (lambda2/2)*||b||^2), so lambda2 = n*penalty maps
# its strength onto survivex's (n*penalty/2)*||b||^2 convention (see the
# lambda2 column computed in validate_coxph_vs_R.py). Data is pre-standardized
# (standardize=FALSE) so coefficients live in the coordinate system survivex
# reports.
cat("\n", strrep("=", 60), "\n", sep = "")
cat("4. CoxPH + Ridge (penalized::penalized, Breslow)\n")
cat(strrep("=", 60), "\n")

suppressMessages(library(penalized))
rossi_std <- read.csv("tests/data/rossi_std.csv")
ridge_spec <- read.csv("tests/data/ridge_spec.csv")
Xr <- as.matrix(rossi_std[, c("fin", "age", "prio")])
yr <- Surv(rossi_std$time, rossi_std$event)

for (i in seq_len(nrow(ridge_spec))) {
  case <- ridge_spec$label[i]
  lam2 <- ridge_spec$lambda2[i]
  fit <- penalized(yr, penalized = Xr, lambda1 = 0, lambda2 = lam2,
                   standardize = FALSE, model = "cox", trace = FALSE)
  beta <- coefficients(fit, "penalized")
  names(beta) <- colnames(Xr)
  cat("\n", case, " (lambda2=", lam2, ")\n", sep = "")
  print(beta, digits = 15)
  rows[[case]] <- add_rows(case, beta)
}

# ---- write tidy reference -----------------------------------------------
out <- do.call(rbind, rows)
rownames(out) <- NULL
write.csv(out, "tests/coxph_R_reference.csv", row.names = FALSE)
cat("\nWrote tests/coxph_R_reference.csv (", nrow(out), " rows)\n", sep = "")
