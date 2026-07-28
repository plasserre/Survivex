# Gold-standard gamma shared-frailty reference values from R.
#
# Driven by tests/validate_frailty_vs_R.py. Two references are produced per
# dataset, because there are two distinct estimators for a gamma frailty:
#
#   EM  -- frailtyEM::emfrail: the EM algorithm on the *exact* marginal
#          likelihood (integrating the gamma frailty out). This is the SAME
#          estimator survivex implements, so agreement is expected to a few
#          significant figures, limited only by the semiparametric baseline.
#
#   PPL -- survival::coxph(... + frailty(g, distribution="gamma")): a
#          *penalized partial likelihood* approximation. A different estimator;
#          it does not maximise the same objective, so it is expected to differ
#          from EM by ~1e-2. This row exists to show that the survivex-vs-R
#          frailty discrepancy is the EM-vs-PPL estimator difference, not a bug.
#
# survival's lazy-loaded `rats` and `kidney` are written to tests/data/*.csv so
# R and survivex fit byte-identical inputs.

suppressMessages({
  library(survival)
  library(frailtyEM)
})

# ---- shared data (LazyData objects in survival) -------------------------
write.csv(rats,   "tests/data/rats.csv",   row.names = FALSE)
write.csv(kidney, "tests/data/kidney.csv", row.names = FALSE)

rats   <- read.csv("tests/data/rats.csv")
kidney <- read.csv("tests/data/kidney.csv")

rows <- list()
add_row <- function(case, method, variable, value) {
  data.frame(case = case, method = method, variable = variable,
             value = as.numeric(value), stringsAsFactors = FALSE)
}

# ---- fit one dataset under both estimators ------------------------------
fit_case <- function(case, surv_rhs, cluster, data, variables) {
  f_em  <- as.formula(paste0("Surv(time, status) ~ ", surv_rhs,
                             " + cluster(", cluster, ")"))
  f_ppl <- as.formula(paste0("Surv(time, status) ~ ", surv_rhs,
                             " + frailty(", cluster, ", distribution=\"gamma\")"))

  em  <- emfrail(f_em, data = data,
                 distribution = emfrail_dist(dist = "gamma"))
  ppl <- coxph(f_ppl, data = data)

  em_theta  <- summary(em)$fr_var[["fr_var"]]
  ppl_theta <- ppl$history[[1]]$theta

  cat("\n", case, "\n", sep = "")
  cat("  EM  coef:", paste(sprintf("%s=%.12g", variables, em$coefficients[variables]),
                           collapse = "  "), " theta=", sprintf("%.12g", em_theta), "\n")
  cat("  PPL coef:", paste(sprintf("%s=%.12g", variables, ppl$coefficients[variables]),
                           collapse = "  "), " theta=", sprintf("%.12g", ppl_theta), "\n")

  out <- list()
  for (v in variables) {
    out[[length(out) + 1]] <- add_row(case, "em",  v, em$coefficients[[v]])
    out[[length(out) + 1]] <- add_row(case, "ppl", v, ppl$coefficients[[v]])
  }
  out[[length(out) + 1]] <- add_row(case, "em",  "theta", em_theta)
  out[[length(out) + 1]] <- add_row(case, "ppl", "theta", ppl_theta)
  do.call(rbind, out)
}

cat(strrep("=", 60), "\n")
cat("Gamma shared frailty: EM (frailtyEM) vs PPL (survival::coxph)\n")
cat(strrep("=", 60), "\n")

rows[["rats"]]   <- fit_case("rats",   "rx",        "litter", rats,   c("rx"))
rows[["kidney"]] <- fit_case("kidney", "age + sex", "id",     kidney, c("age", "sex"))

out <- do.call(rbind, rows)
rownames(out) <- NULL
write.csv(out, "tests/frailty_R_reference.csv", row.names = FALSE)
cat("\nWrote tests/frailty_R_reference.csv (", nrow(out), " rows)\n", sep = "")
