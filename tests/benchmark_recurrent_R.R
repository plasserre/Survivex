library(survival)

df <- read.csv("tests/recurrent_benchmark_data.csv")
cat("Data:", nrow(df), "rows,", length(unique(df$id)), "subjects,", sum(df$event), "events\n\n")

# Andersen-Gill
cat("Andersen-Gill (Breslow):\n")
t_ag <- system.time({
  for (i in 1:3) {
    ag <- coxph(Surv(start, stop, event) ~ x1 + x2 + x3 + x4 + x5,
                data = df, method = "breslow", cluster = id)
  }
})
cat("  3 runs:", t_ag[3], "s (", t_ag[3]/3, "s per run)\n")

# PWP-TT
cat("\nPWP Total Time (Breslow):\n")
t_tt <- system.time({
  for (i in 1:3) {
    pwp_tt <- coxph(Surv(start, stop, event) ~ x1 + x2 + x3 + x4 + x5 + strata(enum),
                    data = df, method = "breslow", cluster = id)
  }
})
cat("  3 runs:", t_tt[3], "s (", t_tt[3]/3, "s per run)\n")

# PWP-GT
cat("\nPWP Gap Time (Breslow):\n")
t_gt <- system.time({
  for (i in 1:3) {
    pwp_gt <- coxph(Surv(gap_time, event) ~ x1 + x2 + x3 + x4 + x5 + strata(enum),
                    data = df, method = "breslow", cluster = id)
  }
})
cat("  3 runs:", t_gt[3], "s (", t_gt[3]/3, "s per run)\n")

cat("\nSummary (per run):\n")
cat("  AG:     ", t_ag[3]/3, "s\n")
cat("  PWP-TT: ", t_tt[3]/3, "s\n")
cat("  PWP-GT: ", t_gt[3]/3, "s\n")
