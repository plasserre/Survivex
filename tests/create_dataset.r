library(survival)
data(cgd)
write.csv(cgd, "cgd_data.csv", row.names=FALSE)

# Also get the R reference values
fit_ag <- coxph(Surv(tstart, tstop, status) ~ treat + age + cluster(id), 
                data = cgd, method = "efron")

cat("\n=== R REFERENCE VALUES ===\n")
cat("treat coef:     ", coef(fit_ag)["treat"], "\n")
cat("age coef:       ", coef(fit_ag)["age"], "\n")
cat("treat robust SE:", sqrt(fit_ag$var[1,1]), "\n")
cat("age robust SE:  ", sqrt(fit_ag$var[2,2]), "\n")
cat("treat naive SE: ", summary(fit_ag)$coefficients["treat", "se(coef)"], "\n")
cat("age naive SE:   ", summary(fit_ag)$coefficients["age", "se(coef)"], "\n")
cat("log-likelihood: ", fit_ag$loglik[2], "\n")
