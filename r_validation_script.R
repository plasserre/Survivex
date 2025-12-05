# R Validation Script for Recurrent Event Models
# Run this in R to generate reference values for Python validation

library(survival)

# ============================================================================
# 1. Andersen-Gill Model with CGD Data
# ============================================================================
cat("\n", rep("=", 80), "\n", sep="")
cat("ANDERSEN-GILL MODEL\n")
cat(rep("=", 80), "\n", sep="")

data(cgd)

# Fit AG model
ag_fit <- coxph(Surv(tstart, tstop, status) ~ treat + age, 
                data=cgd, 
                id=id,
                method="efron")

cat("\nCoefficients:\n")
print(ag_fit$coefficients)

cat("\nRobust SE:\n")
print(sqrt(diag(ag_fit$var)))

cat("\nNaive SE:\n")
print(sqrt(diag(ag_fit$naive.var)))

cat("\nLog-likelihood:\n")
print(ag_fit$loglik)

cat("\nRobust Variance Matrix:\n")
print(ag_fit$var)

cat("\nNaive Variance Matrix:\n")
print(ag_fit$naive.var)

# ============================================================================
# 2. PWP-TT Model with Bladder Data
# ============================================================================
cat("\n", rep("=", 80), "\n", sep="")
cat("PWP-TT MODEL\n")
cat(rep("=", 80), "\n", sep="")

data(bladder)

# Fit PWP-TT
pwptt_fit <- coxph(Surv(start, stop, event) ~ treatment + strata(enum), 
                   data=bladder,
                   id=id,
                   method="efron")

cat("\nCoefficients:\n")
print(pwptt_fit$coefficients)

cat("\nRobust SE:\n")
print(sqrt(diag(pwptt_fit$var)))

cat("\nLog-likelihood:\n")
print(pwptt_fit$loglik)

# ============================================================================
# 3. Frailty Model (Gamma)
# ============================================================================
cat("\n", rep("=", 80), "\n", sep="")
cat("GAMMA FRAILTY MODEL\n")
cat(rep("=", 80), "\n", sep="")

# Use kidney data (has recurrent events)
data(kidney)

frailty_gamma <- coxph(Surv(time, status) ~ age + sex + disease + 
                        frailty(id, distribution="gamma"),
                       data=kidney)

cat("\nCoefficients:\n")
print(frailty_gamma$coefficients)

cat("\nSE:\n")
print(sqrt(diag(frailty_gamma$var)))

cat("\nFrailty variance (theta):\n")
print(frailty_gamma$history$frailty[[1]]$theta)

cat("\nLog-likelihood:\n")
print(frailty_gamma$loglik)

# ============================================================================
# 4. Frailty Model (Gaussian)
# ============================================================================
cat("\n", rep("=", 80), "\n", sep="")
cat("GAUSSIAN FRAILTY MODEL\n")
cat(rep("=", 80), "\n", sep="")

frailty_gaussian <- coxph(Surv(time, status) ~ age + sex + disease + 
                          frailty(id, distribution="gaussian"),
                         data=kidney)

cat("\nCoefficients:\n")
print(frailty_gaussian$coefficients)

cat("\nSE:\n")
print(sqrt(diag(frailty_gaussian$var)))

cat("\nFrailty variance (sigma^2):\n")
print(frailty_gaussian$history$frailty[[1]]$theta)

cat("\nLog-likelihood:\n")
print(frailty_gaussian$loglik)

cat("\n", rep("=", 80), "\n", sep="")
cat("VALIDATION COMPLETE\n")
cat(rep("=", 80), "\n", sep="")
