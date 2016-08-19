# Mehmet Gonen (mehmet.gonen@gmail.com)

logistic <- function(x) {
  z <- 1 / (1 + exp(-x))
  z[z < 1e-2] <- 0
  z[z > 1 - 1e-2] <- 1
  return (z)
}

sbmkl_supervised_classification_variational_train <- function(Km, y, parameters) {    
  set.seed(parameters$seed)

  D <- dim(Km)[1]
  N <- dim(Km)[2]
  P <- dim(Km)[3]
  sigma_g <- parameters$sigma_g

  lambda <- list(alpha = matrix(parameters$alpha_lambda + 0.5, D, 1), beta = matrix(parameters$beta_lambda, D, 1))
  a <- list(mu = matrix(rnorm(D), D, 1), sigma = diag(1, D, D))
  G <- list(mu = (abs(matrix(rnorm(P * N), P, N)) + parameters$margin) * sign(matrix(y, P, N, byrow = TRUE)), sigma = diag(1, P, P))
  kappa <- list(zeta = parameters$zeta_kappa, eta = parameters$eta_kappa)
  s <- list(pi = matrix(1, P, 1))
  omega <- list(alpha = parameters$alpha_omega + 0.5 * P, beta = parameters$beta_omega)
  e_success <- list(mu = matrix(1, P, 1), sigma = matrix(1, P, 1))
  e_failure <- list(mu = matrix(0, P, 1), sigma = matrix(1, P, 1))
  gamma <- list(alpha = parameters$alpha_gamma + 0.5, beta = parameters$beta_gamma)
  b <- list(mu = 0, sigma = 1)
  f <- list(mu = (abs(matrix(rnorm(N), N, 1)) + parameters$margin) * sign(y), sigma = matrix(1, N, 1))

  KmKm <- matrix(0, D, D)
  for (m in 1:P) {
    KmKm <- KmKm + tcrossprod(Km[,,m], Km[,,m])
  }
  Km <- matrix(Km, D, N * P)

  lower <- matrix(-1e40, N, 1)
  lower[which(y > 0)] <- +parameters$margin
  upper <- matrix(+1e40, N, 1)
  upper[which(y < 0)] <- -parameters$margin

  for (iter in 1:parameters$iteration) {
    print(sprintf("running iteration %d", iter))
    # update lambda
    lambda$beta <- 1 / (1 / parameters$beta_lambda + 0.5 * matrix(diag(tcrossprod(a$mu, a$mu) + a$sigma), N, 1))
    # update a
    a$sigma <- chol2inv(chol(diag(as.vector(lambda$alpha * lambda$beta), D, D) + KmKm / sigma_g^2))
    a$mu <- a$sigma %*% (Km %*% matrix(t(G$mu), N * P, 1)) / sigma_g^2
    # update G
    G$sigma <- chol2inv(chol(diag(1, P, P) / sigma_g^2 + (matrix(1, P, P) - diag(1, P, P)) * tcrossprod(s$pi * e_success$mu, s$pi * e_success$mu) + diag(as.vector(s$pi * (e_success$mu^2 + e_success$sigma)), P, P)))
    G$mu <- G$sigma %*% (t(matrix(crossprod(a$mu, Km), N, P)) / sigma_g^2 + tcrossprod(s$pi * e_success$mu, f$mu) - matrix((s$pi * e_success$mu) * b$mu, P, N, byrow = FALSE))
    # update kappa
    kappa$zeta <- parameters$zeta_kappa + sum(s$pi)
    kappa$eta <- parameters$eta_kappa + P - sum(s$pi)
    # update s
    for (m in sample(P)) {
      active_indices <- setdiff(which(s$pi > 0), m)
      s$pi[m] <- logistic(digamma(kappa$zeta) - digamma(kappa$eta) - 0.5 * ((e_success$mu[m]^2 + e_success$sigma[m]) * sum(G$mu[m,]^2 + G$sigma[m, m])) + e_success$mu[m] * (G$mu[m,] %*% (f$mu - b$mu) - sum(s$pi[active_indices] * e_success$mu[active_indices] * (rowSums(G$mu[active_indices,] * matrix(G$mu[m,], nrow = length(active_indices), ncol = N, byrow = TRUE)) + N * G$sigma[active_indices, m]))))
    }
    # update omega
    omega$beta <- 1 / (1 / parameters$beta_omega + 0.5 * sum(s$pi * (e_success$mu^2 + e_success$sigma) + (1 - s$pi) * (e_failure$mu^2 + e_failure$sigma)))
    # update e_success
    for (m in sample(P)) {
      active_indices <- setdiff(which(s$pi > 0), m)
      e_success$sigma[m] <- 1 / (omega$alpha * omega$beta + sum(G$mu[m,]^2 + G$sigma[m, m]))
      e_success$mu[m] <- e_success$sigma[m] * (G$mu[m,] %*% (f$mu - b$mu) - sum(s$pi[active_indices] * e_success$mu[active_indices] * (rowSums(G$mu[active_indices,] * matrix(G$mu[m,], nrow = length(active_indices), ncol = N, byrow = TRUE)) + N * G$sigma[active_indices, m])))
    }
    # update e_failure
    e_failure$sigma <- matrix(1 / (omega$alpha * omega$beta), P, 1)
    # update gamma
    gamma$beta <- 1 / (1 / parameters$beta_gamma + 0.5 * (b$mu^2 + b$sigma))
    # update b
    b$sigma <- 1 / (gamma$alpha * gamma$beta + N)
    b$mu <- b$sigma * sum(f$mu - crossprod(G$mu, s$pi * e_success$mu))
    # update f
    output <- crossprod(G$mu, s$pi * e_success$mu) + b$mu
    alpha_norm <- lower - output
    beta_norm <- upper - output
    normalization <- pnorm(beta_norm) - pnorm(alpha_norm)
    normalization[which(normalization == 0)] <- 1
    f$mu <- output + (dnorm(alpha_norm) - dnorm(beta_norm)) / normalization
    f$sigma <- 1 + (alpha_norm * dnorm(alpha_norm) - beta_norm * dnorm(beta_norm)) / normalization - (dnorm(alpha_norm) - dnorm(beta_norm))^2 / normalization^2
  }
  
  state <- list(lambda = lambda, a = a, kappa = kappa, s = s, omega = omega, e_success = e_success, e_failure = e_failure, gamma = gamma, b = b, parameters = parameters)
}