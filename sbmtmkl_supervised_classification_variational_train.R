# Mehmet Gonen (mehmet.gonen@gmail.com)

logistic <- function(x) {
  z <- 1 / (1 + exp(-x))
  z[z < 1e-2] <- 0
  z[z > 1 - 1e-2] <- 1
  return (z)
}

sbmtmkl_supervised_classification_variational_train <- function(Km, y, parameters) {    
  set.seed(parameters$seed)

  T <- length(Km)
  D <- matrix(0, T, 1)
  N <- matrix(0, T, 1)
  for (o in 1:T) {
    D[o] <- dim(Km[[o]])[1]
    N[o] <- dim(Km[[o]])[2]
  }
  P <- dim(Km[[1]])[3]
  sigma_g <- parameters$sigma_g

  lambda <- vector("list", T)
  a <- vector("list", T)
  G <- vector("list", T)
  for (o in 1:T) {
    lambda[[o]] <- list(alpha = matrix(parameters$alpha_lambda + 0.5, D[o], 1), beta = matrix(parameters$beta_lambda, D[o], 1))
    a[[o]] <- list(mu = matrix(rnorm(D[o]), D[o], 1), sigma = diag(1, D[o], D[o]))
    G[[o]] <- list(mu = (abs(matrix(rnorm(P * N[o]), P, N[o])) + parameters$margin) * sign(matrix(y[[o]], P, N[o], byrow = TRUE)), sigma = diag(1, P, P))
  }
  kappa <- list(zeta = parameters$zeta_kappa, eta = parameters$eta_kappa)
  s <- list(pi = matrix(1, P, 1))
  omega <- vector("list", T)
  e_success <- vector("list", T)
  e_failure <- vector("list", T)
  gamma <- vector("list", T)
  b <- vector("list", T)
  f <- vector("list", T)
  for (o in 1:T) {
    omega[[o]] <- list(alpha = parameters$alpha_omega + 0.5 * P, beta = parameters$beta_omega)
    e_success[[o]] <- list(mu = matrix(1, P, 1), sigma = matrix(1, P, 1))
    e_failure[[o]] <- list(mu = matrix(0, P, 1), sigma = matrix(1, P, 1))
    gamma[[o]] <- list(alpha = parameters$alpha_gamma + 0.5, beta = parameters$beta_gamma)
    b[[o]] <- list(mu = 0, sigma = 1)
    f[[o]] <- list(mu = (abs(matrix(rnorm(N[o]), N[o], 1)) + parameters$margin) * sign(y[[o]]), sigma = matrix(1, N[o], 1))
  }

  KmKm <- vector("list", T)
  for (o in 1:T) {
    KmKm[[o]] <- matrix(0, D[o], D[o])
    for (m in 1:P) {
        KmKm[[o]] <- KmKm[[o]] + tcrossprod(Km[[o]][,,m], Km[[o]][,,m])
    }
    Km[[o]] <- matrix(Km[[o]], D[o], N[o] * P)
  }

  lower <- vector("list", T)
  upper <- vector("list", T)
  for (o in 1:T) {
    lower[[o]] <- matrix(-1e40, N[o], 1)
    lower[[o]][which(y[[o]] > 0)] <- +parameters$margin
    upper[[o]] <- matrix(+1e40, N[o], 1)
    upper[[o]][which(y[[o]] < 0)] <- -parameters$margin
  }

  for (iter in 1:parameters$iteration) {
    print(sprintf("running iteration %d", iter))
    # update lambda
    for (o in 1:T) {
      lambda[[o]]$beta <- 1 / (1 / parameters$beta_lambda + 0.5 * matrix(diag(tcrossprod(a[[o]]$mu, a[[o]]$mu) + a[[o]]$sigma), N[o], 1))
    }
    # update a
    for (o in 1:T) {
      a[[o]]$sigma <- chol2inv(chol(diag(as.vector(lambda[[o]]$alpha * lambda[[o]]$beta), D[o], D[o]) + KmKm[[o]] / sigma_g^2))
      a[[o]]$mu <- a[[o]]$sigma %*% (Km[[o]] %*% matrix(t(G[[o]]$mu), N[o] * P, 1)) / sigma_g^2
    }
    # update G
    for (o in 1:T) {
      G[[o]]$sigma <- chol2inv(chol(diag(1, P, P) / sigma_g^2 + (matrix(1, P, P) - diag(1, P, P)) * tcrossprod(s$pi * e_success[[o]]$mu, s$pi * e_success[[o]]$mu) + diag(as.vector(s$pi * (e_success[[o]]$mu^2 + e_success[[o]]$sigma)), P, P)))
      G[[o]]$mu <- G[[o]]$sigma %*% (t(matrix(crossprod(a[[o]]$mu, Km[[o]]), N[o], P)) / sigma_g^2 + tcrossprod(s$pi * e_success[[o]]$mu, f[[o]]$mu) - matrix((s$pi * e_success[[o]]$mu) * b[[o]]$mu, P, N[o], byrow = FALSE))
    }
    # update kappa
    kappa$zeta <- parameters$zeta_kappa + sum(s$pi)
    kappa$eta <- parameters$eta_kappa + P - sum(s$pi)
    # update s
    for (m in sample(P)) {
      active_indices <- setdiff(which(s$pi > 0), m)
      total <- 0
      for (o in 1:T) {
        total <- total - 0.5 * ((e_success[[o]]$mu[m]^2 + e_success[[o]]$sigma[m]) * sum(G[[o]]$mu[m,]^2 + G[[o]]$sigma[m, m])) + e_success[[o]]$mu[m] * (G[[o]]$mu[m,] %*% (f[[o]]$mu - b[[o]]$mu) - sum(s$pi[active_indices] * e_success[[o]]$mu[active_indices] * (rowSums(G[[o]]$mu[active_indices,] * matrix(G[[o]]$mu[m,], length(active_indices), N[o], byrow = TRUE)) + N[o] * G[[o]]$sigma[active_indices, m])))
      }
      s$pi[m] <- logistic(digamma(kappa$zeta) - digamma(kappa$eta) + total)
    }
    # update omega
    for (o in 1:T) {
      omega[[o]]$beta <- 1 / (1 / parameters$beta_omega + 0.5 * sum(s$pi * (e_success[[o]]$mu^2 + e_success[[o]]$sigma) + (1 - s$pi) * (e_failure[[o]]$mu^2 + e_failure[[o]]$sigma)))
    }
    # update e_success
    for (m in sample(P)) {
      active_indices <- setdiff(which(s$pi > 0), m)
      for (o in 1:T) {
        e_success[[o]]$sigma[m] <- 1 / (omega[[o]]$alpha * omega[[o]]$beta + sum(G[[o]]$mu[m,]^2 + G[[o]]$sigma[m, m]))
        e_success[[o]]$mu[m] <- e_success[[o]]$sigma[m] * (G[[o]]$mu[m,] %*% (f[[o]]$mu - b[[o]]$mu) - sum(s$pi[active_indices] * e_success[[o]]$mu[active_indices,] * (rowSums(G[[o]]$mu[active_indices,] * matrix(G[[o]]$mu[m,], length(active_indices), N[o], byrow = TRUE)) + N[o] * G[[o]]$sigma[active_indices, m])))
      }
    }
    # update e_failure
    for (o in 1:T) {
      e_failure[[o]]$sigma <- matrix(1 / (omega[[o]]$alpha * omega[[o]]$beta), P, 1)
    }
    # update gamma
    for (o in 1:T) {
      gamma[[o]]$beta <- 1 / (1 / parameters$beta_gamma + 0.5 * (b[[o]]$mu^2 + b[[o]]$sigma))
    }
    # update b
    for (o in 1:T) {
      b[[o]]$sigma <- 1 / (gamma[[o]]$alpha * gamma[[o]]$beta + N[o])
      b[[o]]$mu <- b[[o]]$sigma * sum(f[[o]]$mu - crossprod(G[[o]]$mu, s$pi * e_success[[o]]$mu))
    }
    # update f
    for (o in 1:T) {
      output <- crossprod(G[[o]]$mu, s$pi * e_success[[o]]$mu) + b[[o]]$mu
      alpha_norm <- lower[[o]] - output
      beta_norm <- upper[[o]] - output
      normalization <- pnorm(beta_norm) - pnorm(alpha_norm)
      normalization[which(normalization == 0)] <- 1
      f[[o]]$mu <- output + (dnorm(alpha_norm) - dnorm(beta_norm)) / normalization
      f[[o]]$sigma <- 1 + (alpha_norm * dnorm(alpha_norm) - beta_norm * dnorm(beta_norm)) / normalization - (dnorm(alpha_norm) - dnorm(beta_norm))^2 / normalization^2
    }
  }
  
  state <- list(lambda = lambda, a = a, kappa = kappa, s = s, omega = omega, e_success = e_success, e_failure = e_failure, gamma = gamma, b = b, parameters = parameters)
}