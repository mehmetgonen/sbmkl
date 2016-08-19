# Mehmet Gonen (mehmet.gonen@gmail.com)

sbmtmkl_supervised_classification_variational_test <- function(Km, state) {
  T <- length(Km)
  N <- matrix(0, T, 1)
  for (o in 1:T) {
    N[o] <- dim(Km[[o]])[2]
  }
  P <- dim(Km[[1]])[3]

  G <- vector("list", T)
  for (o in 1:T) {
    G[[o]] <- list(mu = matrix(0, P, N[o]))
    for (m in 1:P) {
      G[[o]]$mu[m,] <- crossprod(state$a[[o]]$mu, Km[[o]][,,m])
    }
  }
  
  f <- vector("list", T)
  for (o in 1:T) {
    f[[o]] <- list(mu = crossprod(G[[o]]$mu, state$s$pi * state$e_success[[o]]$mu) + state$b[[o]]$mu)
  }

  prediction <- list(G = G, f = f)
}