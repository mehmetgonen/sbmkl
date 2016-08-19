# Mehmet Gonen (mehmet.gonen@gmail.com)

sbmkl_supervised_classification_variational_test <- function(Km, state) {
  N <- dim(Km)[2]
  P <- dim(Km)[3]

  G <- list(mu = matrix(0, P, N))
  for (m in 1:P) {
    G$mu[m,] <- crossprod(state$a$mu, Km[,,m])
  }
  
  f <- list(mu = crossprod(G$mu, state$s$pi * state$e_success$mu) + state$b$mu)

  prediction <- list(G = G, f = f)
}