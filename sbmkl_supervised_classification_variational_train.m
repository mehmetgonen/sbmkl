% Mehmet Gonen (mehmet.gonen@gmail.com)

function state = sbmkl_supervised_classification_variational_train(Km, y, parameters)
    rand('state', parameters.seed); %#ok<RAND>
    randn('state', parameters.seed); %#ok<RAND>

    D = size(Km, 1);
    N = size(Km, 2);
    P = size(Km, 3);
    sigma_g = parameters.sigma_g;

    lambda.alpha = (parameters.alpha_lambda + 0.5) * ones(D, 1);
    lambda.beta = parameters.beta_lambda * ones(D, 1);
    a.mu = randn(D, 1);
    a.sigma = eye(D, D);
    G.mu = (abs(randn(P, N)) + parameters.margin) .* sign(repmat(y', P, 1));
    G.sigma = eye(P, P);
    kappa.zeta = parameters.zeta_kappa;
    kappa.eta = parameters.eta_kappa;
    s.pi = ones(P, 1);
    omega.alpha = (parameters.alpha_omega + 0.5 * P);
    omega.beta = parameters.beta_omega;
    e_success.mu = ones(P, 1);
    e_success.sigma = ones(P, 1);
    e_failure.mu = zeros(P, 1);
    e_failure.sigma = ones(P, 1);
    gamma.alpha = (parameters.alpha_gamma + 0.5);
    gamma.beta = parameters.beta_gamma;
    b.mu = 0;
    b.sigma = 1;
    f.mu = (abs(randn(N, 1)) + parameters.margin) .* sign(y);
    f.sigma = ones(N, 1);

    KmKm = zeros(D, D);
    for m = 1:P
        KmKm = KmKm + Km(:, :, m) * Km(:, :, m)';
    end
    Km = reshape(Km, [D, N * P]);

    lower = -1e40 * ones(N, 1);
    lower(y > 0) = +parameters.margin;
    upper = +1e40 * ones(N, 1);
    upper(y < 0) = -parameters.margin;
    
    for iter = 1:parameters.iteration
        fprintf(1, 'running iteration %d\n', iter);
        %%%% update lambda
        lambda.beta = 1 ./ (1 / parameters.beta_lambda + 0.5 * diag(a.mu * a.mu' + a.sigma));
        %%%% update a
        a.sigma = (diag(lambda.alpha .* lambda.beta) + KmKm / sigma_g^2) \ eye(D, D);
        a.mu = a.sigma * (Km * reshape(G.mu', N * P, 1)) / sigma_g^2;
        %%%% update G
        G.sigma = (eye(P, P) / sigma_g^2 + (ones(P, P) - eye(P, P)) .* ((s.pi .* e_success.mu) * (s.pi .* e_success.mu)') + diag(s.pi .* (e_success.mu.^2 + e_success.sigma))) \ eye(P, P);
        G.mu = G.sigma * (reshape(a.mu' * Km, [N, P])' / sigma_g^2 + (s.pi .* e_success.mu) * f.mu' - repmat((s.pi .* e_success.mu) * b.mu, 1, N));
        %%%% update kappa
        kappa.zeta = parameters.zeta_kappa + sum(s.pi);
        kappa.eta = parameters.eta_kappa + P - sum(s.pi);
        %%%% update s
        for m = randperm(P)
            active_indices = setdiff(find(s.pi > 0), m);
            s.pi(m) = logistic(psi(kappa.zeta) - psi(kappa.eta) - 0.5 * ((e_success.mu(m)^2 + e_success.sigma(m)) * sum(G.mu(m, :).^2 + G.sigma(m, m))) ...
                               + e_success.mu(m) * (G.mu(m, :) * (f.mu - b.mu) - sum(s.pi(active_indices) .* e_success.mu(active_indices) .* (sum(G.mu(active_indices, :) .* repmat(G.mu(m, :), length(active_indices), 1), 2) + N * G.sigma(active_indices, m)))));
        end
        %%%% update omega
        omega.beta = 1 / (1 / parameters.beta_omega + 0.5 * sum(s.pi .* (e_success.mu.^2 + e_success.sigma) + (1 - s.pi) .* (e_failure.mu.^2 + e_failure.sigma)));
        %%%% update e_success
        for m = randperm(P)
            active_indices = setdiff(find(s.pi > 0), m);
            e_success.sigma(m) = 1 / (omega.alpha * omega.beta + sum(G.mu(m, :).^2 + G.sigma(m, m)));
            e_success.mu(m) = e_success.sigma(m) * (G.mu(m, :) * (f.mu - b.mu) - sum(s.pi(active_indices) .* e_success.mu(active_indices) .* (sum(G.mu(active_indices, :) .* repmat(G.mu(m, :), length(active_indices), 1), 2) + N * G.sigma(active_indices, m))));
        end
        %%%% update e_failure
        e_failure.sigma = (1 / (omega.alpha * omega.beta)) * ones(P, 1);
        %%%% update gamma
        gamma.beta = 1 / (1 / parameters.beta_gamma + 0.5 * (b.mu^2 + b.sigma));
        %%%% update b
        b.sigma = 1 / (gamma.alpha * gamma.beta + N);
        b.mu = b.sigma * sum(f.mu - G.mu' * (s.pi .* e_success.mu));
        %%%% update f
        output = G.mu' * (s.pi .* e_success.mu) + b.mu;
        alpha_norm = lower - output;
        beta_norm = upper - output;
        normalization = normcdf(beta_norm) - normcdf(alpha_norm);
        normalization(normalization == 0) = 1;
        f.mu = output + (normpdf(alpha_norm) - normpdf(beta_norm)) ./ normalization;
        f.sigma = 1 + (alpha_norm .* normpdf(alpha_norm) - beta_norm .* normpdf(beta_norm)) ./ normalization - (normpdf(alpha_norm) - normpdf(beta_norm)).^2 ./ normalization.^2;
    end

    state.lambda = lambda;
    state.a = a;
    state.kappa = kappa;
    state.s = s;
    state.omega = omega;
    state.e_success = e_success;
    state.e_failure = e_failure;
    state.gamma = gamma;
    state.b = b;
    state.parameters = parameters;
end

function z = logistic(x)
    z = 1 ./ (1 + exp(-x));
    z(z < 1e-2) = 0;
    z(z > 1 - 1e-2) = 1;
end