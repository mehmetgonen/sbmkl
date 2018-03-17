function state = sbmtmkl_supervised_classification_variational_train(Km, y, parameters)
    rand('state', parameters.seed); %#ok<RAND>
    randn('state', parameters.seed); %#ok<RAND>

    T = length(Km);
    D = zeros(T, 1);
    N = zeros(T, 1);
    for o = 1:T
        D(o) = size(Km{o}, 1);
        N(o) = size(Km{o}, 2);
    end
    P = size(Km{1}, 3);
    sigma_g = parameters.sigma_g;

    lambda = cell(1, T);
    a = cell(1, T);
    G = cell(1, T);
    for o = 1:T
        lambda{o}.alpha = (parameters.alpha_lambda + 0.5) * ones(D(o), 1);
        lambda{o}.beta = parameters.beta_lambda * ones(D(o), 1);
        a{o}.mu = randn(D(o), 1);
        a{o}.sigma = eye(D(o), D(o));
        G{o}.mu = (abs(randn(P, N(o))) + parameters.margin) .* sign(repmat(y{o}', P, 1));
        G{o}.sigma = eye(P, P);
    end
    kappa.zeta = parameters.zeta_kappa;
    kappa.eta = parameters.eta_kappa;
    s.pi = ones(P, 1);
    omega = cell(1, T);
    e_success = cell(1, T);
    e_failure = cell(1, T);
    gamma = cell(1, T);
    b = cell(1, T);
    f = cell(1, T);
    for o = 1:T
        omega{o}.alpha = (parameters.alpha_omega + 0.5 * P);
        omega{o}.beta = parameters.beta_omega;
        e_success{o}.mu = ones(P, 1);
        e_success{o}.sigma = ones(P, 1);
        e_failure{o}.mu = zeros(P, 1);
        e_failure{o}.sigma = ones(P, 1);
        gamma{o}.alpha = (parameters.alpha_gamma + 0.5);
        gamma{o}.beta = parameters.beta_gamma;
        b{o}.mu = 0;
        b{o}.sigma = 1;
        f{o}.mu = (abs(randn(N(o), 1)) + parameters.margin) .* sign(y{o});
        f{o}.sigma = ones(N(o), 1);
    end

    KmKm = cell(1, T);
    for o = 1:T
        KmKm{o} = zeros(D(o), D(o));
        for m = 1:P
            KmKm{o} = KmKm{o} + Km{o}(:, :, m) * Km{o}(:, :, m)';
        end
        Km{o} = reshape(Km{o}, [D(o), N(o) * P]);
    end

    lower = cell(1, T);
    upper = cell(1, T);
    for o = 1:T
        lower{o} = -1e40 * ones(N(o), 1);
        lower{o}(y{o} > 0) = +parameters.margin;
        upper{o} = +1e40 * ones(N(o), 1);
        upper{o}(y{o} < 0) = -parameters.margin;
    end
    
    for iter = 1:parameters.iteration
        fprintf(1, 'running iteration %d\n', iter);
        %%%% update lambda
        for o = 1:T
            lambda{o}.beta = 1 ./ (1 / parameters.beta_lambda + 0.5 * diag(a{o}.mu * a{o}.mu' + a{o}.sigma));
        end
        %%%% update a
        for o = 1:T
            a{o}.sigma = (diag(lambda{o}.alpha .* lambda{o}.beta) + KmKm{o} / sigma_g^2) \ eye(D(o), D(o));
            a{o}.mu = a{o}.sigma * (Km{o} * reshape(G{o}.mu', N(o) * P, 1)) / sigma_g^2;
        end
        %%%% update G
        for o = 1:T
            G{o}.sigma = (eye(P, P) / sigma_g^2 + (ones(P, P) - eye(P, P)) .* ((s.pi .* e_success{o}.mu) * (s.pi .* e_success{o}.mu)') + diag(s.pi .* (e_success{o}.mu.^2 + e_success{o}.sigma))) \ eye(P, P);
            G{o}.mu = G{o}.sigma * (reshape(a{o}.mu' * Km{o}, [N(o), P])' / sigma_g^2 + (s.pi .* e_success{o}.mu) * f{o}.mu' - repmat((s.pi .* e_success{o}.mu) * b{o}.mu, 1, N(o)));
        end
        %%%% update kappa
        kappa.zeta = parameters.zeta_kappa + sum(s.pi);
        kappa.eta = parameters.eta_kappa + P - sum(s.pi);
        %%%% update s
        for m = randperm(P)
            active_indices = setdiff(find(s.pi > 0), m);
            total = 0;
            for o = 1:T
                total = total - 0.5 * ((e_success{o}.mu(m)^2 + e_success{o}.sigma(m)) * sum(G{o}.mu(m, :).^2 + G{o}.sigma(m, m))) + e_success{o}.mu(m) * (G{o}.mu(m, :) * (f{o}.mu - b{o}.mu) - sum(s.pi(active_indices) .* e_success{o}.mu(active_indices) .* (sum(G{o}.mu(active_indices, :) .* repmat(G{o}.mu(m, :), length(active_indices), 1), 2) + N(o) * G{o}.sigma(active_indices, m))));
            end
            s.pi(m) = logistic(psi(kappa.zeta) - psi(kappa.eta) + total);
        end
        %%%% update omega
        for o = 1:T
            omega{o}.beta = 1 / (1 / parameters.beta_omega + 0.5 * sum(s.pi .* (e_success{o}.mu.^2 + e_success{o}.sigma) + (1 - s.pi) .* (e_failure{o}.mu.^2 + e_failure{o}.sigma)));
        end
        %%%% update e_success
        for m = randperm(P)
            active_indices = setdiff(find(s.pi > 0), m);
            for o = 1:T
                e_success{o}.sigma(m) = 1 / (omega{o}.alpha * omega{o}.beta + sum(G{o}.mu(m, :).^2 + G{o}.sigma(m, m)));
                e_success{o}.mu(m) = e_success{o}.sigma(m) * (G{o}.mu(m, :) * (f{o}.mu - b{o}.mu) - sum(s.pi(active_indices) .* e_success{o}.mu(active_indices) .* (sum(G{o}.mu(active_indices, :) .* repmat(G{o}.mu(m, :), length(active_indices), 1), 2) + N(o) * G{o}.sigma(active_indices, m))));
            end
        end
        %%%% update e_failure
        for o = 1:T
            e_failure{o}.sigma = (1 / (omega{o}.alpha * omega{o}.beta)) * ones(P, 1);
        end
        %%%% update gamma
        for o = 1:T
            gamma{o}.beta = 1 / (1 / parameters.beta_gamma + 0.5 * (b{o}.mu^2 + b{o}.sigma));
        end
        %%%% update b
        for o = 1:T
            b{o}.sigma = 1 / (gamma{o}.alpha * gamma{o}.beta + N(o));
            b{o}.mu = b{o}.sigma * sum(f{o}.mu - G{o}.mu' * (s.pi .* e_success{o}.mu));
        end
        %%%% update f
        for o = 1:T
            output = G{o}.mu' * (s.pi .* e_success{o}.mu) + b{o}.mu;
            alpha_norm = lower{o} - output;
            beta_norm = upper{o} - output;
            normalization = normcdf(beta_norm) - normcdf(alpha_norm);
            normalization(normalization == 0) = 1;
            f{o}.mu = output + (normpdf(alpha_norm) - normpdf(beta_norm)) ./ normalization;
            f{o}.sigma = 1 + (alpha_norm .* normpdf(alpha_norm) - beta_norm .* normpdf(beta_norm)) ./ normalization - (normpdf(alpha_norm) - normpdf(beta_norm)).^2 ./ normalization.^2;
        end
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
