function prediction = sbmtmkl_supervised_classification_variational_test(Km, state)
    T = length(Km);
    N = zeros(T, 1);
    for o = 1:T
        N(o) = size(Km{o}, 2);
    end
    P = size(Km{1}, 3);

    prediction.G = cell(1, T);
    for o = 1:T
        prediction.G{o}.mu = zeros(P, N(o));
        for m = 1:P
            prediction.G{o}.mu(m, :) = state.a{o}.mu' * Km{o}(:, :, m);
        end
    end

    prediction.f = cell(1, T);
    for o = 1:T
        prediction.f{o}.mu = prediction.G{o}.mu' * (state.s.pi .* state.e_success{o}.mu) + state.b{o}.mu;
    end
end
