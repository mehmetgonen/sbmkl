% Mehmet Gonen (mehmet.gonen@gmail.com)

function prediction = sbmkl_supervised_classification_variational_test(Km, state)
    N = size(Km, 2);
    P = size(Km, 3);

    prediction.G.mu = zeros(P, N);
    for m = 1:P
        prediction.G.mu(m, :) = state.a.mu' * Km(:, :, m);
    end

    prediction.f.mu = prediction.G.mu' * (state.s.pi .* state.e_success.mu) + state.b.mu;
end
