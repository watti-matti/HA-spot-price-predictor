function model = fitSeasonalElectricityPriceModel(t, P, opts)
% fitSeasonalElectricityPriceModel
% Seasonal electricity price model per Moazeni et al.:
%   P_t = P_hour(h) + P_day(d) + P_week(w) + Y_t
% where P_hour, P_day, P_week are computed as averages over historical data.
%
% Inputs
%   t    : datetime vector (Nx1) timestamps
%   P    : double   vector (Nx1) spot prices ($/MWh)
%   opts : (optional) struct with fields:
%          .weekStartsMonday (default true)  -> day-of-week indexing
%          .fitOU            (default true)  -> fit OU/AR(1) to deseasonalized Y
%          .dtHours          (default [])    -> override time step in hours
%
% Output
%   model: struct with seasonality vectors and deseasonalized series
%          model.Phour (24x1), model.Pday (7x1), model.Pweek (53x1)
%          model.seasonal (Nx1), model.Y (Nx1)
%          model.ou (optional) fitted OU parameters

    arguments
        t (:,1) datetime
        P (:,1) double
        opts.weekStartsMonday (1,1) logical = true
        opts.fitOU (1,1) logical = true
        opts.dtHours double = []
    end

    % Basic checks
    if numel(t) ~= numel(P)
        error('t and P must have the same length.');
    end
    if any(isnat(t))
        error('t contains NaT values.');
    end

    % Extract time indices
    h = hour(t) + 1;  % 1..24

    if opts.weekStartsMonday
        % weekday(...,'monday') returns 1=Monday,...,7=Sunday
        d = weekday(t, 'monday');
    else
        % weekday(t) returns 1=Sunday,...,7=Saturday
        d = weekday(t);
    end

    w = week(t, 'weekofyear');  % 1..53
    w(w > 53) = 53;  % clamp to 53 weeks max

    % Compute deterministic seasonal components as historical averages
    % (exactly as described in the paper)
    Phour  = accumarray(h, P, [24 1], @mean, NaN);
    Pday   = accumarray(d, P, [7  1], @mean, NaN);
    Pweek  = accumarray(w, P, [53 1], @mean, NaN);

    % Fill any unobserved weeks with nearest-neighbor interpolation
    nanIdx = isnan(Pweek);
    if any(nanIdx)
        validIdx = find(~nanIdx);
        for ii = find(nanIdx)'
            [~, closest] = min(abs(validIdx - ii));
            Pweek(ii) = Pweek(validIdx(closest));
        end
    end

    % Build seasonal signal per timestamp
    seasonal = Phour(h) + Pday(d) + Pweek(w);

    % Deseasonalized price series
    Y = P - seasonal;

    % Pack output
    model = struct();
    model.Phour = Phour;
    model.Pday = Pday;
    model.Pweek = Pweek;
    model.seasonal = seasonal;
    model.Y = Y;
    model.t = t;

    % Optional: Fit a mean-reverting OU (discrete-time AR(1)) to Y
    % This is a practical approximation to the mean-reversion part of the paper's
    % deseasonalized process; jump calibration is not required for seasonality calc.
    if opts.fitOU
        % Determine dt (hours)
        if isempty(opts.dtHours)
            dt = hours(median(diff(t)));
        else
            dt = opts.dtHours;
        end
        if ~isfinite(dt) || dt <= 0
            error('Invalid dtHours inferred/provided.');
        end

        % AR(1): Y_{k+1} = a + b*Y_k + eps
        Yk  = Y(1:end-1);
        Yk1 = Y(2:end);
        X = [ones(size(Yk)), Yk];
        beta = X \ Yk1;
        a = beta(1);
        b = beta(2);
        eps = Yk1 - X*beta;

        % Map to OU parameters:
        % b = exp(-lambda*dt), a = mu*(1-b)
        b_clamped = min(max(b, 1e-6), 0.999999); % keep stable/realistic
        lambda = -log(b_clamped) / dt;           % per hour
        mu = a / (1 - b_clamped);

        % For OU discretization:
        % eps std = sigma * sqrt((1 - exp(-2*lambda*dt)) / (2*lambda))
        epsStd = std(eps, 'omitnan');
        scale = sqrt((1 - exp(-2*lambda*dt)) / (2*lambda));
        sigma = epsStd / max(scale, realmin);

        model.ou = struct();
        model.ou.a = a;
        model.ou.b = b;
        model.ou.lambda_perHour = lambda;
        model.ou.mu = mu;
        model.ou.sigma = sigma;
        model.ou.dtHours = dt;
        model.ou.residuals = eps;
    end
end