function results = npk_cvar_hedge_demo(data, opts)
% NPK-CVaR hedging prototype (MATLAB)
% Based on: nonparametric kernel CVaR + hedge ratio h minimizing CVaR.
% Portfolio return: r_p = r_spot - h * r_fut (long spot, short futures* h).
% The paper uses kernel density estimation and CVaR minimization. [1](https://www.frontiersin.org/journals/energy-research/articles/10.3389/fenrg.2022.887946/full)
%
% Supports two input modes:
%   1. CSV file path (original): CSV with Spot & Futures columns
%   2. Seasonal model struct: output from analyze_sahkotin_seasonal or
%      fitSeasonalElectricityPriceModel. Uses deseasonalized residuals Y_t
%      and the seasonal forecast shifted by futuresLagHours as a day-ahead
%      hedge instrument proxy.
%
% Inputs:
%   data : string (CSV path) OR struct (seasonal model results)
%   opts : (optional) struct with fields:
%          .alpha         - tail probability (default: 0.05)
%          .trainFrac     - train/test split fraction (default: 0.55)
%          .futuresLagHours - lag for seasonal hedge instrument (default: 24)
%                            Only used in seasonal mode.
%          .useDeseasonalized - if true, hedge on Y_t residuals (default: true)
%                              If false, hedge on raw prices with seasonal futures.
%
% Usage:
%   results = npk_cvar_hedge_demo("spot_futures.csv");
%   results = npk_cvar_hedge_demo(analyze_sahkotin_seasonal());
%   results = npk_cvar_hedge_demo(model, futuresLagHours=48);

    arguments
        data
        opts.alpha (1,1) double = 0.05
        opts.trainFrac (1,1) double = 0.55
        opts.futuresLagHours (1,1) double = 24
        opts.useDeseasonalized (1,1) logical = true
    end

    alpha     = opts.alpha;
    trainFrac = opts.trainFrac;

    % Optimization bounds (keep sane; adjust if needed)
    hLB = -5;  hUB = 5;

    %% -----------------------
    % Detect input mode and load data
    %% -----------------------
    isSeasonal = false;

    if isstring(data) || ischar(data)
        %% --- CSV mode (original) ---
        fprintf('\n==== NPK-CVaR Hedge Demo (CSV mode) ====\n');
        T = readtable(data);

        % Try common column names
        date = tryGetCol(T, {'Date','date','TIME','Time'});
        S    = tryGetCol(T, {'SpotPrice','Spot','spot','P_spot','Spot'});
        F    = tryGetCol(T, {'FutPrice','FuturePrice','Futures','fut','P_fut','Future'});

        % Convert date if needed
        if ~isdatetime(date)
            try
                date = datetime(date);
            catch
                date = datetime(date, 'ConvertFrom','datenum');
            end
        end

        % Compute log returns (in % like paper does). [1](https://www.frontiersin.org/journals/energy-research/articles/10.3389/fenrg.2022.887946/full)
        rS = 100 * diff(log(S(:)));
        rF = 100 * diff(log(F(:)));
        dR = date(2:end);

        % Basic cleaning
        idx = isfinite(rS) & isfinite(rF);
        rS = rS(idx); rF = rF(idx); dR = dR(idx);
        returnLabel = 'Log return (%)';

    elseif isstruct(data)
        %% --- Seasonal model mode ---
        isSeasonal = true;
        fprintf('\n==== NPK-CVaR Hedge Demo (Seasonal model mode) ====\n');

        % Accept either analyze_sahkotin_seasonal results or direct model
        if isfield(data, 'model')
            m = data.model;  % results struct from analyze_sahkotin_seasonal
        else
            m = data;        % direct fitSeasonalElectricityPriceModel output
        end

        if ~isfield(m, 'seasonal') || ~isfield(m, 'Y') || ~isfield(m, 't')
            error('Struct must contain fields: seasonal, Y, t (from fitSeasonalElectricityPriceModel).');
        end

        t_all      = m.t;
        P_all      = m.seasonal + m.Y;   % full prices
        Y_all      = m.Y;                % deseasonalized residuals
        seasonal   = m.seasonal;          % deterministic seasonal component
        lagH       = opts.futuresLagHours;

        fprintf('Seasonal model detected (%d observations).\n', numel(P_all));
        fprintf('Period: %s to %s\n', string(t_all(1)), string(t_all(end)));
        fprintf('Futures proxy: seasonal forecast lagged by %d hours.\n', lagH);

        if opts.useDeseasonalized
            % --- Hedge on deseasonalized Y_t ---
            % "Spot" = Y_t changes (residual risk after removing seasonality)
            % "Futures" = lagged Y_t changes (mean-reverting autocorrelation hedge)
            fprintf('Mode: Hedging deseasonalized residuals Y_t.\n');

            % Compute hourly differences of Y (not log returns - Y can be negative)
            dY = diff(Y_all);              % (N-1) x 1

            % rS(t) = dY(t), rF(t) = dY(t-lagH)  (lagged changes as hedge)
            rS = dY(lagH+1:end);           % current changes
            rF = dY(1:end-lagH);           % lagged changes
            dR = t_all(lagH+2:end);        % aligned timestamps
            returnLabel = '\DeltaY_t (EUR/MWh)';
        else
            % --- Hedge raw prices using seasonal forecast as futures ---
            % "Spot" = actual price returns
            % "Futures" = seasonal forecast returns (known day-ahead)
            fprintf('Mode: Hedging raw prices against seasonal forecast.\n');

            % Seasonal forecast shifted forward = "day-ahead price"
            Fwd = [seasonal(lagH+1:end); repmat(seasonal(end), lagH, 1)];

            % Use differences (prices can include zero/negative)
            rS = diff(P_all);
            rF = diff(Fwd);
            dR = t_all(2:end);
            returnLabel = '\DeltaP (EUR/MWh)';
        end

        % Basic cleaning
        idx = isfinite(rS) & isfinite(rF);
        rS = rS(idx); rF = rF(idx); dR = dR(idx);
    else
        error('Input must be a CSV file path (string) or a seasonal model struct.');
    end

    n = numel(rS);
    nTrain = max(50, floor(trainFrac * n));
    rS_tr = rS(1:nTrain); rF_tr = rF(1:nTrain);
    rS_te = rS(nTrain+1:end); rF_te = rF(nTrain+1:end);
    d_te  = dR(nTrain+1:end);

    fprintf('Training: %d observations, Test: %d observations.\n', nTrain, n - nTrain);

%% -----------------------
% Initial guess: minimum-variance hedge ratio
% h_mv = cov(rS,rF)/var(rF)
%% -----------------------
h0 = (cov(rS_tr, rF_tr, 'partialrows') );
h0 = h0(1,2) / var(rF_tr, 'omitnan');
h0 = min(max(h0, hLB), hUB);

% Loss is negative return: L = -r_p
L0 = -(rS_tr - h0 * rF_tr);
v0 = quantile(L0, 1 - alpha); % initial VaR guess on losses

x0 = [h0, v0];

%% -----------------------
% Optimize (h, v) to minimize kernel-smoothed Rockafellar objective:
% CVaR_hat = min_{h,v}  v + (1/alpha) E[(L - v)+]
% Here E[(L - v)+] is estimated under a Gaussian-kernel density estimator. [1](https://www.frontiersin.org/journals/energy-research/articles/10.3389/fenrg.2022.887946/full)
%% -----------------------
obj = @(x) npk_cvar_objective(x, rS_tr, rF_tr, alpha);

options = optimset('Display','iter','TolX',1e-6,'TolFun',1e-6,'MaxIter',5000,'MaxFunEvals',20000);

% Use fminsearch (unconstrained) + penalty for h bounds:
objPen = @(x) obj([clip(x(1),hLB,hUB), x(2)]) + 1e3*( ...
            max(0,hLB-x(1))^2 + max(0,x(1)-hUB)^2 );

xHat = fminsearch(objPen, x0, options);
hHat = clip(xHat(1), hLB, hUB);
vHat = xHat(2);

% Estimated CVaR (loss-domain)
cvarHat_tr = npk_cvar_from_hv(hHat, vHat, rS_tr, rF_tr, alpha);

%% -----------------------
% Compare against historical (empirical) CVaR for sanity check
%% -----------------------
cvarHist_tr_unhedged = hist_cvar(-(rS_tr), alpha);
cvarHist_tr_hedged   = hist_cvar(-(rS_tr - hHat*rF_tr), alpha);

%% -----------------------
% Out-of-sample test performance
%% -----------------------
cvarHist_te_unhedged = hist_cvar(-(rS_te), alpha);
cvarHist_te_hedged   = hist_cvar(-(rS_te - hHat*rF_te), alpha);

%% -----------------------
% Package results
%% -----------------------
results.alpha = alpha;
results.hHat = hHat;
results.vHat = vHat;
results.cvarHat_tr_kernel = cvarHat_tr;
results.cvarHist_tr_unhedged = cvarHist_tr_unhedged;
results.cvarHist_tr_hedged   = cvarHist_tr_hedged;
results.cvarHist_te_unhedged = cvarHist_te_unhedged;
results.cvarHist_te_hedged   = cvarHist_te_hedged;
results.isSeasonal = isSeasonal;
if isSeasonal
    results.futuresLagHours = opts.futuresLagHours;
    results.useDeseasonalized = opts.useDeseasonalized;
end

fprintf('\n==== NPK-CVaR Hedge Results ====\n');
if isSeasonal
    if opts.useDeseasonalized
        fprintf('Mode                  : Seasonal (deseasonalized Y_t)\n');
    else
        fprintf('Mode                  : Seasonal (raw prices, seasonal futures)\n');
    end
    fprintf('Futures lag           : %d hours\n', opts.futuresLagHours);
end
fprintf('alpha (tail prob)     : %.4f\n', alpha);
fprintf('Optimal hedge ratio h : %.6f\n', hHat);
fprintf('Optimal VaR param v   : %.6f (loss-domain)\n', vHat);
fprintf('Kernel-CVaR (train)   : %.6f\n', cvarHat_tr);
fprintf('Hist-CVaR train unhed : %.6f\n', cvarHist_tr_unhedged);
fprintf('Hist-CVaR train hedged: %.6f\n', cvarHist_tr_hedged);
fprintf('Hist-CVaR test  unhed : %.6f\n', cvarHist_te_unhedged);
fprintf('Hist-CVaR test  hedged: %.6f\n', cvarHist_te_hedged);

%% -----------------------
% Quick visualization (test losses)
%% -----------------------
L_te_unh = -(rS_te);
L_te_hed = -(rS_te - hHat*rF_te);

if isSeasonal
    figTitle = sprintf('Test Losses – Seasonal (alpha=%.2f, h=%.3f, lag=%dh)', ...
        alpha, hHat, opts.futuresLagHours);
else
    figTitle = sprintf('Test Losses (alpha=%.2f). Hedge ratio h=%.3f', alpha, hHat);
end

figure('Name','Test Losses: Unhedged vs Hedged');
plot(d_te, L_te_unh, 'Color',[0.7 0.7 0.7]); hold on;
plot(d_te, L_te_hed, 'b');
yline(quantile(L_te_unh, 1-alpha),'--','Color',[0.5 0.5 0.5],'Label','VaR (unhedged)');
yline(quantile(L_te_hed, 1-alpha),'--b','Label','VaR (hedged)');
grid on;
legend('Loss unhedged','Loss hedged','Location','best');
title(figTitle);
xlabel('Date'); ylabel(['Loss = -(' returnLabel ')']);

end

%% ===== Helper: NPK CVaR objective (Rockafellar form) =====
function J = npk_cvar_objective(x, rS, rF, alpha)
% x = [h, v]; L = -(rS - h*rF)
h = x(1); v = x(2);
L = -(rS - h*rF);
T = numel(L);

% Bandwidth per rule-of-thumb in paper: b = 1.06 * sigma * T^(-1/5) [1](https://www.frontiersin.org/journals/energy-research/articles/10.3389/fenrg.2022.887946/full)
sig = std(L, 'omitnan');
b   = 1.06 * sig * T^(-1/5);
b   = max(b, 1e-8);

% Gaussian kernel closed-form for E[(L - v)+] under KDE:
% For each sample Li, z = (v - Li)/b
% E_k[(L - v)+] approx mean_i [ (Li - v)*(1 - Phi(z)) + b*phi(z) ]
z   = (v - L) / b;
Phi = 0.5 * erfc(-z / sqrt(2));          % normcdf(z) without toolbox
phi = exp(-0.5 * z.^2) / sqrt(2 * pi);   % normpdf(z) without toolbox

tailMeanPlus = mean( (L - v).*(1 - Phi) + b.*phi, 'omitnan');

J = v + (1/alpha) * tailMeanPlus;

% Optional slight regularization for numerical stability (tiny)
J = J + 1e-12 * (h^2 + v^2);
end

%% ===== Helper: compute CVaR at solution =====
function cvar = npk_cvar_from_hv(h, v, rS, rF, alpha)
cvar = npk_cvar_objective([h v], rS, rF, alpha);
end

%% ===== Helper: empirical/historical CVaR (sanity check) =====
function cvar = hist_cvar(L, alpha)
L = L(isfinite(L));
if isempty(L), cvar = NaN; return; end
q = quantile(L, 1 - alpha);         % VaR at (1-alpha) for losses
tail = L(L >= q);                   % worst alpha tail (losses)
cvar = mean(tail);
end

%% ===== Helper: safe column fetch =====
function col = tryGetCol(T, names)
col = [];
for i = 1:numel(names)
    if ismember(names{i}, T.Properties.VariableNames)
        col = T.(names{i});
        return;
    end
end
error('Could not find any of the columns: %s', strjoin(names, ', '));
end

function y = clip(x, a, b)
y = min(max(x,a), b);
end