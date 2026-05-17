function results = analyze_sahkotin_seasonal(dataFile, opts)
% analyze_sahkotin_seasonal
% Loads Finnish electricity spot prices from Sähkötin and fits the seasonal
% price model from Moazeni et al. (2015) "Mean-Conditional Value-at-Risk
% Optimal Energy Storage Operation".
%
% Model: P_t = P_hour(h) + P_day(d) + P_week(w) + Y_t
% where Y_t follows an Ornstein-Uhlenbeck (mean-reverting) process.
%
% Inputs:
%   dataFile : (optional) path to .mat file from load_sahkotin
%              If not provided or file doesn't exist, downloads data.
%   opts     : (optional) struct with fields:
%              .useLocalTime (default true) - use Helsinki time for analysis
%              .plotResults  (default true) - generate diagnostic plots
%              .yearFilter   (default [])   - filter to specific years, e.g. [2023 2024]
%
% Outputs:
%   results : struct containing:
%             .T          - raw price table
%             .model      - fitted seasonal model (from fitSeasonalElectricityPriceModel)
%             .stats      - summary statistics
%
% Usage:
%   results = analyze_sahkotin_seasonal();
%   results = analyze_sahkotin_seasonal("fi_spot_2022_2025_sahkotin.mat");
%   results = analyze_sahkotin_seasonal([], yearFilter=[2023 2024]);
%
% Reference:
%   Moazeni, S., Powell, W. B., & Hajimiragha, A. H. (2015).
%   Mean-conditional value-at-risk optimal energy storage operation in the
%   presence of transaction costs and efficient computation.
%   IEEE Transactions on Power Systems, 30(3), 1222-1232.

    arguments
        dataFile string = "fi_spot_2022_2025_sahkotin.mat"
        opts.useLocalTime (1,1) logical = true
        opts.plotResults (1,1) logical = true
        opts.yearFilter double = []
    end

    %% 1. Load or download data
    fprintf("=== Sähkötin Seasonal Price Analysis (Moazeni et al. model) ===\n\n");
    
    if exist(dataFile, "file")
        fprintf("Loading data from: %s\n", dataFile);
        loaded = load(dataFile, "T");
        T = loaded.T;
    else
        fprintf("Data file not found. Downloading from Sähkötin API...\n");
        [folder, base, ~] = fileparts(dataFile);
        if isempty(folder)
            folder = pwd;
        end
        outBase = fullfile(folder, base);
        load_sahkotin(outBase);
        loaded = load(dataFile, "T");
        T = loaded.T;
    end
    
    fprintf("Loaded %d price observations.\n", height(T));
    fprintf("Date range: %s to %s\n", string(min(T.timestamp_utc)), string(max(T.timestamp_utc)));
    
    %% 2. Prepare timestamps and prices
    if opts.useLocalTime && ismember("timestamp_local", T.Properties.VariableNames)
        t = T.timestamp_local;
        tzLabel = "Europe/Helsinki";
    else
        t = T.timestamp_utc;
        tzLabel = "UTC";
    end
    
    % Price in EUR/MWh (Sähkötin returns c/kWh, convert)
    % Note: Sähkötin prices are typically in c/kWh, so multiply by 10 for EUR/MWh
    P = T.price;  % Assuming already in appropriate units
    
    % Check if prices look like c/kWh (typical range 0-50) or EUR/MWh (0-500)
    medianPrice = median(P, 'omitnan');
    if medianPrice < 100
        fprintf("Prices appear to be in c/kWh (median=%.2f). Converting to EUR/MWh.\n", medianPrice);
        P = P * 10;  % Convert c/kWh to EUR/MWh
        priceUnit = "EUR/MWh";
    else
        priceUnit = "EUR/MWh (assumed)";
    end
    
    %% 3. Filter by year if requested
    if ~isempty(opts.yearFilter)
        yearMask = ismember(year(t), opts.yearFilter);
        t = t(yearMask);
        P = P(yearMask);
        T = T(yearMask, :);
        fprintf("Filtered to years %s: %d observations.\n", mat2str(opts.yearFilter), numel(P));
    end
    
    % Remove NaN prices
    validMask = ~isnan(P) & ~isnat(t);
    if sum(~validMask) > 0
        fprintf("Removing %d invalid observations (NaN/NaT).\n", sum(~validMask));
        t = t(validMask);
        P = P(validMask);
        T = T(validMask, :);
    end
    
    %% 4. Fit seasonal model
    fprintf("\nFitting seasonal model (Moazeni et al.)...\n");
    model = fitSeasonalElectricityPriceModel(t, P, weekStartsMonday=true, fitOU=true);
    
    %% 5. Compute summary statistics
    stats = struct();
    stats.nObs = numel(P);
    stats.priceUnit = priceUnit;
    stats.timezone = tzLabel;
    stats.dateRange = [min(t), max(t)];
    
    % Price statistics
    stats.priceMean = mean(P, 'omitnan');
    stats.priceStd = std(P, 'omitnan');
    stats.priceMin = min(P);
    stats.priceMax = max(P);
    stats.priceMedian = median(P, 'omitnan');
    
    % Deseasonalized residual statistics
    stats.residualMean = mean(model.Y, 'omitnan');
    stats.residualStd = std(model.Y, 'omitnan');
    
    % Seasonality variance decomposition
    totalVar = var(P, 'omitnan');
    seasonalVar = var(model.seasonal, 'omitnan');
    residualVar = var(model.Y, 'omitnan');
    stats.seasonalVarPct = 100 * seasonalVar / totalVar;
    stats.residualVarPct = 100 * residualVar / totalVar;
    
    % OU process parameters
    if isfield(model, 'ou')
        stats.ou_lambda = model.ou.lambda_perHour;
        stats.ou_halfLife_hours = log(2) / model.ou.lambda_perHour;
        stats.ou_mu = model.ou.mu;
        stats.ou_sigma = model.ou.sigma;
    end
    
    %% 6. Print results
    fprintf("\n=== RESULTS ===\n");
    fprintf("Data: %d hourly observations (%s)\n", stats.nObs, tzLabel);
    fprintf("Period: %s to %s\n", string(stats.dateRange(1)), string(stats.dateRange(2)));
    fprintf("\nPrice Statistics (%s):\n", priceUnit);
    fprintf("  Mean:   %8.2f\n", stats.priceMean);
    fprintf("  Std:    %8.2f\n", stats.priceStd);
    fprintf("  Min:    %8.2f\n", stats.priceMin);
    fprintf("  Max:    %8.2f\n", stats.priceMax);
    fprintf("  Median: %8.2f\n", stats.priceMedian);
    
    fprintf("\nSeasonality Components (hourly averages, %s):\n", priceUnit);
    fprintf("  Hour-of-day range:    [%.2f, %.2f]\n", min(model.Phour), max(model.Phour));
    fprintf("  Day-of-week range:    [%.2f, %.2f]\n", min(model.Pday), max(model.Pday));
    fprintf("  Week-of-year range:   [%.2f, %.2f]\n", min(model.Pweek), max(model.Pweek));
    
    fprintf("\nVariance Decomposition:\n");
    fprintf("  Seasonal component: %5.1f%%\n", stats.seasonalVarPct);
    fprintf("  Residual (Y_t):     %5.1f%%\n", stats.residualVarPct);
    
    if isfield(model, 'ou')
        fprintf("\nOrnstein-Uhlenbeck Process (deseasonalized Y_t):\n");
        fprintf("  Mean-reversion rate (lambda): %.4f /hour\n", stats.ou_lambda);
        fprintf("  Half-life:                    %.1f hours\n", stats.ou_halfLife_hours);
        fprintf("  Long-run mean (mu):           %.2f %s\n", stats.ou_mu, priceUnit);
        fprintf("  Volatility (sigma):           %.2f\n", stats.ou_sigma);
    end
    
    %% 7. Generate plots
    if opts.plotResults
        plotSeasonalAnalysis(t, P, model, stats, priceUnit);
    end
    
    %% 8. Pack output
    results = struct();
    results.T = T;
    results.model = model;
    results.stats = stats;
    
    fprintf("\nAnalysis complete.\n");
end

%% ========================================================================
% Plotting function
% ========================================================================
function plotSeasonalAnalysis(t, P, model, stats, priceUnit)
    
    figure('Name', 'Sähkötin Seasonal Price Analysis', 'Position', [100 100 1400 900]);
    
    % 1. Raw price time series
    subplot(3,3,1);
    plot(t, P, 'b-', 'LineWidth', 0.3);
    hold on;
    plot(t, model.seasonal, 'r-', 'LineWidth', 1);
    xlabel('Time');
    ylabel(priceUnit);
    title('Spot Price vs Seasonal Component');
    legend('Spot Price', 'Seasonal', 'Location', 'best');
    grid on;
    
    % 2. Deseasonalized residuals
    subplot(3,3,2);
    plot(t, model.Y, 'Color', [0.2 0.6 0.2], 'LineWidth', 0.3);
    xlabel('Time');
    ylabel(priceUnit);
    title('Deseasonalized Price Y_t');
    grid on;
    
    % 3. Hourly pattern
    subplot(3,3,3);
    bar(0:23, model.Phour, 'FaceColor', [0.3 0.5 0.8]);
    xlabel('Hour of Day');
    ylabel(priceUnit);
    title('P_{hour}(h) - Hourly Pattern');
    xlim([-0.5 23.5]);
    grid on;
    
    % 4. Day-of-week pattern
    subplot(3,3,4);
    bar(1:7, model.Pday, 'FaceColor', [0.8 0.4 0.2]);
    xlabel('Day of Week');
    ylabel(priceUnit);
    title('P_{day}(d) - Weekly Pattern');
    set(gca, 'XTick', 1:7, 'XTickLabel', {'Mon','Tue','Wed','Thu','Fri','Sat','Sun'});
    grid on;
    
    % 5. Weekly pattern
    subplot(3,3,5);
    bar(1:53, model.Pweek, 'FaceColor', [0.2 0.7 0.5]);
    xlabel('Week of Year');
    ylabel(priceUnit);
    title('P_{week}(w) - Weekly Pattern');
    xlim([0.5 53.5]);
    grid on;
    
    % 6. Residual histogram
    subplot(3,3,6);
    histogram(model.Y, 50, 'Normalization', 'pdf', 'FaceColor', [0.5 0.5 0.8]);
    hold on;
    x = linspace(min(model.Y), max(model.Y), 200);
    mu_Y = mean(model.Y, 'omitnan');
    sigma_Y = std(model.Y, 'omitnan');
    y = (1 / (sigma_Y * sqrt(2*pi))) * exp(-0.5 * ((x - mu_Y) / sigma_Y).^2);  % Normal PDF
    plot(x, y, 'r-', 'LineWidth', 2);
    xlabel(priceUnit);
    ylabel('Density');
    title('Residual Distribution');
    legend('Empirical', 'Normal fit', 'Location', 'best');
    grid on;
    
    % 7. Residual ACF (manual implementation)
    subplot(3,3,7);
    Yvalid = model.Y(~isnan(model.Y));
    maxLag = min(72, length(Yvalid)-1);
    acf = zeros(maxLag+1, 1);
    Ycentered = Yvalid - mean(Yvalid);
    acf(1) = 1;  % lag 0
    for k = 1:maxLag
        acf(k+1) = sum(Ycentered(1:end-k) .* Ycentered(k+1:end)) / sum(Ycentered.^2);
    end
    lags = 0:maxLag;
    stem(lags, acf, 'filled', 'MarkerSize', 3);
    xlabel('Lag (hours)');
    ylabel('ACF');
    title('Residual Autocorrelation');
    grid on;
    
    % 8. QQ plot (manual implementation)
    subplot(3,3,8);
    Ysorted = sort(Yvalid);
    n = length(Ysorted);
    p = ((1:n)' - 0.5) / n;  % Plotting positions
    theoretical = mu_Y + sigma_Y * sqrt(2) * erfinv(2*p - 1);  % Normal quantiles via erfinv
    plot(theoretical, Ysorted, 'bo', 'MarkerSize', 2);
    hold on;
    qRange = [min(theoretical), max(theoretical)];
    plot(qRange, qRange, 'r-', 'LineWidth', 1.5);
    xlabel('Theoretical Quantiles');
    ylabel('Sample Quantiles');
    title('Residual Q-Q Plot');
    grid on;
    
    % 9. Variance decomposition pie chart
    subplot(3,3,9);
    pie([stats.seasonalVarPct, stats.residualVarPct], ...
        {sprintf('Seasonal\n%.1f%%', stats.seasonalVarPct), ...
         sprintf('Residual\n%.1f%%', stats.residualVarPct)});
    title('Variance Decomposition');
    
    sgtitle(sprintf('Finnish Electricity Spot Price Analysis (Moazeni et al. Model)\n%s to %s', ...
        string(stats.dateRange(1), 'yyyy-MM-dd'), string(stats.dateRange(2), 'yyyy-MM-dd')));
    
    % Second figure: detailed seasonality
    figure('Name', 'Seasonality Details', 'Position', [150 150 1000 600]);
    
    % Heatmap: hour vs day-of-week
    subplot(1,2,1);
    hourDayMat = zeros(24, 7);
    h = hour(t) + 1;
    d = weekday(t, 'monday');
    for hh = 1:24
        for dd = 1:7
            mask = (h == hh) & (d == dd);
            if sum(mask) > 0
                hourDayMat(hh, dd) = mean(P(mask), 'omitnan');
            end
        end
    end
    imagesc(hourDayMat);
    colorbar;
    xlabel('Day of Week');
    ylabel('Hour of Day');
    title('Average Price by Hour & Day');
    set(gca, 'XTick', 1:7, 'XTickLabel', {'Mon','Tue','Wed','Thu','Fri','Sat','Sun'});
    set(gca, 'YTick', [1 6 12 18 24], 'YTickLabel', {'00','05','11','17','23'});
    colormap(turbo);
    
    % Heatmap: hour vs week-of-year
    subplot(1,2,2);
    hourWeekMat = zeros(24, 53);
    w = week(t, 'weekofyear');
    w(w > 53) = 53;
    for hh = 1:24
        for ww = 1:53
            mask = (h == hh) & (w == ww);
            if sum(mask) > 0
                hourWeekMat(hh, ww) = mean(P(mask), 'omitnan');
            end
        end
    end
    imagesc(hourWeekMat);
    colorbar;
    xlabel('Week of Year');
    ylabel('Hour of Day');
    title('Average Price by Hour & Week');
    set(gca, 'XTick', [1 5:5:50 53]);
    set(gca, 'YTick', [1 6 12 18 24], 'YTickLabel', {'00','05','11','17','23'});
    colormap(turbo);
    
    sgtitle(sprintf('Seasonality Heatmaps (%s)', priceUnit));
end
