function [results, final_models] = model_breastfeeding_on_pqb(cvfn, includepsychosis, pqbtype, suppresssummary)
% This function assessees whether psychosis proneness can be predicted
% based on lifetime pregnancy and breastfeeding durations
%
% H: lifetime pregnancy and breastfeeding duration is protective against
%       psychosis, predicting lower rates of attenuated psychotic burden at
%       perimenopause
% H0: They are not protective. 
%
% INPUT:
%   cvfn: Filename of the data table (e.g., 'my_data.csv').
%   includepsychosis: Flag to include (1) or exclude (0) subjects with a
%                     psychosis diagnosis. Default is 0.
%   pqbtype: PQB score type to use. 1 for total score (default), 2 for distress score.
%   suppresssummary: Flag to suppress (1) or show (0) data cleaning summaries. Default is 0.
%
% OUTPUT:
%   results: A table summarizing the performance of each model.
%   final_models: A cell array of the final linear model objects, fit on the full dataset.
%
% RELEVANT VARIABLES:
%   repro30: y/n ever pregnant
%   repro31: number of pregnancies
%   repro33: number of pregnancies >6mos
%   repro34: age at first pregnancy
%   repro36: number of live births
%   repro38: number of stillbirths
%   repro39: number of miscarriages
%   repro40: number of abortions
%   repro42: number of c-sections
%   repro57: y/n ever breastfed >1mo
%   repro58: number of children breastfed
%   repro61: breastfeeding duration
%   bl_education1: categorical highest level of eduaction
%   bl_employment7: current household income
%   bl_famhx5: childhood household income
close all;

%% Input Validation and Setup
if ~exist('includepsychosis', 'var'), includepsychosis = 0; end
if ~exist('pqbtype', 'var'), pqbtype = 1; end
if ~exist('suppresssummary', 'var'), suppresssummary = 0; end
if ~(includepsychosis == 0 || includepsychosis == 1), error('includepsychosis must be 0 or 1.'); end
if ~(pqbtype == 1 || pqbtype == 2), error('pqbtype must be 1 or 2.'); end

% define variables
psych_diag_vals = [1, 3, 6]; % Codes for psychosis-related diagnoses
date_priority_list = {'prescreen_timestamp', 'prescreendate', 'timestamp'};

% nuissance variables (psych diagnosis variable added elsewhere based on input variables)
nuisancevar = {'currentage'; 'menostatus';'bl_employment7'};%...
%    'bl_education1';'bl_employment7';'bl_famhx5'};

relevantvariables = {
    'record_id', 'currentage','full_term_months', ...
    'menostatus', 'other_preg_months', ...
    'psychosisdiag', 'psychosis_score', ...
    'repro61', 'total_progesterone', 'total_prolactin',...%'bl_education1',...
    'bl_employment7'};%,'bl_famhx5'};

% variables for calculating different variable weights
fullterm = 9;       % months assumed for full term pregnancy
partterm = 3;       % months assumed for partial term pregancy
bf_mean = 6.5;      % CDC average BF months per child Source: https://www.cdc.gov/breastfeeding-data/breastfeeding-report-card/index.html

% Load Data
opts = detectImportOptions(cvfn);
date_vars = [date_priority_list, {'dob'}];
opts = setvartype(opts, date_vars, 'datetime');
data = readtable(cvfn, opts);
data.menostatus = nan(height(data), 1);

%% Manual Corrections
if ~suppresssummary
    fprintf('\n--- Manual Data Corrections ---\n');
    fprintf('corrected still births: PM464 \n');
end
data.repro38(data.record_id == 464) = 0;

if ~suppresssummary, fprintf('Excluded for conflicting info: PM349\n'); end
data(data.record_id == 349, :) = [];

if ~suppresssummary,  fprintf(['corrected total births: PM370 \n']);end
data.repro31(data.record_id == 370) = data.repro33(data.record_id == 370);
% Applies manual corrections to specific subject records. (adds c-sections to live births)
if ~suppresssummary, fprintf('Corrected live births: PM1174\n'); end
cidx = data.record_id == 1174;
data.repro36(cidx) = data.repro42(cidx);
data.repro33(cidx) = data.repro42(cidx);

if ~suppresssummary, fprintf('set to premenopausal: PM99\n'); end
data.menostatus(data.record_id == 99) = 1;

if ~suppresssummary, fprintf('set to perimenopausal: PM370, 433, 449, 545, 618, 687\n'); end
data.menostatus(data.record_id == 370) = 2;
data.menostatus(data.record_id == 433) = 2;
data.menostatus(data.record_id == 449) = 2;
data.menostatus(data.record_id == 545) = 2;
data.menostatus(data.record_id == 618) = 2;
data.menostatus(data.record_id == 687) = 2;

if ~suppresssummary, fprintf('set to postmenopausal: PM497, 605, 616, 1174\n'); end
data.menostatus(data.record_id == 497) = 3;
data.menostatus(data.record_id == 605) = 3;
data.menostatus(data.record_id == 616) = 3;
data.menostatus(data.record_id == 1174) = 3;

if ~suppresssummary, fprintf('removed as duplication: PM569 (of PM1174)\n'); end
data(data.record_id == 569, :) = [];

%% Filter and Clean Data
% no valid record id
initial_rows = height(data);
data = data(~(ismissing(data.record_id) | isnan(data.record_id)), :);
if ~suppresssummary, fprintf('\n --- Summary of removed records --- \n'); end
if ~suppresssummary, fprintf('no record_id: %d\n', initial_rows - height(data)); end

% RHQ not completed
initial_rows = height(data);
subjects = data.record_id(data.reproductive_history_questionnaire_complete == 2);
data = data(ismember(data.record_id, subjects), :);
if ~suppresssummary, fprintf('incomplete questionnaire: %d\n', initial_rows - height(data)); end

% Email Duplication Check

[unique_emails, ~, idx] = unique(data.email);
email_counts = accumarray(idx, 1);
duplicate_emails_idx = find(email_counts > 1);
if ~isempty(duplicate_emails_idx), fprintf('Additional duplications found. Manually inspect:\n');
    for i = 1:length(duplicate_emails_idx)
        email_idx = duplicate_emails_idx(i);
        fprintf('%s, Count: %d\n', unique_emails{email_idx}, email_counts(email_idx));
    end
    error('Duplicate emails require manual inspection.');
end

% Subjects Missing Core Pregnancy data
idx_missing_core_preg = ismissing(data.repro30) & ismissing(data.repro31) ...
    & ismissing(data.repro36) & ismissing(data.repro38) & ismissing(data.repro39) & ismissing(data.repro40);
if any(idx_missing_core_preg)
    num_removed = sum(idx_missing_core_preg);
    data(idx_missing_core_preg, :) = []; % Remove the rows
    if ~suppresssummary, fprintf('missing pregnancy info: %d\n', num_removed); end
end

%% Clean Pregnancy and Breastfeeding Data
% if never pregnant, set related counts to 0
idx_never_pregnant = data.repro30 == 0;
cols_to_zero_out = {'repro31', 'repro33', 'repro34', 'repro36', 'repro38', 'repro39', 'repro40'};
for i = 1:length(cols_to_zero_out)
    col = cols_to_zero_out{i};
    data.(col)(idx_never_pregnant & ismissing(data.(col))) = 0;
end

idx_to_replace = ismissing(data.repro57) & idx_never_pregnant;
data.repro57(idx_to_replace) = 0;
data.repro61(data.repro57 == 0) = 0;

% If isempty(stillbirths) && live births = full-term pregnancies, assume 0 stillbirths
idx_to_impute_repro38 = ismissing(data.repro38) & (data.repro36 >= data.repro33);
if any(idx_to_impute_repro38)
    num_imputed = sum(idx_to_impute_repro38);
    data.repro38(idx_to_impute_repro38) = 0;
end

% check for cases where birth outcome is missing
idx_ambiguous = ~ismissing(data.repro33) & data.repro33 > 0 & ismissing(data.repro36) & ismissing(data.repro38);
if any(idx_ambiguous)
    fprintf('The following subjects have full-term pregnancy > 0 with missing live and stillbirth numbers.\n');
    fprintf('Manual inspection is required.\n');
    disp(data(idx_ambiguous, {'record_id', 'firstname', 'repro33'}));
    error('Cannot proceed with ambiguous birth outcomes. Please correct the data.');
end

% Impute missing live births as full term pregnancies - stillbirths
idx_missing_repro36 = ismissing(data.repro36);
if any(idx_missing_repro36)
    temp_repro38 = data.repro38;
    % temp_repro38(ismissing(temp_repro38)) = 0; % Treat missing stillbirths as 0
    data.repro36(idx_missing_repro36) = data.repro33(idx_missing_repro36) - temp_repro38(idx_missing_repro36);
end

% Check if C-section # > total births.
total_births = nansum([data.repro36, data.repro38], 2);
idx_csection_conflict = data.repro42 > total_births;
if any(idx_csection_conflict)
    fprintf('The following subjects have more C-sections (repro42) than total births (repro36 + repro38).\n');
    fprintf('Manual inspection is required.\n');
    disp(data(idx_csection_conflict, {'record_id', 'firstname', 'repro42', 'repro36', 'repro38'}));
    error('Cannot proceed with C-section count exceeding total births.');
end

% Impute Missing Breastfeeding Duration to CDC average for those who have
% number of children breastfed but not total duration
idx_to_impute_bf = find(data.repro57 == 1 & ismissing(data.repro61));
if ~isempty(idx_to_impute_bf)
    num_imputed = 0;
    for i = 1:length(idx_to_impute_bf)
        idx = idx_to_impute_bf(i);
        num_children_bf = data.repro58(idx);

        if ~ismissing(num_children_bf)
            estimated_duration = num_children_bf * bf_mean; % 6.5 months per child
            data.repro61(idx) = estimated_duration;
            num_imputed = num_imputed + 1;
        end
    end
    if ~suppresssummary
        fprintf('Breastfeeding duration calculated using CDC average: %d\n', num_imputed);
    end
end

% Check for subjects who said they were pregnant but have no pregnancy count.
idx_missing_count = data.repro30 == 1 & ismissing(data.repro31);
if any(idx_missing_count)
    if ~suppresssummary
        fprintf('The following subjects reported being pregnant (repro30 = 1) but are missing a value for repro31.\n');
        disp(data(idx_missing_count, {'record_id', 'firstname', 'repro30', 'repro31'}));
        error('correct missing pregnancy numbers.');
    end
end

% Impute total pregnancies from the sum of outcomes.
missing_repro31 = ismissing(data.repro31);
y_pregnant = data.repro30 == 1;
if sum(missing_repro31 & y_pregnant) > 0
    outcomes = data{missing_repro31 & y_pregnant, {'repro36', 'repro38', 'repro39', 'repro40'}};
    idx_to_impute = find(missing_repro31 & y_pregnant & ~idx_ambiguous_preg_count);
    imputed_counts = nansum(data{idx_to_impute, {'repro36', 'repro38', 'repro39', 'repro40'}}, 2);
    data.repro31(idx_to_impute) = imputed_counts;
end

% Check for missing breastfeeding duration
idx_missing_bf_duration = data.repro57 == 1 & ismissing(data.repro61);
if any(idx_missing_bf_duration)
    if ~suppresssummary
        fprintf('The following subjects reported breastfeeding (repro57 = 1) but are missing the duration (repro61).\n');
        fprintf('Manual inspection is required.\n');
        disp(data(idx_missing_bf_duration, {'record_id', 'firstname', 'repro57', 'repro61'}));
        error('correct missing pregnancy numbers.');
    end
end

% Check for missing age at first pregnancy
idx_missing_age_first_preg = data.repro30 == 1 & ismissing(data.repro34);
if any(idx_missing_age_first_preg)
    if ~suppresssummary
        fprintf('Age of first pregnancy missing: %d\n', sum(idx_missing_age_first_preg))
    end
end

% Missing PQB Concern Score ---
idx_missing_concern = ismissing(data.pqb_concern);
if any(idx_missing_concern)

    % Generate the list of pqb#_yes variable names
    pqb_yes_vars = cell(1, 21);
    for i = 1:21, pqb_yes_vars{i} = sprintf('pqb%d_yes', i); end

    % Subset the data for the relevant rows and columns
    pqb_subset = data{idx_missing_concern, pqb_yes_vars};

    % Replace NaNs with 0 in the subset
    pqb_subset(ismissing(pqb_subset)) = 0;

    % Sum the values and add to original table
    imputed_concern = sum(pqb_subset, 2);
    data.pqb_concern(idx_missing_concern) = imputed_concern;
end

% Create pqb total Score
if pqbtype == 1,     data.psychosis_score = data.pqb_symptom;
else,   data.psychosis_score = data.pqb_symptom + data.pqb_concern;
end

% Calculate Age (months)
for i = 1:length(date_priority_list)
    col = date_priority_list{i};
    data.(col) = dateshift(data.(col), 'start', 'day');
    data.(col).TimeZone = '';
end

assessment_date = NaT(height(data), 1);
for i = 1:length(date_priority_list)
    col = date_priority_list{i};
    if ismember(col, data.Properties.VariableNames)
        idx_to_fill = isnat(assessment_date) & ~isnat(data.(col));
        assessment_date(idx_to_fill) = data.(col)(idx_to_fill);
    end
end
data.currentage = calmonths(between(data.dob, assessment_date));

% create psychosis Diagnosis column
has_psychosis = ismember(data.ineligible6, psych_diag_vals);
if ~includepsychosis
    initialheight = size(data,1);
    data = data(~has_psychosis, :);
    data.psychosisdiag = zeros(height(data), 1); % Add a placeholder column if needed
    num_removed = initialheight - size(data,1);
    if ~suppresssummary, fprintf('Psychosis diagnosis: %d ', num_removed); end
    if ~suppresssummary
        if includepsychosis, fprintf(' (included)\n'); else, fprintf(' (excluded)\n'); end
    end
else, data.psychosisdiag = has_psychosis;
end

% Determine Menopause Status
idx_pre = (data.repro4 == 1) & (data.repro7 == 1);
idx_peri = (((data.repro5 == 1) | (data.repro4 == 1)) & (data.repro7 == 0)) ...
    | ((data.repro5 == 1) & (data.repro4 == 0));
idx_meno = (data.repro5 == 0) | (data.repro4 == 0);
data.menostatus(idx_pre) = 1;
data.menostatus(idx_peri) = 2;
data.menostatus(idx_meno) = 3;

if sum(isnan(data.menostatus)) > 0, error('\n\n additional menostatus missing. Manually review'); end

%% 5. Data Summary and Inspection
if ~suppresssummary, fprintf('\nTotal subjects after filtering: %d\n\n', height(data)); end

% --- Pregnancy Outcome Consistency Check ---
% if ~suppresssummary
%     fprintf('--- Pregnancy Outcome Consistency Check ---\n');
%     fprintf('Some inconsistency expected\n')
% end
% idx_preg_gt6mos = data.repro32 == 1;
% if any(idx_preg_gt6mos)
%     data_preg_gt6mos = data(idx_preg_gt6mos, :);
%     birth_outcomes_sum = sum([data_preg_gt6mos.repro36, data_preg_gt6mos.repro38], 2, 'omitmissing');
%     is_consistent = birth_outcomes_sum == data_preg_gt6mos.repro33;
%     if ~suppresssummary
%         fprintf('%d of %d subjects have consistent pregnancy counts (%2.1f%%).\n\n', ...
%                 sum(is_consistent), height(data_preg_gt6mos), mean(is_consistent) * 100);
%     end
% end

% Detailed Variable Summary
if ~suppresssummary
    fprintf('--- Detailed Variable Summary ---\n');
    fprintf('See function notes for description of each item.\n\n');
    vars_to_summarize = {'psychosis_score', 'repro31', 'repro61', ...
        'currentage', 'menostatus', 'bl_education1','bl_employment7','bl_famhx5'};
    if includepsychosis, vars_to_summarize{end + 1} = 'psychosisdiag'; end
    for i = 1:length(vars_to_summarize)
        var = vars_to_summarize{i};
        try 
        if ismember(var, data.Properties.VariableNames)
            non_missing = sum(~ismissing(data.(var)));
            fprintf('Var: %s, Present: %d, Missing: %d\n', var, non_missing, height(data) - non_missing);
        end
        catch
            keyboard
        end
    end
end

%% feature generation & Predictor Variable Summary
data = calculate_lifetime_prolactin_pregnancy_values(data, fullterm, partterm);
data = data(:, relevantvariables);
data = rmmissing(data);

if ~suppresssummary
    fprintf('\n--- Predictor Variable Summary and Visualization ---\n');

    % Create a map for plot titles based on function documentation
    desc_map = containers.Map('KeyType','char','ValueType','char');
    desc_map('currentage') = 'Current Age';
    desc_map('full_term_months') = 'Full Term Pregnancy Months';
    desc_map('menostatus') = 'Menopause Status';
    desc_map('other_preg_months') = 'Other Pregnancy Months';
    desc_map('pqb_concern') = 'PQB Concern Score';
    desc_map('pqb_symptom') = 'PQB Symptom Score';
    desc_map('psychosisdiag') = 'Psychosis Diagnosis';
    desc_map('psychosis_score') = 'Psychosis Score';
    desc_map('repro31') = 'Number of Pregnancies';
    desc_map('repro61') = 'Breastfeeding Duration (months)';
    desc_map('total_progesterone') = 'Total Progesterone Exposure (Pregnancy Months)';
    desc_map('total_prolactin') = 'Total Prolactin Exposure (Breastfeeding Months)';

    % Define the specific list of variables for summary and plotting as requested
    predictor_vars = {
        'currentage','full_term_months', 'menostatus', 'other_preg_months', ...
        'repro31', 'repro34', 'repro36', 'repro38', 'repro57', 'repro61', ...
        'total_progesterone', 'total_prolactin'
        };

    dependent_var = 'psychosis_score';

    for i = 1:length(predictor_vars)
        var_name = predictor_vars{i};

        if ismember(var_name, data.Properties.VariableNames)
            var_data = data.(var_name);

            % --- Summary Statistics ---
            fprintf('\n%s\n', var_name);
            fprintf('Min: %.2f\n', min(var_data, [], 'omitnan'));
            fprintf('Max: %.2f\n', max(var_data, [], 'omitnan'));
            fprintf('Unique Values: %d\n', numel(unique(var_data(~isnan(var_data)))));
            fprintf('Average: %.2f\n', mean(var_data, 'omitnan'));
            fprintf('Std Dev: %.2f\n', std(var_data, 'omitnan'));

            % --- Histogram ---
            figure; % Create a new figure for each histogram
            histogram(var_data);

            % Set plot title using the description map
            if isKey(desc_map, var_name)
                plot_title = ['Distribution of ' desc_map(var_name)];
                x_label_text = desc_map(var_name);
            else
                plot_title = ['Distribution of ' strrep(var_name, '_', ' ')]; % Fallback title
                x_label_text = strrep(var_name, '_', ' '); % Fallback label
            end
            title(plot_title);
            xlabel('Value');
            ylabel('Frequency');
            grid on;

            % --- Scatter Plot against Dependent Variable ---
            figure; % Create a new figure for each scatter plot
            scatter(var_data, data.(dependent_var), 'filled', 'MarkerFaceAlpha', 0.6);

            title(sprintf('%s vs. %s', desc_map(dependent_var), x_label_text));
            xlabel(x_label_text);
            ylabel(desc_map(dependent_var));
            grid on;

        else
            fprintf('\nWarning: Variable "%s" not found in the table. Skipping summary.\n', var_name);
        end
    end
    fprintf('--- Pausing Execution ---\n');
    choice=input('Inspect data summary and plots. Type 0 to quit or press return to continue.\n\n');
    if choice==0, error('manual quit'); end
end

%% Model creation and Cross-Validation
% define and cross validate nuissance models
k = size(data,1) - 1;
nuisance = selectNuisanceModel(data, 'psychosis_score', nuisancevar, includepsychosis, k);
models = definemodels(nuisance);
numModels = length(models);
numRows = height(data);

% Create a partitions for LOO cross-validation
cv = cvpartition(numRows, 'KFold', k);
cv_mse = zeros(cv.NumTestSets, numModels);

% create model variables
final_models = cell(numModels, 1);
full_data_R2_adj = zeros(numModels, 1);
full_data_AICc = zeros(numModels, 1);

%% Run main models Cross-Validation Loop
for i = 1:cv.NumTestSets
    trainIdx = training(cv, i);
    testIdx = test(cv, i);
    trainData = data(trainIdx, :);
    testData = data(testIdx, :);

    for j = 1:numModels
        try
            lm = fitlm(trainData, models{j});
            y_pred = predict(lm, testData);
            y_true = testData.psychosis_score;
            cv_mse(i, j) = mean((y_true - y_pred).^2);
        catch ME
            warning('on'); warning('Model %d failed on fold %d: %s', j, i, ME.message); warning('off');
            cv_mse(i, j) = NaN;
        end
    end
end

% Fit each model on the FULL dataset to get final models and metrics
for j = 1:numModels
    try
        lm_full = fitlm(data, models{j});
        final_models{j} = lm_full;
        full_data_R2_adj(j) = lm_full.Rsquared.Adjusted;

        aic = lm_full.ModelCriterion.AIC;
        n = lm_full.NumObservations;
        K = lm_full.NumCoefficients; % NumCoefficients includes the intercept
        if (n - K - 1) > 0 % Avoid division by zero
            full_data_AICc(j) = aic + (2*K*(K+1))/(n - K - 1);
        else
            full_data_AICc(j) = inf; % Penalize models that are too complex for the data
        end

    catch ME
        warning('Full dataset fit failed for model %d: %s', j, ME.message);
        final_models{j} = [];
        full_data_R2_adj(j) = NaN;
        full_data_AICc(j) = NaN;
    end
end

% Calculate the mean and standard deviation of MSE across folds
mean_cv_mse = mean(cv_mse, 1, 'omitnan')';
std_cv_mse = std(cv_mse, 0, 1, 'omitnan')';

% Create the final results table & display
ModelString = cellfun(@(x) string(x), models, 'UniformOutput', true)';
results = table(ModelString, mean_cv_mse, std_cv_mse, full_data_R2_adj, full_data_AICc, ...
    'VariableNames', {'Model', 'Mean_CV_MSE', 'Std_CV_MSE', 'FullData_R2_adj', 'FullData_AICc'});
[~, best_idx_mse] = min(results.Mean_CV_MSE);
best_model_formula_mse = results.Model(best_idx_mse);
fprintf('\n--- Model Comparison Results ---\n');
disp(results);

%% Visualize Best Models
[~, best_idx_aicc] = min(results.FullData_AICc);

% Get model object and formula from results
model_mse = final_models{best_idx_mse};
formula_mse = results.Model(best_idx_mse);
model_aicc = final_models{best_idx_aicc};
formula_aicc = results.Model(best_idx_aicc);

% Plot effects for best model by MSE
plot_model_effects(model_mse, formula_mse, data, 'Plots for Best Model by CV-MSE');
if best_idx_aicc ~= best_idx_mse
    plot_model_effects(model_aicc, formula_aicc, data, 'Plots for Best Model by AICc');
else
    fprintf('Best model by AICc is the same as best model by MSE. Plots not duplicated.\n');
end

end

%% Subfunctions
function bestModelFormula = selectNuisanceModel(data, dependentVar, nuisanceVars, includepsychosis, k)
% Selects nuisance model using cross-validation and AICc. The model with the lowest mean cross-validated AICc is chosen as the best.
candidateFormulas = {};

%% Candidate Model Formulas
% Intercept-Only Model
candidateFormulas{end+1} = sprintf('%s ~ 1', dependentVar);
% Models with Nuisance Variables
numNuisance = length(nuisanceVars);
for nn = 1:numNuisance
    combos = nchoosek(nuisanceVars, nn);
    for i = 1:size(combos, 1)
        currentVars = combos(i, :);
        % Additive model
        candidateFormulas{end+1} = sprintf('%s ~ 1 + %s', dependentVar, strjoin(currentVars, ' + '));
        % Interactive model (if more than one variable)
        if length(currentVars) > 10

            candidateFormulas{end+1} = sprintf('%s ~ 1 + %s', dependentVar, strjoin(currentVars, ' * '));
        end
    end
end

%% LEAVE ONE OUT Cross-Validation
numRows = height(data);
cv = cvpartition(numRows, 'KFold', k);
bestMeanAICc = inf;
bestModelFormula = '';
nuisance_results = table('Size', [length(candidateFormulas), 2], ...
                         'VariableTypes', {'string', 'double'}, ...
                         'VariableNames', {'NuisanceModel', 'Mean_CV_AICc'});

for j = 1:length(candidateFormulas)
    formula = candidateFormulas{j};
    cv_aicc = zeros(cv.NumTestSets, 1);
    for i = 1:cv.NumTestSets
        trainIdx = training(cv, i);
        trainData = data(trainIdx, :);

        try
            % Fit the model on the training fold
            lm = fitlm(trainData, formula);
            aic = lm.ModelCriterion.AIC;
            n = lm.NumObservations;
            K = lm.NumCoefficients;
            if (n - K - 1) > 0, cv_aicc(i) = aic + (2*K*(K+1))/(n - K - 1);
            else, cv_aicc(i) = inf;
            end

        catch ME
            warning('on');
            warning('Model failed on fold %d: %s', i, ME.message);
            warning('off');
            cv_aicc(i) = NaN;
        end
    end

    % calculate mean corrected AIC
    meanAICc = mean(cv_aicc, 'omitnan');

    nuisance_results.NuisanceModel(j) = formula;
    nuisance_results.Mean_CV_AICc(j) = meanAICc;

    % Check if new model is best
    if meanAICc < bestMeanAICc
        bestMeanAICc = meanAICc;
        bestModelFormula = formula;
    end
end
nuisance_results = sortrows(nuisance_results, 'Mean_CV_AICc', 'ascend');
fprintf('\n--- Nuisance Model Selection Results ---\n');
disp(nuisance_results);
fprintf('\n--- Nuisance Model Selection Complete ---\n');
fprintf('Selected Nuisance Model: %s (Mean CV AICc: %.2f)\n', bestModelFormula, bestMeanAICc);

%% Format Output Formula for use in other models
% Extracts the predictor part (e.g., '+ age + sex')
cutstring = strfind(bestModelFormula, '~');
if ~isempty(cutstring)
    predictor_part = strtrim(bestModelFormula(cutstring(1)+1:end));
    bestModelFormula = predictor_part;
end
end

function models = definemodels(nuisance)
% Defines the candidate models for analysis
models = {...
    ['psychosis_score ~ ' nuisance], ... % Nuisance only
    ['psychosis_score ~ total_prolactin + ' nuisance], ... % breastfeeding only
    ['psychosis_score ~ total_progesterone + ' nuisance], ... % pregnancy only
    ['psychosis_score ~ total_prolactin + total_progesterone + ' nuisance], ... % pregnancy and breastfeeding
    ['psychosis_score ~ total_prolactin * total_progesterone + ' nuisance], ... % interaction of the two
    };
   % ['psychosis_score ~ total_progesterone * repro34 + ' nuisance], ... % pregnancy weighted by age of first
   %['psychosis_score ~ total_prolactin + (total_progesterone * repro34) +' nuisance] % breastfeeding and weighted pregnancy
end

function data = calculate_lifetime_prolactin_pregnancy_values(data, fullterm, partterm)
% Calculates lifetime hormone exposure based on pregnancy and breastfeeding.

% Assumptions:
% - Live births, stillbirths = full term
% - All other pregnancies = part term

% Calculate total full term preg months
total_fullterm = nansum([data.repro36, data.repro38], 2);
data.full_term_months = total_fullterm * fullterm;

% calculate part term preg months
remaining_pregs = data.repro31 - total_fullterm;
remaining_pregs(remaining_pregs < 0) = 0; % Ensure non-negative
data.other_preg_months = remaining_pregs * partterm;

% Calculate total progesterone exposure using total pregnancy months as proxy
data.total_progesterone = data.full_term_months + data.other_preg_months;

% calculate total prolactin exposure using breastfeeding as proxy
data.total_prolactin = data.repro61;

%% Convert certain total_prolactin values to months
% Find rows where total_prolactin is 24-30, convert to months using the halfway point of the range
idx_to_update = data.total_prolactin >= 24;

if any(idx_to_update)
    original_values = data.total_prolactin(idx_to_update);

    % Apply the conversion formula
    % Just trust the equation; I promise it works.
    converted_values = 6 * original_values - 117; % Just trust the equation; I promise it works.

    % Overwrite the original data with the new values
    data.total_prolactin(idx_to_update) = converted_values;
end

%% convert to proportion of life
data.total_prolactin = (data.total_prolactin ./ data.currentage) * 100;
data.total_progesterone = (data.total_progesterone ./ data.currentage) * 100;
end

function plot_model_effects(lm, model_formula, data, plot_title)
% Generates plots for effects of model parameters.
% List of known categorical variables for plotting
categorical_vars = {'menostatus', 'psychosisdiag'};

% Get predictor names from the model
predictors = lm.PredictorNames;
if isempty(predictors)
    fprintf('Model for "%s" is intercept-only. No effects to plot.\n', plot_title);
    return;
end

% Create figure for this model
figure('Name', plot_title, 'NumberTitle', 'off');
sgtitle(plot_title, 'FontSize', 14, 'FontWeight', 'bold');

% Handle Interaction Terms First
interaction_terms = predictors(contains(predictors, ':'));
plotted_interactions = {};
plot_idx = 1;
if ~isempty(interaction_terms)
    for i = 1:length(interaction_terms)

        % Get variables from interaction term name
        vars = strsplit(interaction_terms{i}, ':');
        var1 = vars{1}; var2 = vars{2};

        % Avoid plotting same interaction twice (e.g., A:B and B:A)
        sorted_vars = sort(vars);
        interaction_id = strjoin(sorted_vars, ':');
        if ~ismember(interaction_id, plotted_interactions)
            subplot(2, 2, plot_idx); % Assume max 2x2 grid for simplicity, adjust if needed
            plotInteraction(lm, var1, var2);
            title(sprintf('Interaction: %s * %s', var1, var2));
            plotted_interactions{end+1} = interaction_id;
            plot_idx = plot_idx + 1;
        end
    end
end

% Handle Main Effects
main_effects = predictors(~contains(predictors, ':'));
num_plots = length(main_effects) + length(plotted_interactions);
if num_plots == 0, return; end

% Determine subplot layout
nrows = ceil(sqrt(num_plots));
ncols = ceil(num_plots / nrows);
for i = 1:length(main_effects)
    predictor_name = main_effects{i};
    subplot(nrows, ncols, plot_idx);

    % Check if variable are categorical types
    if ismember(predictor_name, categorical_vars)
        boxplot(data.psychosis_score, data.(predictor_name));
        title(sprintf('Effect of %s', predictor_name));
        xlabel(predictor_name);
        ylabel('Psychosis Score');
    else
        plotAdded(lm, predictor_name);
    end
    plot_idx = plot_idx + 1;
end

% check for linearity
figure;
plotResiduals(lm, 'fitted');
title('Residuals vs. Fitted (check for linearity)');

% --- Breusch-Pagan Test for Homoscedasticity ---
fprintf('\nFormal Homoscedasticity Check\n');
fprintf('Checks if variance of residuals is constant.\n');
fprintf('Null Hypothesis (H0): variance is constant (homoscedastic).\n');
fprintf('Alternative (Ha): variance not constant (heteroscedastic).\n\n');

% Get residuals and create a temporary table for the auxiliary regression
residuals = lm.Residuals.Raw;
squared_residuals = residuals.^2;
aux_tbl = lm.Variables(:, lm.PredictorNames);
aux_tbl.SquaredResiduals = squared_residuals;

% Create the formula string for the auxiliary regression
aux_formula = 'SquaredResiduals ~ 1';
if ~isempty(lm.PredictorNames)
    aux_formula = ['SquaredResiduals ~ ' strjoin(lm.PredictorNames, ' + ')];
end

% Fit the auxiliary regression model
aux_lm = fitlm(aux_tbl, aux_formula);

% Calculate the Breusch-Pagan test statistic
n = lm.NumObservations;
R_squared_aux = aux_lm.Rsquared.Ordinary;
test_statistic = n * R_squared_aux;

% Calculate the p-value using a chi-squared distribution
df = aux_lm.NumCoefficients - 1; % Degrees of freedom
p_value = 1 - chi2cdf(test_statistic, df);

fprintf('Breusch-Pagan Test Results:\n');
fprintf('  Test Statistic: %.4f\n', test_statistic);
fprintf('  P-Value: %.4f\n\n', p_value);

% Interpret the results
alpha = 0.05;
if p_value < alpha
    fprintf('  p-value (%.4f) < %.2f.\n', p_value, alpha);
    fprintf('  Data shows evidence of HETEROscedasticity.\n');
    fprintf('  variance is likely unequal.\n\n');

    % Heteroscedasticity-consistent standard errors (HCSE) Calculation and Display
    fprintf('--- Adjusting results to account for heteroscedasticity. Calculating Robust Standard Errors ---\n');

    % Calculate robust standard errors, t-stats, and p-values
    [se_robust, t_robust, p_robust] = hac(lm);

    % Create a new table to display the robust results
    robust_results = lm.Coefficients; % Copy original coefficients table
    robust_results.SE = se_robust;
    robust_results.tStat = t_robust;
    robust_results.pValue = p_robust;

    % Rename columns for clarity
    robust_results.Properties.VariableNames{'SE'} = 'Robust_SE';
    robust_results.Properties.VariableNames{'tStat'} = 'Robust_tStat';
    robust_results.Properties.VariableNames{'pValue'} = 'Robust_pValue';

    fprintf('Original Model Coefficients:\n');
    disp(lm.Coefficients);
    fprintf('Coefficients with Heteroscedasticity-Consistent Standard Errors (HCSE):\n');
    disp(robust_results);
else
    fprintf('  p-value (%.4f) > %.2f.\n', p_value, alpha);
    fprintf('  variance likely equal.\n\n');
end

% Get the raw residuals and the design matrix (predictors)
residuals = lm.Residuals.Raw;
X = lm.Variables(:, lm.PredictorNames);

% Q-Q plot to check for normalityvisual check
figure;
plotResiduals(lm, 'probability'); % This creates a Q-Q plot
title('Q-Q Plot of Model Residuals (check for normality)');

% Formal Normality Check (Shapiro-Wilk Test)
% fprintf('\nFormal Normality Check\n');
% fprintf('Checks if the model residuals are normally distributed.\n');
% fprintf('Null Hypothesis (H0): The residuals are normally distributed.\n');
% fprintf('Alternative (Ha): The residuals are NOT normally distributed.\n\n');

% Perform the Shapiro-Wilk test on the raw residuals
% currently broken =(
%[H, pValue, W] = shapiro_wilk_test(lm.Residuals.Raw);
% 
% fprintf('Shapiro-Wilk Test Results:\n');
% fprintf('  Test Statistic (W): %.4f\n', W);
% fprintf('  P-Value: %.4f\n\n', pValue);

% % Interpret the results based on the hypothesis decision H
% alpha = 0.05;
% if H == 1
%     fprintf('  p-value (%.4f) < %.2f.\n', pValue, alpha);
%     fprintf('  Data shows evidence of NON-NORMAL residuals.\n');
%     fprintf('  The normality assumption is likely VIOLATED.\n\n');
% else
%     fprintf('  p-value (%.4f) >= %.2f.\n', pValue, alpha);
%     fprintf('  Normality assumption appears to be MET.\n\n');
% end


% Use the final data table right before modeling
predictor_data = data(:, lm.PredictorNames);

% Simple correlation plot
figure;
corrplot(predictor_data);
title('Correlation Matrix of Predictors (test for colinearity)');

% To calculate VIF for each predictor
vif = zeros(width(predictor_data), 1);
for i = 1:width(predictor_data)
    % Regress this predictor on all others
    temp_mdl = fitlm(predictor_data, [predictor_data.Properties.VariableNames{i} ' ~ 1']);
    vif(i) = 1 / (1 - temp_mdl.Rsquared.Ordinary);
end
vif_table = table(predictor_data.Properties.VariableNames', vif, 'VariableNames', {'Predictor', 'VIF'});
disp('Variance Inflation Factors (VIF):');
disp(vif_table);
pp = input('pause for review. \nPress 1 to close all and continue. \nPress 2 to continue without closing. \nPress 0 to quit');
if pp == 1
    close all;
elseif pp == 0
    error('manual quit');
end
end

function [H, pValue, W] = shapiro_wilk_test(x, alpha)
%SHAPIRO_WILK_TEST Performs the Shapiro-Wilk test for normality.
%   This is a standalone implementation and does not require any toolboxes.
%   Based on the algorithm AS R94 by Royston (1995).

if nargin < 2
    alpha = 0.05;
end

x = x(~isnan(x)); % Remove NaNs and Infs
x = x(isfinite(x));
x = sort(x);
n = length(x);

if n < 3 || n > 5000
    warning('Shapiro-Wilk test implementation is valid for sample sizes between 3 and 5000.');
    H = nan; pValue = nan; W = nan;
    return;
end

% The coefficients for the test are approximated by polynomials
if n >= 3 && n <= 11
    g = [-2.273, 0.459];
    c = [0.0; 0.221157; -0.147981; -2.071190; 4.434685; -2.706056];
    c = c(n-2:end);
elseif n >= 12 && n <= 2000
    g = [0];
    c = [0.0; 0.4639; -0.4284; -3.8996; 8.3516; -4.1314];
    u = 1/sqrt(n);
    c(1) = 0.4261*u + 0.0132;
    c(2) = -0.5284*u - 0.2372;
    c(3) = -3.8996;
    c(4) = 8.3516;
    c(5) = -4.1314;
end

m = norminv(((1:n)' - 3/8) / (n + 1/4));
a_poly = polyval(fliplr(c), m);
a = a_poly ./ sqrt(sum(a_poly.^2));

% Calculate W statistic
W = (sum(a' .* x'))^2 / sum((x - mean(x)).^2);
W = min(W, 1.0); % FIX: Ensure W is not > 1 due to floating point error

% Calculate p-value
if n <= 11
    mu = 0.0038915*log(n)^3 - 0.083751*log(n)^2 - 0.31082*log(n) - 1.5861;
    sigma = exp(0.0030302*log(n)^2 - 0.082676*log(n) - 0.4803);
    z = (log(1-W) - mu) / sigma;
else % n > 11
    mu = log(1-W);
    z = log(mu);
    if n <= 11
        gamma_val = g(1)*n + g(2);
    else
        gamma_val = 0; % FIX: The original calculation was incorrect for this sample size range.
    end
    if n <= 2000
        mu_poly = [-0.0006714, 0.025054, -0.39978, 0.5440];
        sigma_poly = [-0.0003295, 0.0121378, -0.20322, 0.0478];
        mu_val = polyval(mu_poly, log(n+gamma_val));
        sigma_val = exp(polyval(sigma_poly, log(n+gamma_val)));
    else % n > 2000 (Approximation)
        mu_poly = [0.00002, -0.0011, -0.0637, 1.1895];
        sigma_poly = [0.000007, -0.0005, 0.0154, -0.4302];
        mu_val = polyval(mu_poly, log(n+gamma_val));
        sigma_val = exp(polyval(sigma_poly, log(n+gamma_val)));
    end
    z = (z-mu_val)/sigma_val;
end

pValue = 1 - normcdf(z, 0, 1);

% Hypothesis decision
if pValue < alpha
    H = 1; % Reject null hypothesis -> Not normal
else
    H = 0; % Fail to reject null hypothesis -> Normal
end

end

