function [summary_table, kit_table] = summarizeOrderedKits(cvfn)
% Function used for tracking the distribution and resulting of blood sample collection kits
%           Calculates total number of kits, analyzes progress, and plots trends.
%
%   Requires a data frame containing the redcap instruments eligibility, hormone_kits,
%       evaluation_start, and symptom_diary.
%
%   1.  Calculates summary statistics for ordered kits (return rate, validity rate)
%   2.  Calculates the number of kits received vs. not yet received by lab
%   3.  Calculates n of each evaluation type
%   4.  Calculates each subjects progress
%   5.  Calculates order and returned statistics for each year
%   6.  Generates a plot of CUMULATIVE kit orders, received, and dispatched kits over time.
%   7.  Generates a plot of the 3-MONTH ROLLING AVERAGE of kits.
%   8.  Generates a detailed projection and summary table for kits.
%
%   Input:
%       fulldata - A MATLAB table. This table will be modified in place.
%
%   Output:
%       summary_table - A table containing the calculated overall statistics.
%           - kit_summary.returnedrate = proportion of total ordered kits
%           documented on redcap that were returned to lab
%           - kit_summary.validrate = proportion of returned kits that are
%           valid
%       kit_table - A table containing detailed kit projections and fiscal year stats.

%% Create environment
warning('on');

% Load data
opts = detectImportOptions(cvfn);
fvn = contains(opts.VariableNames, {'date', 'lgc_cycle_d'});
opts = setvartype(opts, opts.VariableNames(fvn), 'datetime');
fulldata = readtable(cvfn, opts);

warning('on');

%define variables
fvn = fulldata.Properties.VariableNames;
kitnames = {'fh'; 'prg'};
collection_day = {[1, 8, 15, 22]; [6, 20]};
collection_order = {[1, 3, 4, 6]; [2, 5]};
skn = size(kitnames, 1);
numberVis = 5;
numberKits = 6;
goal_subjects = 294;

% define numbers needed for an evaluation to be considered complete
completekits = 4;
completediary = 25;

% output variables
summary_table = {};
kit_summary = {};

%% Calculate Kit summary
fprintf('\nCalculating kit summary...\n');

% Confirm data formatting
if ~ismember('redcap_repeat_instrument', fvn), error('redcap_repeat_instrument column required.');
end

% relevant rows
HormoneKitRow = contains(fulldata.redcap_repeat_instrument, "hormone_kits");
hormone_data = fulldata(HormoneKitRow, :);
if isempty(hormone_data), error('hormone_kits column required.'); end

% relevant columns
regex_pattern = '^ordered_(fh|prg)[1-8]___1$';
is_target_column = ~cellfun('isempty', regexp(fvn, regex_pattern));
target_columns = fvn(is_target_column);

try
    %% Calculate kits ordered
    kit_grandtotal = 0;
    kitoutput_row_names = {};

    % # per kit type
    all_totals_vec = zeros(skn + 1, 1);
    for kt = 1:skn
        cktn = kitnames{kt};
        kt_columns = target_columns(startsWith(target_columns, ['ordered_' cktn]));
        total_kt = sum(hormone_data{:, kt_columns}, 'all', 'omitnan');
        kit_grandtotal = kit_grandtotal + total_kt;
        newrn = [cktn '_ordered'];
        kitoutput_row_names{end+1} = newrn;
        all_totals_vec(kt) = total_kt;
    end

    % add to summary table
    all_totals_vec(end) = kit_grandtotal;
    orderedgt = kit_grandtotal;
    kitoutput_row_names{end+1} = 'Total_ordered';
    kit_summary.orders = array2table(all_totals_vec, 'VariableNames', {'value'}, 'RowNames', kitoutput_row_names);

    %% calculate number returned to lab
    kit_grandtotal = 0;
    kitoutput_row_names = {};
    all_totals_vec = zeros(skn + 1, 1);
    for kt = 1:skn
        cktn = kitnames{kt};

        % relevant columns
        pattern = sprintf('^received_dated.*(%s)$', strjoin(cellstr(num2str(collection_day{kt}(:), '%02d')), '|'));
        is_match = ~cellfun('isempty', regexp(fvn, pattern));
        received_date_columns = fvn(is_match);
        if isempty(received_date_columns), error('No columns with "received_dated" found.'); end

        received_count = sum(~ismissing(hormone_data{:, received_date_columns}), 'all');
        kit_grandtotal = kit_grandtotal + received_count;
        newrn = [cktn '_returned'];
        kitoutput_row_names{end+1} = newrn;
        all_totals_vec(kt) = received_count;
    end

    % add to summary table
    all_totals_vec(end) = kit_grandtotal;
    returnedgt = kit_grandtotal;
    kitoutput_row_names{end+1} = 'Total_returned';
    kit_summary.returned = array2table(all_totals_vec, 'VariableNames', {'value'}, 'RowNames', kitoutput_row_names);

    % calculate return rate
    kit_summary.returnrate = returnedgt / orderedgt * 100;

    %% calculate validity
    kit_grandtotal = 0;
    kitoutput_row_names = {};
    all_totals_vec = zeros(skn + 1, 1);
    for kt = 1:skn
        cktn = kitnames{kt};

        % find relevant columns
        pattern = sprintf('^invalidd.*(%s)$', strjoin(cellstr(num2str(collection_day{kt}(:), '%02d')), '|'));
        is_match = ~cellfun('isempty', regexp(fvn, pattern));
        valid_columns = fvn(is_match);
        if isempty(valid_columns), error('No columns with "valid_columns" found.'); end

        data_subset = hormone_data{:, valid_columns};
        validcount = sum(data_subset == 0, 'all');
        newrn = [cktn '_valid'];
        kitoutput_row_names{end+1} = newrn;
        all_totals_vec(kt) = validcount;
        kit_grandtotal = kit_grandtotal + validcount;
    end

    % add to summary table
    all_totals_vec(end) = kit_grandtotal;
    validgt = kit_grandtotal;
    kitoutput_row_names{end+1} = 'total_valid';
    kit_summary.validity = array2table(all_totals_vec, 'VariableNames', {'value'}, 'RowNames', kitoutput_row_names);

    %% calculate validity rate of returned
    kit_summary.validrate = validgt / returnedgt * 100;

    %% add to output
    summary_table.kit_summary = kit_summary;
catch ME
    warning('Could not generate the kit order summary. Reason: %s', ME.message);
end

%% Calculate Evaluation Progress
% Filter to instruments relevant to analysis
relevant_instruments = {'hormone_kits', 'evaluation_start', 'symptom_diary'};
is_relevant_instrument = ismember(fulldata.redcap_repeat_instrument, relevant_instruments);
data = fulldata(is_relevant_instrument, :);

% Initialize output structure and subjects table
evaluation_status = struct();
evaluation_status.evaluations = table('Size', [0, 6], ...
    'VariableTypes', {'string', 'double', 'categorical', 'datetime', 'double', 'double'}, ...
    'VariableNames', {'record_id', 'evaluation', 'completion_status', 'start_date', 'hormone_kits_returned', 'symptom_diary_days'});

% Get list of unique record IDs with at least one evaluation start date
all_record_ids = unique(data.record_id);
diary_cols = startsWith(fvn, 'depressed_');
received_cols = startsWith(fvn, 'received_dated');

%% generate summary of evaluation status
% Cycle through subjects and evaluations
for i = 1:numel(all_record_ids)
    crid = all_record_ids(i);
    subject_data = data(data.record_id == crid, :);
    eval_start_rows = subject_data(strcmp(subject_data.redcap_repeat_instrument, 'evaluation_start'), :);
    if isempty(eval_start_rows); continue; end

    for j = 1:height(eval_start_rows)

        % relevant variables
        ceval = eval_start_rows(j, :);
        returnedkits= 0; % Default to 0
        symptom_diary_days = 0; % Default to 0
        eval_start_date = ceval.lgc_cycle_d1;
        eval_iteration = ceval.redcap_repeat_instance;

        % Find hormone kit instances for the current subject
        hormone_kit_rows = subject_data(strcmp(subject_data.redcap_repeat_instrument, 'hormone_kits'), :);

        if ~isempty(hormone_kit_rows)
            % Define the two-week window after the evaluation start date
            date_limit = eval_start_date + days(14);

            % Find kit iterations where return date ('received_dated01') is within the window
            is_within_window = hormone_kit_rows.received_dated01 >= eval_start_date & ...
                hormone_kit_rows.received_dated01 <= date_limit;

            matching_kit_row = hormone_kit_rows(is_within_window, :);

            % calculate number of returned kits
            if height(matching_kit_row) == 1
                returned_dates = matching_kit_row{1, received_cols};
                returnedkits = sum(~isnat(returned_dates));
            elseif height(matching_kit_row) > 1
                error('\ntoo many matching hormone kit instances. Uncomment keyboard to pause and explore dataset');
                %keyboard
            end
        end

        % Find symptom diary with same iteration number as evaluation
        symptom_diary_row = subject_data(strcmp(subject_data.redcap_repeat_instrument, 'symptom_diary') & ...
            subject_data.redcap_repeat_instance == eval_iteration, :);

        % calculate the number of completed days
        if height(symptom_diary_row) == 1
            symptom_data = symptom_diary_row{1, diary_cols};
            symptom_diary_days = sum(~isnan(symptom_data), 'omitnan');
        end

        % Determine Completion Status
        if returnedkits >= completekits && symptom_diary_days >= completediary
            completion_status = 'Complete';
        else
            completion_status = 'Incomplete';
        end

        % Store Results
        new_row = {crid, eval_iteration, completion_status, ...
            eval_start_date, returnedkits, symptom_diary_days};
        evaluation_status.evaluations = [evaluation_status.evaluations; new_row];
    end
end
evaluation_status.evaluations.record_id = str2double(evaluation_status.evaluations.record_id);

%% create summary each subject's evaluation progress
subjects = table();
evaluations = evaluation_status.evaluations;
unique_ids = unique(evaluations.record_id);

for i = 1:length(unique_ids)
    current_id = unique_ids(i);

    % find max evaluation number and calculate proportions of evaluations complete
    subject_rows = evaluations(evaluations.record_id == current_id, :);
    latest_evaluation = max(subject_rows.evaluation);
    num_complete = sum(subject_rows.completion_status == "Complete");
    new_row = table(current_id, latest_evaluation, num_complete, ...
        'VariableNames', {'record_id', 'latest_evaluation', 'number_complete'});
    subjects = [subjects; new_row];
end
evaluation_status.subjects = subjects;

% summarize group evaluation
Progress = groupsummary(evaluation_status.evaluations, "evaluation", @(x) sum(x == "Complete"), "completion_status");
Progress.Properties.VariableNames{'fun1_completion_status'} = 'number complete';
evaluation_status.progress = Progress;
summary_table.evaluation_stats = evaluation_status;

%% Kit ordering Statistics
fprintf('\nCalculating statistics for each year in the data...\n');
try
    % ordered information
    order_cols = fvn(contains(fvn, 'orderdate_'));
    order_dates_all = [];
    for i = 1:length(order_cols)
        col_data = fulldata.(order_cols{i});
        if isdatetime(col_data), order_dates_all = [order_dates_all; col_data(:)]; end
    end

    %received at lab information
    received_cols = fvn(contains(fvn, 'received_dated'));
    received_dates_all = [];
    for i = 1:length(received_cols)
        col_data = fulldata.(received_cols{i});
        if isdatetime(col_data), received_dates_all = [received_dates_all; col_data(:)]; end
    end

    % get evaluation information
    all_dates = [order_dates_all; received_dates_all];
    for ee=1:numberVis
        eval_rows_idx = strcmp(fulldata.redcap_repeat_instrument, 'evaluation_start') & fulldata.redcap_repeat_instance == ee & ~ismissing(fulldata.lgc_cycle_d1);
        eval_started_rows = fulldata(eval_rows_idx, :);
        evaldates.(['eval' num2str(ee)]) =  eval_started_rows.lgc_cycle_d1;
        all_dates = [all_dates; evaldates.(['eval' num2str(ee)])];
    end

    all_dates = all_dates(~isnat(all_dates)); % Clean NaTs
    if isempty(all_dates), error('No valid dates found.'); end

    unique_years = unique(year(all_dates));

    % Loop through years and calculate stats
    for i = 1:length(unique_years)
        cyear = unique_years(i);
        yearstart = datetime(cyear, 1, 1);
        yearend = datetime(cyear, 12, 31, 23, 59, 59);

        % Kits Ordered & received
        n_ordered_ytd = sum(order_dates_all >= yearstart & order_dates_all <= yearend, 'omitnan');
        n_received_ytd = sum(received_dates_all >= yearstart & received_dates_all <= yearend, 'omitnan');
        year_rows = {
            sprintf('Kits_Ordered_%d', cyear);
            sprintf('Kits_Received_%d', cyear); };

        % Evals started
        year_stats = [n_ordered_ytd; n_received_ytd];
        for ee=1:numberVis
            ceval = evaldates.(['eval' num2str(ee)]);
            eval_ytd = sum(ceval >= yearstart & ceval <= yearend, 'omitnan');
            year_stats = [year_stats; eval_ytd];
            year_rows{end+1}=sprintf('Eval%ds_Started_%d',ee,cyear);
        end

        year_table = array2table(year_stats, 'VariableNames', {'value'}, 'RowNames', year_rows);
        summary_table.year_table = year_table;
    end
catch ME
    warning('Could not calculate per-year statistics. Reason: %s', ME.message);
end

%% Plot Kit dets Over Time
try
    % Ordered data
    all_order_dates = [];
    order_date_columns = fvn(contains(fvn, 'orderdate_'));

    for i = 1:length(order_date_columns)
        column_data = fulldata.(order_date_columns{i});
        if isdatetime(column_data)
            all_order_dates = [all_order_dates; column_data(~isnat(column_data))];
        end
    end
    if isempty(all_order_dates), warning('No valid order dates found to plot.'); end

    [monthly_counts_orders, unique_months_orders] = groupcounts(dateshift(sort(all_order_dates), 'start', 'month'));
    cumulative_monthly_counts_orders = cumsum(monthly_counts_orders);

    % Received at lab data
    all_received_dates = [];
    received_date_cols = fvn(contains(fvn, 'received_dated'));

    for i = 1:length(received_date_cols)
        column_data = fulldata.(received_date_cols{i});
        if isdatetime(column_data), all_received_dates = [all_received_dates; column_data(~isnat(column_data))]; end
    end
    if isempty(all_received_dates), warning('No valid received dates found to plot.'); end

    [monthly_counts_received, unique_months_received] = groupcounts(dateshift(sort(all_received_dates), 'start', 'month'));
    cumulative_monthly_counts_received = cumsum(monthly_counts_received);

    % Dispatched data
    total_dispatch_counts = [];
    unique_months_dispatch = [];
    dispatch_data = calculatelgcdispatch();

    if ~isempty(dispatch_data) && height(dispatch_data) > 0
        dispatch_data.Month = dateshift(dispatch_data.DispatchDate, 'start', 'month');
        [G, monthGroup, kitTypeGroup] = findgroups(dispatch_data.Month, dispatch_data.KitType);
        getValueAtMaxDate = @(dates, vals) vals(find(dates == max(dates), 1, 'last'));
        last_vals_per_group = splitapply(getValueAtMaxDate, dispatch_data.DispatchDate, dispatch_data.TotalKitsDispatched, G);
        temp_summary = table(monthGroup, kitTypeGroup, last_vals_per_group, 'VariableNames', {'Month', 'KitType', 'LastValue'});
        monthly_summary = groupsummary(temp_summary, 'Month', 'sum', 'LastValue');
        monthly_summary = sortrows(monthly_summary, 'Month');
        unique_months_dispatch = monthly_summary.Month;
        total_dispatch_counts = monthly_summary.sum_LastValue;
    else
        warning('No dispatch data found. Skipping dispatch plot.');
    end

    % Plotting
    figure; hold on;
    plot(unique_months_orders, cumulative_monthly_counts_orders, '-o', 'LineWidth', 2, 'DisplayName', 'Cumulative Orders');
    plot(unique_months_received, cumulative_monthly_counts_received, '-s', 'LineWidth', 2, 'DisplayName', 'Cumulative Returned');
    plot(unique_months_dispatch, total_dispatch_counts, '-d', 'LineWidth', 2, 'DisplayName', 'Cumulative Dispatched', 'Color', [0.4660 0.6740 0.1880]);
    hold off;

    % Formatting
    title('Cumulative Dispatched, Ordered, and Returned Kits Over Time');
    xlabel('Month'); ylabel('Cumulative Number of Kits');
    legend('show', 'Location', 'northwest');
    grid on; box on;
    xtickformat('MMM-yyyy');
    ax = gca; ax.XAxis.TickLabelRotation = 45;
    fprintf('Combined plot generated.\n');
catch ME
    warning('Could not generate the cumulative plot. Reason: %s', ME.message);
end

%% Plot Rolling Average of Ordered and Received Kits
try
    order_cols_for_avg = fvn(contains(fvn, 'orderdate_'));
    received_cols_for_avg = fvn(contains(fvn, 'received_dated'));
    if isempty(order_cols_for_avg) || isempty(received_cols_for_avg), error('Order or Received date columns not found.'); end

    % get order data
    order_dates_avg = [];
    for i = 1:length(order_cols_for_avg)
        col_data = fulldata.(order_cols_for_avg{i});
        if isdatetime(col_data), order_dates_avg = [order_dates_avg; col_data(:)]; end
    end
    [order_counts, order_months] = groupcounts(dateshift(order_dates_avg, 'start', 'month'));

    % get received data
    received_dates_avg = [];
    for i = 1:length(received_cols_for_avg)
        col_data = fulldata.(received_cols_for_avg{i});
        if isdatetime(col_data), received_dates_avg = [received_dates_avg; col_data(:)]; end
    end
    [received_counts, received_months] = groupcounts(dateshift(received_dates_avg, 'start', 'month'));

    % Determine labels based on data
    first_date = min([order_dates_avg; received_dates_avg]);
    last_date = max([order_dates_avg; received_dates_avg]);
    if isnat(first_date) || isnat(last_date), error('Cannot create a valid date timeline.'); end

    timeline_months = dateshift(first_date, 'start', 'month'):calmonths(1) : dateshift(last_date, 'start', 'month');
    timeline_months = timeline_months';

    % find orders per month
    [lia_o, locb_o] = ismember(order_months, timeline_months);
    orders_on_timeline = zeros(size(timeline_months));
    orders_on_timeline(locb_o(lia_o)) = order_counts(lia_o);

    % find received per month
    [lia_r, locb_r] = ismember(received_months, timeline_months);
    received_on_timeline = zeros(size(timeline_months));
    received_on_timeline(locb_r(lia_r)) = received_counts(lia_r);

    % calc rolling average
    rolling_avg_orders = movmean(orders_on_timeline, 3);
    rolling_avg_received = movmean(received_on_timeline, 3);

    % plot
    figure; hold on;
    plot(timeline_months, rolling_avg_orders, 'b-o', 'LineWidth', 2, 'DisplayName', 'Ordered Kits');
    plot(timeline_months, rolling_avg_received, 'r-s', 'LineWidth', 2, 'DisplayName', 'Received Kits');
    hold off;

    % format
    title('3-Month Rolling Average of Kits Ordered and Received');
    xlabel('Month'); ylabel('Average Number of Kits per Month');
    legend('show', 'Location', 'northwest');
    grid on; box on;
    xtickformat('MMM-yyyy');
    ax = gca;
    ax.XAxis.TickLabelRotation = 45;
catch ME
    warning('Could not generate the rolling average plot. Reason: %s', ME.message);
end

%% Generate detailed Kit Table
try
    kit_table = assessRemainingEvaluations(fulldata);
catch ME
    warning('Could not generate the kit projection table. Reason: %s', ME.message);
end

end

% --------------- SUB-FUNCTIONS ---------------
function kit_table = assessRemainingEvaluations(fulldata, numberVis, numberKits, goal_subjects)
% Calculates remaining evaluations and detailed kit stats.

% Kit Configuration
kit_configs = struct();
kit_configs.fh.name_long = 'FMH';
kit_configs.fh.ordered_cols = arrayfun(@(n) sprintf('ordered_fh%d___1', n), [1, 2, 3, 4, 6, 7], 'UniformOutput', false);
kit_configs.fh.received_cols = arrayfun(@(n) sprintf('received_fh%d___1', n), [1, 2, 3, 4, 6, 7], 'UniformOutput', false);
kit_configs.fh.order_date_cols = arrayfun(@(n) sprintf('orderdate_%d', n), [1, 3, 4, 6, 9, 10], 'UniformOutput', false);
kit_configs.fh.received_date_cols = arrayfun(@(n) sprintf('received_dated%02d', n), [1, 8, 15, 22], 'UniformOutput', false);

kit_configs.prg.name_long = 'PRG';
kit_configs.prg.ordered_cols = arrayfun(@(n) sprintf('ordered_prg%d___1', n), [1, 2, 4, 5], 'UniformOutput', false);
kit_configs.prg.received_cols = arrayfun(@(n) sprintf('received_prg%d___1', n), [1, 2, 4, 5], 'UniformOutput', false);
kit_configs.prg.order_date_cols = arrayfun(@(n) sprintf('orderdate_%d', n), [2, 5, 11, 12], 'UniformOutput', false);
kit_configs.prg.received_date_cols = arrayfun(@(n) sprintf('received_dated%02d', n), [6, 20], 'UniformOutput', false);
kit_types = fieldnames(kit_configs);

% Input Validation
required_cols = {'record_id', 'elig_determination___1', 'elig_determination___8', ...
    'elig_determination___4', 'elig_determination___5', ...
    'aim_2_consent_v3_complete', 'lgc_cycle_d1'};
fvn = fulldata.Properties.VariableNames;

if ~all(ismember(required_cols, fvn))
    missing_cols = setdiff(required_cols, fvn);
    error('Input table is missing required columns: %s', strjoin(missing_cols, ', '));
end

% find engaged subjects
is_engaged_idx = (fulldata.elig_determination___1 ~= 1 & ...
    fulldata.elig_determination___8 ~= 1 & ...
    (fulldata.elig_determination___4 == 1 | fulldata.elig_determination___5 == 1) & ...
    fulldata.aim_2_consent_v3_complete > 0);

% calculate relevant evaluation valus
eligible_subject_ids = unique(fulldata.record_id(is_engaged_idx));
total_aim_2_engaged = numel(eligible_subject_ids);
total_visits_needed_for_enrolled = total_aim_2_engaged * numberVis;

eligible_rows_idx = ismember(fulldata.record_id, eligible_subject_ids);
eligible_data = fulldata(eligible_rows_idx, :);
total_visits_completed_for_enrolled = sum(~isnat(eligible_data.lgc_cycle_d1));
remaining_visits_for_enrolled = total_visits_needed_for_enrolled - total_visits_completed_for_enrolled;
total_visits_needed_for_goal = goal_subjects * numberVis;
visits_remaining_to_reach_goal = total_visits_needed_for_goal - total_visits_completed_for_enrolled;

% Assemble Base Output Table
base_vars = {'number_visits', 'number_kits_per_visit', 'GoalSubjects', ...
    'TotalVisitsNeededForGoal', 'VisitsRemainingToReachGoal', ...
    'TotalAim2EngagedSubjects', 'TotalVisitsCompletedForEnrolled', ...
    'RemainingVisitsForEnrolled'};
kit_table = table(numberVis, numberKits, goal_subjects, ...
    total_visits_needed_for_goal, visits_remaining_to_reach_goal, ...
    total_aim_2_engaged, total_visits_completed_for_enrolled, ...
    remaining_visits_for_enrolled, 'VariableNames', base_vars);

% Detailed Kit Calculations
getFiscalYear = @(dt) year(dt + calmonths(3));
for i = 1:numel(kit_types)
    kit_name = kit_types{i};
    config = kit_configs.(kit_name);

    % Calculate Total Ordered & Unresultable
    ordered_cols_exist = config.ordered_cols(ismember(config.ordered_cols, fvn));
    received_cols_exist = config.received_cols(ismember(config.received_cols, fvn));

    total_ordered = 0;
    if ~isempty(ordered_cols_exist)
        ordered_data = fulldata{:, ordered_cols_exist};
        total_ordered = sum(ordered_data == 1, 'all', 'omitnan');
    else
        fprintf('    Warning: No "ordered_%s" columns found.\n', kit_name);
    end

    total_unresultable = 0;
    if ~isempty(received_cols_exist)
        received_data = fulldata{:, received_cols_exist};
        total_unresultable = sum(received_data == 2, 'all', 'omitnan');
    else
        fprintf('    Warning: No "received_%s" columns found.\n', kit_name);
    end

    kit_table.(sprintf('Total_%s_Ordered', config.name_long)) = total_ordered;
    kit_table.(sprintf('Total_%s_Unresultable', config.name_long)) = total_unresultable;

    % Calculate Per-Fiscal-Year Stats
    all_order_dates = [];
    valid_order_cols = config.order_date_cols(ismember(config.order_date_cols, fvn));
    for j = 1:length(valid_order_cols)
        col_data = fulldata.(valid_order_cols{j});
        if isdatetime(col_data)
            all_order_dates = [all_order_dates; col_data(~isnat(col_data))];
        end
    end

    all_received_dates = [];
    valid_received_cols = config.received_date_cols(ismember(config.received_date_cols, fvn));
    for j = 1:length(valid_received_cols)
        col_data = fulldata.(valid_received_cols{j});
        if isdatetime(col_data)
            all_received_dates = [all_received_dates; col_data(~isnat(col_data))];
        end
    end

    all_fy_dates = [all_order_dates; all_received_dates];
    if ~isempty(all_fy_dates)
        unique_fys = unique(getFiscalYear(all_fy_dates));
        for k = 1:length(unique_fys)
            current_fy = unique_fys(k);
            ordered_in_fy = sum(getFiscalYear(all_order_dates) == current_fy);
            received_in_fy = sum(getFiscalYear(all_received_dates) == current_fy);

            ordered_var_name = sprintf('%s_Ordered_FY%d', config.name_long, current_fy);
            received_var_name = sprintf('%s_Received_FY%d', config.name_long, current_fy);
            kit_table.(ordered_var_name) = ordered_in_fy;
            kit_table.(received_var_name) = received_in_fy;
        end
    end
end
end

function dispatchTable = calculatelgcdispatch()
% calculatelgcdispatch - Scans for LGC kit dispatch reports and extracts data.
basepath = findbasepath;
resultsData = cell(0, 3);

folderPath = fullfile(basepath, '1_grants', '05_Menopause', 'LetsGetChecked Kits', 'Results Reports');
kitTypes = {'FMHM', 'PRGS'};
fprintf('Starting dispatch calculation...\n');

for i = 1:length(kitTypes)
    currentKit = kitTypes{i};
    fprintf('Processing kit type: %s\n', currentKit);
    filePattern = sprintf('%s\\*%sDTP_TEST_UTILIZATION_*.csv', folderPath, currentKit);
    reportFiles = dir(filePattern);

    if isempty(reportFiles), warning('No report files found for kit type: %s. Pattern: %s', currentKit, filePattern);
        continue;
    end

    for j = 1:length(reportFiles)
        currentFile = reportFiles(j);
        fullFilePath = fullfile(currentFile.folder, currentFile.name);
        try
            % Extract Date from Filename
            dateStr = currentFile.name(1:8);
            dispatchDate = datetime(dateStr, 'InputFormat', 'yyyyMMdd');

            % Load File and Find Value
            opts = detectImportOptions(fullFilePath);
            opts.VariableNames = {'Metric', 'Value'};
            opts.VariableTypes = {'char', 'char'};
            fileContent = readtable(fullFilePath, opts);

            targetRowIdx = find(strcmp(fileContent.Metric, 'TOTAL KITS DISPATCHED'));
            if ~isempty(targetRowIdx)
                dispatchedValueStr = fileContent.Value{targetRowIdx(1)};
                dispatchedCount = str2double(dispatchedValueStr);
                if isnan(dispatchedCount)
                    warning('Could not convert dispatch value "%s" to a number in file %s. Skipping entry.', dispatchedValueStr, currentFile.name);
                    continue;
                end

                % Save Data
                newDataRow = {currentKit, dispatchDate, dispatchedCount};
                resultsData = [resultsData; newDataRow];
            else
                warning('Could not find "TOTAL KITS DISPATCHED" in file: %s. Skipping file.', currentFile.name);
            end
        catch ME
            warning('An error occurred while processing file: %s\nError message: %s\nSkipping file.', ...
                fullFilePath, ME.message);
            continue;
        end
    end
end

% Finalization
if ~isempty(resultsData)
    dispatchTable = cell2table(resultsData, ...
        'VariableNames', {'KitType', 'DispatchDate', 'TotalKitsDispatched'});
    dispatchTable = sortrows(dispatchTable, 'DispatchDate');
else
    dispatchTable = table('Size', [0 3], 'VariableTypes', {'string', 'datetime', 'double'}, ...
        'VariableNames', {'KitType', 'DispatchDate', 'TotalKitsDispatched'});
end
end

function basepath = findbasepath()
%findbasepath Determines the base path based on the operating system.
if ispc
    basepath = 'Z:\';
    fprintf('Operating System: Windows PC\n');
    fprintf('Default network drive set to: %s\n', basepath);
elseif ismac
    fprintf('Operating System: macOS\n');
    prompt = 'Please enter the full path to the network drive (e.g., /Volumes/YourDriveName): ';
    basepath = input(prompt, 's');

    if isempty(basepath)
        error('No drive path was provided. Please run the function again.');
    end
    fprintf('Network drive set to: %s\n', basepath);
else
    error('Unsupported operating system. This function only supports Windows and macOS.');
end
end
