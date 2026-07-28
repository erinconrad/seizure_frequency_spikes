function Out = run_spike_sz_pipeline_clean()
%RUN_SPIKE_SZ_PIPELINE_CLEAN  All analyses for Conrad et al., 2026.
%
%   "Interictal Spike Rate on Routine Outpatient EEG Is Associated With
%    Seizure Frequency in a Large Epilepsy Cohort"
%
% ---------------------------------------------------------------------
% REQUIREMENTS
%   1. spike_counts.csv and clinical_data_deidentified.csv in ../data
%      https://upenn.box.com/s/yy4o1t6nit7yu35flz59ux6zf54slg9m
%   2. MATLAB R2024a + Statistics and Machine Learning Toolbox
%      (Parallel Computing Toolbox optional; parfor degrades to for)
%   3. Codebase: https://github.com/erinconrad/seizure_severity
%   4. ../output must be creatable (it is made automatically).
%
% USAGE
%   >> Out = run_spike_sz_pipeline_clean;
%
% ---------------------------------------------------------------------
% PIPELINE OVERVIEW  (read top to bottom; each stage feeds the next)
%
%   LOAD        spike counts (one row per EEG) + clinical report table
%               (one row per EEG, carrying that patient's full visit arrays)
%   DURATION    apply DURATION_SOURCE toggle BEFORE computing spike rate
%   FILTER      outpatient clinic visits only; outpatient routine EEG <=4 h
%   COHORT      Vuniq  = one row per (patient, clinic visit)
%               Views  = all analysis-ready tables for the primary cohort
%   MODEL       PairTable = every (EEG, visit) pair -> logistic GLME
%   OUTPUT      figures, Table 1, Table S1, results_summary.html
%
% ---------------------------------------------------------------------
% FILE MAP (local functions, in order)
%   Data loading / filtering .... require_cols, assert_unique_keys,
%                                 apply_duration_source,
%                                 filter_visit_arrays_by_type,
%                                 filter_outpatient_routine
%   Cohort construction ......... build_visit_level_table,
%                                 build_patient_seizure_metrics,
%                                 build_patient_typing_from_report,
%                                 build_filtered_view,
%                                 resolve_reported_spike_status
%   Model ....................... build_eeg_visit_pairs,
%                                 fit_mixed_effects_models, run_bootstrap,
%                                 add_duration_to_model
%   Figures ..................... make_flowchart_figure, make_fig_controls,
%                                 spearman_figure, draw_spearman_panel,
%                                 make_model_figure, make_figSup_lag,
%                                 make_fig_szfreq_by_reported_spikes,
%                                 plot_delta_rho_histogram,
%                                 make_eeg_duration_histogram,
%                                 make_fig_report_bias,
%                                 make_fig_generalized_subtypes,
%                                 spearman_trim_top_spikers
%   Tables / text ............... build_table1_flat, write_tableS1,
%                                 write_results_html
%   Utilities ................... JSON, log-scale, bootstrap, p-formatting
%
% ---------------------------------------------------------------------
% CHANGES FROM THE PREVIOUS VERSION (all flagged in the response that
% accompanied this file; search "REVIEW:" for the points worth a second look)
%   - Removed the M_after / M_before directional sensitivity models.
%   - Removed the unused MRI figure code.
%   - One bootstrap count (nBoot) is now used everywhere.
%   - Bootstrap p-values use the (count+1)/(B+1) convention.
%   - Fig 1B plot, stats, and N labels now come from a single table.
%   - Spearman panel A now fits its line on non-zero points, like B-D.
%   - Exclusion accounting no longer charges "no clinic visits" to the
%     "no epilepsy diagnosis" bucket. Final cohort is unchanged.
%   - Non-finite values are no longer silently plotted at the zero floor.

%% ===================== 0. CONFIGURATION =====================
rng(1);

% ---- Paths -----------------------------------------------------------
P.spikeCounts   = '../data/spike_counts.csv';
P.report        = '../data/clinical_data_deidentified.csv';
P.deadtime      = '../data/spike_counts_deadtime_0p46.csv';
P.outDir        = '../output';

% ---- Analysis switches ----------------------------------------------
% DURATION_SOURCE decides the denominator (and, for "deadtime", also the
% numerator) of spike rate. Applied BEFORE any filtering so that the choice
% propagates to spike rates, the <4 h routine filter, and everything after.
%   "file"     EDF span from spike_counts.csv (primary analysis)
%   "natus"    duration_hms from the report; clips pre-connection segments
%              REVIEW: this clips the denominator but NOT the numerator, so
%              spike rate is biased upward. Sensitivity use only.
%   "deadtime" detections outside the leading dead window / that same window
CFG.DURATION_SOURCE   = "file";
CFG.DEADTIME_COUNTCOL = "count_0_46";   % threshold the deadtime file was built at
CFG.countCol          = "count_0_46";   % spike-count column to analyse
CFG.durCol            = "Duration_sec";
CFG.MAX_ROUTINE_HOURS = 4;

% ---- Draft mode ------------------------------------------------------
% CFG.QUICK = true is for checking that the pipeline RUNS, not for results.
% It skips the GLME cluster bootstrap entirely, which is by far the slowest
% step (thousands of refits of a mixed model on the full pair table). M1/M2
% confidence intervals then fall back to the Laplace approximation, which the
% forest plot, Table S1 and the results HTML all already handle. Everything
% else resamples 200 times instead of 5000, so medians and Spearman CIs are
% rough but present.
%
% If the four remaining GLME fits are still too slow, also set
% CFG.SUBSAMPLE_PATIENTS (e.g. 400). That makes the run fast but every count
% in the output is wrong by construction, so it is strictly a smoke test.
CFG.QUICK              = false;   % <-- flip to true for a fast structural check
CFG.SUBSAMPLE_PATIENTS = 0;       % 0 = all patients; e.g. 400 for a smoke test

CFG.alpha = 0.05;
if CFG.QUICK
    CFG.nBoot      = 200;         % medians, Spearman, delta-rho
    CFG.nBootModel = 0;           % GLME cluster bootstrap; 0 = skip entirely
else
    CFG.nBoot      = 5000;
    CFG.nBootModel = 5000;
    % A floor, rather than silently producing NaN intervals on a real run.
    assert(CFG.nBoot >= 1000, ...
        'nBoot = %d is too low; CIs and bootstrap p-values require >= 1000.', CFG.nBoot);
end

% Generalized-syndrome panels need enough patients per syndrome; relax the
% threshold when the cohort has been artificially shrunk.
CFG.genSubMinN = 10;
if CFG.SUBSAMPLE_PATIENTS > 0, CFG.genSubMinN = 3; end

if CFG.QUICK || CFG.SUBSAMPLE_PATIENTS > 0
    fprintf(2, ['\n*** DRAFT RUN: nBoot=%d, model bootstrap=%d, subsample=%d. ' ...
        'Numbers are NOT publishable. ***\n\n'], ...
        CFG.nBoot, CFG.nBootModel, CFG.SUBSAMPLE_PATIENTS);
end

% ---- Epsilons (two different ones on purpose) ------------------------
% EPS_RATE only nudges exact zeros onto the log axes for DISPLAY.
% EPS_SPIKE enters log(spike rate) in the MODEL, so it is a real modelling
% choice: at 1e-3 a zero-spike EEG sits ~7 log units below a 1/h EEG.
% REVIEW: worth a sensitivity check at, say, 1e-2 and 1e-1.
CFG.EPS_RATE  = 30e-3;    % spikes/hour, display floor
CFG.EPS_FREQ  = 1e-3;     % seizures/month, display floor
CFG.EPS_SPIKE = 1e-3;     % spikes/hour, model floor

% ---- Cohort definitions ---------------------------------------------
CFG.NESD_LABEL = "Non-Epileptic Seizure Disorder";
CFG.badTypes   = lower(["Uncertain if Epilepsy","Unknown or MRN not found",""]);
CFG.canonical3 = ["General","Temporal","Frontal"];   % NOTE: this order is
                                                     % assumed by the HTML
                                                     % writer, which now looks
                                                     % rows up by name anyway.

CFG.allowableVisits = [ ...
    "CONSULT VISIT","ESTABLISHED PATIENT VISIT", ...
    "FOLLOW-UP PATIENT CLINIC","NEW PATIENT CLINIC","NEW PATIENT VISIT", ...
    "NPV MANAGEMENT DURING COVID-19","NPV NEUROLOGY", ...
    "RETURN ANNUAL VISIT","RETURN PATIENT EXTENDED","RETURN PATIENT VISIT", ...
    "RPV MANAGEMENT DURING COVID-19","TELEHEALTH VIDEO VISIT RETURN"];

% ---- Plot limits -----------------------------------------------------
CFG.Y_LIMS        = [-2 4];       % log10 spikes/hour, Fig 1
CFG.spearman_xLim = [-3.5 4];     % log10 seizures/month
CFG.spearman_yLim = [-1.5 3];     % log10 spikes/hour

if ~exist(P.outDir,'dir'), mkdir(P.outDir); end
f = @(name) fullfile(P.outDir, name);

%% ===================== 1. LOAD =====================
Spikes = readtable(P.spikeCounts,'TextType','string','VariableNamingRule','preserve');
require_cols(Spikes, ["Patient","Session",CFG.countCol,CFG.durCol], "Spikes");

Report = readtable(P.report,'TextType','string','VariableNamingRule','preserve');
require_cols(Report, ...
    ["patient_id","session_number","acquired_on", ...
     "report_PATIENT_CLASS","jay_in_or_out", ...
     "visit_type","visit_dates_deid","sz_freqs","visit_hasSz", ...
     "epilepsy_type","epilepsy_specific","nlp_gender","deid_birth_date", ...
     "start_time_deid","report_SPORADIC_EPILEPTIFORM_DISCHARGES", ...
     "jay_focal_epi","jay_multifocal_epi","jay_gen_epi","duration_hms"], "Report");

% Draft-mode subsampling happens here, before any filtering, so the flow
% diagram counts stay internally consistent with each other.
if CFG.SUBSAMPLE_PATIENTS > 0
    [Spikes, Report] = subsample_patients(Spikes, Report, CFG.SUBSAMPLE_PATIENTS);
end

% Duration first, then spike rate, so the toggle propagates everywhere.
Spikes = apply_duration_source(Spikes, Report, CFG, P.deadtime);
Spikes.SpikeRate_perHour = Spikes.(CFG.countCol) ./ Spikes.(CFG.durCol) * 3600;

%% ===================== 2. FILTER =====================
% (a) inside each patient's visit arrays, keep only outpatient clinic visits
Report = filter_visit_arrays_by_type(Report, CFG.allowableVisits);

% (b) keep only outpatient routine EEGs of <= MAX_ROUTINE_HOURS
[Spikes, Report, nPatientsTotal] = filter_outpatient_routine(Spikes, Report, CFG);

assert_unique_keys(Spikes, "Patient",    "Session",        "Spikes");
assert_unique_keys(Report, "patient_id", "session_number", "Report");

%% ===================== 3. BUILD COHORT =====================
Vuniq  = build_visit_level_table(Report);                          % 1 row / visit
Typing = build_patient_typing_from_report(Report, CFG.canonical3); % 1 row / patient
SzFreq = build_patient_seizure_metrics(Vuniq);                     % 1 row / patient

Views  = build_filtered_view(Spikes, Report, Typing, SzFreq, CFG, nPatientsTotal);

%% ===================== 4. MODEL =====================
% One row per (EEG, clinic visit) pair, canonical-subtype patients only.
PairTable = build_eeg_visit_pairs(Vuniq, Views.SessionLevelSpikeRates, ...
    Views.ReportForKeptSessions, Views.PatientTypingFiltered, CFG.EPS_SPIKE);

MMR = fit_mixed_effects_models(PairTable, CFG);

% Does EEG duration add anything on top of M1? (LRT, 1 df)
DurCompare = add_duration_to_model(MMR, Views, CFG.alpha);

%% ===================== 5. FIGURES =====================
% Manuscript figure numbering. Output FILENAMES now carry the figure number
% so the two cannot drift apart (previously Fig1.png was manuscript Fig 2 and
% FigS1.png was manuscript Fig S2).
%
%   MAIN          Fig 1  study schematic (not produced here)
%                 Fig 2  detector controls
%                 Fig 3  spike rate vs seizure frequency
%                 Fig 4  mixed effects model
%
%   SUPPLEMENT    Fig S1  participant flow
%                 Fig S2  EEG duration distribution
%                 Fig S3  EEGs with vs without an available spike report
%                 Fig S4  generalized-epilepsy syndromes
%                 Fig S5  Spearman, non-zero spike rates and frequencies only
%                 Fig S6  Spearman, top 10% of spikers removed
%                 Fig S7  seizure frequency by reported spikes across EEGs
%                 Fig S8  EEG-visit lag context
%                 Fig S9  near vs far visit windows

% ---- Fig S1-S4: cohort, measurement and bias supplements ----
FigFlow = make_flowchart_figure(Views, MMR);
save_fig(FigFlow, f('FigS1_flow.png'));

[FigDur, DurStats] = make_eeg_duration_histogram(Views, f('FigS2_eeg_duration.png'));

[FigBias, BiasStats] = make_fig_report_bias(Views, SzFreq, CFG, ...
    f('FigS3_report_bias.png'));

[GenSub, FigGenSub] = make_fig_generalized_subtypes(Views, CFG, ...
    f('FigS4_generalized_subtypes.png'), CFG.genSubMinN, false);

% ---- Fig 2: detector controls ----
[FigControls, ControlStats] = make_fig_controls(Views, CFG);
save_fig(FigControls, f('Fig2_controls.png'));

% ---- Fig 3 and its two sensitivity supplements (Fig S5, Fig S6) ----
SP.main = spearman_figure(Views.PatientSpikeSz_All, Views.PatientSpikeSz_Typed, ...
    CFG, f('Fig3_spearman.png'), '', false);
SP.nz   = spearman_figure(Views.PatientSpikeSz_All, Views.PatientSpikeSz_Typed, ...
    CFG, f('FigS5_spearman_nonzero.png'), ' (positive spike/seizures only)', true);
SP.trim = spearman_trim_top_spikers(Views, CFG, ...
    f('FigS6_spearman_trim10.png'), 0.10, false);

% ---- Fig S7: patient-level companion to Fig 2A ----
FigSzByReport = make_fig_szfreq_by_reported_spikes(Views, SzFreq, CFG);
save_fig(FigSzByReport, f('FigS7_szfreq_by_reported_spikes.png'));

% ---- Fig 4 and its context supplements (Fig S8, Fig S9) ----
FigMain   = make_model_figure(MMR, f('Fig4_model.png'));
FigSupLag = make_figSup_lag(MMR, Vuniq, CFG, f('FigS8_lag_context.png'));
NearFar   = plot_delta_rho_histogram(Views, Vuniq, Views.ReportForKeptSessions, ...
    0.333, 0.667, CFG, f('FigS9_near_far_tertiles.png'));

%% ===================== 6. TABLES AND TEXT =====================
Table1 = build_table1_flat(Views, SzFreq, Vuniq, CFG);
writetable(Table1, f('Table1.csv'));

write_tableS1(MMR, f('TableS1.csv'));

% Supp carries the numbers that the new supplemental sentences need.
Supp = struct('Duration',DurStats, 'Bias',BiasStats, 'GenSub',GenSub, 'Trim',SP.trim);
write_results_html(f('results_summary.html'), Views, SzFreq, ControlStats, ...
    SP, MMR, Vuniq, NearFar, Supp, CFG);

%% ===================== 7. RETURN EVERYTHING =====================
% Bundled so the whole run can be inspected from one variable.
Out = struct('CFG',CFG, 'Views',Views, 'Vuniq',Vuniq, 'SzFreq',SzFreq, ...
    'PairTable',PairTable, 'MMR',MMR, 'DurCompare',DurCompare, ...
    'Spearman',SP, 'NearFar',NearFar, 'GenSubtypes',GenSub, ...
    'BiasStats',BiasStats, 'DurStats',DurStats, ...
    'ControlStats',ControlStats, 'Table1',Table1, ...
    'Figures',struct('S1_Flow',FigFlow, 'S2_Duration',FigDur, ...
                     'S3_Bias',FigBias, 'S4_GenSub',FigGenSub, ...
                     'F2_Controls',FigControls, 'F4_Model',FigMain, ...
                     'S7_SzByReport',FigSzByReport, 'S8_Lag',FigSupLag));

fprintf('\nDone. Outputs in %s\n', P.outDir);
end


%% #####################################################################
%% ##  DATA LOADING AND FILTERING
%% #####################################################################

function require_cols(T, cols, name)
%REQUIRE_COLS  Fail fast if an expected column is absent.
missing = setdiff(string(cols), string(T.Properties.VariableNames));
assert(isempty(missing), '%s is missing required columns: %s', ...
    name, strjoin(missing, ", "));
end


function assert_unique_keys(T, pidCol, sesCol, name)
%ASSERT_UNIQUE_KEYS  One row per (patient, session).
key = string(T.(pidCol)) + "|" + string(T.(sesCol));
assert(numel(unique(key)) == numel(key), ...
    '%s has duplicated (%s, %s) keys.', name, pidCol, sesCol);
end


function S = apply_duration_source(S, R, CFG, deadtimeCsv)
%APPLY_DURATION_SOURCE  Overwrite the duration (and, for "deadtime", also the
% spike count) used to compute spike rate.
%
% Numerator and denominator must always describe the same window. "file" and
% "deadtime" satisfy this; "natus" deliberately does not (see the note in the
% config block) and is for sensitivity analysis only.

durCol   = CFG.durCol;
countCol = CFG.countCol;

switch lower(string(CFG.DURATION_SOURCE))

    case "file"
        % Nothing to do: Duration_sec already holds the EDF span.
        fprintf('[Duration] File duration (%s from spike counts).\n', durCol);

    case "natus"
        require_cols(R, ["patient_id","session_number","duration_hms"], "Report");

        dur_raw = R.duration_hms;
        if isduration(dur_raw)
            natus_sec = seconds(dur_raw);
        else
            natus_sec = seconds(duration(strtrim(string(dur_raw)), ...
                'InputFormat','hh:mm:ss'));
        end

        Map = unique(table(double(R.patient_id), double(R.session_number), ...
            natus_sec, 'VariableNames',{'Patient','Session','Dur_sec'}), 'rows');
        assert_unique_keys(Map, "Patient","Session", "Natus duration map");

        [tf, loc] = ismember([double(S.Patient), double(S.Session)], ...
                             [Map.Patient, Map.Session], 'rows');
        newDur       = nan(height(S),1);
        newDur(tf)   = Map.Dur_sec(loc(tf));
        S.(durCol)   = newDur;

        fprintf(['[Duration] Natus duration (duration_hms). %d/%d spike rows ' ...
            'have no match and are set missing (dropped at the routine filter).\n'], ...
            nnz(~tf), height(S));

    case "deadtime"
        assert(strlength(string(deadtimeCsv)) > 0 && isfile(deadtimeCsv), ...
            'Dead-time CSV not found: %s', deadtimeCsv);
        assert(strlength(string(CFG.DEADTIME_COUNTCOL)) == 0 || ...
               string(countCol) == string(CFG.DEADTIME_COUNTCOL), ...
            ['countCol is "%s" but the dead-time file was built for "%s". ' ...
             'The split is threshold-specific.'], countCol, CFG.DEADTIME_COUNTCOL);

        D = readtable(deadtimeCsv,'TextType','string','VariableNamingRule','preserve');
        require_cols(D, ["Patient","Session","EDF_Duration_sec","Deadtime_sec", ...
            "n_spikes_deadtime","n_spikes_recording","n_spikes_total"], "Deadtime");

        nDead   = double(D.n_spikes_deadtime);
        nRec    = double(D.n_spikes_recording);
        nTot    = double(D.n_spikes_total);
        edfSec  = double(D.EDF_Duration_sec);
        deadSec = double(D.Deadtime_sec);
        ok      = isfinite(nDead) & isfinite(nRec);

        assert(all(nDead(ok) + nRec(ok) == nTot(ok)), ...
            'Dead-time file: spike split is not exhaustive in %d rows.', ...
            nnz(ok & (nDead + nRec ~= nTot)));
        assert(all(deadSec(ok) >= 0 & deadSec(ok) <= edfSec(ok)), ...
            'Dead-time file: Deadtime_sec outside [0, EDF_Duration_sec].');

        % Denominator is EDF - dead time, i.e. exactly the window the
        % recording count was taken over (not Recorded_Duration_sec, which
        % differs on rows where dead time was clamped to zero).
        recSec = edfSec - deadSec;
        usable = ok & isfinite(recSec) & recSec > 0;

        Map = unique(table(double(D.Patient(usable)), double(D.Session(usable)), ...
            nRec(usable), recSec(usable), ...
            'VariableNames',{'Patient','Session','Count','Dur_sec'}), 'rows');
        assert_unique_keys(Map, "Patient","Session", "dead-time map");

        [tf, loc] = ismember([double(S.Patient), double(S.Session)], ...
                             [Map.Patient, Map.Session], 'rows');
        newCount     = nan(height(S),1);
        newDur       = nan(height(S),1);
        newCount(tf) = Map.Count(loc(tf));
        newDur(tf)   = Map.Dur_sec(loc(tf));

        % Never let one of the two be set without the other.
        assert(isequal(isfinite(newCount), isfinite(newDur)), ...
            'Dead-time join produced a count without a duration (or vice versa).');
        assert(any(tf), 'No spike rows matched the dead-time file.');

        S.(countCol) = newCount;
        S.(durCol)   = newDur;

        fprintf(['[Duration] Dead-time corrected. %d/%d spike rows matched; ' ...
            '%d set missing.\n'], nnz(tf), height(S), nnz(~tf));

    otherwise
        error('DURATION_SOURCE must be "file", "natus" or "deadtime" (got "%s").', ...
            CFG.DURATION_SOURCE);
end
end


function [S, R] = subsample_patients(S, R, nKeep)
%SUBSAMPLE_PATIENTS  Draft mode only. Keep a random subset of patients so the
% whole pipeline runs in a fraction of the time. Reproducible because rng is
% seeded at the top. Every count in the output is wrong by construction.
pids = unique(double(R.patient_id));
if numel(pids) <= nKeep
    fprintf('[Draft] %d patients available; no subsampling needed.\n', numel(pids));
    return
end
keep = pids(randperm(numel(pids), nKeep));
S = S(ismember(double(S.Patient),    keep), :);
R = R(ismember(double(R.patient_id), keep), :);
fprintf('[Draft] Subsampled to %d of %d patients.\n', nKeep, numel(pids));
end


function R = filter_visit_arrays_by_type(R, allowableVisits)
%FILTER_VISIT_ARRAYS_BY_TYPE  Keep only outpatient clinic visits.
%
% Each report row carries that patient's FULL visit history as four parallel
% JSON arrays (type, date, seizure frequency, has-seizure). This filters all
% four in lock-step and rewrites them as JSON.

% Normalise the several ways "no visits" is encoded.
isNull = ismember(strtrim(string(R.visit_type)), ["[null]","null"]);
[R.visit_type(isNull), R.visit_dates_deid(isNull), ...
 R.sz_freqs(isNull),   R.visit_hasSz(isNull)] = deal("[]");

nBefore = 0; nAfter = 0;
for i = 1:height(R)
    vt_raw = strtrim(string(R.visit_type(i)));
    if ismember(vt_raw, ["", "[]", "<missing>"])
        [R.visit_type(i), R.visit_dates_deid(i), ...
         R.sz_freqs(i),   R.visit_hasSz(i)] = deal("[]");
        continue
    end

    vt    = json_to_string_array(vt_raw);
    dates = json_to_string_array(R.visit_dates_deid(i));
    sz    = json_to_double_array(R.sz_freqs(i));
    hs    = json_to_double_array(R.visit_hasSz(i));

    assert(numel(vt)==numel(dates) && numel(vt)==numel(sz) && numel(vt)==numel(hs), ...
        'Row %d: visit arrays have mismatched lengths.', i);

    nBefore = nBefore + numel(vt);
    keep    = ismember(string(vt), allowableVisits);

    if ~any(keep)
        [R.visit_type(i), R.visit_dates_deid(i), ...
         R.sz_freqs(i),   R.visit_hasSz(i)] = deal("[]");
        continue
    end

    nAfter = nAfter + nnz(keep);
    R.visit_type(i)       = string(jsonencode(cellstr(string(vt(keep)))));
    R.visit_dates_deid(i) = string(jsonencode(cellstr(string(dates(keep)))));
    R.sz_freqs(i)         = string(jsonencode(sz(keep)));
    R.visit_hasSz(i)      = string(jsonencode(hs(keep)));
end

fprintf('[Visit-type filter] %d -> %d visits (kept %.1f%%)\n', ...
    nBefore, nAfter, 100*nAfter/max(1,nBefore));
end


function [S, R, nPatientsTotal] = filter_outpatient_routine(S, R, CFG)
%FILTER_OUTPATIENT_ROUTINE  Keep outpatient routine EEGs of <= MAX_ROUTINE_HOURS.
%
% "Outpatient" is taken from any of three partially redundant sources
% (acquisition site, report patient class, manual in/out label).

nR0 = height(R); nS0 = height(S);
nPatientsTotal = numel(unique(double(R.patient_id)));   % denominator of the flow diagram

acq   = lower(strtrim(string(R.acquired_on)));
class = lower(strtrim(string(R.report_PATIENT_CLASS)));
jay   = lower(strtrim(string(R.jay_in_or_out)));
isOutpt = contains(acq,"spe") | contains(acq,"radnor") | ...
          (class == "outpatient") | (jay == "out");

OutptKeys = unique(R(isOutpt, {'patient_id','session_number'}));
OutptKeys.Properties.VariableNames = {'Patient','Session'};
assert(~isempty(OutptKeys), 'No outpatient sessions identified.');

% Non-finite durations fail this test, which is how missing Natus/deadtime
% durations drop out of the analysis.
isRoutine   = isfinite(S.(CFG.durCol)) & S.(CFG.durCol) <= CFG.MAX_ROUTINE_HOURS*3600;
RoutineKeys = unique(S(isRoutine, {'Patient','Session'}));

Keys = innerjoin(OutptKeys, RoutineKeys, 'Keys', {'Patient','Session'});
S = innerjoin(S, Keys, 'Keys', {'Patient','Session'});
R = innerjoin(R, Keys, 'LeftKeys', {'patient_id','session_number'}, ...
                       'RightKeys', {'Patient','Session'});

fprintf('[Outpatient+routine] Kept %d/%d spike rows, %d/%d report rows\n', ...
    height(S), nS0, height(R), nR0);
end


%% #####################################################################
%% ##  COHORT CONSTRUCTION
%% #####################################################################

function Vuniq = build_visit_level_table(R)
%BUILD_VISIT_LEVEL_TABLE  One row per (patient, clinic visit date).
%
% The same visit history is repeated on every EEG row for a patient, so the
% rows are expanded and then collapsed on (patient, date).
%
% Two imputation rules, applied in this order:
%   Rule 1  HasSz == 0 but no documented frequency  ->  SzFreq = 0
%   Rule 2  documented SzFreq but no HasSz          ->  HasSz = (SzFreq > 0)

PV = table('Size',[0 4], ...
    'VariableTypes',{'double','datetime','double','double'}, ...
    'VariableNames',{'Patient','VisitDate','Freq','HasSz'});
rows = cell(height(R),1);

for j = 1:height(R)
    ds = strtrim(string(R.visit_dates_deid(j)));
    if ismember(ds, ["","[]"]), continue; end

    d = datetime(string(jsondecode(char(ds))), 'InputFormat','yyyy-MM-dd');

    s = regexprep(strtrim(string(R.sz_freqs(j))), 'null', 'NaN', 'ignorecase');
    v = double(jsondecode(char(s)));
    v(~isfinite(v) | v < 0) = NaN;         % negatives are data-entry noise

    h = double(jsondecode(char(strtrim(string(R.visit_hasSz(j))))));
    h(h == 2) = NaN;                       % 2 = "unknown" in the source coding

    assert(numel(d)==numel(v) && numel(d)==numel(h), ...
        'Row %d: visit arrays mismatched.', j);
    if isempty(d), continue; end

    rows{j} = table(repmat(double(R.patient_id(j)),numel(d),1), d(:), v(:), h(:), ...
        'VariableNames', PV.Properties.VariableNames);
end
PV = vertcat(PV, rows{~cellfun(@isempty, rows)});

% Collapse duplicated (patient, date) rows. Duplicates come from the same
% visit being listed on several EEG rows, so mean/max are no-ops in practice.
[g, pid, dt] = findgroups(PV.Patient, PV.VisitDate);
Freq  = splitapply(@(x) mean(x(isfinite(x))), PV.Freq,  g);
HasSz = splitapply(@max_omitnan,               PV.HasSz, g);

Vuniq = table(pid, dt, Freq, HasSz, ...
    'VariableNames', {'Patient','VisitDate','Freq','HasSz'});

Vuniq.Freq_R1 = Vuniq.Freq;
r1 = ~isfinite(Vuniq.Freq_R1) & (Vuniq.HasSz == 0);
Vuniq.Freq_R1(r1) = 0;

r2pos = isfinite(Vuniq.Freq_R1) & Vuniq.Freq_R1 >  0 & ~isfinite(Vuniq.HasSz);
r2neg = isfinite(Vuniq.Freq_R1) & Vuniq.Freq_R1 == 0 & ~isfinite(Vuniq.HasSz);
Vuniq.HasSz(r2pos) = 1;
Vuniq.HasSz(r2neg) = 0;

fprintf(['[Imputation] Rule 1 set SzFreq=0 for %d visits; ' ...
    'Rule 2 set HasSz for %d (SzFreq>0) and %d (SzFreq=0) visits.\n'], ...
    nnz(r1), nnz(r2pos), nnz(r2neg));
end


function SzP = build_patient_seizure_metrics(Vuniq)
%BUILD_PATIENT_SEIZURE_METRICS  Mean documented seizure frequency per patient.
[g, pids] = findgroups(Vuniq.Patient);
MeanSzFreq = splitapply(@(x) mean(x,'omitnan'), Vuniq.Freq_R1, g);
SzP = table(pids, MeanSzFreq, 'VariableNames', {'Patient','MeanSzFreq'});
end


function T = build_patient_typing_from_report(R, canonical3)
%BUILD_PATIENT_TYPING_FROM_REPORT  One epilepsy type/subtype per patient.
%
% epilepsy_type and epilepsy_specific are LLM-extracted and repeated on every
% EEG row; both must be internally consistent within a patient.
%
% EpiType3 is the canonical three-way label used throughout:
%   Temporal / Frontal  from epilepsy_specific
%   General             from epilepsy_type, only if no lobe was matched
% REVIEW: a patient whose type is "general" but whose specific text mentions
% "temporal" is labelled Temporal. Check whether that combination occurs.

pid   = double(R.patient_id);
etype = strtrim(string(R.epilepsy_type));
espec = strtrim(string(R.epilepsy_specific));

[uid,   etype_one] = collapse_one_per_patient(pid, etype, 'epilepsy_type');
[uidS,  espec_one] = collapse_one_per_patient(pid, espec, 'epilepsy_specific');

% Align on patient rather than assuming the two sets coincide.
T = outerjoin(table(uid,  etype_one, 'VariableNames',{'Patient','EpilepsyType'}), ...
              table(uidS, espec_one, 'VariableNames',{'Patient','EpilepsySpecific'}), ...
              'Keys','Patient','MergeKeys',true);
T.EpilepsyType(ismissing(T.EpilepsyType))         = "";
T.EpilepsySpecific(ismissing(T.EpilepsySpecific)) = "";

spec = lower(T.EpilepsySpecific);
type = lower(T.EpilepsyType);
E3   = strings(height(T),1);
E3(contains(spec,"temporal"))                   = "Temporal";
E3(contains(spec,"frontal"))                    = "Frontal";
E3((strlength(E3)==0) & strcmp(type,"general")) = "General";
T.EpiType3 = categorical(E3, canonical3);
end


function [uid, valOne] = collapse_one_per_patient(pid, val, label)
%COLLAPSE_ONE_PER_PATIENT  One non-empty value per patient; error on conflict.
ok  = ~ismissing(val) & strlength(val) > 0;
Tin = sortrows(table(pid(ok), val(ok), 'VariableNames',{'Patient','V'}), 'Patient');
[uid, ~, g] = unique(Tin.Patient, 'stable');
valOne = strings(numel(uid),1);
for k = 1:numel(uid)
    vals = unique(Tin.V(g == k));
    assert(numel(vals) <= 1, 'Conflicting %s for Patient %g.', label, uid(k));
    if ~isempty(vals), valOne(k) = vals(1); end
end
end


function Views = build_filtered_view(Sessions, Report, Typing, SzFreq, CFG, nPatientsTotal)
%BUILD_FILTERED_VIEW  Assemble every analysis-ready table for the cohort.
%
% Primary cohort = patient has (a) an outpatient routine EEG, (b) a confirmed
% epilepsy diagnosis, and (c) at least one documented seizure frequency.
%
% Exclusion accounting note: patients with no clinic-visit record at all fail
% criterion (c), not (b). The previous version charged them to (b) because the
% seizure-frequency table simply had no row for them. The final cohort is
% identical either way; only the flow-diagram attribution changes.

%% --- Patients surviving the outpatient/routine filter ---
PatientsKept       = unique(double(Sessions.Patient));
nAfterOutptRoutine = numel(PatientsKept);

%% --- (b) epilepsy diagnosis -------------------------------------------
% A patient with no typing row at all counts as "not confirmed", so the
% membership test is explicit rather than being hidden inside an innerjoin.
TypingKept = Typing(ismember(Typing.Patient, PatientsKept), :);

etype    = lower(strtrim(string(TypingKept.EpilepsyType)));
isNESD   = (etype == lower(strtrim(CFG.NESD_LABEL)));
isBad    = ismember(etype, CFG.badTypes) | ismissing(etype) | strlength(etype)==0;
EpiPatients = TypingKept.Patient(~isNESD & ~isBad);

nWithEpilepsy = numel(EpiPatients);

%% --- (c) documented seizure frequency ----------------------------------
SzKept   = SzFreq(ismember(SzFreq.Patient, EpiPatients), :);
CohortPatients = unique(SzKept.Patient(isfinite(SzKept.MeanSzFreq)));
CohortTable    = table(CohortPatients, 'VariableNames',{'Patient'});

fprintf('[Cohort] %d epilepsy patients with a documented seizure frequency\n', ...
    numel(CohortPatients));

%% --- Restrict everything to the cohort ---------------------------------
SessionsFiltered = innerjoin(Sessions, CohortTable, 'Keys','Patient');

SessKeys = unique(SessionsFiltered(:,{'Patient','Session'}));
ReportForKeptSessions = innerjoin(Report, SessKeys, ...
    'LeftKeys',{'patient_id','session_number'}, 'RightKeys',{'Patient','Session'});
ReportForKeptSessions.Patient = double(ReportForKeptSessions.patient_id);
ReportForKeptSessions.Session = double(ReportForKeptSessions.session_number);

TypingFiltered = innerjoin(Typing,  CohortTable, 'Keys','Patient');
SzFreqCohort   = innerjoin(SzFreq,  CohortTable, 'Keys','Patient');

%% --- Session- and patient-level spike rates ----------------------------
% Built from the restricted session table, so every downstream table refers
% to exactly the same set of EEGs.
SessionLevelSpikeRates = SessionsFiltered(:, ...
    {'Patient','Session','SpikeRate_perHour','Duration_sec'});
SessionLevelSpikeRates.Properties.VariableNames{'SpikeRate_perHour'} = 'SpikesPerHour';

[g, pids] = findgroups(double(SessionsFiltered.Patient));
PatientLevelSpikeRates = table(double(pids), ...
    splitapply(@(x) mean(x,'omitnan'), SessionsFiltered.SpikeRate_perHour, g), ...
    'VariableNames', {'Patient','MeanSpikeRate_perHour'});
PatientLevelSpikeRates = innerjoin(PatientLevelSpikeRates, ...
    TypingFiltered(:,{'Patient','EpilepsyType','EpilepsySpecific','EpiType3'}), ...
    'Keys','Patient');

%% --- Spearman input tables ---------------------------------------------
% All-epilepsy view.
PatientSpikeSz_All = innerjoin( ...
    PatientLevelSpikeRates(:,{'Patient','MeanSpikeRate_perHour'}), ...
    SzFreqCohort, 'Keys','Patient');
PatientSpikeSz_All = PatientSpikeSz_All( ...
    isfinite(PatientSpikeSz_All.MeanSpikeRate_perHour) & ...
    isfinite(PatientSpikeSz_All.MeanSzFreq), :);

% Canonical-subtype view (Temporal / Frontal / General only).
isCanon = ~ismissing(PatientLevelSpikeRates.EpiType3) & ...
          ismember(string(PatientLevelSpikeRates.EpiType3), CFG.canonical3) & ...
          isfinite(PatientLevelSpikeRates.MeanSpikeRate_perHour);

PatientSpikeSz_Typed = innerjoin( ...
    PatientLevelSpikeRates(isCanon, {'Patient','MeanSpikeRate_perHour','EpiType3'}), ...
    SzFreqCohort, 'Keys','Patient');
PatientSpikeSz_Typed = PatientSpikeSz_Typed( ...
    isfinite(PatientSpikeSz_Typed.MeanSzFreq), :);

%% --- Fig 1B source table (plot, stats and N labels all come from here) --
Canonical3_SubsetTable = PatientSpikeSz_Typed(:, ...
    {'Patient','EpiType3','MeanSpikeRate_perHour'});
Canonical3_SubsetTable.Properties.VariableNames{'EpiType3'} = 'EpiType4';

[g3, cats3] = findgroups(Canonical3_SubsetTable.EpiType4);
x3 = Canonical3_SubsetTable.MeanSpikeRate_perHour;
Canonical3_Stats = table(cats3, ...
    splitapply(@(x) sum(isfinite(x)), x3, g3), ...
    splitapply(@(x) median(x,'omitnan'), x3, g3), ...
    splitapply(@(x) prctile(x,25),       x3, g3), ...
    splitapply(@(x) prctile(x,75),       x3, g3), ...
    'VariableNames',{'EpiType4','GroupCount','Median','P25','P75'});

Canonical3_Pairs = ["General","Temporal"; "General","Frontal"; "Temporal","Frontal"];
p_pair = NaN(3,1);
for i = 1:3
    xa = x3(Canonical3_SubsetTable.EpiType4 == categorical(Canonical3_Pairs(i,1), CFG.canonical3));
    xb = x3(Canonical3_SubsetTable.EpiType4 == categorical(Canonical3_Pairs(i,2), CFG.canonical3));
    if nnz(isfinite(xa)) >= 3 && nnz(isfinite(xb)) >= 3
        p_pair(i) = ranksum(xa, xb, 'method','approx');
    end
end

%% --- Flow-diagram counts ------------------------------------------------
EC.nTotal              = nPatientsTotal;
EC.nAfterOutptRoutine  = nAfterOutptRoutine;
EC.nExcludedNoEpilepsy = nAfterOutptRoutine - nWithEpilepsy;
EC.nExcludedNoSzFreq   = nWithEpilepsy - numel(CohortPatients);
EC.nFinalCohort        = numel(CohortPatients);
assert(EC.nExcludedNoEpilepsy + EC.nExcludedNoSzFreq + EC.nFinalCohort == ...
    EC.nAfterOutptRoutine, 'Flow diagram counts do not sum to the post-filter total.');

%% --- Bundle -------------------------------------------------------------
Views = struct( ...
    'ExclusionCounts',        EC, ...
    'SessionsForFigures',     SessionsFiltered, ...
    'ReportForKeptSessions',  ReportForKeptSessions, ...
    'SessionLevelSpikeRates', SessionLevelSpikeRates, ...
    'PatientLevelSpikeRates', PatientLevelSpikeRates, ...
    'PatientTypingFiltered',  TypingFiltered(ismember(TypingFiltered.Patient, ...
                                  PatientSpikeSz_Typed.Patient), :), ...
    'PatientSpikeSz_All',     PatientSpikeSz_All, ...
    'PatientSpikeSz_Typed',   PatientSpikeSz_Typed, ...
    'Canonical3_SubsetTable', Canonical3_SubsetTable, ...
    'Canonical3_Stats',       Canonical3_Stats, ...
    'Canonical3_Pairs',       Canonical3_Pairs, ...
    'PvalsPairwise',          p_pair, ...
    'PvalsPairwiseBonf',      min(p_pair*3, 1));
end


function ReportSlim = resolve_reported_spike_status(R)
%RESOLVE_REPORTED_SPIKE_STATUS  Clinically reported spike presence per EEG.
%
% Two overlapping sources: the free-text field
% report_SPORADIC_EPILEPTIFORM_DISCHARGES, and three structured columns
% (focal / multifocal / generalized). Resolution rules, in order:
%   present  either source says present anywhere
%   absent   one source says absent and the other is blank
%   unknown  neither source is informative

main = string(R.report_SPORADIC_EPILEPTIFORM_DISCHARGES);
isMainPresent = (main == "present");
isMainAbsent  = (main == "absent");

rawF = lower(strtrim(string(R.jay_focal_epi)));
rawM = lower(strtrim(string(R.jay_multifocal_epi)));
rawG = lower(strtrim(string(R.jay_gen_epi)));

jayPresent = (rawF=="present") | (rawM=="present") | (rawG=="present");
jayAbsent  = (rawF=="absent")  & (rawM=="absent")  & (rawG=="absent");
jayBlank   = ~(rawF=="present"|rawF=="absent") & ...
             ~(rawM=="present"|rawM=="absent") & ...
             ~(rawG=="present"|rawG=="absent");
mainBlank  = ~(isMainPresent | isMainAbsent);

% Symmetric discordance check. The previous version only caught the case
% where ALL THREE structured columns said present against an absent free-text
% field, so most disagreements passed silently.
discordant = (jayAbsent & isMainPresent) | (isMainAbsent & jayPresent);
assert(~any(discordant), ...
    'Discordant spike presence between free-text and structured columns in %d EEGs.', ...
    nnz(discordant));

status = strings(height(R),1);
status(isMainPresent | jayPresent)  = "present";
status(jayAbsent  & mainBlank)      = "absent";
status(isMainAbsent & jayBlank)     = "absent";
status(status == "")                = "unknown";

ReportSlim = R(:,{'Patient','Session'});
ReportSlim.ReportStatus = categorical(status, ["absent","present","unknown"]);
end


%% #####################################################################
%% ##  MIXED EFFECTS MODEL
%% #####################################################################

function PairTable = build_eeg_visit_pairs(Vuniq, SessionLevelSpikeRates, ...
    ReportForKeptSessions, PatientTyping, EPS_SPIKE)
%BUILD_EEG_VISIT_PAIRS  Every (EEG, clinic visit) pair within a patient.
%
% This is the unit of analysis for the mixed effects model: each EEG is paired
% with each of that patient's clinic visits, and SignedLag records how far
% apart they were (positive = visit occurred after the EEG).

EEG_raw = ReportForKeptSessions.start_time_deid;
if isdatetime(EEG_raw)
    EEG_dt = EEG_raw;
else
    EEG_dt = datetime(strtrim(string(EEG_raw)), 'InputFormat', "yyyy-MM-dd'T'HH:mm:ss");
end

EEG_tbl = table(double(ReportForKeptSessions.Patient), ...
                double(ReportForKeptSessions.Session), EEG_dt, ...
                'VariableNames',{'Patient','Session','EEG_Date'});
EEG_tbl = EEG_tbl(~isnat(EEG_tbl.EEG_Date), :);
EEG_tbl = innerjoin(EEG_tbl, SessionLevelSpikeRates(:,{'Patient','Session','SpikesPerHour'}), ...
    'Keys',{'Patient','Session'});

Vtyped = innerjoin(Vuniq(:,{'Patient','VisitDate','Freq_R1','HasSz'}), ...
    PatientTyping(:,{'Patient','EpiType3'}), 'Keys','Patient');

patients = intersect(unique(EEG_tbl.Patient), unique(Vtyped.Patient));

% Build per patient and concatenate once; far cheaper than growing a table.
blocks = cell(numel(patients),1);
for i = 1:numel(patients)
    p = patients(i);
    E = EEG_tbl(EEG_tbl.Patient == p, :);
    V = Vtyped(Vtyped.Patient  == p, :);
    if isempty(E) || isempty(V), continue; end

    [ei, vi] = ndgrid(1:height(E), 1:height(V));
    ei = ei(:); vi = vi(:);

    blocks{i} = table( ...
        repmat(p, numel(ei), 1), ...
        E.Session(ei), ...
        V.VisitDate(vi), ...
        E.SpikesPerHour(ei), ...
        V.Freq_R1(vi), ...
        V.HasSz(vi), ...
        days(V.VisitDate(vi) - E.EEG_Date(ei)), ...
        string(V.EpiType3(vi)), ...
        'VariableNames',{'Patient','Session','VisitDate','SpikesPerHour', ...
                         'SzFreq','HasSz','SignedLag_days','EpiType3'});
end
PairTable = vertcat(blocks{~cellfun(@isempty, blocks)});

nBefore = height(PairTable);
PairTable = PairTable( ...
    isfinite(PairTable.SpikesPerHour) & isfinite(PairTable.SzFreq) & ...
    isfinite(PairTable.SignedLag_days) & strlength(PairTable.EpiType3) > 0, :);

fprintf('[Pairs] %d patients, %d pairs (%d dropped for missing data)\n', ...
    numel(patients), height(PairTable), nBefore - height(PairTable));

% EPS_SPIKE keeps zero-spike EEGs on the log scale; see the config block.
PairTable.LogSpikesPerHour = log(PairTable.SpikesPerHour + EPS_SPIKE);
PairTable.SignedLag_years  = PairTable.SignedLag_days / 365.25;
PairTable.PatientID        = categorical(PairTable.Patient);
PairTable.EEG_ID           = categorical(string(PairTable.Patient) + "_" + ...
                                         string(PairTable.Session));
end


function MMR = fit_mixed_effects_models(PairTable, CFG)
%FIT_MIXED_EFFECTS_MODELS  Primary logistic GLME and its no-interaction twin.
%
%   Outcome   HasSz_bin, whether a seizure was reported at that clinic visit
%   Exposure  LogSpikesPerHour, from the paired EEG
%   Moderator AbsLag_years (how far apart) and VisitAfterEEG (which side)
%   Random    patient intercept
%
%   M1 (primary)  spike rate x |lag|  +  spike rate x direction  +  subtype
%   M2            same terms, no interactions
%   LRT           M1 vs M2, chi^2 on 2 df, tests both interactions jointly
%
% Reference category is Temporal, so the subtype coefficients read as
% "Frontal vs Temporal" and "Generalized vs Temporal".

canonical3 = ["Frontal","General","Temporal"];

keep = ismember(string(PairTable.EpiType3), canonical3) & ...
       isfinite(PairTable.LogSpikesPerHour) & ...
       isfinite(PairTable.SignedLag_years)  & ...
       isfinite(PairTable.HasSz);
T = PairTable(keep, :);

T.HasSz_bin     = double(T.HasSz == 1);
T.AbsLag_years  = abs(T.SignedLag_years);
T.VisitAfterEEG = double(T.SignedLag_years >= 0);   % 1 = visit at/after EEG
T.EpiType3_cat  = reordercats(categorical(string(T.EpiType3), canonical3), ...
                              ["Temporal","Frontal","General"]);

fprintf('[Model] %d pairs, %d patients\n', height(T), numel(unique(T.Patient)));

formula_M1 = ['HasSz_bin ~ LogSpikesPerHour * AbsLag_years + ' ...
              'LogSpikesPerHour * VisitAfterEEG + EpiType3_cat + (1|PatientID)'];
formula_M2 = ['HasSz_bin ~ LogSpikesPerHour + AbsLag_years + VisitAfterEEG + ' ...
              'EpiType3_cat + (1|PatientID)'];

glme_opts = {'Distribution','Binomial','Link','logit', ...
             'FitMethod','Laplace','CovariancePattern','Diagonal'};

fprintf('\nFitting M1 (interactions)...\n');
mdl_M1 = fitglme(T, formula_M1, glme_opts{:});
disp(mdl_M1);

fprintf('\nFitting M2 (no interactions)...\n');
mdl_M2 = fitglme(T, formula_M2, glme_opts{:});
disp(mdl_M2);

% --- Fixed effects (Wald / Laplace) ---
[b1, bn1, s1] = fixedEffects(mdl_M1, 'Alpha', CFG.alpha);
[b2, bn2, s2] = fixedEffects(mdl_M2, 'Alpha', CFG.alpha);
FE_M1 = make_fe_table_logistic(bn1, b1, s1);
FE_M2 = make_fe_table_logistic(bn2, b2, s2);
fprintf('\nM1 fixed effects:\n'); disp(FE_M1);
fprintf('\nM2 fixed effects:\n'); disp(FE_M2);

% --- LRT on the two interaction terms ---
fprintf('\n=== LRT: M1 vs M2 (both interactions jointly) ===\n');
lrt = compare(mdl_M2, mdl_M1);
disp(lrt);

% --- Patient-level bootstrap (the CIs actually reported) ---
[BT1, betas1, nConv1] = run_bootstrap(T, mdl_M1, formula_M1, CFG, 'M1');
[BT2, betas2, nConv2] = run_bootstrap(T, mdl_M2, formula_M2, CFG, 'M2');
if size(betas1,1) > 10
    plot_bootstrap_diagnostics(betas1, mdl_M1, 'M1');
end

MMR = struct( ...
    'ModelTable', T, ...
    'mdl_M1', mdl_M1, 'mdl_M2', mdl_M2, ...
    'FE_M1',  FE_M1,  'FE_M2',  FE_M2, ...
    'BootstrapTable1', BT1, 'BootstrapBetas1', betas1, ...
    'BootstrapTable2', BT2, 'BootstrapBetas2', betas2, ...
    'LRT', lrt, 'LRT_p', lrt.pValue(2), ...
    'BootstrapConvergence', struct( ...
        'M1_nConverged', nConv1, 'M1_nTotal', CFG.nBootModel, ...
        'M2_nConverged', nConv2, 'M2_nTotal', CFG.nBootModel));
end


function T_fe = make_fe_table_logistic(bn, b, s)
%MAKE_FE_TABLE_LOGISTIC  Fixed effects on both the logit and odds-ratio scale.
T_fe = table(string(bn.Name), b, s.SE, s.tStat, s.pValue, ...
    exp(b), exp(b - 1.96*s.SE), exp(b + 1.96*s.SE), ...
    'VariableNames',{'Term','Beta','SE','t','p','OR','OR_lo','OR_hi'});
end


function [T_boot, boot_betas, nConverged] = run_bootstrap(T, mdl, formula, CFG, label)
%RUN_BOOTSTRAP  Cluster bootstrap, resampling PATIENTS (not rows).
%
% Resampling whole patients preserves the within-patient correlation that the
% random intercept models. Each drawn patient is given a fresh ID so that a
% patient drawn twice contributes two independent clusters.
%
% p-values use the Phipson & Smyth (2010) (count + 1)/(B + 1) convention, so
% they can never be exactly zero.

nBoot      = CFG.nBootModel;
T_boot     = [];
boot_betas = [];
nConverged = 0;
if nBoot == 0 || isempty(mdl)
    % Draft mode. Every consumer of BootstrapTable1/2 already falls back to
    % the Laplace CIs when the table is empty.
    fprintf('Skipping %s bootstrap (nBootModel = 0); CIs fall back to Laplace.\n', label);
    return
end

patients   = unique(T.PatientID);
nPat       = numel(patients);
boot_betas = nan(nBoot, size(fixedEffects(mdl),1));

fprintf('\nBootstrapping %s (%d iterations)...\n', label, nBoot);
parfor b = 1:nBoot
    bootPats = patients(randi(nPat, nPat, 1));
    blocks   = cell(nPat, 1);
    for k = 1:nPat
        blk = T(T.PatientID == bootPats(k), :);
        blk.PatientID = categorical(repmat(k, height(blk), 1));
        blocks{k} = blk;
    end
    try
        m = fitglme(vertcat(blocks{:}), formula, 'Distribution','Binomial', ...
            'Link','logit','FitMethod','Laplace','CovariancePattern','Diagonal');
        boot_betas(b,:) = fixedEffects(m)';
    catch
        % Non-converged draws stay NaN and are dropped below.
    end
end

converged  = all(isfinite(boot_betas), 2);
boot_betas = boot_betas(converged, :);
nConverged = nnz(converged);
fprintf('%s bootstrap: %d/%d converged (%.1f%%)\n', ...
    label, nConverged, nBoot, 100*nConverged/nBoot);
assert(nConverged >= 0.90*nBoot, ...
    '%s bootstrap converged in only %.1f%% of draws; CIs are unreliable.', ...
    label, 100*nConverged/nBoot);

ci_lo = prctile(boot_betas, 100*(CFG.alpha/2),     1);
ci_hi = prctile(boot_betas, 100*(1-CFG.alpha/2),   1);
[beta_obs, names_obs] = fixedEffects(mdl);

boot_p = nan(numel(beta_obs),1);
for k = 1:numel(beta_obs)
    boot_p(k) = boot_p_two_sided(boot_betas(:,k), 0);
end

T_boot = table(string(names_obs.Name), beta_obs, ci_lo(:), ci_hi(:), ...
    exp(beta_obs), exp(ci_lo(:)), exp(ci_hi(:)), boot_p, ...
    'VariableNames',{'Term','Beta','Boot_CI_lo','Boot_CI_hi', ...
                     'OR','OR_CI_lo','OR_CI_hi','Boot_p'});
fprintf('%s bootstrapped ORs:\n', label); disp(T_boot);
end


function plot_bootstrap_diagnostics(boot_betas, mdl, label)
%PLOT_BOOTSTRAP_DIAGNOSTICS  One histogram per coefficient, with the observed
% value and the percentile interval marked.
[beta, names] = fixedEffects(mdl);
terms = string(names.Name);
nCols = ceil(sqrt(numel(terms)));
nRows = ceil(numel(terms)/nCols);

fig = figure('Color','w','Position',[100 100 300*nCols 250*nRows], ...
    'Name', sprintf('Bootstrap - %s', label));
tl = tiledlayout(fig, nRows, nCols, 'TileSpacing','compact','Padding','loose');
for k = 1:numel(terms)
    ax = nexttile(tl);
    histogram(ax, boot_betas(:,k), 40, 'FaceColor',[0.2 0.5 0.8], ...
        'EdgeColor','none','FaceAlpha',0.7);
    xline(ax, beta(k), 'r-', 'LineWidth',2);
    xline(ax, prctile(boot_betas(:,k),  2.5), 'k--', 'LineWidth',1.5);
    xline(ax, prctile(boot_betas(:,k), 97.5), 'k--', 'LineWidth',1.5);
    title(ax, terms(k), 'FontSize',9,'Interpreter','none');
    xlabel(ax, '\beta','FontSize',8); box(ax,'off');
end
sgtitle(fig, sprintf('Bootstrap - %s (red = observed, dashed = 95%% CI)', label), ...
    'FontSize',11);
end


function DurCompare = add_duration_to_model(MMR, Views, alpha)
%ADD_DURATION_TO_MODEL  Does EEG duration explain the spike-rate effect?
%
% Refits M1 with and without EEG duration on IDENTICAL rows (required for a
% valid LRT) and tests the 1-df improvement.

T = MMR.ModelTable;
Dur = Views.SessionLevelSpikeRates(:,{'Patient','Session','Duration_sec'});
Dur.Duration_sec = double(Dur.Duration_sec);

nBefore = height(T);
T = innerjoin(T, Dur, 'Keys', {'Patient','Session'});
assert(height(T) == nBefore, ...
    'Duration join changed row count (%d -> %d); duplicate or missing session keys.', ...
    nBefore, height(T));

T.EEG_DurationHours = T.Duration_sec / 3600;
assert(all(isfinite(T.EEG_DurationHours)), 'Non-finite EEG durations in the model table.');

base = ['HasSz_bin ~ LogSpikesPerHour * AbsLag_years + ' ...
        'LogSpikesPerHour * VisitAfterEEG + EpiType3_cat'];
glme_opts = {'Distribution','Binomial','Link','logit', ...
             'FitMethod','Laplace','CovariancePattern','Diagonal'};

mdl_reduced = fitglme(T, [base ' + (1|PatientID)'], glme_opts{:});
mdl_full    = fitglme(T, [base ' + EEG_DurationHours + (1|PatientID)'], glme_opts{:});

[b, bn, s] = fixedEffects(mdl_full, 'Alpha', alpha);
FE_full = make_fe_table_logistic(bn, b, s);

lrt   = compare(mdl_reduced, mdl_full);
lrt_p = lrt.pValue(2);
row   = FE_full(string(FE_full.Term) == "EEG_DurationHours", :);

fprintf(['\n[EEG duration] OR = %.3f [%.3f-%.3f] per hour, p = %.4g; ' ...
    'LRT (1 df) p = %.4g\n'], row.OR, row.OR_lo, row.OR_hi, row.p, lrt_p);

DurCompare = struct('ModelTable',T, 'mdl_reduced',mdl_reduced, 'mdl_full',mdl_full, ...
    'FE_full',FE_full, 'LRT',lrt, 'LRT_p',lrt_p, ...
    'Duration_OR',row.OR, 'Duration_OR_lo',row.OR_lo, ...
    'Duration_OR_hi',row.OR_hi, 'Duration_p',row.p);
end


%% #####################################################################
%% ##  FIGURES
%% #####################################################################

function FigFlow = make_flowchart_figure(Views, MMR)
%MAKE_FLOWCHART_FIGURE  STROBE-style participant flow diagram.

EC        = Views.ExclusionCounts;
n_subtype = numel(unique(MMR.ModelTable.Patient));
n_pairs   = height(MMR.ModelTable);

FigFlow = figure('Color','w','Position',[100 100 820 820]);
ax = axes('Position',[0 0 1 1]); axis(ax,'off'); hold(ax,'on');
xlim(ax,[0 1.15]); ylim(ax,[0 1]);

BOX_W = 0.52; BOX_H = 0.08; EXC_W = 0.34; EXC_H = 0.075;
CX    = 0.46; EXC_X = 0.76;
COL_MAIN = [0.22 0.45 0.70];
COL_EXC  = [0.80 0.30 0.10];
COL_SUB  = [0.15 0.55 0.40];

% Main column, top to bottom, and the midpoints where exclusions branch off.
y = [0.90 0.74 0.57 0.40 0.18];
ym = (y(1:end-1) + y(2:end)) / 2;

draw_box(CX, y(1), BOX_W, BOX_H, sprintf('All patients with EEG data\nN = %d', ...
    EC.nTotal), COL_MAIN, 15);
draw_box(CX, y(2), BOX_W, BOX_H, sprintf('Outpatient routine EEG <=4 hours\nN = %d', ...
    EC.nAfterOutptRoutine), COL_MAIN, 15);
draw_box(CX, y(3), BOX_W, BOX_H, sprintf('LLM-confirmed epilepsy diagnosis\nN = %d', ...
    EC.nAfterOutptRoutine - EC.nExcludedNoEpilepsy), COL_MAIN, 15);
draw_box(CX, y(4), BOX_W, BOX_H, sprintf('Documented seizure frequency\nN = %d (primary cohort)', ...
    EC.nFinalCohort), COL_SUB, 15);
draw_box(CX, y(5), BOX_W, BOX_H*1.5, sprintf( ...
    ['Known epilepsy subtype\n(temporal, frontal, generalized)\n' ...
     'for mixed effects model\nN = %d patients, %d EEG-visit pairs'], ...
    n_subtype, n_pairs), COL_SUB, 15);

draw_arrow_down(y(1)-BOX_H/2, y(2)+BOX_H/2);
draw_arrow_down(y(2)-BOX_H/2, y(3)+BOX_H/2);
draw_arrow_down(y(3)-BOX_H/2, y(4)+BOX_H/2);
draw_arrow_down(y(4)-BOX_H/2, y(5)+BOX_H*1.5/2);

% Exclusion labels state the full criterion, not just its most common cause.
excTxt = { ...
    sprintf('Excluded: not an outpatient\nroutine EEG <=4 hours\nN = %d', ...
        EC.nTotal - EC.nAfterOutptRoutine), ...
    sprintf('Excluded: no confirmed epilepsy\ndiagnosis (NESD, uncertain,\nor unknown)\nN = %d', ...
        EC.nExcludedNoEpilepsy), ...
    sprintf('Excluded: no documented\nseizure frequency at any\nclinic visit\nN = %d', ...
        EC.nExcludedNoSzFreq)};
for i = 1:3
    draw_arrow_right(ym(i));
    draw_box(EXC_X + EXC_W/2, ym(i), EXC_W, EXC_H, excTxt{i}, COL_EXC, 12);
end

text(ax, CX, 0.97, 'Study participant flow', 'HorizontalAlignment','center', ...
    'VerticalAlignment','top', 'FontSize',16, 'FontWeight','bold');

    function draw_box(cx, cy, w, h, txt, col, fsz)
        patch(ax, cx + [-w -w w w -w]/2, cy + [-h h h -h -h]/2, col, ...
            'FaceAlpha',0.25, 'EdgeColor',col, 'LineWidth',1.8);
        text(ax, cx, cy, txt, 'HorizontalAlignment','center', ...
            'VerticalAlignment','middle', 'FontSize',fsz, ...
            'Color',[0.1 0.1 0.1], 'Interpreter','none');
    end
    function draw_arrow_down(yTop, yBot)
        plot(ax, [CX CX], [yTop yBot], '-', 'Color',[0.3 0.3 0.3], 'LineWidth',1.4);
        fill(ax, CX + [-0.012 0.012 0], [yBot+0.018 yBot+0.018 yBot], ...
            [0.3 0.3 0.3], 'EdgeColor','none');
    end
    function draw_arrow_right(yMid)
        plot(ax, [CX EXC_X], [yMid yMid], '-', 'Color',COL_EXC, 'LineWidth',1.2);
        fill(ax, [EXC_X-0.018 EXC_X-0.018 EXC_X], yMid + [-0.012 0.012 0], ...
            COL_EXC, 'EdgeColor','none');
    end
end


function [f1, ControlStats] = make_fig_controls(Views, CFG)
%MAKE_FIG1_CONTROLS  Two sanity checks on the automated spike detector.
%
%   A  EEG level. Do detected spike rates track what the reading physician
%      reported?  (Wilcoxon rank-sum + Cliff's delta, present vs absent)
%   B  Patient level. Do spike rates differ across epilepsy subtypes?
%      (Kruskal-Wallis + Bonferroni-corrected pairwise rank-sums)
%
% REVIEW: panel A treats each EEG as independent, but patients can contribute
% several. If a reviewer presses on this, a patient-level version (each
% patient's median rate, split by whether any EEG reported spikes) is the
% natural companion; Fig S7 is already close to that.

Y_ZERO = log10(CFG.EPS_RATE);
Y_LIMS = CFG.Y_LIMS;

%% --- Panel A data ---
JoinA = innerjoin(Views.SessionLevelSpikeRates(:,{'Patient','Session','SpikesPerHour'}), ...
    resolve_reported_spike_status(Views.ReportForKeptSessions), 'Keys',{'Patient','Session'});
x_abs = JoinA.SpikesPerHour(JoinA.ReportStatus == "absent");
x_pre = JoinA.SpikesPerHour(JoinA.ReportStatus == "present");
x_abs = x_abs(isfinite(x_abs));
x_pre = x_pre(isfinite(x_pre));

pA      = ranksum(x_abs, x_pre, 'method','approx');
effectA = cliff_delta(x_pre, x_abs);       % > 0 means present exceeds absent
[med_abs, lo_abs, hi_abs] = bootstrap_median_ci(x_abs, CFG.nBoot, CFG.alpha);
[med_pre, lo_pre, hi_pre] = bootstrap_median_ci(x_pre, CFG.nBoot, CFG.alpha);

%% --- Panel B data (single source for plot, stats and labels) ---
Sub  = Views.Canonical3_SubsetTable;
cats = categories(Sub.EpiType4);
[p_kw, tbl_kw] = kruskalwallis(Sub.MeanSpikeRate_perHour, Sub.EpiType4, 'off');
eta2_kw = tbl_kw{2,2} / tbl_kw{end,2};

%% --- Draw ---
f1 = figure('Color','w','Position',[60 60 950 520]);
tiledlayout(f1,1,2,'TileSpacing','compact','Padding','loose');

axA = nexttile;
G_A = categorical([repmat("Absent",numel(x_abs),1); repmat("Present",numel(x_pre),1)]);
Y_A = jitter_at_floor(log10_floor([x_abs; x_pre], CFG.EPS_RATE), Y_ZERO, Y_LIMS, 0.02);
box_swarm_panel(axA, G_A, Y_A, Y_ZERO, Y_LIMS, CFG.EPS_RATE, 'Spikes/hour (log scale)');
add_median_ci_overlay(axA, 1, med_abs, lo_abs, hi_abs, CFG.EPS_RATE);
add_median_ci_overlay(axA, 2, med_pre, lo_pre, hi_pre, CFG.EPS_RATE);
add_sigbar(axA, 1, 2, Y_LIMS(2)-0.08*range(Y_LIMS), p_label(pA));
finish_panel(axA, 'A. Reported presence or absence of spikes', ...
    ["Absent","Present"], [numel(x_abs) numel(x_pre)], 20, 0.02);

axB = nexttile;
Y_B = jitter_at_floor(log10_floor(Sub.MeanSpikeRate_perHour, CFG.EPS_RATE), ...
    Y_ZERO, Y_LIMS, 0.02);
box_swarm_panel(axB, Sub.EpiType4, Y_B, Y_ZERO, Y_LIMS, CFG.EPS_RATE, ...
    'Spikes/hour (log scale)');

subMed = nan(numel(cats),1); subLo = subMed; subHi = subMed;
for k = 1:numel(cats)
    [subMed(k), subLo(k), subHi(k)] = bootstrap_median_ci( ...
        Sub.MeanSpikeRate_perHour(Sub.EpiType4 == cats{k}), CFG.nBoot, CFG.alpha);
    add_median_ci_overlay(axB, k, subMed(k), subLo(k), subHi(k), CFG.EPS_RATE);
end

% Pairwise bars, stacked downward from the top of the axis.
catList = categorical(string(cats));
yStep   = 0.08*range(Y_LIMS);
for i = 1:size(Views.Canonical3_Pairs,1)
    pval = Views.PvalsPairwiseBonf(i);
    if isnan(pval), continue; end
    x1 = find(catList == categorical(string(Views.Canonical3_Pairs(i,1))));
    x2 = find(catList == categorical(string(Views.Canonical3_Pairs(i,2))));
    add_sigbar(axB, x1, x2, Y_LIMS(2) - 0.05*range(Y_LIMS) - (i-1)*yStep, stars(pval));
end
finish_panel(axB, 'B. Epilepsy subtype', string(Views.Canonical3_Stats.EpiType4), ...
    Views.Canonical3_Stats.GroupCount, 20, 0.02);

%% --- Stats bundle ---
ControlStats = struct( ...
    'p_rankSum_A', pA, 'effectA_cliff', effectA, ...
    'm_pre', med_pre, 'lo_pre', lo_pre, 'hi_pre', hi_pre, ...
    'm_abs', med_abs, 'lo_abs', lo_abs, 'hi_abs', hi_abs, ...
    'p_kw_C', p_kw, 'eta2_kw_C', eta2_kw, ...
    'p_pair_bonf', Views.PvalsPairwiseBonf, ...
    'SubtypeStatsTable', table(string(cats), subMed, subLo, subHi, ...
        'VariableNames',{'Group','Median','CI_lo','CI_hi'}));
end


function SP = spearman_figure(SpikeSz_All, SpikeSz_Typed, CFG, fig_out, ...
    labelSuffix, nonZeroOnly)
%SPEARMAN_FIGURE  Spike rate vs seizure frequency, overall and by subtype.
%
%   A  all epilepsy patients
%   B-D  Frontal / Temporal / General
%
% Zeros are pinned half a step below the smallest positive value and drawn as
% asterisks. Fit lines use only doubly non-zero points in EVERY panel; the
% previous version fitted panel A on the pinned points too, which is why panel
% A's slope did not match the subtype panels.
%
% nonZeroOnly = true drops zero-rate / zero-frequency patients entirely.

nBoot = CFG.nBoot; alpha = CFG.alpha;
COL_ALL   = [0.45 0.45 0.45];
COL.Frontal  = [0.93 0.69 0.13];
COL.Temporal = [0.85 0.33 0.10];
COL.General  = [0.00 0.45 0.74];
panelOrder = ["Frontal","Temporal","General"];
panelTitle = ["B. Frontal","C. Temporal","D. General"];

%% --- All-patient correlation ---
x = double(SpikeSz_All.MeanSpikeRate_perHour);
y = double(SpikeSz_All.MeanSzFreq);
m = isfinite(x) & isfinite(y);
if nonZeroOnly, m = m & x > 0 & y > 0; end
[rho_all, lo_all, hi_all, p_all] = spearman_with_ci(x(m), y(m), nBoot, alpha);
n_all = nnz(m);

%% --- Subtype correlations (Bonferroni over the three subtypes) ---
Res = table('Size',[numel(CFG.canonical3) 7], ...
    'VariableTypes',{'string','double','double','double','double','double','double'}, ...
    'VariableNames',{'Group','N','Spearman_r','ci_lo','ci_hi','p_raw','p_bonf'});
for i = 1:numel(CFG.canonical3)
    g  = CFG.canonical3(i);
    mi = (SpikeSz_Typed.EpiType3 == g);
    xi = double(SpikeSz_Typed.MeanSpikeRate_perHour(mi));
    yi = double(SpikeSz_Typed.MeanSzFreq(mi));
    ki = isfinite(xi) & isfinite(yi);
    if nonZeroOnly, ki = ki & xi > 0 & yi > 0; end
    [r, lo, hi, p] = spearman_with_ci(xi(ki), yi(ki), nBoot, alpha);
    Res(i,:) = {g, nnz(ki), r, lo, hi, p, NaN};
end
Res.p_bonf = min(Res.p_raw * height(Res), 1);

%% --- Shared zero floors, from the all-patient data actually plotted ---
[eps_sz, eps_rate] = zero_floors(y(m), x(m));

%% --- Draw ---
fig = figure('Color','w','Position',[60 60 1200 900]);
tiledlayout(fig,2,2,'Padding','compact','TileSpacing','compact');

draw_spearman_panel(nexttile(1), y(m), x(m), COL_ALL, ...
    sprintf('A. All epilepsy%s (N=%d)', labelSuffix, n_all), ...
    sprintf('\\rho=%.2f [%.2f-%.2f], %s', rho_all, lo_all, hi_all, p_label(p_all)), ...
    CFG, eps_sz, eps_rate);

for k = 1:3
    g  = panelOrder(k);
    r  = Res(Res.Group == g, :);
    mi = (SpikeSz_Typed.EpiType3 == g) & ...
         isfinite(SpikeSz_Typed.MeanSpikeRate_perHour) & ...
         isfinite(SpikeSz_Typed.MeanSzFreq);
    if nonZeroOnly
        mi = mi & SpikeSz_Typed.MeanSpikeRate_perHour > 0 & SpikeSz_Typed.MeanSzFreq > 0;
    end
    ax = nexttile(k+1);
    if ~any(mi), axis(ax,'off'); continue; end
    draw_spearman_panel(ax, ...
        double(SpikeSz_Typed.MeanSzFreq(mi)), ...
        double(SpikeSz_Typed.MeanSpikeRate_perHour(mi)), COL.(g), ...
        sprintf('%s%s (N=%d)', panelTitle(k), labelSuffix, nnz(mi)), ...
        sprintf('\\rho=%.2f [%.2f-%.2f], p_{bonf}%s', r.Spearman_r, r.ci_lo, r.ci_hi, ...
            regexprep(char(p_label(r.p_bonf)), '^p', '')), ...
        CFG, eps_sz, eps_rate);
end

save_fig(fig, fig_out);

SP = struct('Results',Res, 'rho_all',rho_all, 'p_all',p_all, 'n_all',n_all, ...
    'ci_lo_all',lo_all, 'ci_hi_all',hi_all, 'Figure',fig);
end


function draw_spearman_panel(ax, xSz, ySpike, col, ttl, annot, CFG, eps_sz, eps_rate)
%DRAW_SPEARMAN_PANEL  One log-log scatter panel, shared by the main Spearman
% figure and the generalized-subtype figure.
FONT = 20;
hold(ax,'on'); grid(ax,'on'); box(ax,'off');

logX = log10(xSz    + (xSz    <= 0) .* eps_sz);
logY = log10(ySpike + (ySpike <= 0) .* eps_rate);
zx   = (xSz == 0);
zy   = (ySpike == 0);

xline(ax, log10(eps_sz),   ':', 'Color',[0.4 0.4 0.4], 'LineWidth',1.2);
yline(ax, log10(eps_rate), ':', 'Color',[0.4 0.4 0.4], 'LineWidth',1.2);

scatter(ax, logX(~zx & ~zy), logY(~zx & ~zy), 18, col, 'filled','MarkerFaceAlpha',0.35);
for sel = {zx & ~zy, ~zx & zy, zx & zy}
    if any(sel{1})
        plot(ax, logX(sel{1}), logY(sel{1}), '*', 'Color',col, ...
            'MarkerSize',8, 'LineWidth',1.1);
    end
end

% Fit on doubly non-zero points only, in every panel.
nz = ~zx & ~zy;
if nnz(nz) >= 3
    b  = [ones(nnz(nz),1), logX(nz)] \ logY(nz);
    xg = linspace(CFG.spearman_xLim(1), CFG.spearman_xLim(2), 250)';
    plot(ax, xg, b(1) + b(2)*xg, '-', 'Color',col, 'LineWidth',2);
end

xlim(ax, CFG.spearman_xLim); ylim(ax, CFG.spearman_yLim);
xlabel(ax, 'Seizures per month (log scale)', 'FontSize',FONT);
ylabel(ax, 'Spikes per hour (log scale)',    'FontSize',FONT);
set_log10_ticks(ax, 'x', eps_sz,   CFG.spearman_xLim);
set_log10_ticks(ax, 'y', eps_rate, CFG.spearman_yLim);

% Blank the top x label so it does not collide with the neighbouring tile.
labs = string(ax.XTickLabel); [~, iMax] = max(ax.XTick); labs(iMax) = "";
ax.XTickLabel = labs;

title(ax, ttl, 'FontSize',FONT, 'FontWeight','bold');
text(ax, 0.98, 0.95, annot, 'Units','normalized', ...
    'HorizontalAlignment','right','VerticalAlignment','top', ...
    'FontSize',FONT-3, 'FontWeight','bold');
set(ax,'FontSize',FONT);
end


function FigMain = make_model_figure(MMR, outPath)
%MAKE_MODEL_FIGURE  The two panels of the primary model figure.
%
%   A  forest plot of M1 odds ratios (bootstrap CIs when available)
%   B  predicted P(seizure reported) vs spike rate, at three EEG-visit gaps
%
% Panel B and its OR annotations are drawn at the reference condition
% VisitAfterEEG = 1 and Temporal epilepsy, and the annotated OR now includes
% the direction interaction so that it matches the curve it labels.

FONT = 20;
mdl  = MMR.mdl_M1;
FigMain = figure('Color','w','Position',[60 60 1300 560]);
axC = axes('Position',[0.21 0.12 0.35 0.72]);
axD = axes('Position',[0.64 0.12 0.35 0.72]);

%% ---------------- Panel A: forest plot ----------------
hold(axC,'on'); box(axC,'off'); grid(axC,'on');

[beta, names, stats] = fixedEffects(mdl);
raw = string(names.Name);

useBoot = ~isempty(MMR.BootstrapTable1);
if useBoot
    ci_label = '95% Bootstrap CI';
    BT = MMR.BootstrapTable1;
    [OR, OR_lo, OR_hi, pvals] = deal(nan(numel(raw),1));
    for k = 1:numel(raw)
        r = BT(string(BT.Term) == raw(k), :);
        if isempty(r)   % should not happen; fall back rather than crash
            OR(k)    = exp(beta(k));
            OR_lo(k) = exp(beta(k) - 1.96*stats.SE(k));
            OR_hi(k) = exp(beta(k) + 1.96*stats.SE(k));
            pvals(k) = stats.pValue(k);
        else
            OR(k) = r.OR; OR_lo(k) = r.OR_CI_lo; OR_hi(k) = r.OR_CI_hi;
            pvals(k) = r.Boot_p;
        end
    end
else
    ci_label = '95% Laplace CI';
    OR    = exp(beta);
    OR_lo = exp(beta - 1.96*stats.SE);
    OR_hi = exp(beta + 1.96*stats.SE);
    pvals = stats.pValue;
end

disp_names = pretty_term_names(raw);
keep       = (disp_names ~= "Intercept");     % intercept is not informative here
OR = OR(keep); OR_lo = OR_lo(keep); OR_hi = OR_hi(keep);
pvals = pvals(keep); disp_names = disp_names(keep);

nTerms = numel(OR);
for k = 1:nTerms
    idx = nTerms - k + 1;                     % draw first term at the top
    col = [0.1 0.3 0.7]; if pvals(idx) >= 0.05, col = [0.6 0.6 0.6]; end
    plot(axC, [OR_lo(idx) OR_hi(idx)], [k k], '-', 'Color',col, 'LineWidth',2.5);
    scatter(axC, OR(idx), k, 100, col, 'filled');
    text(axC, OR_hi(idx)+0.005, k, p_text(pvals(idx)), ...
        'FontSize',FONT-5, 'VerticalAlignment','middle');
end
xline(axC, 1, 'k--', 'LineWidth',1.5);

set(axC, 'YTick',1:nTerms, 'YTickLabel',flipud(disp_names), 'FontSize',FONT-4);
xlabel(axC, sprintf('Odds Ratio (%s)', ci_label), 'FontSize',FONT);
xpad = 0.05 * (max(OR_hi) - min(OR_lo));
xlim(axC, [min(OR_lo)-xpad, max(OR_hi)+xpad]);
th = title(axC, {'A. Spike rate, epilepsy type, and EEG-visit gap', ...
    'predict seizure occurrence'}, 'FontSize',FONT, 'FontWeight','bold');
th.Units = 'normalized';
th.Position(1:2) = th.Position(1:2) + [-0.13 0.02];

%% ---------------- Panel B: predicted probability ----------------
hold(axD,'on'); box(axD,'off'); grid(axD,'on');

B = coef_struct(mdl);
EPS_SPIKE  = 1e-3;                       % must match build_eeg_visit_pairs
spike_grid = linspace(0, 50, 200);
log_grid   = log(spike_grid + EPS_SPIKE);
lag_vals   = [0.5 2 4];
lag_labels = ["6 months","2 years","4 years"];
lag_colors = [0.05 0.30 0.70; 0.15 0.50 0.80; 0.40 0.65 0.85];
dir_val    = 1;                          % curves are for visits after the EEG

for k = 1:numel(lag_vals)
    eta = B.intercept + B.spike.*log_grid + B.abslag*lag_vals(k) + B.dir*dir_val + ...
          B.int_lag.*log_grid*lag_vals(k) + B.int_dir.*log_grid*dir_val;
    plot(axD, spike_grid, 1./(1+exp(-eta)), '-', 'Color',lag_colors(k,:), ...
        'LineWidth',2.5, 'DisplayName',lag_labels(k));
end

x_anno = 22;
for k = 1:numel(lag_vals)
    % OR per unit of log spike rate, at this lag AND this direction.
    or_k  = exp(B.spike + B.int_lag*lag_vals(k) + B.int_dir*dir_val);
    lx    = log(x_anno + EPS_SPIKE);
    eta_k = B.intercept + B.spike*lx + B.abslag*lag_vals(k) + B.dir*dir_val + ...
            B.int_lag*lx*lag_vals(k) + B.int_dir*lx*dir_val;
    text(axD, x_anno+1, 1/(1+exp(-eta_k))+0.01, sprintf('OR=%.2f', or_k), ...
        'FontSize',FONT-7, 'Color',lag_colors(k,:), ...
        'VerticalAlignment','middle', 'HorizontalAlignment','left');
end

xlabel(axD, 'Spike rate (spikes/hour)', 'FontSize',FONT);
ylabel(axD, 'P(seizure reported at visit)', 'FontSize',FONT);
xlim(axD,[0 30]); ylim(axD,[0.3 0.61]);
lg = legend(axD, 'Location','southeast', 'FontSize',FONT-6);
title(lg, 'EEG-visit lag');
th = title(axD, {'B. Spike rates are most predictive', ...
    'when EEG is obtained close to the visit'}, 'FontSize',FONT, 'FontWeight','bold');
th.Units = 'normalized'; th.Position(2) = th.Position(2) + 0.02;
set(axD, 'FontSize',FONT);

save_fig(FigMain, outPath);
end


function FigSup = make_figSup_lag(MMR, Vuniq, CFG, outPath)
%MAKE_FIGSUP_LAG  Context for the lag interaction.
%
%   A  seizure burden over calendar time (proportion reporting seizures, and
%      median seizure frequency), for the model cohort
%   B  distribution of the absolute EEG-visit gap
%
% NOTE: "years after first visit" is measured from the de-identification
% anchor 2000-01-01, which is the shifted date of each patient's first visit.

FONT = 20;
refDate = datetime(2000,1,1);
FigSup = figure('Color','w','Position',[60 60 1300 560]);
axA = axes('Position',[0.07 0.12 0.38 0.72]);
axB = axes('Position',[0.57 0.12 0.38 0.72]);

%% ---------------- Panel A ----------------
V = Vuniq(ismember(Vuniq.Patient, unique(MMR.ModelTable.Patient)) & ...
          (Vuniq.HasSz == 0 | Vuniq.HasSz == 1), :);
V.Years = days(V.VisitDate - refDate)/365.25;

edges   = 0:4;
centers = (edges(1:end-1) + edges(2:end))/2;
nB      = numel(centers);
[prop, propLo, propHi, medF, medFLo, medFHi] = deal(nan(nB,1));
nPerBin = zeros(nB,1);

for b = 1:nB
    inBin = V.Years >= edges(b) & V.Years < edges(b+1);
    hz = V.HasSz(inBin);   hz = hz(isfinite(hz));
    fr = V.Freq_R1(inBin); fr = fr(isfinite(fr));
    if numel(hz) >= 10
        prop(b) = mean(hz); nPerBin(b) = numel(hz);
        [~, propLo(b), propHi(b)] = bootstrap_stat_ci(hz, @mean, CFG.nBoot, CFG.alpha);
    end
    if numel(fr) >= 10
        [medF(b), medFLo(b), medFHi(b)] = bootstrap_stat_ci(fr, ...
            @(z) median(z,'omitnan'), CFG.nBoot, CFG.alpha);
    end
end

COL_PROP = [0.8 0.3 0.1];
COL_FREQ = [0.1 0.55 0.55];
okP = isfinite(prop); okF = isfinite(medF);

hold(axA,'on'); box(axA,'off'); grid(axA,'on');
yyaxis(axA,'left');
patch(axA, [centers(okP), fliplr(centers(okP))], ...
    [propLo(okP)', fliplr(propHi(okP)')], COL_PROP, 'FaceAlpha',0.2, 'EdgeColor','none');
plot(axA, centers(okP), prop(okP), 'o-', 'Color',COL_PROP, 'LineWidth',2, ...
    'MarkerFaceColor',COL_PROP, 'MarkerSize',6);
scatter(axA, centers(okP), prop(okP), nPerBin(okP)/5, COL_PROP, 'filled', ...
    'MarkerFaceAlpha',0.25);   % marker area encodes bin size
ylim(axA,[0 1]);
ylabel(axA, 'Proportion with seizures', 'FontSize',FONT, 'Color',COL_PROP);
axA.YAxis(1).Color = COL_PROP;

yyaxis(axA,'right');
Y_FREQ = [-2 2];
patch(axA, [centers(okF), fliplr(centers(okF))], ...
    [log10_floor(medFLo(okF), CFG.EPS_FREQ)', ...
     fliplr(log10_floor(medFHi(okF), CFG.EPS_FREQ)')], ...
    COL_FREQ, 'FaceAlpha',0.18, 'EdgeColor','none');
plot(axA, centers(okF), log10_floor(medF(okF), CFG.EPS_FREQ), 's--', ...
    'Color',COL_FREQ, 'LineWidth',1.8, 'MarkerFaceColor',COL_FREQ, 'MarkerSize',5);
ylim(axA, Y_FREQ);
set_log10_ticks(axA, 'y', CFG.EPS_FREQ, Y_FREQ);
ylabel(axA, 'Median sz/month (log scale)', 'FontSize',FONT, 'Color',COL_FREQ);
axA.YAxis(2).Color = COL_FREQ;

xlabel(axA, 'Years after first visit', 'FontSize',FONT);
th = title(axA, 'A. Seizure burden tends to decrease over time', ...
    'FontSize',FONT, 'FontWeight','bold');
th.Units = 'normalized'; th.Position(2) = th.Position(2) + 0.02;
set(axA,'FontSize',FONT);

%% ---------------- Panel B ----------------
hold(axB,'on'); box(axB,'off'); grid(axB,'on');
histogram(axB, MMR.ModelTable.AbsLag_years, 40, 'FaceColor',[0.3 0.3 0.3], ...
    'FaceAlpha',0.6, 'EdgeColor','none', 'Normalization','probability');
xline(axB, 1, 'k--', 'LineWidth',1.2);
xlabel(axB, 'Absolute EEG-visit gap (years)', 'FontSize',FONT);
ylabel(axB, 'Proportion of pairs', 'FontSize',FONT);
th = title(axB, 'B. EEG and visit are often separated by years', ...
    'FontSize',FONT, 'FontWeight','bold');
th.Units = 'normalized'; th.Position(2) = th.Position(2) + 0.02;
set(axB,'FontSize',FONT);

save_fig(FigSup, outPath);
end


function fig = make_fig_szfreq_by_reported_spikes(Views, SzFreq, CFG)
%MAKE_FIGS2_SZ_BY_REPORTED_SPIKES  Patient-level companion to Fig 1A.
%
% Splits patients by whether ANY of their EEGs had clinically reported spikes,
% then compares mean seizure frequency.

RS = resolve_reported_spike_status(Views.ReportForKeptSessions);
[g, pid] = findgroups(RS.Patient);
Rpt = table(pid, ...
    splitapply(@(x) any(x == "present"), RS.ReportStatus, g), ...
    splitapply(@(x) any(x == "absent"),  RS.ReportStatus, g), ...
    'VariableNames',{'Patient','HasPresent','HasAbsent'});

S2 = innerjoin(SzFreq, Rpt, 'Keys','Patient');
S2 = innerjoin(S2, table(Views.PatientLevelSpikeRates.Patient, ...
    'VariableNames',{'Patient'}), 'Keys','Patient');

freqAbsent  = S2.MeanSzFreq(S2.HasAbsent & ~S2.HasPresent);
freqPresent = S2.MeanSzFreq(S2.HasPresent);
freqAbsent  = freqAbsent(isfinite(freqAbsent));
freqPresent = freqPresent(isfinite(freqPresent));

p = ranksum(freqAbsent, freqPresent, 'method','approx');
d = cliff_delta(freqPresent, freqAbsent);   % same sign convention as Fig 2A
[m1, lo1, hi1] = bootstrap_median_ci(freqAbsent,  CFG.nBoot, CFG.alpha);
[m2, lo2, hi2] = bootstrap_median_ci(freqPresent, CFG.nBoot, CFG.alpha);

Y_ZERO = log10(CFG.EPS_FREQ);
Y_LIMS = CFG.spearman_xLim;               % same decade range as the Spearman x axis
G = categorical([repmat("All EEGs: no spikes",   numel(freqAbsent),1); ...
                 repmat("At least 1 EEG: spikes",numel(freqPresent),1)]);
Y = jitter_at_floor(log10_floor([freqAbsent; freqPresent], CFG.EPS_FREQ), ...
    Y_ZERO, Y_LIMS, 0.02);

fig = figure('Color','w','Position',[100 100 800 520]);
ax  = axes(fig);
box_swarm_panel(ax, G, Y, Y_ZERO, Y_LIMS, CFG.EPS_FREQ, 'Seizures/month (log scale)');
add_median_ci_overlay(ax, 1, m1, lo1, hi1, CFG.EPS_FREQ);
add_median_ci_overlay(ax, 2, m2, lo2, hi2, CFG.EPS_FREQ);

% Make room for the significance bar above the highest point.
yBar = max(Y(isfinite(Y))) + 0.06*range(ylim(ax));
if yBar + 0.10*range(ylim(ax)) > max(ylim(ax))
    ylim(ax, [min(ylim(ax)), yBar + 0.10*range(ylim(ax))]);
end
add_sigbar(ax, 1, 2, yBar, p_label(p));

finish_panel(ax, 'Mean seizure frequency by reported spikes across EEGs', ...
    ["All EEGs: no spikes","At least 1 EEG: spikes"], ...
    [numel(freqAbsent) numel(freqPresent)], 20, 0.03);

fprintf(['\n[Fig S7] Median [95%% CI] seizure frequency: %.2f [%.2f-%.2f] (no spikes) ' ...
    'vs %.2f [%.2f-%.2f] (spikes present); %s, Cliff''s d = %.2f\n'], ...
    m1, lo1, hi1, m2, lo2, hi2, p_label(p), d);
end


function NearFar = plot_delta_rho_histogram(Views, Vuniq, Report, nearQ, farQ, CFG, outPng)
%PLOT_DELTA_RHO_HISTOGRAM  Is the spike-seizure correlation stronger when the
% EEG is close in time to the clinic visit?
%
% Each visit is labelled by its minimum |visit - EEG| gap. Visits in the lower
% tertile of that distribution are "short gap", the upper tertile "long gap".
% For each patient we compute a mean seizure frequency within each window and
% correlate both against that patient's mean spike rate; the statistic is the
% difference in Spearman rho, with a patient-level bootstrap.
%
% Only patients contributing BOTH a short-gap and a long-gap visit enter the
% comparison, so the two correlations are computed on the same people.

%% --- Base cohort and its spike rates ---
basePatients = unique(double(Views.PatientSpikeSz_All.Patient));
SpikeTbl = Views.PatientLevelSpikeRates(:,{'Patient','MeanSpikeRate_perHour'});
SpikeTbl.Patient = double(SpikeTbl.Patient);
SpikeTbl = SpikeTbl(ismember(SpikeTbl.Patient, basePatients), :);

%% --- Gap per visit, computed once and reused ---
gaps_all = compute_visit_eeg_gaps(Vuniq, Report);
finiteGaps = gaps_all(isfinite(gaps_all));
assert(~isempty(finiteGaps), 'No finite visit-EEG gaps found.');

nearDays = quantile(finiteGaps, nearQ);
farDays  = quantile(finiteGaps, farQ);

isNear = isfinite(gaps_all) & gaps_all <= nearDays;
isFar  = isfinite(gaps_all) & gaps_all >= farDays;
fprintf('[Near/Far] Short gap <= %.0f days (%d visits); long gap >= %.0f days (%d visits)\n', ...
    nearDays, nnz(isNear), farDays, nnz(isFar));

% Restrict to cohort patients and to visits with a documented frequency.
inCohort = ismember(Vuniq.Patient, basePatients) & isfinite(Vuniq.Freq_R1);
Vn = Vuniq(isNear & inCohort, :);
Vf = Vuniq(isFar  & inCohort, :);
pBoth = intersect(unique(Vn.Patient), unique(Vf.Patient));
fprintf('[Near/Far] %d/%d cohort patients have both a short-gap and a long-gap visit (%.1f%%)\n', ...
    numel(pBoth), numel(basePatients), 100*numel(pBoth)/numel(basePatients));

%% --- Descriptive checks: are long-gap visits simply later, or worse? ---
VnB = Vn(ismember(Vn.Patient, pBoth), :);
VfB = Vf(ismember(Vf.Patient, pBoth), :);
[gn, pN] = findgroups(VnB.Patient);
[gf, pF] = findgroups(VfB.Patient);

Tdesc = innerjoin( ...
    table(double(pN), splitapply(@median, VnB.VisitDate, gn), ...
          splitapply(@(x) mean(x,'omitnan'), VnB.Freq_R1, gn), ...
          'VariableNames',{'Patient','NearDate','NearFreq'}), ...
    table(double(pF), splitapply(@median, VfB.VisitDate, gf), ...
          splitapply(@(x) mean(x,'omitnan'), VfB.Freq_R1, gf), ...
          'VariableNames',{'Patient','FarDate','FarFreq'}), 'Keys','Patient');

dDays = days(Tdesc.FarDate - Tdesc.NearDate);
p_time = signrank(dDays, 0, 'method','approx');
p_sz   = signrank(Tdesc.NearFreq, Tdesc.FarFreq, 'method','approx');
fprintf(['[Near/Far] Long-gap visits are %.0f days later on median (%s); ' ...
    'near vs far seizure frequency %.2f vs %.2f (%s)\n'], ...
    median(dDays,'omitnan'), p_label(p_time), ...
    median(Tdesc.NearFreq,'omitnan'), median(Tdesc.FarFreq,'omitnan'), p_label(p_sz));

%% --- Paired correlations ---
Sn = renamevars(build_patient_seizure_metrics(Vn), "MeanSzFreq", "SzNear");
Sf = renamevars(build_patient_seizure_metrics(Vf), "MeanSzFreq", "SzFar");
J  = innerjoin(innerjoin(SpikeTbl, Sn, 'Keys','Patient'), Sf, 'Keys','Patient');
J  = J(isfinite(J.MeanSpikeRate_perHour) & isfinite(J.SzNear) & isfinite(J.SzFar), :);

n = height(J);
assert(n >= 3, 'Only %d patients have both near and far seizure metrics.', n);

x  = J.MeanSpikeRate_perHour;
rho_near = corr(x, J.SzNear, 'Type','Spearman', 'Rows','complete');
rho_far  = corr(x, J.SzFar,  'Type','Spearman', 'Rows','complete');
delta_obs = rho_near - rho_far;

delta = nan(CFG.nBoot,1);
for b = 1:CFG.nBoot
    idx = randi(n, n, 1);
    delta(b) = corr(x(idx), J.SzNear(idx), 'Type','Spearman','Rows','complete') - ...
               corr(x(idx), J.SzFar(idx),  'Type','Spearman','Rows','complete');
end
ci_lo = prctile(delta, 100*(CFG.alpha/2));
ci_hi = prctile(delta, 100*(1-CFG.alpha/2));
p_one = boot_p_one_sided(delta, 0);          % H1: short gap > long gap
p_two = boot_p_two_sided(delta, 0);

%% --- Draw ---
fig = figure('Color','w','Position',[120 80 950 780]);
tl = tiledlayout(fig, 2, 1, 'TileSpacing','compact','Padding','compact');

ax1 = nexttile(tl,1); hold(ax1,'on'); box(ax1,'off'); grid(ax1,'on');
h = histogram(ax1, finiteGaps, 60, 'EdgeColor','none');
xlabel(ax1, '|Visit - EEG| gap (days)', 'FontSize',20);
ylabel(ax1, 'Visit count', 'FontSize',20);

xL = min(h.BinEdges); xU = max(h.BinEdges);
yl = ylim(ax1);
ylim(ax1, [yl(1), yl(1) + 1.3*diff(yl)]);    % headroom for the region labels
yl = ylim(ax1);

shade = @(xa, xb, a) patch(ax1, [xa xb xb xa], [yl(1) yl(1) yl(2) yl(2)], ...
    [0 0 0], 'FaceAlpha',a, 'EdgeColor','none');
hShade = [shade(xL, nearDays, 0.12), shade(nearDays, farDays, 0.06), ...
          shade(farDays, xU, 0.12)];
uistack(hShade, 'bottom'); uistack(h, 'top');

xline(ax1, nearDays, 'k--', 'LineWidth',2);
xline(ax1, farDays,  'k--', 'LineWidth',2);
text(ax1, mean([xL nearDays]), yl(2)*0.99, sprintf("Short gap\n(lower third)"), ...
    'HorizontalAlignment','center','VerticalAlignment','top','FontSize',20);
text(ax1, mean([farDays xU]), yl(2)*0.99, sprintf("Long gap\n(upper third)"), ...
    'HorizontalAlignment','center','VerticalAlignment','top','FontSize',20);
title(ax1, 'A. Visit-EEG gap distribution with lower and upper third cutoffs', ...
    'FontSize',20, 'Interpreter','none');

ax2 = nexttile(tl,2); hold(ax2,'on'); box(ax2,'off'); grid(ax2,'on');
histogram(ax2, delta, 40, 'EdgeColor','none');
xline(ax2, 0, 'k--', 'LineWidth',2);
xline(ax2, median(delta,'omitnan'), 'k-', 'LineWidth',2);
maxAbs = max(abs(delta(isfinite(delta))));
if isempty(maxAbs) || ~isfinite(maxAbs) || maxAbs == 0, maxAbs = 1e-3; end
xlim(ax2, [-1 1]*maxAbs*1.08);               % symmetric about the null
xlabel(ax2, '\Delta\rho = \rho_{short gap} - \rho_{long gap}');
ylabel(ax2, 'Bootstrap count');
title(ax2, sprintf(['B. Distribution of differences in spike-seizure correlation\n' ...
    'between short and long visit-EEG gaps\n95%% CI [%.3f, %.3f], p = %.3g'], ...
    ci_lo, ci_hi, p_one), 'FontSize',20, 'Interpreter','tex');
set([ax1 ax2], 'FontSize',20);

save_fig(fig, outPng);

fprintf(['\n[Fig S9] N = %d patients; rho short gap = %.2f, long gap = %.2f; ' ...
    'delta = %.3f [%.2f-%.2f], one-sided p = %.4f\n'], ...
    n, rho_near, rho_far, delta_obs, ci_lo, ci_hi, p_one);

NearFar = struct('nPatients',n, 'nearQ',nearQ, 'farQ',farQ, ...
    'nearDays',nearDays, 'farDays',farDays, ...
    'rho_near',rho_near, 'rho_far',rho_far, 'delta_obs',delta_obs, ...
    'delta_boot',delta, 'delta_median',median(delta,'omitnan'), ...
    'delta_ci_lo',ci_lo, 'delta_ci_hi',ci_hi, ...
    'p_one_sided',p_one, 'p_two_sided',p_two, 'tableUsed',J);
end


function gaps = compute_visit_eeg_gaps(Vuniq, Report)
%COMPUTE_VISIT_EEG_GAPS  Minimum |visit - EEG| in days, one value per row of
% Vuniq (same row order). NaN where the patient has no dated EEG.

if ismember("Patient", string(Report.Properties.VariableNames))
    pid = double(Report.Patient);
else
    pid = double(Report.patient_id);
end
require_cols(Report, "start_time_deid", "Report");

EEG_raw = Report.start_time_deid;
if isdatetime(EEG_raw)
    EEG_dt = EEG_raw;
else
    EEG_dt = datetime(strtrim(string(EEG_raw)), 'InputFormat',"yyyy-MM-dd'T'HH:mm:ss");
end
ok = ~isnat(EEG_dt) & isfinite(pid);
assert(any(ok), 'No parseable EEG dates in start_time_deid.');
EEG_pid = pid(ok); EEG_dt = EEG_dt(ok);

gaps = nan(height(Vuniq),1);
[g, pidV] = findgroups(Vuniq.Patient);
for k = 1:numel(pidV)
    rows = find(g == k);
    e = EEG_dt(EEG_pid == pidV(k));
    if isempty(e), continue; end
    for j = rows'
        gaps(j) = min(abs(days(e - Vuniq.VisitDate(j))));
    end
end
end


function [figH, DurStats] = make_eeg_duration_histogram(Views, outPath)
%MAKE_EEG_DURATION_HISTOGRAM  Fig S2. File vs Natus duration for cohort EEGs.
%
% The file duration is the full EDF span; the Natus duration clips segments
% recorded before the electrodes were connected. The difference between them
% is the "clipped" time that motivates the deadtime correction.
%
% DurStats feeds the EEG-duration row of Table 1 and the cohort paragraph of
% the results HTML, so those two can never disagree with this figure.

FONT = 20;
COL_FILE  = [0.22 0.45 0.70];
COL_NATUS = [0.85 0.33 0.10];

Sess = Views.SessionsForFigures;
require_cols(Sess, ["Patient","Session","Duration_sec"], "SessionsForFigures");
assert_unique_keys(Sess, "Patient","Session", "SessionsForFigures");

Rep = Views.ReportForKeptSessions;
require_cols(Rep, ["Patient","Session","duration_hms"], "ReportForKeptSessions");
assert_unique_keys(Rep, "Patient","Session", "ReportForKeptSessions");

D = innerjoin( ...
    table(double(Sess.Patient), double(Sess.Session), double(Sess.Duration_sec)/60, ...
        'VariableNames',{'Patient','Session','FileMin'}), ...
    table(double(Rep.Patient), double(Rep.Session), minutes(Rep.duration_hms), ...
        'VariableNames',{'Patient','Session','NatusMin'}), ...
    'Keys',{'Patient','Session'});
assert(height(D) == height(Sess), ...
    'Join dropped EEGs: %d session rows vs %d matched.', height(Sess), height(D));

fileMin  = D.FileMin(isfinite(D.FileMin));
natusMin = D.NatusMin(isfinite(D.NatusMin));
assert(~isempty(fileMin) && ~isempty(natusMin), 'No finite durations found.');

edges = 0 : 5 : ceil(max([fileMin; natusMin])/5)*5;

figH = figure('Color','w','Position',[100 100 850 540]);
ax = axes(figH); hold(ax,'on'); box(ax,'off'); grid(ax,'on');
histogram(ax, fileMin,  'BinEdges',edges, 'FaceColor',COL_FILE, 'FaceAlpha',0.5, ...
    'EdgeColor','none', 'DisplayName',sprintf('File duration (N=%d)', numel(fileMin)));
histogram(ax, natusMin, 'BinEdges',edges, 'FaceColor',COL_NATUS,'FaceAlpha',0.5, ...
    'EdgeColor','none', 'DisplayName',sprintf('Natus duration (N=%d)', numel(natusMin)));
xline(ax, median(fileMin),  '--', 'Color',COL_FILE,  'LineWidth',2, 'HandleVisibility','off');
xline(ax, median(natusMin), '--', 'Color',COL_NATUS, 'LineWidth',2, 'HandleVisibility','off');

xlabel(ax, 'EEG duration (minutes)', 'FontSize',FONT);
ylabel(ax, 'Number of EEGs', 'FontSize',FONT);
title(ax, sprintf('EEG durations in study cohort (N = %d EEGs)', height(D)), ...
    'FontSize',FONT, 'FontWeight','bold');
legend(ax, 'Location','northeast', 'FontSize',FONT-6);
set(ax,'FontSize',FONT);

paired = isfinite(D.FileMin) & isfinite(D.NatusMin);
clip   = D.FileMin(paired) - D.NatusMin(paired);
fprintf(['[Durations] File median %.1f (IQR %.1f-%.1f) min; ' ...
    'Natus median %.1f (IQR %.1f-%.1f) min; ' ...
    'clipped median %.1f (IQR %.1f-%.1f) min over %d EEGs\n'], ...
    median(fileMin),  prctile(fileMin,25),  prctile(fileMin,75), ...
    median(natusMin), prctile(natusMin,25), prctile(natusMin,75), ...
    median(clip), prctile(clip,25), prctile(clip,75), nnz(paired));

DurStats = struct( ...
    'nEEG',        height(D), ...
    'file_med',    median(fileMin),  'file_q', prctile(fileMin,  [25 75]), ...
    'natus_med',   median(natusMin), 'natus_q',prctile(natusMin, [25 75]), ...
    'clip_med',    median(clip),     'clip_q', prctile(clip,     [25 75]));

save_fig(figH, outPath);
end


function [FigBias, BiasStats] = make_fig_report_bias(Views, SzFreq, CFG, outPath)
%MAKE_FIG_REPORT_BIAS  Are EEGs WITHOUT a usable spike report different?
%
%   A  spike rate           B  seizure frequency      C  epilepsy type
%
% REVIEW: all three panels are at the EEG level while B and C carry
% patient-level values, so a patient with many EEGs is counted many times.
% This is fine as a descriptive check but the p-values are anticonservative;
% consider reporting them as descriptive only, or repeating at patient level.

FONT   = 18;
grpNames  = ["Report available","No report"];
canon3    = CFG.canonical3;

E = innerjoin(resolve_reported_spike_status(Views.ReportForKeptSessions), ...
    Views.SessionLevelSpikeRates(:,{'Patient','Session','SpikesPerHour'}), ...
    'Keys',{'Patient','Session'});
E = innerjoin(E, SzFreq(:,{'Patient','MeanSzFreq'}), 'Keys','Patient');
E = innerjoin(E, Views.PatientLevelSpikeRates(:,{'Patient','EpiType3'}), 'Keys','Patient');
assert_unique_keys(E, "Patient","Session", "report-bias table");

hasRep = (E.ReportStatus == "present") | (E.ReportStatus == "absent");
E.Grp  = categorical(repmat(grpNames(2), height(E),1), grpNames);
E.Grp(hasRep) = grpNames(1);
fprintf('[Report bias] %d EEGs with a report, %d without (of %d)\n', ...
    nnz(hasRep), nnz(~hasRep), height(E));

FigBias = figure('Color','w','Position',[60 60 1550 520]);
tiledlayout(FigBias, 1, 3, 'TileSpacing','compact','Padding','compact');

sA = two_group_panel(nexttile(1), E.SpikesPerHour, E.Grp, grpNames, ...
    CFG, CFG.EPS_RATE, 'Spikes/hour (log scale)', 'A. Spike rate', FONT);
sB = two_group_panel(nexttile(2), E.MeanSzFreq, E.Grp, grpNames, ...
    CFG, CFG.EPS_FREQ, 'Seizures/month (log scale)', 'B. Seizure frequency', FONT);

% --- Panel C: 2 x 3 contingency, drawn row-normalised ---
axC = nexttile(3);
Et  = E(ismember(string(E.EpiType3), canon3) & ~ismissing(E.EpiType3), :);
O   = zeros(2,3);
for r = 1:2
    for c = 1:3
        O(r,c) = nnz(Et.Grp == grpNames(r) & string(Et.EpiType3) == canon3(c));
    end
end
[chi2, p_chi2, df, Eexp] = chi2_contingency(O);
rowProp = O ./ max(sum(O,2), 1);

imagesc(axC, rowProp); hold(axC,'on');
colormap(axC, flipud(gray)); clim(axC, [0 1]);
for r = 1:2
    for c = 1:3
        txtcol = [0 0 0]; if rowProp(r,c) > 0.6, txtcol = [1 1 1]; end
        text(axC, c, r, sprintf('%d\n(%.0f%%)', O(r,c), 100*rowProp(r,c)), ...
            'HorizontalAlignment','center','VerticalAlignment','middle', ...
            'FontSize',FONT-3, 'Color',txtcol);
    end
end
axC.XTick = 1:3; axC.XTickLabel = canon3;
axC.YTick = 1:2; axC.YTickLabel = ["Report avail.","No report"];
xlabel(axC, 'Epilepsy type', 'FontSize',FONT);
warnstr = ''; if any(Eexp(:) < 5), warnstr = ' (expected<5)'; end
title(axC, sprintf('C. Epilepsy type\n\\chi^2(%d)=%.1f, %s%s', ...
    df, chi2, char(p_label(p_chi2)), warnstr), 'FontSize',FONT, 'FontWeight','bold');
set(axC,'FontSize',FONT); box(axC,'on'); axis(axC,'tight');

save_fig(FigBias, outPath);

BiasStats = struct('nWithReport',nnz(hasRep), 'nNoReport',nnz(~hasRep), ...
    'SpikeRate',sA, 'SzFreq',sB, ...
    'EpiType',struct('Counts',O,'Expected',Eexp,'chi2',chi2,'df',df,'p',p_chi2));
end


function stats = two_group_panel(ax, val, grp, grpNames, CFG, epsFloor, ylab, ttl, FONT)
%TWO_GROUP_PANEL  Box + swarm + bootstrap median CI for a two-group comparison.
x1 = val(grp == grpNames(1)); x1 = x1(isfinite(x1));
x2 = val(grp == grpNames(2)); x2 = x2(isfinite(x2));
p  = NaN; d = NaN;
if ~isempty(x1) && ~isempty(x2)
    p = ranksum(x1, x2, 'method','approx');
    d = cliff_delta(x1, x2);
end

Y   = log10_floor(val, epsFloor);
yz  = log10(epsFloor);
yl  = [min([yz; Y], [], 'omitnan')-0.3, max(Y,[],'omitnan')+0.7];
Yj  = jitter_at_floor(Y, yz, yl, 0.02);

box_swarm_panel(ax, grp, Yj, yz, yl, epsFloor, ylab);
for k = 1:2
    [m, lo, hi] = bootstrap_median_ci(val(grp == grpNames(k)), CFG.nBoot, CFG.alpha);
    add_median_ci_overlay(ax, k, m, lo, hi, epsFloor);
end
if isfinite(p), add_sigbar(ax, 1, 2, yl(2)-0.10*range(yl), p_label(p)); end
finish_panel(ax, ttl, grpNames, [numel(x1) numel(x2)], FONT, 0);
ax.XTickLabelRotation = 15;

stats = struct('p_ranksum',p, 'cliff_delta',d, 'n1',numel(x1), 'n2',numel(x2));
end


function [GenSub, fig] = make_fig_generalized_subtypes(Views, CFG, fig_out, minN, nonZeroOnly)
%MAKE_FIG_GENERALIZED_SUBTYPES  Spike-seizure correlation within generalized
% epilepsy, split by syndrome.
%
% Subtypes come from the free-text epilepsy_specific field, canonicalised by
% keyword. Anything below minN patients is pooled into "Other generalized".
% The raw-to-canonical mapping is printed so the keyword rules can be checked
% against the actual vocabulary before the figure is trusted.

%% --- Generalized-only patient table ---
D = innerjoin(Views.PatientSpikeSz_All(:,{'Patient','MeanSpikeRate_perHour','MeanSzFreq'}), ...
    Views.PatientLevelSpikeRates(:,{'Patient','EpiType3','EpilepsySpecific'}), ...
    'Keys','Patient');
D = D(~ismissing(D.EpiType3) & string(D.EpiType3) == "General", :);
D = D(isfinite(D.MeanSpikeRate_perHour) & isfinite(D.MeanSzFreq), :);
if nonZeroOnly
    D = D(D.MeanSpikeRate_perHour > 0 & D.MeanSzFreq > 0, :);
end
assert(~isempty(D), 'No generalized patients with both a spike rate and a seizure frequency.');
assert(numel(unique(D.Patient)) == height(D), 'Duplicate patients in the generalized table.');

D.Subtype = canonicalize_generalized_subtype(D.EpilepsySpecific);

% Inventory, so unmapped vocabulary is visible rather than silently pooled.
raw = strtrim(string(D.EpilepsySpecific));
raw(ismissing(raw) | raw == "") = "<blank>";
[gr, rawU] = findgroups(raw);
fprintf('\n[Generalized subtypes] epilepsy_specific -> canonical (N=%d patients)\n', height(D));
canonU = splitapply(@(s) s(1), D.Subtype, gr);
nRaw   = splitapply(@numel, D.Subtype, gr);
for i = 1:numel(rawU)
    fprintf('  %-45s -> %-26s N=%d\n', rawU(i), canonU(i), nRaw(i));
end

%% --- Pool rare subtypes, then drop panels still under minN ---
[g0, lab0] = findgroups(D.Subtype);
small = lab0(splitapply(@numel, D.Subtype, g0) < minN);
D.Subtype(ismember(D.Subtype, small)) = "Other generalized";

[g1, lab1] = findgroups(D.Subtype);
n1   = splitapply(@numel, D.Subtype, g1);
labs = lab1(n1 >= minN);
nPer = n1(n1 >= minN);
[nPer, ord] = sort(nPer, 'descend'); labs = labs(ord);

% Keep the catch-all buckets last regardless of size.
isLast = ismember(labs, ["Other generalized","Unspecified generalized"]);
labs = [labs(~isLast); labs(isLast)];
nPer = [nPer(~isLast); nPer(isLast)];
nG   = numel(labs);
assert(nG >= 1, 'No generalized subtype reaches minN=%d (largest N=%d).', minN, max(n1));

fprintf('[Generalized subtypes] %d panels at minN=%d; %d/%d patients shown\n', ...
    nG, minN, sum(nPer), height(D));

%% --- Correlations (Bonferroni over the plotted subtypes) ---
[rho_all, lo_all, hi_all, p_all] = spearman_with_ci( ...
    D.MeanSpikeRate_perHour, D.MeanSzFreq, CFG.nBoot, CFG.alpha);

[rho, lo, hi, p_raw] = deal(nan(nG,1));
for k = 1:nG
    m = (D.Subtype == labs(k));
    [rho(k), lo(k), hi(k), p_raw(k)] = spearman_with_ci( ...
        D.MeanSpikeRate_perHour(m), D.MeanSzFreq(m), CFG.nBoot, CFG.alpha);
end
p_bonf = min(p_raw * nG, 1);

GenSub = table(["All generalized"; labs(:)], [height(D); nPer(:)], ...
    [rho_all; rho], [lo_all; lo], [hi_all; hi], [p_all; p_raw], [NaN; p_bonf], ...
    'VariableNames',{'Group','N','Spearman_rho','ci_lo','ci_hi','p_raw','p_bonf'});

%% --- Draw ---
[eps_sz, eps_rate] = zero_floors(D.MeanSzFreq, D.MeanSpikeRate_perHour);
COL = [0.00 0.45 0.74; 0.85 0.33 0.10; 0.93 0.69 0.13; 0.49 0.18 0.56;
       0.47 0.67 0.19; 0.30 0.75 0.93; 0.64 0.08 0.18];

nCols = min(3, nG+1);
nRows = ceil((nG+1)/nCols);
fig = figure('Color','w','Position',[60 60 min(1800, 600*nCols) 460*nRows]);
tiledlayout(fig, nRows, nCols, 'Padding','compact','TileSpacing','compact');

draw_spearman_panel(nexttile(1), D.MeanSzFreq, D.MeanSpikeRate_perHour, ...
    [0.45 0.45 0.45], sprintf('A. All generalized (N=%d)', height(D)), ...
    sprintf('\\rho=%.2f [%.2f-%.2f], %s', rho_all, lo_all, hi_all, p_label(p_all)), ...
    CFG, eps_sz, eps_rate);

for k = 1:nG
    m = (D.Subtype == labs(k));
    draw_spearman_panel(nexttile(k+1), D.MeanSzFreq(m), D.MeanSpikeRate_perHour(m), ...
        COL(mod(k-1,size(COL,1))+1,:), ...
        sprintf('%s. %s (N=%d)', char('A'+k), labs(k), nPer(k)), ...
        sprintf('\\rho=%.2f [%.2f-%.2f], p_{bonf}%s', rho(k), lo(k), hi(k), ...
            regexprep(char(p_label(p_bonf(k))), '^p', '')), ...
        CFG, eps_sz, eps_rate);
end

save_fig(fig, fig_out);
end


function lab = canonicalize_generalized_subtype(specRaw)
%CANONICALIZE_GENERALIZED_SUBTYPE  Free-text epilepsy_specific -> syndrome label.
% Unrecognised text keeps its original wording so that it appears in the
% printed inventory rather than disappearing into an "other" bucket.
s   = lower(strtrim(string(specRaw)));
lab = strings(numel(s),1);
wordHit = @(t,w) ~isempty(regexp(t, ['\<' w '\>'], 'once'));

for i = 1:numel(s)
    t = s(i);
    if ismissing(t) || ismember(t, ["","null","[null]","<missing>"]) || ...
            contains(t,"unclassified") || contains(t,"unspecified") || contains(t,"unknown")
        lab(i) = "Unspecified generalized";
    elseif contains(t,"juvenile myoclonic") || wordHit(t,"jme")
        lab(i) = "Juvenile myoclonic";
    elseif contains(t,"childhood absence") || wordHit(t,"cae")
        lab(i) = "Childhood absence";
    elseif contains(t,"juvenile absence") || wordHit(t,"jae")
        lab(i) = "Juvenile absence";
    elseif contains(t,"myoclonic-atonic") || contains(t,"myoclonic atonic") || contains(t,"doose")
        lab(i) = "Myoclonic-atonic";
    elseif contains(t,"lennox")
        lab(i) = "Lennox-Gastaut";
    elseif contains(t,"tonic-clonic") || contains(t,"tonic clonic") || wordHit(t,"gtca")
        lab(i) = "GTC alone";
    elseif contains(t,"absence")
        lab(i) = "Absence, other";
    else
        lab(i) = strtrim(string(specRaw(i)));
    end
end
end


function SP = spearman_trim_top_spikers(Views, CFG, fig_out, trimFrac, nonZeroOnly)
%SPEARMAN_TRIM_TOP_SPIKERS  Reviewer sensitivity check: is the association
% driven by the highest-density patients? The cutoff is set on the primary
% all-epilepsy cohort and applied to both input tables.

All   = Views.PatientSpikeSz_All;
Typed = Views.PatientSpikeSz_Typed;
thr   = prctile(All.MeanSpikeRate_perHour, 100*(1-trimFrac));

keepAll   = All.MeanSpikeRate_perHour   <= thr;
keepTyped = Typed.MeanSpikeRate_perHour <= thr;
fprintf(['[Trim top %.0f%%] cutoff = %.2f spikes/hour; removed %d/%d all-epilepsy ' ...
    'and %d/%d subtype-typed patients.\n'], 100*trimFrac, thr, ...
    nnz(~keepAll), height(All), nnz(~keepTyped), height(Typed));

SP = spearman_figure(All(keepAll,:), Typed(keepTyped,:), CFG, fig_out, ...
    sprintf(' (top %.0f%% spikers removed)', 100*trimFrac), nonZeroOnly);
SP.trimInfo = struct('trimFrac',trimFrac, 'threshold',thr, ...
    'nRemovedAll',nnz(~keepAll), 'nRemovedTyped',nnz(~keepTyped));
end


%% #####################################################################
%% ##  TABLES AND HTML
%% #####################################################################

function Table1 = build_table1_flat(Views, SzFreq, Vuniq, CFG)
%BUILD_TABLE1_FLAT  Cohort characteristics, one Variable/Statistic row each.
%
% NOTE on age: dates are de-identified by shifting each patient's first visit
% to 2000-01-01, so age relative to that anchor IS age at first visit.

PL = Views.PatientLevelSpikeRates;
allPatients = PL.Patient;
N = numel(allPatients);
Rk = Views.ReportForKeptSessions;

%% --- Age at first visit ---
birth = strtrim(string(Rk.deid_birth_date));
miss  = ismember(birth, ["","null","[null]"]);
birth_dt = NaT(size(birth));
birth_dt(~miss) = datetime(birth(~miss), 'InputFormat','yyyy-MM-dd');
age = NaN(size(birth_dt));
age(~isnat(birth_dt)) = days(datetime(2000,1,1) - birth_dt(~isnat(birth_dt)))/365.25;
ageVec = per_patient(Rk.Patient, age, @min_omitnan, allPatients);

%% --- Sex (first non-missing value per patient) ---
sexVec = per_patient(Rk.Patient, upper(strtrim(string(Rk.nlp_gender))), ...
    @first_nonmissing, allPatients);
n_f = nnz(sexVec == "F"); n_m = nnz(sexVec == "M"); n_u = N - n_f - n_m;

%% --- Epilepsy subtype ---
E3    = strtrim(string(PL.EpiType3));
espec = strtrim(string(PL.EpilepsySpecific));
isCanon   = ismember(E3, CFG.canonical3);
isUnknown = ~isCanon & (ismissing(espec) | espec == "" | ...
    espec == "Unclassified or Unspecified" | espec == "Unknown or MRN not found");
n_temp = nnz(E3=="Temporal"); n_front = nnz(E3=="Frontal"); n_gen = nnz(E3=="General");
n_other = nnz(~isCanon & ~isUnknown); n_unk = nnz(isUnknown);

%% --- Per-patient counts, restricted to the cohort ---
Vc = Vuniq(ismember(Vuniq.Patient, allPatients), :);
assert(numel(unique(Vc.Patient)) == N, ...
    'Cohort visit table covers %d patients, expected %d.', numel(unique(Vc.Patient)), N);

visVec = per_patient(Vc.Patient, Vc.VisitDate, @(x) numel(unique(x)), allPatients);
fuVec  = per_patient(Vc.Patient, Vc.VisitDate, @(d) days(max(d)-min(d))/365.25, allPatients);
docVec = per_patient(Vc.Patient, Vc.Freq_R1, @(f) 100*mean(isfinite(f)), allPatients);
eegVec = per_patient(Views.SessionsForFigures.Patient, ...
    Views.SessionsForFigures.Session, @(x) numel(unique(x)), allPatients);

% EEG duration is summarised across EEGs, not across patients, so it is taken
% straight from the session table rather than through per_patient. This uses
% whichever duration DURATION_SOURCE selected, i.e. the one spike rates were
% actually computed from.
durMin = double(Views.SessionsForFigures.(CFG.durCol)) / 60;
durMin = durMin(isfinite(durMin));

%% --- Outcome distributions ---
sfVec = SzFreq.MeanSzFreq(ismember(SzFreq.Patient, allPatients));
sfVec = sfVec(isfinite(sfVec));
srVec = PL.MeanSpikeRate_perHour(isfinite(PL.MeanSpikeRate_perHour));
[~, sf_lo, sf_hi] = bootstrap_median_ci(sfVec, CFG.nBoot, CFG.alpha);
[~, sr_lo, sr_hi] = bootstrap_median_ci(srVec, CFG.nBoot, CFG.alpha);

%% --- Reported spikes, per EEG and per patient ---
RS = resolve_reported_spike_status(Rk);
nEEG = height(RS);
[gp, pidRS] = findgroups(RS.Patient);
hasPre = splitapply(@(x) any(string(x)=="present"), RS.ReportStatus, gp);
hasAbs = splitapply(@(x) any(string(x)=="absent"),  RS.ReportStatus, gp);
patStatus = repmat("unknown", numel(pidRS), 1);
patStatus(hasAbs & ~hasPre) = "absent";
patStatus(hasPre)           = "present";

%% --- Assemble ---
row = @(name, stat) {string(name), string(stat)};
mq  = @(v, fmt) sprintf(fmt, median(v,'omitnan'), prctile(v,25), prctile(v,75));
pct = @(n, d) sprintf('%d (%.1f%%)', n, 100*n/max(1,d));

R = [ ...
    row("Total N patients",                  sprintf('%d', N));
    row("Age at first visit (years)",        mq(ageVec, '%.1f (%.1f-%.1f)'));
    row("Sex",                               "");
    row("    Women",                         pct(n_f, N));
    row("    Men",                           pct(n_m, N));
    row("    Unknown/Other",                 pct(n_u, N));
    row("Epilepsy subtype",                  "");
    row("    Temporal lobe",                 pct(n_temp,  N));
    row("    Frontal lobe",                  pct(n_front, N));
    row("    Generalized",                   pct(n_gen,   N));
    row("    Other",                         pct(n_other, N));
    row("    Unknown",                       pct(n_unk,   N));
    row("Number of clinic visits",           mq(visVec, '%.1f (%.1f-%.1f)'));
    row("Follow-up duration (years)",        mq(fuVec,  '%.1f (%.1f-%.1f)'));
    row("Visits with documented seizure frequency", mq(docVec, '%.1f%% (%.1f-%.1f)'));
    row("Number of EEGs",                    mq(eegVec, '%.1f (%.1f-%.1f)'));
    row("EEG duration (minutes, across EEGs)", mq(durMin, '%.1f (%.1f-%.1f)'));
    row("Mean seizure frequency (seizures/month)", ...
        sprintf('%.2f (%.2f-%.2f); median CI [%.2f-%.2f]', median(sfVec,'omitnan'), ...
            prctile(sfVec,25), prctile(sfVec,75), sf_lo, sf_hi));
    row("Mean spike rate (spikes/hour)", ...
        sprintf('%.2f (%.2f-%.2f); median CI [%.2f-%.2f]', median(srVec,'omitnan'), ...
            prctile(srVec,25), prctile(srVec,75), sr_lo, sr_hi));
    row("EEGs with reported spikes",         "N (% EEGs)");
    row("    Present", pct(nnz(RS.ReportStatus=="present"), nEEG));
    row("    Absent",  pct(nnz(RS.ReportStatus=="absent"),  nEEG));
    row("    Unknown", pct(nnz(RS.ReportStatus=="unknown"), nEEG));
    row("Patients with reported spikes",     "N (% patients)");
    row("    Present", pct(nnz(patStatus=="present"), numel(pidRS)));
    row("    Absent",  pct(nnz(patStatus=="absent"),  numel(pidRS)));
    row("    Unknown", pct(nnz(patStatus=="unknown"), numel(pidRS)))];

Table1 = table(vertcat(R{:,1}), vertcat(R{:,2}), ...
    'VariableNames',{'Variable','Statistic'});
end


function write_tableS1(MMR, outPath)
%WRITE_TABLES1  Full fixed-effects table for M1 and M2.
% Bootstrap CIs and p-values are used where available, Laplace otherwise; the
% CI_method column records which was used for each row.

specs = {'M1 (logistic, subtypes, interactions)',    MMR.FE_M1, MMR.BootstrapTable1;
         'M2 (logistic, subtypes, no interactions)', MMR.FE_M2, MMR.BootstrapTable2};
rows = {};

for mi = 1:size(specs,1)
    label = specs{mi,1}; FE = specs{mi,2}; BT = specs{mi,3};
    if isempty(FE), continue; end
    for i = 1:height(FE)
        term = string(FE.Term(i));
        if term == "(Intercept)", continue; end

        p_val = FE.p(i); ci_lo = FE.OR_lo(i); ci_hi = FE.OR_hi(i); src = 'Laplace';
        if ~isempty(BT)
            r = BT(string(BT.Term) == term, :);
            if ~isempty(r)
                p_val = r.Boot_p; ci_lo = r.OR_CI_lo; ci_hi = r.OR_CI_hi;
                src = 'Bootstrap';
            end
        end

        rows(end+1,:) = {label, char(pretty_term_names(term)), ...
            sprintf('%.3f', FE.OR(i)), sprintf('%.3f', ci_lo), ...
            sprintf('%.3f', ci_hi), src, p_str(p_val)}; %#ok<AGROW>
    end
end

writetable(cell2table(rows, 'VariableNames', ...
    {'Model','Term','Estimate','CI_lower','CI_upper','CI_method','p_value'}), outPath);
end


function write_results_html(outPath, Views, SzFreq, ControlStats, SP, MMR, ...
    Vuniq, NearFar, Supp, CFG)
%WRITE_RESULTS_HTML  Draft results text with every number filled in from the
% live analysis, so the manuscript can be updated by copy-paste.
%
% Figure numbers used here match the driver's numbering block:
%   Fig 2 controls, Fig 3 Spearman, Fig 4 model;
%   Fig S1 flow, S2 EEG duration, S3 report availability, S4 generalized
%   syndromes, S5 non-zero Spearman, S6 trimmed Spearman,
%   S7 sz-by-reported-spikes, S8 lag context, S9 near/far tertiles.
%
% Sentences about the new supplements are written so the WORDING follows the
% data (see similar_or_differed): if a comparison turns out significant, the
% text will say "differed from" rather than silently asserting similarity.

fid = fopen(outPath, 'w');
assert(fid ~= -1, 'Could not open %s for writing.', outPath);
closer = onCleanup(@() fclose(fid));

PL = Views.PatientLevelSpikeRates;
N  = numel(PL.Patient);
nEEG = height(Views.ReportForKeptSessions);

sfVec = SzFreq.MeanSzFreq(ismember(SzFreq.Patient, PL.Patient));
sfVec = sfVec(isfinite(sfVec));
[sf_med, sf_lo, sf_hi] = bootstrap_median_ci(sfVec, CFG.nBoot, CFG.alpha);
srVec = PL.MeanSpikeRate_perHour(isfinite(PL.MeanSpikeRate_perHour));
[sr_med, sr_lo, sr_hi] = bootstrap_median_ci(srVec, CFG.nBoot, CFG.alpha);

Vc = Vuniq(ismember(Vuniq.Patient, PL.Patient), :);
fu  = per_patient(Vc.Patient, Vc.VisitDate, @(d) days(max(d)-min(d))/365.25, PL.Patient);
doc = per_patient(Vc.Patient, Vc.Freq_R1,   @(f) 100*mean(isfinite(f)),      PL.Patient);

RS = resolve_reported_spike_status(Views.ReportForKeptSessions);
[gp, pidRS] = findgroups(RS.Patient);
nPatPresent = nnz(splitapply(@(x) any(string(x)=="present"), RS.ReportStatus, gp));

EC = Views.ExclusionCounts;
fprintf(fid, '<html><head><meta charset="UTF-8"><title>Results</title></head><body>\n');

% A draft run must never be mistaken for a final one when read out of context.
if CFG.QUICK || CFG.SUBSAMPLE_PATIENTS > 0
    fprintf(fid, ['<p style="background:#fdd;padding:8px;border:2px solid #900">' ...
        '<strong>DRAFT RUN &mdash; DO NOT USE THESE NUMBERS.</strong> ' ...
        'nBoot=%d, model bootstrap=%d, patient subsample=%d.</p>\n'], ...
        CFG.nBoot, CFG.nBootModel, CFG.SUBSAMPLE_PATIENTS);
end

%% ---------------- Cohort ----------------
fprintf(fid, '<h2>Cohort summary</h2>\n');
fprintf(fid, ['<p>Of %d patients with EEG data in the Penn Epilepsy Center database, ' ...
    '%d were excluded because their EEG was not an outpatient routine recording of ' ...
    'less than 4 hours, %d were excluded without an LLM-confirmed epilepsy diagnosis, ' ...
    'and %d were excluded without a documented seizure frequency at any clinic visit, ' ...
    'yielding a final cohort of %d patients with %d EEGs (Fig. S1). ' ...
    'Median EEG duration was %.0f minutes (IQR %.0f&ndash;%.0f; Fig. S2). ' ...
    'Median follow-up from first to last clinic visit was %.1f years (IQR %.1f&ndash;%.1f). ' ...
    'Across patients, a median of %.1f%% (IQR %.1f&ndash;%.1f%%) of clinic visits had a ' ...
    'documented seizure frequency. %d patients (%.1f%%) had spikes reported on at least ' ...
    'one EEG. Median [95%% CI] monthly seizure frequency was %.2f [%.2f-%.2f], and median ' ...
    'spikes/hour was %.2f [%.2f-%.2f] (Table 1).</p>\n'], ...
    EC.nTotal, EC.nTotal - EC.nAfterOutptRoutine, EC.nExcludedNoEpilepsy, ...
    EC.nExcludedNoSzFreq, N, nEEG, ...
    Supp.Duration.file_med, Supp.Duration.file_q(1), Supp.Duration.file_q(2), ...
    median(fu,'omitnan'),  prctile(fu,25),  prctile(fu,75), ...
    median(doc,'omitnan'), prctile(doc,25), prctile(doc,75), ...
    nPatPresent, 100*nPatPresent/numel(pidRS), ...
    sf_med, sf_lo, sf_hi, sr_med, sr_lo, sr_hi);

%% ---------------- Figure 2 (controls) ----------------
fprintf(fid, '<h2>Spike rates by patient groups</h2>\n');
fprintf(fid, ['<p>Detected spike rates were higher in EEGs with clinically-reported spikes ' ...
    '(median %.2f [95%% CI %.2f-%.2f] spikes/hour) than without (%.2f [%.2f-%.2f] ' ...
    'spikes/hour) (%s, Cliff''s &delta;=%.2f; Fig. 2A). '], ...
    ControlStats.m_pre, ControlStats.lo_pre, ControlStats.hi_pre, ...
    ControlStats.m_abs, ControlStats.lo_abs, ControlStats.hi_abs, ...
    format_p_html(ControlStats.p_rankSum_A), ControlStats.effectA_cliff);

% Report availability. Wording adapts to the p-values rather than presuming
% the null; note this comparison is at the EEG level, not the patient level.
B = Supp.Bias;
verb = similar_or_differed( ...
    [B.SpikeRate.p_ranksum, B.SzFreq.p_ranksum, B.EpiType.p], CFG.alpha);
fprintf(fid, ['EEGs for which the reported presence or absence of spikes was ' ...
    'unavailable (N=%d) %s those with an available report (N=%d) in automatically ' ...
    'detected spike rate (%s, Cliff''s &delta;=%.2f), seizure frequency ' ...
    '(%s, &delta;=%.2f), and epilepsy subtype distribution ' ...
    '(&chi;&sup2;(%d)=%.1f, %s; Fig. S3). '], ...
    B.nNoReport, verb, B.nWithReport, ...
    format_p_html(B.SpikeRate.p_ranksum), B.SpikeRate.cliff_delta, ...
    format_p_html(B.SzFreq.p_ranksum),    B.SzFreq.cliff_delta, ...
    B.EpiType.df, B.EpiType.chi2, format_p_html(B.EpiType.p));

fprintf(fid, ['Spike rates differed across epilepsy subtypes, with the highest rates in ' ...
    'generalized epilepsy (Kruskal-Wallis %s, &eta;&sup2;&asymp;%.3f; Fig. 2B).</p>\n'], ...
    format_p_html(ControlStats.p_kw_C), ControlStats.eta2_kw_C);

%% ---------------- Figure 3 (Spearman) ----------------
% Rows are looked up by group NAME, so reordering canonical3 cannot silently
% swap the numbers in this paragraph.
getG = @(R, g) R(R.Group == g, :);
Rm = SP.main.Results;

fprintf(fid, '<h2>Spike rate and seizure frequency</h2>\n');
fprintf(fid, ['<p>Spike rate and seizure frequency were positively correlated across all ' ...
    'epilepsy patients (N=%d, &rho;=%.2f [95%% CI %.2f-%.2f], %s). '], ...
    SP.main.n_all, SP.main.rho_all, SP.main.ci_lo_all, SP.main.ci_hi_all, ...
    format_p_html(SP.main.p_all));

gen = getG(Rm,"General"); tem = getG(Rm,"Temporal"); fro = getG(Rm,"Frontal");
fprintf(fid, ['Subtype-specific correlations were significant for generalized epilepsy ' ...
    '(N=%d, &rho;=%.2f [%.2f-%.2f], Bonferroni-adjusted %s) and temporal lobe epilepsy ' ...
    '(N=%d, &rho;=%.2f [%.2f-%.2f], Bonferroni-adjusted %s), but not frontal lobe epilepsy ' ...
    '(N=%d, &rho;=%.2f [%.2f-%.2f], Bonferroni-adjusted %s; Fig. 3). '], ...
    gen.N, gen.Spearman_r, gen.ci_lo, gen.ci_hi, format_p_html(gen.p_bonf), ...
    tem.N, tem.Spearman_r, tem.ci_lo, tem.ci_hi, format_p_html(tem.p_bonf), ...
    fro.N, fro.Spearman_r, fro.ci_lo, fro.ci_hi, format_p_html(fro.p_bonf));

% Generalized-epilepsy syndromes. Listed descriptively; the numbers, not an
% adjective, carry the message.
Gs = Supp.GenSub(Supp.GenSub.Group ~= "All generalized", :);
fprintf(fid, 'Correlations within individual generalized-epilepsy syndromes are shown in Fig. S4 (');
for i = 1:height(Gs)
    if i < height(Gs), sep = '; '; else, sep = '). '; end
    fprintf(fid, '%s: N=%d, &rho;=%.2f [%.2f-%.2f], Bonferroni-adjusted %s%s', ...
        Gs.Group(i), Gs.N(i), Gs.Spearman_rho(i), Gs.ci_lo(i), Gs.ci_hi(i), ...
        format_p_html(Gs.p_bonf(i)), sep);
end

temNZ = getG(SP.nz.Results, "Temporal");
fprintf(fid, ['When restricting to patients with non-zero spike rates and seizure ' ...
    'frequencies, results were similar, although the temporal epilepsy correlation was ' ...
    'no longer significant in this subgroup (Bonferroni-adjusted %s) despite a similar ' ...
    'magnitude (&rho;=%.2f [%.2f-%.2f]; Fig. S5). '], ...
    format_p_html(temNZ.p_bonf), temNZ.Spearman_r, temNZ.ci_lo, temNZ.ci_hi);

fprintf(fid, ['After excluding the 10%% of patients with the highest spike rates, ' ...
    'the overall correlation was &rho;=%.2f [%.2f-%.2f] (N=%d, %s; Fig. S6). '], ...
    Supp.Trim.rho_all, Supp.Trim.ci_lo_all, Supp.Trim.ci_hi_all, ...
    Supp.Trim.n_all, format_p_html(Supp.Trim.p_all));

fprintf(fid, ['Patients with spikes on at least one EEG had higher mean seizure ' ...
    'frequencies (Fig. S7).</p>\n']);

%% ---------------- Figure 4 (model) ----------------
fprintf(fid, '<h2>Mixed effects model</h2>\n');

if ~isempty(MMR.BootstrapTable1)
    BT = MMR.BootstrapTable1; ci_source = 'bootstrap';
else
    BT = MMR.FE_M1;
    BT.Properties.VariableNames{'OR_lo'} = 'OR_CI_lo';
    BT.Properties.VariableNames{'OR_hi'} = 'OR_CI_hi';
    ci_source = 'Laplace approximation';
end
fprintf(fid, '<p><em>CIs from %s.</em></p>\n', ci_source);

getRow = @(nm) BT(string(BT.Term) == nm, :);
getP   = @(nm) get_p_preferred(MMR.FE_M1, MMR.BootstrapTable1, nm);
r_spike   = getRow('LogSpikesPerHour');
r_dir     = getRow('VisitAfterEEG');
r_intLag  = getRow('LogSpikesPerHour:AbsLag_years');
r_intDir  = getRow('LogSpikesPerHour:VisitAfterEEG');
r_frontal = getRow('EpiType3_cat_Frontal');
r_general = getRow('EpiType3_cat_General');

fprintf(fid, ['<p>Seizure frequency varies over time within individuals, and we ' ...
    'hypothesized that spike rates track this variability, predicting a stronger ' ...
    'spike-seizure association for clinic visits close in time to EEGs. To test this, we ' ...
    'fit logistic mixed effects models on all EEG-visit pairs for patients with known ' ...
    'epilepsy subtype (N=%d pairs, %d patients), with interaction terms allowing the ' ...
    'spike-seizure association to vary with the temporal distance between EEG and visit ' ...
    '(Fig. S8). A likelihood ratio test confirmed that these interactions jointly improved ' ...
    'model fit over a model without them (&chi;&sup2;(2), %s).</p>\n'], ...
    height(MMR.ModelTable), numel(unique(MMR.ModelTable.Patient)), ...
    format_p_html(MMR.LRT_p));

% The main effect is conditional on the interaction terms, so state the
% reference condition explicitly rather than reporting a bare OR.
fprintf(fid, ['<p>Higher spike rates were associated with higher odds of reporting seizures ' ...
    'at a clinic visit (OR=%.2f [95%% CI %.2f-%.2f], %s; Fig. 4A), implying that an e-fold ' ...
    '(%.1f-fold) increase in spike rate is associated with %.0f%% higher odds of seizure ' ...
    'reporting. Because the model includes interactions, this odds ratio refers to the ' ...
    'reference condition of an EEG and visit on the same day, with the visit preceding the ' ...
    'EEG, in temporal lobe epilepsy. '], ...
    r_spike.OR, r_spike.OR_CI_lo, r_spike.OR_CI_hi, ...
    format_p_html(getP('LogSpikesPerHour')), exp(1), (r_spike.OR - 1)*100);

% ORs at each lag are evaluated at VisitAfterEEG = 1, matching Fig 4B.
B = coef_struct(MMR.mdl_M1);
or_at_lag = exp(B.spike + B.int_lag.*[0.5 2 4] + B.int_dir*1);
fprintf(fid, ['The spike-seizure association attenuated with greater EEG-visit distance, ' ...
    'although the effect was small (OR=%.3f [%.3f-%.3f], %s): for visits after the EEG, ' ...
    'the model-predicted OR for spike rate was %.2f at a 6-month lag, %.2f at 2 years, and ' ...
    '%.2f at 4 years (Fig. 4B). '], ...
    r_intLag.OR, r_intLag.OR_CI_lo, r_intLag.OR_CI_hi, ...
    format_p_html(getP('LogSpikesPerHour:AbsLag_years')), ...
    or_at_lag(1), or_at_lag(2), or_at_lag(3));

fprintf(fid, ['Spike rates from EEGs obtained before versus after a clinic visit were ' ...
    'similarly associated with seizure occurrence (interaction OR=%.3f [%.3f-%.3f], %s). '], ...
    r_intDir.OR, r_intDir.OR_CI_lo, r_intDir.OR_CI_hi, ...
    format_p_html(getP('LogSpikesPerHour:VisitAfterEEG')));
fprintf(fid, ['Visits occurring after the EEG had lower baseline odds of seizure reporting ' ...
    '(OR=%.2f [%.2f-%.2f], %s), consistent with gradual clinical improvement over time ' ...
    '(Fig. S8). '], r_dir.OR, r_dir.OR_CI_lo, r_dir.OR_CI_hi, format_p_html(getP('VisitAfterEEG')));
fprintf(fid, ['Compared with temporal lobe epilepsy, generalized epilepsy had lower baseline ' ...
    'odds of seizure reporting (OR=%.2f [%.2f-%.2f], %s), while frontal lobe epilepsy did ' ...
    'not differ significantly (OR=%.2f [%.2f-%.2f], %s). '], ...
    r_general.OR, r_general.OR_CI_lo, r_general.OR_CI_hi, format_p_html(getP('EpiType3_cat_General')), ...
    r_frontal.OR, r_frontal.OR_CI_lo, r_frontal.OR_CI_hi, format_p_html(getP('EpiType3_cat_Frontal')));
fprintf(fid, ['Our secondary analysis also found that spike-seizure correlations were ' ...
    'stronger for clinic visits close in time to the EEG (Fig. S9). Together, these results ' ...
    'confirm a positive spike rate-seizure association and suggest it is strongest when the ' ...
    'EEG is obtained close to the clinic visit, consistent with spike rates tracking ' ...
    'within-individual seizure burden over time.</p>\n']);

%% ---------------- Diagnostics ----------------
BC = MMR.BootstrapConvergence;
fprintf(fid, '<h2>Bootstrap diagnostics</h2>\n');
if BC.M1_nTotal == 0
    fprintf(fid, '<p>Model bootstrap skipped; M1/M2 CIs are Laplace approximations.</p>\n');
else
    fprintf(fid, '<p>M1: %d/%d iterations converged (%.1f%%). M2: %d/%d (%.1f%%).</p>\n', ...
        BC.M1_nConverged, BC.M1_nTotal, 100*BC.M1_nConverged/max(1,BC.M1_nTotal), ...
        BC.M2_nConverged, BC.M2_nTotal, 100*BC.M2_nConverged/max(1,BC.M2_nTotal));
end

%% ---------------- Fig S9 legend ----------------
fprintf(fid, '<h2>Figure S9 legend</h2>\n');
fprintf(fid, ['<p><strong>Fig. S9. The association between interictal spike rate and seizure ' ...
    'frequency is higher for clinic visits close in time to EEGs.</strong> ' ...
    '<strong>A:</strong> Distribution of the absolute time difference between clinic visits ' ...
    'and EEG recordings, taking the minimum in the case of multiple EEGs per patient. Visits ' ...
    'were stratified into short gap and long gap groups based on tertiles of the ' ...
    'visit&ndash;EEG gap distribution. Shaded regions indicate the lower, middle, and upper ' ...
    'thirds, with dashed vertical lines marking the tertile cutoffs (%.0f days and %.0f days). ' ...
    '<strong>B:</strong> Bootstrap distribution (%d iterations) of the difference in Spearman ' ...
    'correlation coefficients between spike rate and seizure frequency for near versus far ' ...
    'visit windows (&Delta;&rho; = &rho;<sub>short gap</sub> &minus; &rho;<sub>long gap</sub>). ' ...
    'The correlation was stronger when clinic visits occurred closer in time to EEG ' ...
    'acquisition (N = %d patients with both short-gap and long-gap visits; ' ...
    '&rho;<sub>short gap</sub> = %.2f; &rho;<sub>long gap</sub> = %.2f; observed [95%% CI] ' ...
    '&Delta;&rho; = %.3f [%.2f&ndash;%.2f], one-sided p = %.3f).</p>\n'], ...
    NearFar.nearDays, NearFar.farDays, numel(NearFar.delta_boot), NearFar.nPatients, ...
    NearFar.rho_near, NearFar.rho_far, NearFar.delta_obs, ...
    NearFar.delta_ci_lo, NearFar.delta_ci_hi, NearFar.p_one_sided);

fprintf(fid, '</body></html>\n');
fprintf('Wrote HTML: %s\n', outPath);
end


%% #####################################################################
%% ##  UTILITIES
%% #####################################################################

%% ---- JSON ----------------------------------------------------------

function arr = json_to_string_array(s)
s = strtrim(string(s));
if ismember(s, ["","[]","<missing>"]), arr = strings(0,1); return; end
dec = jsondecode(char(s));
if iscell(dec)
    arr = strings(numel(dec),1);
    for k = 1:numel(dec)
        if ischar(dec{k}) || (isstring(dec{k}) && isscalar(dec{k}))
            arr(k) = string(dec{k});
        end
    end
elseif ischar(dec) || isstring(dec) || isnumeric(dec)
    arr = string(dec(:));
else
    error('Unsupported JSON string-array type.');
end
arr = string(arr(:));
end

function arr = json_to_double_array(s)
s = strtrim(string(s));
if ismember(s, ["","[]","<missing>"]), arr = double([]); return; end
arr = double(jsondecode(char(regexprep(s,'null','NaN','ignorecase'))));
arr = arr(:);
end

%% ---- Small reductions ----------------------------------------------

function out = max_omitnan(x)
x = x(isfinite(x));
if isempty(x), out = NaN; else, out = max(x); end
end

function out = min_omitnan(x)
x = x(isfinite(x));
if isempty(x), out = NaN; else, out = min(x); end
end

function out = first_nonmissing(s)
s = s(:); s = s(~ismissing(s) & strlength(s) > 0);
if isempty(s), out = ""; else, out = s(1); end
end

function v = per_patient(pidCol, valCol, fn, targetPatients)
%PER_PATIENT  Apply fn within patient, then return one value per element of
% targetPatients (NaN / "" where the patient has no rows).
[g, pids] = findgroups(double(pidCol));
res = splitapply(fn, valCol, g);
[tf, loc] = ismember(double(targetPatients), pids);
if isstring(res) || iscellstr(res)
    v = strings(numel(targetPatients),1);
else
    v = nan(numel(targetPatients),1);
end
v(tf) = res(loc(tf));
end

%% ---- Log-scale plotting --------------------------------------------

function y = log10_floor(x, epsVal)
%LOG10_FLOOR  log10 with zeros/negatives pinned to epsVal.
% Non-finite input stays NaN: it must not be silently drawn at the zero floor
% (which is what happened previously when a bootstrap CI came back NaN).
x = double(x);
y = nan(size(x));
ok = isfinite(x);
x(ok & x <= 0) = epsVal;
y(ok) = log10(x(ok));
end

function [eps_sz, eps_rate] = zero_floors(szVals, rateVals)
%ZERO_FLOORS  Half a step below the smallest positive value on each axis.
eps_sz   = 0.5 * local_minpos(szVals);
eps_rate = 0.5 * local_minpos(rateVals);
end

function m = local_minpos(v)
v = v(isfinite(v) & v > 0);
if isempty(v), m = 1e-6; else, m = min(v); end
end

function Yj = jitter_at_floor(Y, Y_ZERO, Y_LIMS, frac)
%JITTER_AT_FLOOR  Spread out the stack of points sitting exactly on the zero
% floor so their density is visible.
Yj = Y;
atFloor = abs(Y - Y_ZERO) < 1e-9;
if any(atFloor)
    Yj(atFloor) = Yj(atFloor) + (rand(nnz(atFloor),1) - 0.5) * frac*diff(Y_LIMS);
end
end

function set_log10_ticks(ax, whichAxis, eps_val, axisLims, maxPow)
%SET_LOG10_TICKS  Decade ticks plus a "0" tick at the epsilon floor.
if nargin < 5 || isempty(maxPow), maxPow = 6; end
decades = 10.^(0:maxPow);
logDec  = log10(decades);
keep    = logDec >= axisLims(1) & logDec <= axisLims(2);
ticks   = logDec(keep);
labels  = string(decades(keep));

logEps = log10(eps_val);
if logEps >= axisLims(1) && logEps <= axisLims(2)
    ticks  = [logEps; ticks(:)];
    labels = ["0"; labels(:)];
end
if lower(string(whichAxis)) == "x"
    ax.XTick = ticks; ax.XTickLabel = labels;
else
    ax.YTick = ticks; ax.YTickLabel = labels;
end
end

function box_swarm_panel(ax, grp, Y, Y_ZERO, Y_LIMS, epsFloor, ylab)
%BOX_SWARM_PANEL  Shared box + swarm + zero-line skeleton.
hold(ax,'on'); box(ax,'off'); grid(ax,'on');
boxchart(ax, grp, Y, 'BoxFaceAlpha',0.25, 'MarkerStyle','none');
swarmchart(ax, grp, Y, 18, 'filled', 'MarkerFaceAlpha',0.18);
yline(ax, Y_ZERO, ':', 'Color',[0.4 0.4 0.4], 'LineWidth',1.2);
ylim(ax, Y_LIMS);
ylabel(ax, ylab);
set_log10_ticks(ax, 'y', epsFloor, Y_LIMS);
end

function finish_panel(ax, ttl, groupLabels, groupN, fontSize, titleOffset)
%FINISH_PANEL  Title, font, and "Label (N=...)" tick labels.
t = title(ax, ttl);
set(ax, 'FontSize', fontSize);
if titleOffset ~= 0
    t.Units = 'normalized'; t.Position(2) = t.Position(2) + titleOffset;
end
labs = string(ax.XTickLabel);
for i = 1:numel(groupLabels)
    labs(labs == groupLabels(i)) = sprintf('%s (N=%d)', groupLabels(i), groupN(i));
end
ax.XTickLabel = labs;
ax.XTickLabelRotation = 20;
end

function add_median_ci_overlay(ax, xpos, med, lo, hi, epsFloor)
%ADD_MEDIAN_CI_OVERLAY  Black median marker with its bootstrap interval.
plot(ax, [xpos xpos], log10_floor([lo hi], epsFloor), 'k-', 'LineWidth',3);
plot(ax, xpos, log10_floor(med, epsFloor), 'ko', 'MarkerFaceColor','k','MarkerSize',6);
end

function add_sigbar(ax, x1, x2, y, ptext)
%ADD_SIGBAR  Bracket with a p-value or star label above a comparison.
tick = 0.03*diff(ax.YLim);
plot(ax, [x1 x1 x2 x2], [y-tick, y, y, y-tick], 'k-', 'LineWidth',1.3);
if ismember(ptext, ["**","***"]), yOff = -0.012*diff(ax.YLim);
else,                             yOff =  0.003*diff(ax.YLim); end
text(ax, mean([x1 x2]), y+yOff, ptext, 'HorizontalAlignment','center', ...
    'VerticalAlignment','bottom', 'FontSize',20);
end

function save_fig(figH, outPath)
if strlength(string(outPath)) == 0, return; end
if ~exist(fileparts(outPath),'dir'), mkdir(fileparts(outPath)); end
exportgraphics(figH, outPath, 'Resolution',300);
fprintf('Saved: %s\n', outPath);
end

%% ---- Statistics ----------------------------------------------------

function [med, lo, hi] = bootstrap_median_ci(x, nBoot, alpha)
[med, lo, hi] = bootstrap_stat_ci(x, @(z) median(z,'omitnan'), nBoot, alpha);
end

function [stat, lo, hi] = bootstrap_stat_ci(x, fn, nBoot, alpha)
%BOOTSTRAP_STAT_CI  Percentile bootstrap interval for any scalar statistic.
x = double(x(:)); x = x(isfinite(x));
if isempty(x), stat = NaN; lo = NaN; hi = NaN; return; end
stat = fn(x);
n = numel(x);
b = nan(nBoot,1);
for k = 1:nBoot
    b(k) = fn(x(randi(n,n,1)));
end
lo = prctile(b, 100*(alpha/2));
hi = prctile(b, 100*(1-alpha/2));
end

function [rho, lo, hi, p] = spearman_with_ci(x, y, nBoot, alpha)
%SPEARMAN_WITH_CI  Spearman rho with a percentile bootstrap interval.
x = double(x(:)); y = double(y(:));
m = isfinite(x) & isfinite(y);
x = x(m); y = y(m); n = numel(x);
if n < 3, rho = NaN; lo = NaN; hi = NaN; p = NaN; return; end
[rho, p] = corr(x, y, 'Type','Spearman', 'Rows','complete');
b = nan(nBoot,1);
for k = 1:nBoot
    idx = randi(n,n,1);
    b(k) = corr(x(idx), y(idx), 'Type','Spearman', 'Rows','complete');
end
lo = prctile(b, 100*(alpha/2));
hi = prctile(b, 100*(1-alpha/2));
end

function d = cliff_delta(x1, x2)
%CLIFF_DELTA  Non-parametric effect size; positive means x1 tends to exceed x2.
x1 = x1(isfinite(x1(:))); x2 = x2(isfinite(x2(:)));
n1 = numel(x1); n2 = numel(x2);
if n1 == 0 || n2 == 0, d = NaN; return; end
[~,~,stats] = ranksum(x1, x2, 'method','approx');
U1 = stats.ranksum - n1*(n1+1)/2;
d  = 2*U1/(n1*n2) - 1;
end

function p = boot_p_two_sided(b, null_val)
%BOOT_P_TWO_SIDED  Phipson & Smyth (2010) (count+1)/(B+1) convention.
b = b(isfinite(b));
B = numel(b);
p = min(2 * min((sum(b <= null_val)+1)/(B+1), (sum(b >= null_val)+1)/(B+1)), 1);
end

function p = boot_p_one_sided(b, null_val)
b = b(isfinite(b));
p = (sum(b <= null_val) + 1) / (numel(b) + 1);
end

function [chi2, p, df, E] = chi2_contingency(O)
%CHI2_CONTINGENCY  Pearson chi-square test of independence.
E    = sum(O,2) * sum(O,1) / sum(O(:));
chi2 = sum(((O - E).^2) ./ E, 'all');
df   = (size(O,1)-1) * (size(O,2)-1);
p    = 1 - chi2cdf(chi2, df);
end

function p = get_p_preferred(FE, BT, nm)
%GET_P_PREFERRED  Bootstrap p if available, otherwise Wald.
p = FE.p(string(FE.Term) == nm);
if ~isempty(BT) && ismember('Boot_p', BT.Properties.VariableNames)
    r = BT(string(BT.Term) == nm, :);
    if ~isempty(r), p = r.Boot_p; end
end
end

function B = coef_struct(mdl)
%COEF_STRUCT  Named M1 coefficients, so the prediction code reads clearly.
[beta, names] = fixedEffects(mdl);
n = string(names.Name);
pick = @(nm) beta(n == nm);
B = struct('intercept', pick("(Intercept)"), ...
           'spike',     pick("LogSpikesPerHour"), ...
           'abslag',    pick("AbsLag_years"), ...
           'dir',       pick("VisitAfterEEG"), ...
           'int_lag',   pick("LogSpikesPerHour:AbsLag_years"), ...
           'int_dir',   pick("LogSpikesPerHour:VisitAfterEEG"));
end

%% ---- Formatting ----------------------------------------------------

function names = pretty_term_names(raw)
%PRETTY_TERM_NAMES  Model coefficient names for figures and tables.
map = { ...
    "(Intercept)",                            "Intercept"; ...
    "LogSpikesPerHour:AbsLag_years",          "Spike rate effect per year of gap"; ...
    "LogSpikesPerHour:VisitAfterEEG",         "Spike rate effect: visit before or after"; ...
    "LogSpikesPerHour",                       "Log spike rate"; ...
    "AbsLag_years",                           "EEG-visit gap (years)"; ...
    "VisitAfterEEG",                          "Visit after vs before EEG"; ...
    "EEG_DurationHours",                      "EEG duration (hours)"; ...
    "EpiType3_cat_Frontal",                   "Frontal vs Temporal"; ...
    "EpiType3_cat_General",                   "Generalized vs Temporal"};
names = string(raw);
for i = 1:size(map,1)
    names(names == map{i,1}) = map{i,2};
end
end

function s = similar_or_differed(pvals, alpha)
%SIMILAR_OR_DIFFERED  Pick the verb from the data, not from the hypothesis.
% Used for the report-availability sentence so that a significant difference
% cannot be described as similarity.
pvals = pvals(isfinite(pvals));
if isempty(pvals) || all(pvals >= alpha)
    s = 'were similar to';
else
    s = 'differed from';
end
end

function s = p_label(p)
%P_LABEL  "p=0.03" style label for figures.
if isnan(p),   s = "p=NaN";                 return; end
if p < 0.001,  s = "p<0.001";               return; end
if p < 0.01,   s = sprintf("p=%.2g", p);    return; end
s = sprintf("p=%.2f", p);
end

function s = stars(p)
%STARS  Significance stars for pairwise comparison bars.
if     p < 1e-3, s = "***";
elseif p < 1e-2, s = "**";
elseif p < 5e-2, s = "*";
else,            s = "ns";
end
end

function s = p_text(p)
%P_TEXT  Inline p annotation for the forest plot.
if     p < 0.001, s = 'p<0.001';
elseif p < 0.05,  s = sprintf('p=%.3f', p);
else,             s = sprintf('p=%.2f', p);
end
end

function s = p_str(p)
%P_STR  p for CSV output.
if     p < 0.001, s = '<0.001';
elseif p < 0.01,  s = sprintf('%.3f', p);
else,             s = sprintf('%.2f', p);
end
end

function s = format_p_html(p)
%FORMAT_P_HTML  p for the results HTML.
if isnan(p),  s = 'p = NaN';       return; end
if p < 0.001, s = 'p &lt; 0.001';  return; end
if p < 0.01
    t = sprintf('%.2g', p);
    if startsWith(t,'.'), t = ['0' t]; end
    s = ['p = ' t]; return;
end
s = sprintf('p = %.2f', p);
end
