function run_spike_sz_pipeline_clean()
%% ===================== INFORMATION ==================
%{
This contains the code for running all analyses associated with the
manuscript "Interictal Spike Rate on Routine Outpatient EEG Is Associated
With Seizure Frequency in a Large Epilepsy Cohort" by Conrad et al., 2026

Requirements:
1. spike_counts.csv and clinical_data_deidentified.csv
     - both are available for download at: https://upenn.box.com/s/yy4o1t6nit7yu35flz59ux6zf54slg9m
2. Matlab (vR2024a) and the Statistics and Machine Learning Toolbox
3. This codebase: https://github.com/erinconrad/seizure_severity
4. Create output and data directories in the paths below, place CSVs in data directory.

To run:
>> run_spike_sz_pipeline_clean
%}

%% ======================= RNG =======================
rng(1);

%% ======================= PATHS =======================
spikeSummaryMultiCsv = '../data/spike_counts.csv';
reportCsv            = '../data/clinical_data_deidentified.csv';
deadtimeCsv = '../data/spike_counts_deadtime_0p46.csv';
%reportCsv = '../data/Routineeegpec-Deidreport_DATA_LABELS_2026-07-21_0930.csv';

fig1_out    = '../output/Fig1.png';
fig2_out    = '../output/Fig2.png';
figS1_out   = '../output/FigS1.png';
figS2_out   = '../output/FigS2.png';
figMain_out = '../output/FigModel.png';
Table1Csv   = fullfile('..','output','Table1.csv');
tableS1Csv  = fullfile('..','output','TableS1.csv');
resultsHtml = '../output/results_summary.html';

%% ======================= PARAMETERS =======================
MAX_ROUTINE_HOURS = 4;
NESD_LABEL        = "Non-Epileptic Seizure Disorder";
badTypes          = lower(["Uncertain if Epilepsy","Unknown or MRN not found",""]);
canonical3        = ["General","Temporal","Frontal"];

DURATION_SOURCE = "file"; % "file" | "natus" | "deadtime"
DEADTIME_COUNTCOL = "count_0_46"; % countCol the deadtime file corresponds to

EPS_RATE       = 30e-3; % How much epsilon to add to zero spike rates so it doesn't screw up log scale on figures
Y_ZERO         = log10(EPS_RATE);
Y_LIMS         = [-2 4];
TITLE_Y_OFFSET = 0.02;

spearman_xLims = [-3.5, 4];
spearman_yLims = [-1.5, 3];

nBoot    = 0;%5000;
alpha    = 0.05;
countCol = "count_0_46";
durCol   = "Duration_sec";

allowable_visits = [
    "CONSULT VISIT","ESTABLISHED PATIENT VISIT",...
    "FOLLOW-UP PATIENT CLINIC","NEW PATIENT CLINIC","NEW PATIENT VISIT",...
    "NPV MANAGEMENT DURING COVID-19","NPV NEUROLOGY",...
    "RETURN ANNUAL VISIT","RETURN PATIENT EXTENDED","RETURN PATIENT VISIT",...
    "RPV MANAGEMENT DURING COVID-19","TELEHEALTH VIDEO VISIT RETURN"
];

%% ======================= LOAD DATA =======================
SpikeSummaryTable = readtable(spikeSummaryMultiCsv,'TextType','string','VariableNamingRule','preserve');
require_cols(SpikeSummaryTable, ["Patient","Session",countCol,durCol], "SpikeSummaryTable");

ReportTable = readtable(reportCsv,'TextType','string','VariableNamingRule','preserve');
require_cols(ReportTable, ...
    ["patient_id","session_number","acquired_on", ...
     "report_PATIENT_CLASS","jay_in_or_out", ...
     "visit_type","visit_dates_deid","sz_freqs","visit_hasSz", ...
     "epilepsy_type","epilepsy_specific","nlp_gender","deid_birth_date","start_time_deid", ...
     "report_SPORADIC_EPILEPTIFORM_DISCHARGES","jay_focal_epi","jay_multifocal_epi","jay_gen_epi", ...
     "duration_hms"], ...
    "ReportTable");

% Apply the duration-source toggle BEFORE computing spike rate, so the chosen
% duration propagates to spike rates, the <4h filter, and everything downstream.
SpikeSummaryTable = apply_duration_source(SpikeSummaryTable, ReportTable, ...
       DURATION_SOURCE, durCol, countCol, deadtimeCsv, DEADTIME_COUNTCOL);
SpikeSummaryTable.SpikeRate_perHour = SpikeSummaryTable.(countCol) ./ SpikeSummaryTable.(durCol) * 3600;

%% ======================= FILTER =======================
ReportTable = filter_visit_arrays_by_type(ReportTable, allowable_visits); % only allow outpatient clinic visits
[SpikeSummaryTable, ReportTable, nPatientsTotal] = filter_outpatient_routine(...
    SpikeSummaryTable, ReportTable, durCol, MAX_ROUTINE_HOURS);

assert_unique_keys(SpikeSummaryTable, "Patient","Session","SpikeSummaryTable");
assert_unique_keys(ReportTable, "patient_id","session_number","ReportTable");

%% ======================= BUILD COHORT =======================
Vuniq            = build_visit_level_table_R1(ReportTable); % all unique visits
PatientTypingAll = build_patient_typing_from_report(ReportTable, canonical3); % patient epilepsy types
SzFreqPerPatient = build_patient_seizure_metrics(Vuniq);

Views = build_filtered_view(SpikeSummaryTable, ReportTable, PatientTypingAll, ...
    SzFreqPerPatient, NESD_LABEL, badTypes, canonical3, nPatientsTotal);

%% ======================= BUILD PAIR TABLE FOR MODEL =======================
% Canonical-subtype patients only
PairTable = build_eeg_visit_pairs(Vuniq, Views.SessionLevelSpikeRates, ...
    Views.ReportForKeptSessions, Views.PatientTypingFiltered);
fprintf('Canonical-subtype pairs: %d, patients: %d\n', ...
    height(PairTable), numel(unique(PairTable.Patient)));

%% ======================= FIT MODELS =======================
MMR = fit_mixed_effects_models(PairTable, nBoot, alpha);


%% ======================= GEN SUBTYPES =======================
   figGenSub_out = '../output/FigGenSubtypes.png';
   [GenSubtypeResults, FigGenSub] = make_fig_generalized_subtypes( ...
       Views, spearman_xLims, spearman_yLims, figGenSub_out, 10, nBoot, alpha, false);
   disp(GenSubtypeResults);

%% ======================= CHECK DURATION ===================
DurCompare = add_duration_to_model(MMR, Views);

%% ======================= HISTOGRAM OF DURATIONS =======================
figDur_out = '../output/FigDuration.png';
FigDur = make_eeg_duration_histogram(Views, figDur_out);

%% ========================= MRI FIGURE ======================
%{
figMRI_out = '../output/FigMRI.png';
[FigMRI, MRIStats] = make_fig_mri_groups(Views, EPS_RATE, [], [], alpha, figMRI_out);
%}

%% ========================= CHECK FOR REPORTING BIAS ========
figBias_out = '../output/FigReportBias.png';
[FigBias, BiasStats] = make_fig_report_bias(Views, SzFreqPerPatient, EPS_RATE, [], [], alpha, figBias_out);

%% ====================== REMOVE HIGH SPIKERS ===================
figS1trim_out = '../output/FigS1_trim10.png';
[SpearmanResults_S1trim, rs_all_S1trim, p_all_S1trim, n_all_S1trim, ...
 rho_lo_S1trim, rho_hi_S1trim, subtype_ci_S1trim, trimInfo] = ...
    spearman_trim_top_spikers(Views, canonical3, spearman_xLims, spearman_yLims, ...
        figS1trim_out, 0.05, false);

%% ======================= FLOW DIAGRAM =======================
figFlow_out = '../output/FigFlow.png';
FigFlow = make_flowchart_figure(Views, MMR);
save_fig(FigFlow, figFlow_out);
fprintf('Saved flow diagram: %s\n', figFlow_out);

%% ======================= FIGURE 1 =======================
[Fig1, Fig1Stats] = make_fig1_controls(Views, EPS_RATE, Y_ZERO, Y_LIMS, TITLE_Y_OFFSET, nBoot, alpha);
save_fig(Fig1, fig1_out);
fprintf('Saved Fig 1: %s\n', fig1_out);

%% ======================= FIGURE 2 + FIG S1 =======================
[SpearmanResults_main, rs_all_main, p_all_main, n_all_main, rho_lo_main, rho_hi_main, subtype_ci_main] = ...
    spearman_plotting_function(Views.PatientSpikeSz_All, Views.PatientSpikeSz_Typed, ...
        canonical3, spearman_xLims, spearman_yLims, fig2_out, 'MeanSzFreq', '', false);

[SpearmanResults_S1, rs_all_S1, p_all_S1, n_all_S1, rho_lo_S1, rho_hi_S1, subtype_ci_S1] = ...
    spearman_plotting_function(Views.PatientSpikeSz_All, Views.PatientSpikeSz_Typed, ...
        canonical3, spearman_xLims, spearman_yLims, figS1_out, 'MeanSzFreq', ...
        ' (positive spike/seizures only)', true);

fprintf('Saved Fig 2: %s\n', fig2_out);
fprintf('Saved Fig S1: %s\n', figS1_out);

%% Model figs
% Main figure
FigMain = make_model_figure(MMR, figMain_out);

% Supplemental
figSupLag_out = '../output/FigSupLag.png'; % Fig S3
FigSupLag = make_figSup_lag(MMR, Vuniq, Views.ReportForKeptSessions, figSupLag_out);

%% ======================= FIG S2 =======================
FigS2 = make_figS2_sz_by_reported_spikes(Views, SzFreqPerPatient, nBoot, alpha, spearman_xLims);
save_fig(FigS2, figS2_out);
fprintf('Saved Fig S2: %s\n', figS2_out);

%% ======================= FIG S_TERTILE (NEAR VS FAR CORRELATION) =======================
figSTertile_out = '../output/FigSTertile.png';
NearFarStats = plot_delta_rho_histogram( ...
    Views, Vuniq, Views.ReportForKeptSessions, ...
    0.333, 0.667, nBoot, alpha, ...
    figSTertile_out);
fprintf('Saved tertile figure: %s\n', figSTertile_out);
fprintf('Tertile cutoffs: near (q=%.3f) = %.0f days (%.2f yr); far (q=%.3f) = %.0f days (%.2f yr)\n', ...
    NearFarStats.nearQ, NearFarStats.nearDays, NearFarStats.nearDays/365.25, ...
    NearFarStats.farQ,  NearFarStats.farDays,  NearFarStats.farDays/365.25);

%% ======================= TABLE 1 =======================
Table1_flat = build_table1_flat(Views, SzFreqPerPatient, Vuniq, EPS_RATE, nBoot, alpha);
if ~exist(fileparts(Table1Csv),'dir'), mkdir(fileparts(Table1Csv)); end
writetable(Table1_flat, Table1Csv);
fprintf('Wrote Table 1: %s\n', Table1Csv);

%% ======================= TABLE S1 =======================
write_tableS1(MMR, tableS1Csv);
fprintf('Wrote Table S1: %s\n', tableS1Csv);

%% ======================= HTML RESULTS =======================
write_results_html(resultsHtml, Views, SzFreqPerPatient, Fig1Stats, ...
    SpearmanResults_main, rs_all_main, p_all_main, n_all_main, ...
    rho_lo_main, rho_hi_main, subtype_ci_main, ...
    SpearmanResults_S1, rs_all_S1, p_all_S1, n_all_S1, ...
    rho_lo_S1, rho_hi_S1, subtype_ci_S1, ...
    Views.ReportForKeptSessions, MMR, Vuniq, NearFarStats);
fprintf('Wrote HTML: %s\n', resultsHtml);


end

%% =====================================================================
%% CORE PIPELINE HELPERS
%% =====================================================================

function require_cols(T, cols, name)
missing = setdiff(string(cols), string(T.Properties.VariableNames));
if ~isempty(missing)
    error('%s is missing required columns: %s', name, strjoin(missing, ", "));
end
end

function assert_unique_keys(T, pidCol, sesCol, name)
key = string(T.(pidCol)) + "|" + string(T.(sesCol));
if numel(unique(key)) ~= numel(key)
    error('%s has duplicated (Patient,Session) keys.', name);
end
end

function save_fig(figH, outPath)
if ~exist(fileparts(outPath),'dir'), mkdir(fileparts(outPath)); end
exportgraphics(figH, outPath, 'Resolution', 300);
end

function R = filter_visit_arrays_by_type(R, allowable_visits)
% Only allow outpatient clinic visits
vt_raw_all = strtrim(string(R.visit_type));
mask_null_only = (vt_raw_all == "[null]") | (vt_raw_all == "null");
R.visit_type(mask_null_only)       = "[]";
R.visit_dates_deid(mask_null_only) = "[]";
R.sz_freqs(mask_null_only)         = "[]";
R.visit_hasSz(mask_null_only)      = "[]";

total_before = 0; total_after = 0;
for i = 1:height(R)
    vt_raw    = strtrim(string(R.visit_type(i)));
    dates_raw = strtrim(string(R.visit_dates_deid(i)));
    sz_raw    = strtrim(string(R.sz_freqs(i)));
    hs_raw    = strtrim(string(R.visit_hasSz(i)));

    if vt_raw=="" || vt_raw=="[]" || vt_raw=="<missing>"
        R.visit_type(i)="[]"; R.visit_dates_deid(i)="[]";
        R.sz_freqs(i)="[]";   R.visit_hasSz(i)="[]";
        continue
    end

    vt    = json_to_string_array(vt_raw);
    dates = json_to_string_array(dates_raw);
    sz    = json_to_double_array(sz_raw);
    hs    = json_to_double_array(hs_raw);

    if ~(numel(vt)==numel(dates) && numel(vt)==numel(sz) && numel(vt)==numel(hs))
        error('Row %d: visit arrays have mismatched lengths.', i);
    end

    total_before = total_before + numel(vt);
    keepMask = ismember(string(vt), allowable_visits);

    if ~any(keepMask)
        R.visit_type(i)="[]"; R.visit_dates_deid(i)="[]";
        R.sz_freqs(i)="[]";   R.visit_hasSz(i)="[]";
        continue
    end

    vt_f    = cellstr(string(vt(keepMask)));
    dates_f = cellstr(string(dates(keepMask)));
    sz_f    = sz(keepMask);
    hs_f    = hs(keepMask);
    total_after = total_after + numel(vt_f);

    R.visit_type(i)       = string(jsonencode(vt_f));
    R.visit_dates_deid(i) = string(jsonencode(dates_f));
    R.sz_freqs(i)         = string(jsonencode(sz_f));
    R.visit_hasSz(i)      = string(jsonencode(hs_f));
end
fprintf('[Visit-type filter] %d -> %d visits (kept %.1f%%)\n', ...
    total_before, total_after, 100*total_after/max(1,total_before));
end


function [S, R, nPatientsTotal] = filter_outpatient_routine(S, R, durCol, MAX_ROUTINE_HOURS)
% only allow outpatient routine EEGs
nR0 = height(R); nS0 = height(S);

% Count total unique patients before any filtering
nPatientsTotal = numel(unique(double(R.patient_id)));

acqStr   = lower(strtrim(string(R.acquired_on)));
classStr = lower(strtrim(string(R.report_PATIENT_CLASS)));
jayStr   = lower(strtrim(string(R.jay_in_or_out)));

isOutpt = contains(acqStr,"spe") | contains(acqStr,"radnor") | ...
          (classStr == "outpatient") | (jayStr == "out");

OutptKeys = unique(R(isOutpt, {'patient_id','session_number'}));
OutptKeys.Properties.VariableNames = {'Patient','Session'};
if isempty(OutptKeys), error('No outpatient sessions identified.'); end

isRoutine   = isfinite(S.(durCol)) & S.(durCol) <= MAX_ROUTINE_HOURS*3600;
RoutineKeys = unique(S(isRoutine, {'Patient','Session'}));

OutptRoutineKeys = innerjoin(OutptKeys, RoutineKeys, 'Keys', {'Patient','Session'});
S = innerjoin(S, OutptRoutineKeys, 'Keys', {'Patient','Session'});
R = innerjoin(R, OutptRoutineKeys, 'LeftKeys', {'patient_id','session_number'}, ...
    'RightKeys', {'Patient','Session'});

fprintf('[Outpatient+routine] Kept %d/%d spike rows, %d/%d report rows\n', ...
    height(S), nS0, height(R), nR0);
end


function Vuniq = build_visit_level_table_R1(R)
% Make one row for each visit
PV = table('Size',[0 4], 'VariableTypes',{'double','datetime','double','double'}, ...
    'VariableNames',{'Patient','VisitDate','Freq','HasSz'});

% loop over eeg rows (each eeg row has all visits for that patient)
for j = 1:height(R)
    pid = double(R.patient_id(j));
    ds  = strtrim(string(R.visit_dates_deid(j)));
    if ds=="[]" || ds=="", continue; end
    d = datetime(string(jsondecode(char(ds))),'InputFormat','yyyy-MM-dd'); % get dates of clinic visits

    s = strtrim(string(R.sz_freqs(j))); % get sz frequencies
    s = regexprep(s,'null','NaN','ignorecase');
    v = double(jsondecode(char(s)));
    v(~isfinite(v)|v<0) = NaN;

    hs = strtrim(string(R.visit_hasSz(j)));
    h  = double(jsondecode(char(hs)));
    h(h==2) = NaN;

    if ~(numel(d)==numel(v) && numel(d)==numel(h))
        error('Row %d: visit arrays mismatched.', j); end
    if numel(d)==0, continue; end

    PV = [PV; table(repmat(pid,numel(d),1), d(:), v(:), h(:), ...
        'VariableNames', PV.Properties.VariableNames)]; %#ok<AGROW>
end

% collapse to unique patient-date combos
[gv, pid_keys, date_keys] = findgroups(PV.Patient, PV.VisitDate);
Freq_agg = splitapply(@(x) mean(x(isfinite(x)),'omitnan'), PV.Freq, gv);
Has_agg  = splitapply(@max_hasSz, PV.HasSz, gv);

Vuniq = table(pid_keys, date_keys, Freq_agg, Has_agg, ...
    'VariableNames', {'Patient','VisitDate','Freq','HasSz'});
Vuniq.Freq_R1 = Vuniq.Freq;
% Rule 1: HasSz==0 with no documented frequency → impute SzFreq=0
mask_rule1 = ~isfinite(Vuniq.Freq_R1) & (Vuniq.HasSz==0);
Vuniq.Freq_R1(mask_rule1) = 0;

% Rule 2 (new): documented SzFreq with no HasSz → impute HasSz
mask_rule2_sz1 = isfinite(Vuniq.Freq_R1) & (Vuniq.Freq_R1 > 0) & ~isfinite(Vuniq.HasSz);
mask_rule2_sz0 = isfinite(Vuniq.Freq_R1) & (Vuniq.Freq_R1 == 0) & ~isfinite(Vuniq.HasSz);
Vuniq.HasSz(mask_rule2_sz1) = 1;
Vuniq.HasSz(mask_rule2_sz0) = 0;
fprintf('[Rule 2] Imputed HasSz for %d visits with SzFreq>0, %d visits with SzFreq=0\n', ...
    nnz(mask_rule2_sz1), nnz(mask_rule2_sz0));
end

function SzP = build_patient_seizure_metrics(Vuniq)
[gp, pids] = findgroups(Vuniq.Patient);
MeanSzFreq        = splitapply(@(x) mean(x,'omitnan'), Vuniq.Freq_R1, gp); % get mean sz frequency per patient
FracVisits_HasSz1 = splitapply(@local_frac_hasSz1, Vuniq.HasSz, gp);
SzP = table(pids, MeanSzFreq, FracVisits_HasSz1, ...
    'VariableNames', {'Patient','MeanSzFreq','FracVisits_HasSz1'});
end

% determine epilepsy type
function T = build_patient_typing_from_report(R, canonical3)
pid   = double(R.patient_id);
etype = strtrim(string(R.epilepsy_type));
espec = strtrim(string(R.epilepsy_specific));

okT = ~ismissing(etype) & strlength(etype)>0;
okS = ~ismissing(espec) & strlength(espec)>0;

Ttype = table(pid(okT), etype(okT), 'VariableNames', {'Patient','EpilepsyType_raw'});
Ttype = sortrows(Ttype, 'Patient');
[uid, ~, g] = unique(Ttype.Patient, 'stable');
etype_one = strings(numel(uid),1);
for k=1:numel(uid)
    vals = unique(Ttype.EpilepsyType_raw(g==k));
    vals = vals(strlength(vals)>0);
    if isempty(vals), continue; end
    if numel(vals)>1, error('Conflicting epilepsy_type for Patient %g', uid(k)); end
    etype_one(k) = vals(1);
end

Tspec = table(pid(okS), espec(okS), 'VariableNames', {'Patient','EpilepsySpecific_raw'});
Tspec = sortrows(Tspec, 'Patient');
[uidS, ~, gS] = unique(Tspec.Patient, 'stable');
espec_one = strings(numel(uidS),1);
for k=1:numel(uidS)
    vals = unique(Tspec.EpilepsySpecific_raw(gS==k));
    vals = vals(strlength(vals)>0);
    if isempty(vals), continue; end
    if numel(vals)>1, error('Conflicting epilepsy_specific for Patient %g', uidS(k)); end
    espec_one(k) = vals(1);
end

assert(all(uid==uidS))
T = table(uid, etype_one, 'VariableNames', {'Patient','EpilepsyType'});
T = outerjoin(T, table(uidS, espec_one, 'VariableNames', {'Patient','EpilepsySpecific'}), ...
    'Keys','Patient','MergeKeys',true);

spec_norm = lower(strtrim(string(T.EpilepsySpecific)));
type_norm = lower(strtrim(string(T.EpilepsyType)));
E3 = strings(height(T),1);
E3(contains(spec_norm,"temporal"))                   = "Temporal";
E3(contains(spec_norm,"frontal"))                    = "Frontal";
E3((strlength(E3)==0) & strcmp(type_norm,"general")) = "General";
T.EpiType3 = categorical(E3, canonical3);
end

%% =====================================================================
%% VIEW BUNDLE
%% =====================================================================

function Views = build_filtered_view(SessionsFiltered, ReportIn, PatientTypingAll, ...
    SzFreqPerPatient, NESD_LABEL, badTypes, canonical3, nPatientsTotal)

%% Join report to kept sessions
SessKeys = unique(SessionsFiltered(:,{'Patient','Session'}));
ReportForKeptSessions = innerjoin(ReportIn, SessKeys, ...
    'LeftKeys',{'patient_id','session_number'}, 'RightKeys',{'Patient','Session'});
ReportForKeptSessions.Patient = double(ReportForKeptSessions.patient_id);
ReportForKeptSessions.Session = double(ReportForKeptSessions.session_number);

%% Patient-level mean spike rate + typing
PatientsKept   = unique(double(SessionsFiltered.Patient));
nAfterOutptRoutine  = numel(PatientsKept);  
TypingFiltered = innerjoin(PatientTypingAll, ...
    table(PatientsKept,'VariableNames',{'Patient'}), 'Keys','Patient');

[gpS, pidsS] = findgroups(double(SessionsFiltered.Patient));
MeanSpikeRate_perHour = splitapply(@(x) mean(x,'omitnan'), SessionsFiltered.SpikeRate_perHour, gpS);
PatientLevelSpikeRates = table(double(pidsS), MeanSpikeRate_perHour, ...
    'VariableNames', {'Patient','MeanSpikeRate_perHour'});
PatientLevelSpikeRates = innerjoin(PatientLevelSpikeRates, ...
    TypingFiltered(:,{'Patient','EpilepsyType','EpilepsySpecific','EpiType3'}), 'Keys','Patient');

%% Epilepsy masks
etype_norm     = lower(strtrim(string(PatientLevelSpikeRates.EpilepsyType)));
IsNESDMask     = (etype_norm == lower(strtrim(NESD_LABEL)));
IsBadType      = ismember(etype_norm, badTypes) | ismissing(etype_norm) | (strlength(etype_norm)==0);
IsEpilepsyMask = ~IsNESDMask & ~IsBadType;

%% Session-level spike rates
SessionLevelSpikeRates = SessionsFiltered(:, {'Patient','Session','SpikeRate_perHour','Duration_sec'});
SessionLevelSpikeRates.Properties.VariableNames{'SpikeRate_perHour'} = 'SpikesPerHour';

%% Define cohort: epilepsy + documented seizure frequency
SzFreqFiltered = innerjoin(SzFreqPerPatient, ...
    table(PatientsKept,'VariableNames',{'Patient'}), 'Keys','Patient');
EpPatients     = table(PatientLevelSpikeRates.Patient(IsEpilepsyMask), 'VariableNames',{'Patient'});
SzFreqEpilepsy = innerjoin(SzFreqFiltered, EpPatients, 'Keys','Patient');


% *** CAPTURE THESE BEFORE THE COHORT RESTRICTION ***
nPatientsWithEpilepsy  = numel(unique(SzFreqEpilepsy.Patient));
nPatientsWithSzFreq    = numel(unique(SzFreqEpilepsy.Patient(...
    isfinite(SzFreqEpilepsy.MeanSzFreq))));

CohortPatients = unique(SzFreqEpilepsy.Patient(isfinite(SzFreqEpilepsy.MeanSzFreq)));
CohortTable    = table(CohortPatients, 'VariableNames',{'Patient'});

%% Restrict all tables to cohort
PatientLevelSpikeRates = innerjoin(PatientLevelSpikeRates, CohortTable, 'Keys','Patient');
TypingFiltered         = innerjoin(TypingFiltered,         CohortTable, 'Keys','Patient');
SessionsFiltered       = innerjoin(SessionsFiltered,       CohortTable, 'Keys','Patient');
ReportForKeptSessions  = innerjoin(ReportForKeptSessions,  CohortTable, 'Keys','Patient');
SzFreqEpilepsy         = innerjoin(SzFreqEpilepsy,         CohortTable, 'Keys','Patient');

etype_norm     = lower(strtrim(string(PatientLevelSpikeRates.EpilepsyType)));
IsNESDMask     = (etype_norm == lower(strtrim(NESD_LABEL)));
IsBadType      = ismember(etype_norm, badTypes) | ismissing(etype_norm) | (strlength(etype_norm)==0);
IsEpilepsyMask = ~IsNESDMask & ~IsBadType;
assert(all(IsEpilepsyMask))

fprintf('[Cohort] %d epilepsy patients with documented seizure frequency\n', numel(CohortPatients));

%% All-epilepsy typing table (no subtype restriction) for primary model
PatientTyping_AllEpilepsy = TypingFiltered;

%% Canonical-subtype typing table for Spearman only
keepCanon3    = ~ismissing(PatientLevelSpikeRates.EpiType3) & ...
                ismember(string(PatientLevelSpikeRates.EpiType3), canonical3);
TypedPatients = PatientLevelSpikeRates.Patient(IsEpilepsyMask & keepCanon3);
TypingFiltered_Canonical = TypingFiltered(ismember(TypingFiltered.Patient, TypedPatients), :);

%% Spearman input tables
PatientSpikeSz_All = innerjoin(...
    PatientLevelSpikeRates(IsEpilepsyMask, {'Patient','MeanSpikeRate_perHour'}), ...
    SzFreqEpilepsy, 'Keys','Patient');
keepAll = isfinite(PatientSpikeSz_All.MeanSpikeRate_perHour) & ...
          isfinite(PatientSpikeSz_All.MeanSzFreq);
PatientSpikeSz_All = PatientSpikeSz_All(keepAll,:);

SzFreqCanon = innerjoin(SzFreqEpilepsy, ...
    PatientLevelSpikeRates(IsEpilepsyMask & keepCanon3, {'Patient','EpiType3'}), 'Keys','Patient');
PatientSpikeSz_Typed = innerjoin(...
    PatientLevelSpikeRates(IsEpilepsyMask & ismember(PatientLevelSpikeRates.Patient, TypedPatients), ...
        {'Patient','MeanSpikeRate_perHour'}), ...
    SzFreqCanon, 'Keys','Patient');
keepTyped = isfinite(PatientSpikeSz_Typed.MeanSpikeRate_perHour) & ...
            isfinite(PatientSpikeSz_Typed.MeanSzFreq) & ...
            ~ismissing(PatientSpikeSz_Typed.EpiType3);
PatientSpikeSz_Typed = PatientSpikeSz_Typed(keepTyped,:);

%% Canonical3 group stats for Fig 1B
Canonical3_SubsetTable = PatientLevelSpikeRates(IsEpilepsyMask & keepCanon3, ...
    {'Patient','EpiType3','MeanSpikeRate_perHour'});
Canonical3_SubsetTable.Properties.VariableNames{'EpiType3'} = 'EpiType4';

[g3, cats3] = findgroups(Canonical3_SubsetTable.EpiType4);
medVals = splitapply(@(x) median(x,'omitnan'), Canonical3_SubsetTable.MeanSpikeRate_perHour, g3);
p25Vals = splitapply(@(x) prctile(x,25),      Canonical3_SubsetTable.MeanSpikeRate_perHour, g3);
p75Vals = splitapply(@(x) prctile(x,75),      Canonical3_SubsetTable.MeanSpikeRate_perHour, g3);
nVals   = splitapply(@(x) sum(isfinite(x)),   Canonical3_SubsetTable.MeanSpikeRate_perHour, g3);
Canonical3_Stats = table(cats3, nVals, medVals, p25Vals, p75Vals, ...
    'VariableNames',{'EpiType4','GroupCount','Median','P25','P75'});

Canonical3_Pairs = ["General","Temporal"; "General","Frontal"; "Temporal","Frontal"];
p_pair = NaN(3,1);
for i = 1:3
    xa = Canonical3_SubsetTable.MeanSpikeRate_perHour(...
        Canonical3_SubsetTable.EpiType4 == categorical(Canonical3_Pairs(i,1), canonical3));
    xb = Canonical3_SubsetTable.MeanSpikeRate_perHour(...
        Canonical3_SubsetTable.EpiType4 == categorical(Canonical3_Pairs(i,2), canonical3));
    if nnz(isfinite(xa))>=3 && nnz(isfinite(xb))>=3
        p_pair(i) = ranksum(xa, xb, 'method','approx');
    end
end

%% Exclusion counts for flow diagram
EC.nTotal                = nPatientsTotal; % starting
EC.nAfterOutptRoutine    = nAfterOutptRoutine; % those with outpatient routine eegs
EC.nExcludedNoEpilepsy   = nAfterOutptRoutine - nPatientsWithEpilepsy; % excluded for not having epilepsy
EC.nExcludedNoSzFreq     = nPatientsWithEpilepsy - nPatientsWithSzFreq; % excluded for not having sz frequency
EC.nFinalCohort          = numel(CohortPatients);

% Sanity check: exclusions + final cohort must sum to post-filter total
assert(EC.nExcludedNoEpilepsy + EC.nExcludedNoSzFreq + EC.nFinalCohort == EC.nAfterOutptRoutine, ...
    'Flow count mismatch: %d + %d + %d = %d, expected %d', ...
    EC.nExcludedNoEpilepsy, EC.nExcludedNoSzFreq, EC.nFinalCohort, ...
    EC.nExcludedNoEpilepsy + EC.nExcludedNoSzFreq + EC.nFinalCohort, ...
    EC.nAfterOutptRoutine);

Views.ExclusionCounts = EC;

%% Bundle
Views.SessionsForFigures         = SessionsFiltered;
Views.ReportForKeptSessions      = ReportForKeptSessions;
Views.PatientTypingFiltered      = TypingFiltered_Canonical;
Views.PatientTyping_AllEpilepsy  = PatientTyping_AllEpilepsy;
Views.SessionLevelSpikeRates     = SessionLevelSpikeRates;
Views.PatientLevelSpikeRates     = PatientLevelSpikeRates;
Views.PatientSpikeSz_All         = PatientSpikeSz_All;
Views.PatientSpikeSz_Typed       = PatientSpikeSz_Typed;
Views.IsEpilepsyMask             = IsEpilepsyMask;
Views.IsNESDMask                 = IsNESDMask;
Views.Canonical3_SubsetTable     = Canonical3_SubsetTable;
Views.Canonical3_Stats           = Canonical3_Stats;
Views.Canonical3_Pairs           = Canonical3_Pairs;
Views.PvalsPairwise              = p_pair;
Views.PvalsPairwiseBonf          = min(p_pair*3, 1);
end

%% =====================================================================
%% PAIR TABLE BUILDER
%% =====================================================================
% This builds the EEG-visit pair table that is the input to the mixed
% effects model
function PairTable = build_eeg_visit_pairs(Vuniq, SessionLevelSpikeRates, ...
    ReportForKeptSessions, PatientTyping)

%% Get EEG table
EEG_raw = ReportForKeptSessions.start_time_deid;
if isdatetime(EEG_raw), EEG_dt = EEG_raw;
else, EEG_dt = datetime(strtrim(string(EEG_raw)), 'InputFormat', "yyyy-MM-dd'T'HH:mm:ss");
end

EEG_dates = table(double(ReportForKeptSessions.Patient), ...
    double(ReportForKeptSessions.Session), EEG_dt, ...
    'VariableNames',{'Patient','Session','EEG_Date'});
EEG_dates = EEG_dates(~isnat(EEG_dates.EEG_Date), :);

EEG_tbl = innerjoin(EEG_dates, ...
    SessionLevelSpikeRates(:,{'Patient','Session','SpikesPerHour'}), ...
    'Keys',{'Patient','Session'});

%% Get patient visit table
Vtyped = innerjoin(Vuniq(:,{'Patient','VisitDate','Freq_R1','HasSz'}), ...
    PatientTyping(:,{'Patient','EpiType3'}), 'Keys','Patient');

%% Estimate size of array to allow pre-allocation to speed it up
patients  = intersect(unique(EEG_tbl.Patient), unique(Vtyped.Patient)); % find unique patient identifiers
nEstimate = 0;
for i = 1:numel(patients)
    p = patients(i);
    nEstimate = nEstimate + sum(EEG_tbl.Patient==p) * sum(Vtyped.Patient==p);
end

Patient_out       = nan(nEstimate,1);
Session_out       = nan(nEstimate,1);
VisitDate_out     = NaT(nEstimate,1);
SpikesPerHour_out = nan(nEstimate,1);
SzFreq_out        = nan(nEstimate,1);
HasSz_out         = nan(nEstimate,1);
SignedLag_out     = nan(nEstimate,1);
EpiType3_out      = strings(nEstimate,1);

%% Fill up table
row = 0;
% loop over all patients
for i = 1:numel(patients)
    p        = patients(i);
    eeg_rows = EEG_tbl(EEG_tbl.Patient==p, :);
    vis_rows = Vtyped(Vtyped.Patient==p, :);
    for e = 1:height(eeg_rows) % loop over all eegs for that patient
        for v = 1:height(vis_rows) % loop over all visits for that patient
            row = row + 1;
            Patient_out(row)       = p;
            Session_out(row)       = eeg_rows.Session(e);
            VisitDate_out(row)     = vis_rows.VisitDate(v);
            SpikesPerHour_out(row) = eeg_rows.SpikesPerHour(e);
            SzFreq_out(row)        = vis_rows.Freq_R1(v);
            HasSz_out(row)         = vis_rows.HasSz(v);
            SignedLag_out(row)     = days(vis_rows.VisitDate(v) - eeg_rows.EEG_Date(e)); % positive if visit after eeg
            EpiType3_out(row)      = string(vis_rows.EpiType3(v));
        end
    end
end

PairTable = table(Patient_out(1:row), Session_out(1:row), VisitDate_out(1:row), ...
    SpikesPerHour_out(1:row), SzFreq_out(1:row), HasSz_out(1:row), ...
    SignedLag_out(1:row), EpiType3_out(1:row), ...
    'VariableNames',{'Patient','Session','VisitDate','SpikesPerHour','SzFreq', ...
                     'HasSz','SignedLag_days','EpiType3'});

PairTable.EEG_ID = categorical(string(PairTable.Patient) + "_" + string(PairTable.Session));

keepMask  = isfinite(PairTable.SpikesPerHour) & isfinite(PairTable.SzFreq) & ...
            isfinite(PairTable.SignedLag_days) & strlength(PairTable.EpiType3)>0;
n_before  = height(PairTable);
PairTable = PairTable(keepMask,:);
fprintf('[build_eeg_visit_pairs] %d patients, %d pairs (%d removed)\n', ...
    numel(patients), height(PairTable), n_before-height(PairTable));

EPS_SPIKE = 1e-3;
PairTable.LogSpikesPerHour = log(PairTable.SpikesPerHour + EPS_SPIKE); % add tiny epsilon to not break log
PairTable.SignedLag_years  = PairTable.SignedLag_days / 365.25;
PairTable.PatientID        = categorical(PairTable.Patient);
end

%% =====================================================================
%% MIXED EFFECTS MODELS
%% =====================================================================

function MMR = fit_mixed_effects_models(PairTable, nBoot, alpha)
% PairTable — canonical-subtype patients only
if nargin < 2, nBoot = 0;    end
if nargin < 3, alpha = 0.05; end

%% ---- MODEL TABLE ----
canonical3 = ["Frontal","General","Temporal"];

keepMask = ismember(string(PairTable.EpiType3), canonical3) & ...
           isfinite(PairTable.LogSpikesPerHour) & ...
           isfinite(PairTable.SignedLag_years)  & ...
           isfinite(PairTable.HasSz);
T = PairTable(keepMask, :);

T.HasSz_bin    = double(T.HasSz == 1);
T.AbsLag_years = abs(T.SignedLag_years);
% Binary: 1 if visit occurred after EEG (or same day), 0 if before
T.VisitAfterEEG = double(T.SignedLag_years >= 0);
T.EpiType3_cat = reordercats(categorical(string(T.EpiType3), canonical3), ...
    ["Temporal","Frontal","General"]);

fprintf('[Model table] %d pairs, %d patients\n', height(T), numel(unique(T.Patient)));

%% ---- MODEL FORMULAS ----
formula_M1 = ['HasSz_bin ~ ' ...
    'LogSpikesPerHour * AbsLag_years + ' ...
    'LogSpikesPerHour * VisitAfterEEG + ' ...
    'EpiType3_cat + (1|PatientID)'];

% Alternate model, no interaction terms
formula_M2 = ['HasSz_bin ~ ' ...
    'LogSpikesPerHour + AbsLag_years + VisitAfterEEG + ' ...
    'EpiType3_cat + (1|PatientID)'];

%% ---- FIT MODELS ----
glme_opts = {'Distribution','Binomial','Link','logit', ...
    'FitMethod','Laplace','CovariancePattern','Diagonal'}; % laplace is maximum likelihood

fprintf('\nFitting M1 (logistic + subtypes + interactions)...\n');
try; mdl_M1 = fitglme(T, formula_M1, glme_opts{:}); fprintf('M1 converged.\n'); disp(mdl_M1);
catch ME; fprintf('M1 failed: %s\n', ME.message); mdl_M1 = []; end

fprintf('\nFitting M2 (logistic + subtypes, no interactions)...\n');
try; mdl_M2 = fitglme(T, formula_M2, glme_opts{:}); fprintf('M2 converged.\n'); disp(mdl_M2);
catch ME; fprintf('M2 failed: %s\n', ME.message); mdl_M2 = []; end

%% ---- DIRECTIONAL SENSITIVITY MODELS ----
% M_after: only visits occurring after (or same day as) the EEG
% M_before: only visits occurring before the EEG

T_after  = T(T.SignedLag_years >= 0, :);
T_before = T(T.SignedLag_years <= 0, :);

fprintf('\n[Directional] After-only: %d pairs, %d patients\n', ...
    height(T_after), numel(unique(T_after.Patient)));
fprintf('[Directional] Before-only: %d pairs, %d patients\n', ...
    height(T_before), numel(unique(T_before.Patient)));

% Same formula structure as M1 but signed lag
formula_dir = ['HasSz_bin ~ ' ...
    'LogSpikesPerHour * SignedLag_years + ' ...
    'EpiType3_cat + (1|PatientID)'];

fprintf('\nFitting M_after (visit after EEG only)...\n');
try
    mdl_after = fitglme(T_after, formula_dir, glme_opts{:});
    fprintf('M_after converged.\n'); disp(mdl_after);
    [b,bn,s] = fixedEffects(mdl_after, 'Alpha', alpha);
    T_fe_after = make_fe_table_logistic(bn, b, s);
    fprintf('\nM_after fixed effects:\n'); disp(T_fe_after);
catch ME
    fprintf('M_after failed: %s\n', ME.message);
    mdl_after = []; T_fe_after = [];
end

fprintf('\nFitting M_before (visit before EEG only)...\n');
try
    mdl_before = fitglme(T_before, formula_dir, glme_opts{:});
    fprintf('M_before converged.\n'); disp(mdl_before);
    [b,bn,s] = fixedEffects(mdl_before, 'Alpha', alpha);
    T_fe_before = make_fe_table_logistic(bn, b, s);
    fprintf('\nM_before fixed effects:\n'); disp(T_fe_before);
catch ME
    fprintf('M_before failed: %s\n', ME.message);
    mdl_before = []; T_fe_before = [];
end


%% ---- LRT ----
fprintf('\n=== LIKELIHOOD RATIO TEST ===\n');
lrt_p = NaN;
if ~isempty(mdl_M1) && ~isempty(mdl_M2)
    fprintf('\nLRT: M1 vs M2 (tests both interactions jointly)\n');
    lrt = compare(mdl_M2, mdl_M1); disp(lrt);
    lrt_p = lrt.pValue(2);
end

%% ---- FIXED EFFECTS TABLES ----
fprintf('\n=== FIXED EFFECTS ===\n');
T_fe1=[]; T_fe2=[];

if ~isempty(mdl_M1)
    [b,bn,s] = fixedEffects(mdl_M1, 'Alpha', alpha);
    T_fe1 = make_fe_table_logistic(bn, b, s);
    fprintf('\nM1 (logistic + subtypes + interactions):\n'); disp(T_fe1);
end
if ~isempty(mdl_M2)
    [b,bn,s] = fixedEffects(mdl_M2, 'Alpha', alpha);
    T_fe2 = make_fe_table_logistic(bn, b, s);
    fprintf('\nM2 (logistic + subtypes, no interactions):\n'); disp(T_fe2);
end

%% ---- BOOTSTRAP ----
[T_boot1, boot_betas1, nConv1, nTotal1] = run_bootstrap(T, mdl_M1, formula_M1, nBoot, alpha, 'M1');
[T_boot2, boot_betas2, nConv2, nTotal2] = run_bootstrap(T, mdl_M2, formula_M2, nBoot, alpha, 'M2');

if ~isempty(boot_betas1) && size(boot_betas1,1) > 10
    plot_bootstrap_diagnostics(boot_betas1, mdl_M1, 'M1');
end

%% ---- BOOTSTRAP DIRECTIONAL MODELS ----
[T_boot_after,  boot_betas_after,  nConv_after,  nTotal_after]  = ...
    run_bootstrap(T_after,  mdl_after,  formula_dir, nBoot, alpha, 'M_after');
[T_boot_before, boot_betas_before, nConv_before, nTotal_before] = ...
    run_bootstrap(T_before, mdl_before, formula_dir, nBoot, alpha, 'M_before');

%% ---- BUNDLE ----
MMR.ModelTable  = T;
MMR.mdl_M1      = mdl_M1;
MMR.mdl_M2      = mdl_M2;
MMR.FE_M1       = T_fe1;
MMR.FE_M2       = T_fe2;
MMR.BootstrapBetas1 = boot_betas1;
MMR.BootstrapTable1 = T_boot1;
MMR.BootstrapBetas2 = boot_betas2;
MMR.BootstrapTable2 = T_boot2;
MMR.LRT_p       = lrt_p;   % M1 vs M2, chi^2(2)
MMR.BootstrapConvergence.M1_nConverged = nConv1;
MMR.BootstrapConvergence.M1_nTotal     = nTotal1;
MMR.BootstrapConvergence.M2_nConverged = nConv2;
MMR.BootstrapConvergence.M2_nTotal     = nTotal2;

% ---- DIRECTIONAL SENSITIVITY ----
MMR.mdl_after   = mdl_after;
MMR.mdl_before  = mdl_before;
MMR.FE_after    = T_fe_after;
MMR.FE_before   = T_fe_before;

MMR.BootstrapTable_after  = T_boot_after;
MMR.BootstrapBetas_after  = boot_betas_after;
MMR.BootstrapTable_before = T_boot_before;
MMR.BootstrapBetas_before = boot_betas_before;
MMR.BootstrapConvergence.after_nConverged  = nConv_after;
MMR.BootstrapConvergence.after_nTotal      = nTotal_after;
MMR.BootstrapConvergence.before_nConverged = nConv_before;
MMR.BootstrapConvergence.before_nTotal     = nTotal_before;



fprintf('\nDone. Primary model: M1 (logistic + subtypes + interactions).\n');
if ~isempty(T_boot1)
    fprintf('Bootstrap CIs in MMR.BootstrapTable1\n');
else
    fprintf('No bootstrap CIs (nBoot=0).\n');
end
end

function T_fe = make_fe_table_logistic(bn, b, s)
T_fe = table(string(bn.Name), b, s.SE, s.tStat, s.pValue, ...
    exp(b), exp(b - 1.96*s.SE), exp(b + 1.96*s.SE), ...
    'VariableNames',{'Term','Beta','SE','t','p','OR','OR_lo','OR_hi'});
end

function [T_boot, boot_betas, nConverged, nTotal] = run_bootstrap(T, mdl, formula, nBoot, alpha, label)
T_boot     = [];
boot_betas = [];
nConverged = 0;
nTotal     = 0;
if isempty(mdl) || nBoot == 0, return; end

fprintf('\nBootstrapping %s (%d iterations)...\n', label, nBoot);
patients   = unique(T.PatientID);
nPat       = numel(patients);
nFixed     = size(fixedEffects(mdl), 1);
boot_betas = nan(nBoot, nFixed);

parfor b = 1:nBoot
    idx      = randi(nPat, nPat, 1); % draw n patients with replacement
    bootPats = patients(idx);
    Tboot    = cell(nPat, 1);

    % for each patient, copy all their rows over (so bootstrap at the
    % PATIENT LEVEL)
    for k = 1:nPat
        Tboot{k} = T(T.PatientID == bootPats(k), :);
        Tboot{k}.PatientID = categorical(repmat(k, height(Tboot{k}), 1));
    end
    Tboot = vertcat(Tboot{:});
    try
        mboot = fitglme(Tboot, formula, 'Distribution','Binomial','Link','logit', ...
            'FitMethod','Laplace','CovariancePattern','Diagonal');
        boot_betas(b,:) = fixedEffects(mboot)';
    catch
    end
end

converged  = all(isfinite(boot_betas), 2);
boot_betas = boot_betas(converged, :);
fprintf('%s bootstrap: %d/%d converged (%.1f%%)\n', ...
    label, sum(converged), nBoot, 100*mean(converged));

ci_lo = prctile(boot_betas, 100*(alpha/2),   1);
ci_hi = prctile(boot_betas, 100*(1-alpha/2), 1);
[beta_obs, betanames_obs] = fixedEffects(mdl);

% Bootstrap two-tailed p-value: 2 * min(prop <= 0, prop >= 0)
boot_p = nan(numel(beta_obs), 1);
for k = 1:numel(beta_obs)
    p_lo = mean(boot_betas(:,k) <= 0);
    p_hi = mean(boot_betas(:,k) >= 0);
    boot_p(k) = min(2 * min(p_lo, p_hi), 1);
end

T_boot = table(string(betanames_obs.Name), beta_obs, ci_lo(:), ci_hi(:), ...
    exp(beta_obs), exp(ci_lo(:)), exp(ci_hi(:)), boot_p, ...
    'VariableNames',{'Term','Beta','Boot_CI_lo','Boot_CI_hi','OR','OR_CI_lo','OR_CI_hi','Boot_p'});
fprintf('%s bootstrapped ORs:\n', label); disp(T_boot);
nConverged = sum(converged);
nTotal     = nBoot;

end

function plot_bootstrap_diagnostics(boot_betas, mdl, label)
[beta_diag, betanames_diag] = fixedEffects(mdl);
term_names = string(betanames_diag.Name);
nTerms = numel(term_names);
nCols  = ceil(sqrt(nTerms));
nRows  = ceil(nTerms/nCols);
figDiag = figure('Color','w','Position',[100 100 300*nCols 250*nRows], ...
    'Name',sprintf('Bootstrap — %s', label));
tl = tiledlayout(figDiag, nRows, nCols, 'TileSpacing','compact','Padding','loose');
for k = 1:nTerms
    ax = nexttile(tl);
    histogram(ax, boot_betas(:,k), 40, ...
        'FaceColor',[0.2 0.5 0.8],'EdgeColor','none','FaceAlpha',0.7);
    xline(ax, beta_diag(k), 'r-', 'LineWidth', 2);
    xline(ax, prctile(boot_betas(:,k), 2.5),  'k--', 'LineWidth', 1.5);
    xline(ax, prctile(boot_betas(:,k), 97.5), 'k--', 'LineWidth', 1.5);
    title(ax, term_names(k), 'FontSize',9,'Interpreter','none');
    xlabel(ax, '\beta','FontSize',8); box(ax,'off');
end
sgtitle(figDiag, sprintf('Bootstrap — %s (red=observed, dashed=95%% CI)', label), 'FontSize',11);
end

%% =====================================================================
%% FIGURE 1
%% =====================================================================

function [f1, Fig1Stats] = make_fig1_controls(Views, EPS_RATE, Y_ZERO, Y_LIMS, TITLE_Y_OFFSET, nBoot, alpha)
SessionLevel = Views.SessionLevelSpikeRates;
Report       = Views.ReportForKeptSessions;

ReportSlim = resolve_reported_spike_status(Report);
JoinA = innerjoin(SessionLevel(:,{'Patient','Session','SpikesPerHour'}), ReportSlim, ...
    'Keys',{'Patient','Session'});

x_abs = JoinA.SpikesPerHour(JoinA.ReportStatus=="absent");
x_pre = JoinA.SpikesPerHour(JoinA.ReportStatus=="present");
pA      = ranksum(x_abs, x_pre, 'method','approx');
effectA = cliff_delta(x_pre, x_abs);

[med_abs, lo_abs, hi_abs] = bootstrap_median_ci(x_abs, nBoot, alpha);
[med_pre, lo_pre, hi_pre] = bootstrap_median_ci(x_pre, nBoot, alpha);

Y_A = add_y_jitter_eps(to_log10_per_hour([x_abs; x_pre], EPS_RATE), log10(EPS_RATE), Y_LIMS, 0.02);
G_A = [repmat("Absent",numel(x_abs),1); repmat("Present",numel(x_pre),1)];

Sub = Views.PatientSpikeSz_Typed(:,{'EpiType3','MeanSpikeRate_perHour'});
Sub.Properties.VariableNames{'EpiType3'} = 'EpiType4';
Y_C = add_y_jitter_eps(to_log10_per_hour(Sub.MeanSpikeRate_perHour, EPS_RATE), Y_ZERO, Y_LIMS, 0.02);

[p_kw, tbl_kw] = kruskalwallis(Sub.MeanSpikeRate_perHour, Sub.EpiType4, 'off');
eta2_kw = tbl_kw{2,2} / tbl_kw{end,2};

f1 = figure('Color','w','Position',[60 60 950 520]);
tiledlayout(f1,1,2,'TileSpacing','compact','Padding','loose');

axA = nexttile; hold(axA,'on'); box(axA,'off'); grid(axA,'on');
boxchart(axA, categorical(G_A), Y_A, 'BoxFaceAlpha',0.25,'MarkerStyle','none');
swarmchart(axA, categorical(G_A), Y_A, 18, 'filled','MarkerFaceAlpha',0.18);
yline(axA, Y_ZERO, ':', 'Color',[0.4 0.4 0.4], 'LineWidth',1.2);
ylim(axA, Y_LIMS); ylabel(axA,'Spikes/hour (log scale)');
set_log10_ticks(axA,'y',EPS_RATE,Y_LIMS);
add_median_ci_overlay(axA, 1, med_abs, lo_abs, hi_abs, EPS_RATE);
add_median_ci_overlay(axA, 2, med_pre, lo_pre, hi_pre, EPS_RATE);
t = title(axA,'A. Reported presence or absence of spikes');
add_sigbar(axA, 1, 2, Y_LIMS(2)-0.08*range(Y_LIMS), p_label(pA));
set(axA,'FontSize',20);
t.Units='normalized'; t.Position(2) = t.Position(2)+TITLE_Y_OFFSET;
labelsA = string(axA.XTickLabel);
labelsA(labelsA=="Absent")  = sprintf('Absent (N=%d)',  nnz(isfinite(x_abs)));
labelsA(labelsA=="Present") = sprintf('Present (N=%d)', nnz(isfinite(x_pre)));
axA.XTickLabel = labelsA; axA.XTickLabelRotation = 20;

axC = nexttile; hold(axC,'on'); box(axC,'off'); grid(axC,'on');
boxchart(axC, Sub.EpiType4, Y_C, 'BoxFaceAlpha',0.25,'MarkerStyle','none');
swarmchart(axC, Sub.EpiType4, Y_C, 18, 'filled','MarkerFaceAlpha',0.18);
yline(axC, Y_ZERO, ':', 'Color',[0.4 0.4 0.4], 'LineWidth',1.2);
ylim(axC, Y_LIMS); ylabel(axC,'Spikes/hour (log scale)');
set_log10_ticks(axC,'y',EPS_RATE,Y_LIMS);
cats = categories(Sub.EpiType4);
for k = 1:numel(cats)
    xg = Sub.MeanSpikeRate_perHour(Sub.EpiType4 == cats{k});
    [mg, log, hig] = bootstrap_median_ci(xg, nBoot, alpha);
    add_median_ci_overlay(axC, k, mg, log, hig, EPS_RATE);
end
t = title(axC,'B. Epilepsy subtype');
set(axC,'FontSize',20);
t.Units='normalized'; t.Position(2) = t.Position(2)+TITLE_Y_OFFSET;
SubStats = Views.Canonical3_Stats;
labelsC = string(axC.XTickLabel);
for i = 1:height(SubStats)
    lab = string(SubStats.EpiType4(i));
    labelsC(labelsC==lab) = sprintf('%s (N=%d)', lab, SubStats.GroupCount(i));
end
axC.XTickLabel = labelsC; axC.XTickLabelRotation = 20;

SubtypePairs = Views.Canonical3_Pairs;
p_pair_bonf  = Views.PvalsPairwiseBonf;
cats3 = categorical(string(categories(Sub.EpiType4)));
yTop = Y_LIMS(2); yStep = 0.08*range(Y_LIMS); y0 = yTop - 0.05*range(Y_LIMS);
for i = 1:size(SubtypePairs,1)
    x1  = find(cats3 == categorical(string(SubtypePairs(i,1))));
    x2  = find(cats3 == categorical(string(SubtypePairs(i,2))));
    pval = p_pair_bonf(i);
    if isnan(pval), continue; end
    if pval<1e-3, lab="***"; elseif pval<1e-2, lab="**"; elseif pval<5e-2, lab="*"; else, lab="ns"; end
    add_sigbar(axC, x1, x2, y0-(i-1)*yStep, lab);
end

Fig1Stats.p_rankSum_A   = pA;
Fig1Stats.effectA_cliff = effectA;
Fig1Stats.m_pre = med_pre; Fig1Stats.lo_pre = lo_pre; Fig1Stats.hi_pre = hi_pre;
Fig1Stats.m_abs = med_abs; Fig1Stats.lo_abs = lo_abs; Fig1Stats.hi_abs = hi_abs;
Fig1Stats.p_kw_C    = p_kw;
Fig1Stats.eta2_kw_C = eta2_kw;
Fig1Stats.p_pair_bonf = Views.PvalsPairwiseBonf;

Sub2 = Views.Canonical3_SubsetTable;
subNames = string(categories(Sub2.EpiType4));
subMed = nan(numel(subNames),1); subCI = nan(numel(subNames),3);
for k=1:numel(subNames)
    x = Sub2.MeanSpikeRate_perHour(Sub2.EpiType4==subNames(k));
    [m,lo,hi] = bootstrap_median_ci(x, nBoot, alpha);
    subMed(k)=m; subCI(k,:)=[m lo hi];
end
Fig1Stats.SubtypeStatsTable = table(subNames, subMed, subCI(:,2), subCI(:,3), ...
    'VariableNames',{'Group','Median','CI_lo','CI_hi'});
end

%% =====================================================================
%% MAIN MODEL FIGURE (panels C and D only)
%% =====================================================================

function FigMain = make_model_figure(MMR, outPath)
if nargin < 2, outPath = ''; end

FONT_SIZE = 20;
T   = MMR.ModelTable;
mdl = MMR.mdl_M1;

FigMain = figure('Color','w','Position',[60 60 1300 560]);

% Two panels side by side, more room now that A/B are removed
ax_w     = 0.35;
ax_h     = 0.72;
col1_bot = 0.21;   % C left edge — extra room for long ytick labels
col2     = 0.64;
row1     = 0.12;

axC = axes('Position',[col1_bot, row1, ax_w, ax_h]);
axD = axes('Position',[col2,     row1, ax_w, ax_h]);

%% Panel C: Forest plot
axes(axC); hold(axC,'on'); box(axC,'off'); grid(axC,'on');

[beta_m, betanames_m, stats_m] = fixedEffects(mdl);
raw_names = string(betanames_m.Name);

if ~isempty(MMR.BootstrapTable1)
    BT = MMR.BootstrapTable1; ci_label = '95% Bootstrap CI';
    OR_C    = nan(numel(raw_names),1);
    OR_lo_C = nan(numel(raw_names),1);
    OR_hi_C = nan(numel(raw_names),1);
    for k = 1:numel(raw_names)
        bt_row = BT(string(BT.Term)==raw_names(k),:);
        if ~isempty(bt_row)
            OR_C(k)=bt_row.OR; OR_lo_C(k)=bt_row.OR_CI_lo; OR_hi_C(k)=bt_row.OR_CI_hi;
        else
            OR_C(k)=exp(beta_m(k));
            OR_lo_C(k)=exp(beta_m(k)-1.96*stats_m.SE(k));
            OR_hi_C(k)=exp(beta_m(k)+1.96*stats_m.SE(k));
        end
    end
else
    OR_C    = exp(beta_m);
    OR_lo_C = exp(beta_m-1.96*stats_m.SE);
    OR_hi_C = exp(beta_m+1.96*stats_m.SE);
    ci_label = '95% Laplace CI';
end

disp_names = raw_names;
disp_names(disp_names=="(Intercept)")                   = "Intercept";
disp_names(disp_names=="LogSpikesPerHour")              = "Log spike rate";
disp_names(disp_names=="AbsLag_years")                  = "EEG-visit gap (years)";
disp_names(disp_names=="VisitAfterEEG")                 = "Visit after vs before EEG";
disp_names(disp_names=="EpiType3_cat_Frontal")          = "Frontal vs Temporal";
disp_names(disp_names=="EpiType3_cat_General")          = "Generalized vs Temporal";
disp_names(disp_names=="LogSpikesPerHour:AbsLag_years") = "Spike rate effect per year of gap";
disp_names(disp_names=="LogSpikesPerHour:VisitAfterEEG") = "Spike rate effect: visit before or after";

isInt       = (disp_names=="Intercept");
OR_C        = OR_C(~isInt);
OR_lo_C     = OR_lo_C(~isInt);
OR_hi_C     = OR_hi_C(~isInt);

pvals_C        = stats_m.pValue(~isInt);   % Wald fallback
raw_names_plot = raw_names(~isInt);
if ~isempty(MMR.BootstrapTable1)
    BT_fig = MMR.BootstrapTable1;
    for k = 1:numel(raw_names_plot)
        bt_row = BT_fig(string(BT_fig.Term)==raw_names_plot(k), :);
        if ~isempty(bt_row) && ismember('Boot_p', bt_row.Properties.VariableNames)
            pvals_C(k) = bt_row.Boot_p;
        end
    end
end

disp_names  = disp_names(~isInt);
nTerms      = numel(OR_C);
plot_order  = nTerms:-1:1;

for k = 1:nTerms
    idx = plot_order(k);
    col = [0.1 0.3 0.7]; if pvals_C(idx)>=0.05, col=[0.6 0.6 0.6]; end
    plot(axC,[OR_lo_C(idx),OR_hi_C(idx)],[k k],'-','Color',col,'LineWidth',2.5);
    scatter(axC,OR_C(idx),k,100,col,'filled');
    p = pvals_C(idx);
    if p<0.001, pstr='p<0.001'; elseif p<0.05, pstr=sprintf('p=%.3f',p); else, pstr=sprintf('p=%.2f',p); end
    text(axC,OR_hi_C(idx)+0.005,k,pstr,'FontSize',FONT_SIZE-5,'VerticalAlignment','middle');
end
xline(axC,1,'k--','LineWidth',1.5);
set(axC,'FontSize',FONT_SIZE);
set(axC,'YTick',1:nTerms,'YTickLabel',disp_names(plot_order),'FontSize',FONT_SIZE-4);
xlabel(axC,sprintf('Odds Ratio (%s)',ci_label),'FontSize',FONT_SIZE);
th = title(axC,{'A. Spike rate, epilepsy type, and EEG-visit gap', 'predict seizure occurrence'}, ...
    'FontSize',FONT_SIZE,'FontWeight','bold');
th.Units = 'normalized'; th.Position(2) = th.Position(2) + 0.02; th.Position(1) = th.Position(1) - 0.13;
all_ors = [OR_lo_C; OR_hi_C];
xpad = 0.05 * (max(OR_hi_C) - min(OR_lo_C));
xlim(axC, [min(OR_lo_C) - xpad, max(OR_hi_C) + xpad]);

%% Panel D: Predicted probability vs spike rate by lag distance
axes(axD); hold(axD,'on'); box(axD,'off'); grid(axD,'on');

spike_grid_raw = linspace(0,50,200);
EPS_SPIKE      = 1e-3;
spike_grid_log = log(spike_grid_raw + EPS_SPIKE);
lag_vals       = [6/12, 2, 4];
lag_labels     = ["6 months","2 years","4 years"];
lag_colors     = [0.05 0.30 0.70;
                  0.15 0.50 0.80;
                  0.40 0.65 0.85];

[beta_pred, betanames_pred] = fixedEffects(mdl);
bnames      = string(betanames_pred.Name);
b_intercept = beta_pred(bnames=="(Intercept)");
b_spike     = beta_pred(bnames=="LogSpikesPerHour");
b_abslag    = beta_pred(bnames=="AbsLag_years");
b_dir       = beta_pred(bnames=="VisitAfterEEG");
b_int_prox  = beta_pred(bnames=="LogSpikesPerHour:AbsLag_years");
b_int_dir   = beta_pred(bnames=="LogSpikesPerHour:VisitAfterEEG");
dir_val     = 1;   % visit after EEG; temporal epilepsy reference

for k = 1:numel(lag_vals)
    lag = lag_vals(k);
    eta = b_intercept ...
        + b_spike   .* spike_grid_log ...
        + b_abslag  .* lag ...
        + b_dir     .* dir_val ...
        + b_int_prox.* spike_grid_log .* lag ...
        + b_int_dir .* spike_grid_log .* dir_val;
    prob = 1 ./ (1 + exp(-eta));
    plot(axD, spike_grid_raw, prob, '-', ...
        'Color',lag_colors(k,:),'LineWidth',2.5,'DisplayName',lag_labels(k));
end

% OR annotations
x_anno = 22;
for k = 1:numel(lag_vals)
    lag_k  = lag_vals(k);
    or_k   = exp(b_spike + b_int_prox*lag_k);
    eta_k  = b_intercept + b_spike*log(x_anno+EPS_SPIKE) + b_abslag*lag_k + ...
        b_dir*dir_val + b_int_prox*log(x_anno+EPS_SPIKE)*lag_k + ...
        b_int_dir*log(x_anno+EPS_SPIKE)*dir_val;
    prob_k = 1/(1+exp(-eta_k));
    text(axD, x_anno+1.0, prob_k+0.01, sprintf('OR=%.2f',or_k), ...
        'FontSize',FONT_SIZE-7,'Color',lag_colors(k,:), ...
        'VerticalAlignment','middle','HorizontalAlignment','left');
end

xlabel(axD,'Spike rate (spikes/hour)','FontSize',FONT_SIZE);
ylabel(axD,'P(seizure reported at visit)','FontSize',FONT_SIZE);
th = title(axD,{'B. Spike rates are most predictive', ...
    'when EEG is obtained close to the visit'}, ...
    'FontSize',FONT_SIZE,'FontWeight','bold');
th.Units = 'normalized'; th.Position(2) = th.Position(2) + 0.02;
lg = legend(axD,'Location','southeast','FontSize',FONT_SIZE-6);
title(lg,'EEG-visit lag');
xlim(axD,[0 30]); ylim(axD,[0.3 0.61]);
set(axD,'FontSize',FONT_SIZE);

%% Save
if strlength(string(outPath)) > 0
    if ~exist(fileparts(outPath),'dir'), mkdir(fileparts(outPath)); end
    exportgraphics(FigMain, outPath, 'Resolution',300);
    fprintf('Saved main model figure: %s\n', outPath);
end
end


%% =====================================================================
%% SUPPLEMENTAL FIGURE: EEG-visit context panels (old A and B)
%% =====================================================================

function FigSup = make_figSup_lag(MMR, Vuniq, ReportForKeptSessions, outPath)
if nargin < 4, outPath = ''; end

FONT_SIZE = 20;
T       = MMR.ModelTable;
refDate = datetime(2000,1,1);

FigSup = figure('Color','w','Position',[60 60 1300 560]);

ax_w  = 0.38;
ax_h  = 0.72;
col1  = 0.07;
col2  = 0.57;
row1  = 0.12;

axA = axes('Position',[col1, row1, ax_w, ax_h]);
axB = axes('Position',[col2, row1, ax_w, ax_h]);

%% Panel A: Seizure occurrence over time + median seizure frequency (right y-axis)
cohort_patients = unique(T.Patient);
V_cohort = Vuniq(ismember(Vuniq.Patient,cohort_patients) & ...
    (Vuniq.HasSz==0|Vuniq.HasSz==1),:);
V_cohort.YearsSinceFirst = days(V_cohort.VisitDate-refDate)/365.25;
bin_edges_sz   = [0, 1, 2, 3, 4];
bin_centers_sz = (bin_edges_sz(1:end-1) + bin_edges_sz(2:end)) / 2;
nBins_sz       = numel(bin_centers_sz);

bin_prop_sz  = nan(nBins_sz,1); bin_lo_sz   = nan(nBins_sz,1);
bin_hi_sz    = nan(nBins_sz,1); bin_n_sz    = zeros(nBins_sz,1);
bin_med_freq = nan(nBins_sz,1); bin_lo_freq = nan(nBins_sz,1);
bin_hi_freq  = nan(nBins_sz,1);
EPS_FREQ = 1e-3;

for b = 1:nBins_sz
    mask    = V_cohort.YearsSinceFirst >= bin_edges_sz(b) & ...
              V_cohort.YearsSinceFirst <  bin_edges_sz(b+1);
    vals_hz = V_cohort.HasSz(mask);   vals_hz = vals_hz(isfinite(vals_hz));
    vals_fr = V_cohort.Freq_R1(mask); vals_fr = vals_fr(isfinite(vals_fr));
    if numel(vals_hz) >= 10
        bin_prop_sz(b) = mean(vals_hz);
        bin_n_sz(b)    = numel(vals_hz);
        boot_p = nan(5000,1);
        for bb = 1:5000
            boot_p(bb) = mean(vals_hz(randi(numel(vals_hz),numel(vals_hz),1)));
        end
        bin_lo_sz(b) = prctile(boot_p,2.5);
        bin_hi_sz(b) = prctile(boot_p,97.5);
    end
    if numel(vals_fr) >= 10
        bin_med_freq(b) = median(vals_fr,'omitnan');
        boot_f = nan(5000,1);
        for bb = 1:5000
            boot_f(bb) = median(vals_fr(randi(numel(vals_fr),numel(vals_fr),1)),'omitnan');
        end
        bin_lo_freq(b) = prctile(boot_f,2.5);
        bin_hi_freq(b) = prctile(boot_f,97.5);
    end
end

validA    = isfinite(bin_prop_sz);
validFreq = isfinite(bin_med_freq);
COL_PROP  = [0.8 0.3 0.1];
COL_FREQ  = [0.1 0.55 0.55];

axes(axA); hold(axA,'on'); box(axA,'off'); grid(axA,'on');
yyaxis(axA,'left');
patch(axA, [bin_centers_sz(validA), fliplr(bin_centers_sz(validA))], ...
    [bin_lo_sz(validA)', fliplr(bin_hi_sz(validA)')], ...
    COL_PROP,'FaceAlpha',0.2,'EdgeColor','none');
plot(axA, bin_centers_sz(validA), bin_prop_sz(validA), 'o-', ...
    'Color',COL_PROP,'LineWidth',2,'MarkerFaceColor',COL_PROP,'MarkerSize',6);
for b = find(validA)'
    scatter(axA, bin_centers_sz(b), bin_prop_sz(b), bin_n_sz(b)/5, ...
        COL_PROP,'filled','MarkerFaceAlpha',0.25);
end
ylim(axA,[0 1]);
ylabel(axA,'Proportion with seizures','FontSize',FONT_SIZE,'Color',COL_PROP);
axA.YAxis(1).Color = COL_PROP;

yyaxis(axA,'right');
freq_log_med = log10(bin_med_freq  + EPS_FREQ);
freq_log_lo  = log10(bin_lo_freq   + EPS_FREQ);
freq_log_hi  = log10(bin_hi_freq   + EPS_FREQ);
patch(axA, [bin_centers_sz(validFreq), fliplr(bin_centers_sz(validFreq))], ...
    [freq_log_lo(validFreq)', fliplr(freq_log_hi(validFreq)')], ...
    COL_FREQ,'FaceAlpha',0.18,'EdgeColor','none');
plot(axA, bin_centers_sz(validFreq), freq_log_med(validFreq), 's--', ...
    'Color',COL_FREQ,'LineWidth',1.8,'MarkerFaceColor',COL_FREQ,'MarkerSize',5);
Y_LIM_FREQ = [-2 2];
ylim(axA, Y_LIM_FREQ);
set_log10_ticks(axA,'y',EPS_FREQ,Y_LIM_FREQ);
ylabel(axA,'Median sz/month (log scale)','FontSize',FONT_SIZE,'Color',COL_FREQ);
axA.YAxis(2).Color = COL_FREQ;

xlabel(axA,'Years after first visit','FontSize',FONT_SIZE);
th = title(axA,'A. Seizure burden tends to decrease over time', ...
    'FontSize',FONT_SIZE,'FontWeight','bold');
th.Units = 'normalized'; th.Position(2) = th.Position(2) + 0.02;
set(axA,'FontSize',FONT_SIZE);

%% Panel B: EEG-visit lag distribution
axes(axB); hold(axB,'on'); box(axB,'off'); grid(axB,'on');

abs_lags = T.AbsLag_years;
histogram(axB, abs_lags, 40,'FaceColor',[0.3 0.3 0.3],'FaceAlpha',0.6, ...
    'EdgeColor','none','Normalization','probability');
xline(axB, 1,'k--','LineWidth',1.2);
xlabel(axB,'Absolute EEG-visit gap (years)','FontSize',FONT_SIZE);
ylabel(axB,'Proportion of pairs','FontSize',FONT_SIZE);
th = title(axB,'B. EEG and visit are often separated by years', ...
    'FontSize',FONT_SIZE,'FontWeight','bold');
th.Units = 'normalized'; th.Position(2) = th.Position(2) + 0.02;
set(axB,'FontSize',FONT_SIZE);

%% Save
if strlength(string(outPath)) > 0
    if ~exist(fileparts(outPath),'dir'), mkdir(fileparts(outPath)); end
    exportgraphics(FigSup, outPath, 'Resolution',300);
    fprintf('Saved supplemental lag figure: %s\n', outPath);
end
end

%% =====================================================================
%% FIGURE S2
%% =====================================================================

function fS2 = make_figS2_sz_by_reported_spikes(Views, SzFreqPerPatient, nBoot, alpha, xLims_log10)
RS = resolve_reported_spike_status(Views.ReportForKeptSessions);
[gp, pid] = findgroups(RS.Patient);
hasPresent = splitapply(@(x) any(x=="present"), RS.ReportStatus, gp);
hasAbsent  = splitapply(@(x) any(x=="absent"),  RS.ReportStatus, gp);
RptP = table(pid, hasPresent, hasAbsent, 'VariableNames',{'Patient','HasPresent','HasAbsent'});

S2 = innerjoin(SzFreqPerPatient, RptP, 'Keys','Patient');
EpPatients = Views.PatientLevelSpikeRates.Patient(Views.IsEpilepsyMask);
S2 = innerjoin(S2, table(EpPatients,'VariableNames',{'Patient'}), 'Keys','Patient');

freq_allAbsent  = S2.MeanSzFreq(S2.HasAbsent & ~S2.HasPresent);
freq_anyPresent = S2.MeanSzFreq(S2.HasPresent);
freq_allAbsent  = freq_allAbsent(isfinite(freq_allAbsent));
freq_anyPresent = freq_anyPresent(isfinite(freq_anyPresent));
n_allAbsent  = numel(freq_allAbsent);
n_anyPresent = numel(freq_anyPresent);

p = ranksum(freq_allAbsent, freq_anyPresent, 'method','approx');
assert(p<0.001)

EPS_FREQ = 1e-3; Y_ZERO = log10(EPS_FREQ);
Y = [to_log10_per_month(freq_allAbsent, EPS_FREQ); to_log10_per_month(freq_anyPresent,EPS_FREQ)];
Y = add_y_jitter_eps(Y, Y_ZERO, xLims_log10, 0.02);
G = [repmat("All EEGs: no spikes",numel(freq_allAbsent),1); ...
     repmat("≥1 EEG: spikes present",numel(freq_anyPresent),1)];

[med1,lo1,hi1] = bootstrap_median_ci(freq_allAbsent,  nBoot, alpha);
[med2,lo2,hi2] = bootstrap_median_ci(freq_anyPresent, nBoot, alpha);

fS2 = figure('Color','w','Position',[100 100 800 520]);
ax = axes(fS2); hold(ax,'on'); box(ax,'off'); grid(ax,'on');
boxchart(ax,categorical(G),Y,'BoxFaceAlpha',0.25,'MarkerStyle','none');
swarmchart(ax,categorical(G),Y,18,'filled','MarkerFaceAlpha',0.18);
yline(ax,Y_ZERO,':','Color',[0.4 0.4 0.4],'LineWidth',1.2);
ylim(ax,xLims_log10); ylabel(ax,'Seizures/month (log scale)');
set_log10_ticks(ax,'y',EPS_FREQ,xLims_log10);
add_median_ci_overlay_month(ax,1,med1,lo1,hi1,EPS_FREQ);
add_median_ci_overlay_month(ax,2,med2,lo2,hi2,EPS_FREQ);

yl=ylim(ax); yMaxData=max(Y(isfinite(Y)));
yBar=yMaxData+0.06*range(yl);
yNeedTop=yBar+0.10*range(yl);
if yNeedTop>yl(2), ylim(ax,[yl(1) yNeedTop]); end
add_sigbar(ax,1,2,yBar,p_label(p));
t=title(ax,'Mean seizure frequency by reported spikes across EEGs');
t.Units='normalized'; t.Position(2)=1.03;
set(ax,'FontSize',20);

labels = string(ax.XTickLabel);
labels(labels=="All EEGs: no spikes")    = sprintf('All EEGs: no spikes (N=%d)',    n_allAbsent);
labels(labels=="≥1 EEG: spikes present") = sprintf('≥1 EEG: spikes present (N=%d)', n_anyPresent);
ax.XTickLabel = labels; ax.XTickLabelRotation = 20;

fprintf(['\nMedian [95%% CI] seizure frequency: %.2f [%.2f-%.2f] (no spikes) ' ...
    'vs %.2f [%.2f-%.2f] (spikes present) (p<0.001, Cliff''s d=%.2f)\n'], ...
    med1,lo1,hi1,med2,lo2,hi2,cliff_delta(freq_allAbsent,freq_anyPresent));
end

%% =====================================================================
%% TABLE 1
%% =====================================================================

function Table1_flat = build_table1_flat(Views, SzFreqPerPatient, Vuniq, EPS_RATE, nBoot, alpha)
Rk = Views.ReportForKeptSessions;
PL = Views.PatientLevelSpikeRates;
AllPatients = PL.Patient;
N_total = numel(AllPatients);

birth_str = strtrim(string(Rk.deid_birth_date));
isMiss    = birth_str=="" | birth_str=="null" | birth_str=="[null]";
birth_dt  = NaT(size(birth_str));
birth_dt(~isMiss) = datetime(birth_str(~isMiss),'InputFormat','yyyy-MM-dd');
refDate   = datetime(2000,1,1);
age_first = NaN(size(birth_dt));
age_first(~isnat(birth_dt)) = days(refDate - birth_dt(~isnat(birth_dt)))/365.25;
[ga, pidA] = findgroups(Rk.Patient);
AgeFirst = splitapply(@local_min_omitnan, age_first, ga);
AgeTable = innerjoin(table(AllPatients,'VariableNames',{'Patient'}), ...
    table(pidA,AgeFirst,'VariableNames',{'Patient','AgeFirst'}),'Keys','Patient');
age_vec = AgeTable.AgeFirst;
age_med = median(age_vec,'omitnan'); age_q = prctile(age_vec,[25,75]);

sex_raw = upper(strtrim(string(Rk.nlp_gender)));
[gs, pidS] = findgroups(Rk.Patient);
sex_per = splitapply(@local_first_nonmissing, sex_raw, gs);
SexTable = innerjoin(table(AllPatients,'VariableNames',{'Patient'}), ...
    table(pidS,sex_per,'VariableNames',{'Patient','SexCode'}),'Keys','Patient');
n_f = nnz(SexTable.SexCode=="F");
n_m = nnz(SexTable.SexCode=="M");
n_u = N_total - n_f - n_m;

E3    = strtrim(string(PL.EpiType3));
espec = strtrim(string(PL.EpilepsySpecific));
isTemp    = (E3=="Temporal") & Views.IsEpilepsyMask;
isFront   = (E3=="Frontal")  & Views.IsEpilepsyMask;
isGen     = (E3=="General")  & Views.IsEpilepsyMask;
isCanon   = isTemp | isFront | isGen;
isUnknown = Views.IsEpilepsyMask & ~isCanon & ...
    (ismissing(espec)|espec==""|espec=="Unclassified or Unspecified"|espec=="Unknown or MRN not found");
isOther   = Views.IsEpilepsyMask & ~isCanon & ~isUnknown;
n_epi     = nnz(Views.IsEpilepsyMask);
n_temp    = nnz(isTemp); n_front = nnz(isFront); n_gen    = nnz(isGen);
n_other   = nnz(isOther); n_subunk = nnz(isUnknown);

[gv, pidV] = findgroups(Vuniq.Patient);
nVisits = splitapply(@(x) numel(unique(x)), Vuniq.VisitDate, gv);
VisitsTable = innerjoin(table(AllPatients,'VariableNames',{'Patient'}), ...
    table(pidV,nVisits,'VariableNames',{'Patient','NumVisits'}),'Keys','Patient');
vis_med = median(VisitsTable.NumVisits,'omitnan'); vis_q = prctile(VisitsTable.NumVisits,[25,75]);

Sess = Views.SessionsForFigures;
[ge, pidE] = findgroups(Sess.Patient);
nEEG = splitapply(@(x) numel(unique(x)), Sess.Session, ge);
EEGTable = innerjoin(table(AllPatients,'VariableNames',{'Patient'}), ...
    table(pidE,nEEG,'VariableNames',{'Patient','NumEEG'}),'Keys','Patient');
eeg_med = median(EEGTable.NumEEG,'omitnan'); eeg_q = prctile(EEGTable.NumEEG,[25,75]);

SzJ = innerjoin(table(AllPatients,'VariableNames',{'Patient'}), SzFreqPerPatient,'Keys','Patient');
sf_vec = SzJ.MeanSzFreq(isfinite(SzJ.MeanSzFreq));
sf_med = median(sf_vec,'omitnan'); sf_q = prctile(sf_vec,[25,75]);
[~,sf_lo,sf_hi] = bootstrap_median_ci(sf_vec, nBoot, alpha);

sr_vec = PL.MeanSpikeRate_perHour(isfinite(PL.MeanSpikeRate_perHour));
sr_med = median(sr_vec,'omitnan'); sr_q = prctile(sr_vec,[25,75]);
[~,sr_lo,sr_hi] = bootstrap_median_ci(sr_vec, nBoot, alpha);

RS_all = resolve_reported_spike_status(Views.ReportForKeptSessions);
status = string(RS_all.ReportStatus); status(ismissing(RS_all.ReportStatus)) = "unknown";
RS_all.SpikeStatus = categorical(status,["absent","present","unknown"]);
n_rep_pre  = nnz(RS_all.SpikeStatus=="present");
n_rep_abs  = nnz(RS_all.SpikeStatus=="absent");
n_rep_unk  = nnz(RS_all.SpikeStatus=="unknown");
n_eegs_all = height(RS_all);

% Patient-level spike status
[gp_rs, pid_rs] = findgroups(RS_all.Patient);
hasPresent = splitapply(@(x) any(string(x)=="present"), RS_all.SpikeStatus, gp_rs);
hasAbsent  = splitapply(@(x) any(string(x)=="absent"),  RS_all.SpikeStatus, gp_rs);
% Present: any EEG with spikes reported
% Absent: no EEG with spikes reported, but at least one EEG with a report
% Unknown: no EEG with any report
pat_spike_status = repmat("unknown", numel(pid_rs), 1);
pat_spike_status(hasAbsent & ~hasPresent)  = "absent";
pat_spike_status(hasPresent)               = "present";
n_pats_pre = nnz(pat_spike_status=="present");
n_pats_abs = nnz(pat_spike_status=="absent");
n_pats_unk = nnz(pat_spike_status=="unknown");
n_pats_rs  = numel(pid_rs);

%% Restrict Vuniq to primary cohort patients only
Vuniq_cohort = Vuniq(ismember(Vuniq.Patient, AllPatients), :);

%% Follow-up duration: first to last clinic visit per patient
[gf, pidf] = findgroups(Vuniq_cohort.Patient);
FollowupDays = splitapply(@(d) days(max(d) - min(d)), Vuniq_cohort.VisitDate, gf);
FollowupYears = FollowupDays / 365.25;
FollowupTable = innerjoin(table(AllPatients,'VariableNames',{'Patient'}), ...
    table(pidf, FollowupYears,'VariableNames',{'Patient','FollowupYears'}), 'Keys','Patient');
fu_med = median(FollowupTable.FollowupYears,'omitnan');
fu_q   = prctile(FollowupTable.FollowupYears,[25,75]);

%% % visits with documented seizure frequency per patient
[gd, pidd] = findgroups(Vuniq_cohort.Patient);
FracDocumented = splitapply(@(f) mean(isfinite(f)), Vuniq_cohort.Freq_R1, gd);
FracDocTable = innerjoin(table(AllPatients,'VariableNames',{'Patient'}), ...
    table(pidd, FracDocumented,'VariableNames',{'Patient','FracDocumented'}), 'Keys','Patient');
doc_med = median(FracDocTable.FracDocumented,'omitnan') * 100;
doc_q   = prctile(FracDocTable.FracDocumented,[25,75]) * 100;

assert(numel(unique(Vuniq_cohort.Patient)) == N_total, ...
    'Vuniq_cohort patient count (%d) does not match N_total (%d)', ...
    numel(unique(Vuniq_cohort.Patient)), N_total);

OutVar = {}; OutStat = {};
OutVar{end+1,1}="Total N patients"; OutStat{end+1,1}=sprintf('%d',N_total);
OutVar{end+1,1}="Age at first visit (years)"; OutStat{end+1,1}=sprintf('%.1f (%.1f-%.1f)',age_med,age_q(1),age_q(2));
OutVar{end+1,1}="Sex"; OutStat{end+1,1}="";
OutVar{end+1,1}="    Women";         OutStat{end+1,1}=sprintf('%d (%.1f%%)',n_f,100*n_f/N_total);
OutVar{end+1,1}="    Men";           OutStat{end+1,1}=sprintf('%d (%.1f%%)',n_m,100*n_m/N_total);
OutVar{end+1,1}="    Unknown/Other"; OutStat{end+1,1}=sprintf('%d (%.1f%%)',n_u,100*n_u/N_total);
OutVar{end+1,1}="Epilepsy subtype"; OutStat{end+1,1}="";
OutVar{end+1,1}="    Temporal lobe"; OutStat{end+1,1}=sprintf('%d (%.1f%%)',n_temp, 100*n_temp /max(1,n_epi));
OutVar{end+1,1}="    Frontal lobe";  OutStat{end+1,1}=sprintf('%d (%.1f%%)',n_front,100*n_front/max(1,n_epi));
OutVar{end+1,1}="    Generalized";   OutStat{end+1,1}=sprintf('%d (%.1f%%)',n_gen,  100*n_gen  /max(1,n_epi));
OutVar{end+1,1}="    Other";         OutStat{end+1,1}=sprintf('%d (%.1f%%)',n_other,100*n_other/max(1,n_epi));
OutVar{end+1,1}="    Unknown";       OutStat{end+1,1}=sprintf('%d (%.1f%%)',n_subunk,100*n_subunk/max(1,n_epi));
OutVar{end+1,1}="Number of clinic visits"; OutStat{end+1,1}=sprintf('%.1f (%.1f-%.1f)',vis_med,vis_q(1),vis_q(2));
OutVar{end+1,1}="Follow-up duration (years)"; 
OutStat{end+1,1}=sprintf('%.1f (%.1f-%.1f)',fu_med,fu_q(1),fu_q(2));
OutVar{end+1,1}="Visits with documented seizure frequency"; 
OutStat{end+1,1}=sprintf('%.1f%% (%.1f-%.1f)',doc_med,doc_q(1),doc_q(2));
OutVar{end+1,1}="Number of EEGs"; OutStat{end+1,1}=sprintf('%.1f (%.1f-%.1f)',eeg_med,eeg_q(1),eeg_q(2));
OutVar{end+1,1}="Mean seizure frequency (seizures/month)"; OutStat{end+1,1}=sprintf('%.2f (%.2f-%.2f); median CI [%.2f-%.2f]',sf_med,sf_q(1),sf_q(2),sf_lo,sf_hi);
OutVar{end+1,1}="Mean spike rate (spikes/hour)"; OutStat{end+1,1}=sprintf('%.2f (%.2f-%.2f); median CI [%.2f-%.2f]',sr_med,sr_q(1),sr_q(2),sr_lo,sr_hi);
OutVar{end+1,1}="EEGs with reported spikes";  OutStat{end+1,1}="N (% EEGs)";
OutVar{end+1,1}="    Present"; OutStat{end+1,1}=sprintf('%d (%.1f%%)',n_rep_pre,100*n_rep_pre/max(1,n_eegs_all));
OutVar{end+1,1}="    Absent";  OutStat{end+1,1}=sprintf('%d (%.1f%%)',n_rep_abs,100*n_rep_abs/max(1,n_eegs_all));
OutVar{end+1,1}="    Unknown"; OutStat{end+1,1}=sprintf('%d (%.1f%%)',n_rep_unk,100*n_rep_unk/max(1,n_eegs_all));
OutVar{end+1,1}="Patients with reported spikes"; OutStat{end+1,1}="N (% patients)";
OutVar{end+1,1}="    Present"; OutStat{end+1,1}=sprintf('%d (%.1f%%)',n_pats_pre,100*n_pats_pre/max(1,n_pats_rs));
OutVar{end+1,1}="    Absent";  OutStat{end+1,1}=sprintf('%d (%.1f%%)',n_pats_abs,100*n_pats_abs/max(1,n_pats_rs));
OutVar{end+1,1}="    Unknown"; OutStat{end+1,1}=sprintf('%d (%.1f%%)',n_pats_unk,100*n_pats_unk/max(1,n_pats_rs));


Table1_flat = table(string(OutVar), string(OutStat), 'VariableNames',{'Variable','Statistic'});
end

function write_tableS1(MMR, outPath)

function disp = clean_term(raw)
    disp = raw;
    disp = strrep(disp,'(Intercept)',                   'Intercept');
    disp = strrep(disp,'LogSpikesPerHour:AbsLag_years',  'Log spike rate x Absolute lag');
    disp = strrep(disp,'LogSpikesPerHour:VisitAfterEEG', 'Log spike rate x Visit after EEG');
    disp = strrep(disp,'LogSpikesPerHour',               'Log spike rate');
    disp = strrep(disp,'AbsLag_years',                   'Absolute lag (years)');
    disp = strrep(disp,'VisitAfterEEG',                  'Visit after vs before EEG');
    disp = strrep(disp,'SignedLag_years',               'Signed lag (years)');
    disp = strrep(disp,'EpiType3_cat_Frontal',          'Frontal vs Temporal epilepsy');
    disp = strrep(disp,'EpiType3_cat_General',          'Generalized vs Temporal epilepsy');
end

rows = {};

model_specs = {
    'M1','M1 (logistic, subtypes, interactions)',                       'logistic', MMR.FE_M1,     MMR.BootstrapTable1;
    'M2','M2 (logistic, subtypes, no interactions)',                    'logistic', MMR.FE_M2,     MMR.BootstrapTable2;
};

for mi = 1:size(model_specs,1)
    model_label = model_specs{mi,2};
    FE  = model_specs{mi,4};
    BT  = model_specs{mi,5};
    if isempty(FE), continue; end

    for i = 1:height(FE)
        term = string(FE.Term(i));
        if term=="(Intercept)", continue; end

        p_val = FE.p(i);
        if ~isempty(BT)
            bt_row_p = BT(string(BT.Term)==term, :);
            if ~isempty(bt_row_p) && ismember('Boot_p', bt_row_p.Properties.VariableNames)
                p_val = bt_row_p.Boot_p;
            end
        end

        est_pt = FE.OR(i);
        if ~isempty(BT)
            bt_row = BT(string(BT.Term)==term,:);
            if ~isempty(bt_row)
                ci_lo=bt_row.OR_CI_lo; ci_hi=bt_row.OR_CI_hi; ci_src='Bootstrap';
            else
                ci_lo=FE.OR_lo(i); ci_hi=FE.OR_hi(i); ci_src='Laplace';
            end
        else
            ci_lo=FE.OR_lo(i); ci_hi=FE.OR_hi(i); ci_src='Laplace';
        end

        if p_val<0.001, p_str='<0.001'; elseif p_val<0.01, p_str=sprintf('%.3f',p_val); else, p_str=sprintf('%.2f',p_val); end
        rows(end+1,:) = {model_label, clean_term(char(term)), ...
            sprintf('%.3f',est_pt), sprintf('%.3f',ci_lo), sprintf('%.3f',ci_hi), ...
            ci_src, p_str}; %#ok<AGROW>
    end
end

T_out = cell2table(rows, 'VariableNames', ...
    {'Model','Term','Estimate','CI_lower','CI_upper','CI_method','p_value'});
if ~exist(fileparts(outPath),'dir'), mkdir(fileparts(outPath)); end
writetable(T_out, outPath);
end


%% =====================================================================
%% HTML RESULTS
%% =====================================================================

function write_results_html(outPath, Views, SzFreqPerPatient, Fig1Stats, ...
    SpearmanResults_main, rs_all_main, p_all_main, n_all_main, ...
    rho_lo_main, rho_hi_main, subtype_ci_main, ...
    SpearmanResults_S1, rs_all_S1, p_all_S1, n_all_S1, ...
    rho_lo_S1, rho_hi_S1, subtype_ci_S1, ...
    ReportForKeptSessions, MMR, Vuniq, NearFarStats)

if ~exist(fileparts(outPath),'dir'), mkdir(fileparts(outPath)); end
fid = fopen(outPath,'w');
if fid==-1, error('Could not open %s', outPath); end

PL         = Views.PatientLevelSpikeRates;
N_total    = numel(PL.Patient);
n_eegs_all = height(ReportForKeptSessions);

Sf     = innerjoin(table(PL.Patient,'VariableNames',{'Patient'}), SzFreqPerPatient,'Keys','Patient');
sf_vec = Sf.MeanSzFreq(isfinite(Sf.MeanSzFreq));
sf_med = median(sf_vec,'omitnan');
[~,sf_ci_lo,sf_ci_hi] = bootstrap_median_ci(sf_vec,5000,0.05);

sr_vec = PL.MeanSpikeRate_perHour(isfinite(PL.MeanSpikeRate_perHour));
sr_med = median(sr_vec,'omitnan');
[~,sr_ci_lo,sr_ci_hi] = bootstrap_median_ci(sr_vec,5000,0.05);

%% Follow-up duration and documentation rate (cohort patients only)
AllPats   = unique(Views.PatientLevelSpikeRates.Patient);
Vuniq_coh = Vuniq(ismember(Vuniq.Patient, AllPats), :);

[gf, pidf] = findgroups(Vuniq_coh.Patient);
fu_days  = splitapply(@(d) days(max(d)-min(d)), Vuniq_coh.VisitDate, gf);
fu_years = fu_days / 365.25;
FuTable  = innerjoin(table(AllPats,'VariableNames',{'Patient'}), ...
    table(pidf, fu_years,'VariableNames',{'Patient','FU'}), 'Keys','Patient');
fu_med = median(FuTable.FU,'omitnan');
fu_q   = prctile(FuTable.FU, [25 75]);

[gd, pidd] = findgroups(Vuniq_coh.Patient);
frac_doc = splitapply(@(f) mean(isfinite(f)), Vuniq_coh.Freq_R1, gd);
DocTable = innerjoin(table(AllPats,'VariableNames',{'Patient'}), ...
    table(pidd, frac_doc,'VariableNames',{'Patient','FracDoc'}), 'Keys','Patient');
doc_med = median(DocTable.FracDoc,'omitnan') * 100;
doc_q   = prctile(DocTable.FracDoc, [25 75]) * 100;


%% Patient-level spike status for cohort summary
RS_html = resolve_reported_spike_status(ReportForKeptSessions);
[gp_html, pid_html] = findgroups(RS_html.Patient);
hasPresent_html = splitapply(@(x) any(string(x)=="present"), RS_html.ReportStatus, gp_html);
n_pats_present_html = nnz(hasPresent_html);
n_pats_total_html   = numel(pid_html);
pct_present_html    = 100 * n_pats_present_html / n_pats_total_html;


fprintf(fid,'<html><head><meta charset="UTF-8"><title>Results</title></head><body>\n');

%% Cohort
fprintf(fid,'<h2>Cohort summary</h2>\n');
EC = Views.ExclusionCounts;
fprintf(fid,['<p>Of %d patients with EEG data in the Penn Epilepsy Center database, ' ...
    '%d were excluded because their EEG was not an outpatient routine recording of less than 4 hours, ' ...
    '%d were excluded without an LLM-confirmed epilepsy diagnosis, and ' ...
    '%d were excluded without a documented seizure frequency, yielding a final cohort of ' ...
    '%d patients with %d EEGs (Fig. S1). ' ...
    'Median follow-up from first to last clinic visit was %.1f years (IQR %.1f&ndash;%.1f). ' ...
    'Across patients, a median of %.1f%% (IQR %.1f&ndash;%.1f%%) of clinic visits had a documented seizure frequency. ' ...
    '%d patients (%.1f%%) had spikes reported on at least one EEG. ' ...
    'Median [95%% CI] monthly seizure frequency was %1.2f [%1.2f-%1.2f], and median spikes/hour was %1.2f [%1.2f-%1.2f] (Table 1).</p>\n'], ...
    EC.nTotal, ...
    EC.nTotal - EC.nAfterOutptRoutine, ...
    EC.nExcludedNoEpilepsy, ...
    EC.nExcludedNoSzFreq, ...
    N_total, n_eegs_all, ...
    fu_med, fu_q(1), fu_q(2), ...
    doc_med, doc_q(1), doc_q(2), ...
    n_pats_present_html, pct_present_html, ...
    sf_med, sf_ci_lo, sf_ci_hi, ...
    sr_med, sr_ci_lo, sr_ci_hi);

%% Figure 1
fprintf(fid,'<h2>Spike rates by patient groups</h2>\n');
fprintf(fid,'<p>Detected spike rates were higher in EEGs with clinically-reported spikes (median %.2f [95%% CI %.2f-%.2f] spikes/hour) than without (%.2f [%.2f-%.2f] spikes/hour) (%s, Cliff''s &delta;=%.2f; Fig. 2A). ', ...
    Fig1Stats.m_pre, Fig1Stats.lo_pre, Fig1Stats.hi_pre, ...
    Fig1Stats.m_abs, Fig1Stats.lo_abs, Fig1Stats.hi_abs, ...
    format_p_html(Fig1Stats.p_rankSum_A), Fig1Stats.effectA_cliff);
fprintf(fid,'Spike rates differed across epilepsy subtypes, with the highest rates in generalized epilepsy (Kruskal-Wallis %s, &eta;&sup2;&asymp;%.3f; Fig. 2B).</p>\n', ...
    format_p_html(Fig1Stats.p_kw_C), Fig1Stats.eta2_kw_C);

%% Figure 2
fprintf(fid,'<h2>Spike rate and seizure frequency</h2>\n');
fprintf(fid,'<p>Spike rate and seizure frequency were positively correlated across all epilepsy patients (N=%d, &rho;=%.2f [95%% CI %.2f-%.2f], %s). ', ...
    n_all_main, rs_all_main, rho_lo_main, rho_hi_main, format_p_html(p_all_main));
fprintf(fid,'Subtype-specific correlations were significant for generalized epilepsy (N=%d, &rho;=%.2f [%.2f-%.2f], Bonferroni-adjusted %s) and temporal lobe epilepsy (N=%d, &rho;=%.2f [%.2f-%.2f], Bonferroni-adjusted %s), but not frontal lobe epilepsy (N=%d, &rho;=%.2f [%.2f-%.2f], Bonferroni-adjusted %s; Fig. 3). ', ...
    SpearmanResults_main.N(1), SpearmanResults_main.Spearman_r(1), subtype_ci_main.ci_lo(1), subtype_ci_main.ci_hi(1), format_p_html(SpearmanResults_main.p_bonf(1)), ...
    SpearmanResults_main.N(2), SpearmanResults_main.Spearman_r(2), subtype_ci_main.ci_lo(2), subtype_ci_main.ci_hi(2), format_p_html(SpearmanResults_main.p_bonf(2)), ...
    SpearmanResults_main.N(3), SpearmanResults_main.Spearman_r(3), subtype_ci_main.ci_lo(3), subtype_ci_main.ci_hi(3), format_p_html(SpearmanResults_main.p_bonf(3)));

% S1 Spearman (non-zero only)
canonical_order = ["General","Temporal","Frontal"];
g = canonical_order(2);
row   = SpearmanResults_S1(strcmp(string(SpearmanResults_S1.Group), g), :);
rowCI = subtype_ci_S1(subtype_ci_S1.Group == g, :);
    
if isfinite(rowCI.ci_lo) && isfinite(rowCI.ci_hi)
    ci_str = sprintf('[%.2f-%.2f]', rowCI.ci_lo, rowCI.ci_hi);
else
    ci_str = '';
end
fprintf(fid,['When restricting to patients with non-zero spike rates and seizure frequencies, '...
    'results were similar, although the temporal epilepsy correlation was '...
    'no longer significant in this subgroup (Bonferroni-adjusted p = %.2f), although the magnitude was similar (&rho;=%.2f %s; Fig. S2). '],row.p_bonf,row.Spearman_r, ci_str);


fprintf(fid,'Patients with spikes on at least one EEG had higher mean seizure frequencies (Fig. S3).</p>\n');

%% Figure 3
fprintf(fid,'<h2>Mixed effects model</h2>\n');

if ~isempty(MMR.BootstrapTable1)
    BT = MMR.BootstrapTable1; ci_source = 'bootstrap';
else
    BT = MMR.FE_M1;
    BT.Properties.VariableNames{'OR_lo'} = 'OR_CI_lo';
    BT.Properties.VariableNames{'OR_hi'} = 'OR_CI_hi';
    ci_source = 'Laplace approximation';
end
fprintf(fid,'<p><em>CIs from %s.</em></p>\n', ci_source);

getRow = @(tbl,nm) tbl(string(tbl.Term)==nm,:);
r_spike    = getRow(BT,'LogSpikesPerHour');
r_dir      = getRow(BT,'VisitAfterEEG');
r_int_prox = getRow(BT,'LogSpikesPerHour:AbsLag_years');
r_int_dir  = getRow(BT,'LogSpikesPerHour:VisitAfterEEG');
r_frontal  = getRow(BT,'EpiType3_cat_Frontal');
r_general  = getRow(BT,'EpiType3_cat_General');
FE1  = MMR.FE_M1;
BT1  = MMR.BootstrapTable1;
getP = @(nm) get_p_preferred(FE1, BT1, nm);

n_pairs = height(MMR.ModelTable);
n_pats  = numel(unique(MMR.ModelTable.Patient));

fprintf(fid,['<p>Seizure frequency varies over time within individuals, '...
    'and we hypothesized that spike rates track this variability, ' ...
    'predicting a stronger spike-seizure association for clinic visits close in time to EEGs. '...
    'To test this, we fit logistic mixed effects models on all EEG-visit pairs for patients '...
    'with known epilepsy subtype (N=%d pairs, %d patients), with interaction terms allowing '...
    'the spike-seizure association to vary with the temporal distance between EEG and visit (Fig. S4). A likelihood ratio '...
    'test confirmed that these interactions '...
    'jointly improved model fit over a model without them (&chi;&sup2;(2), %s). </p>\n'], ...
    n_pairs, n_pats, format_p_html(MMR.LRT_p));

% e^1 increase in spike rate = 2.718x increase in raw spike rate
% OR corresponds to that fold-increase; express as percent
pct_increase = (r_spike.OR - 1) * 100;
fold_increase = exp(1);  % one unit on log scale = e-fold increase in raw rate

fprintf(fid,['<p>Higher spike rates were associated with higher odds of reporting seizures at a clinic visit ' ...
    '(OR=%.2f [95%% CI %.2f-%.2f], %s; Fig. 4A), implying that a %.1f-fold increase in spike rate ' ...
    'is associated with %.0f%% higher odds of seizure reporting at a clinic visit. '], ...
    r_spike.OR, r_spike.OR_CI_lo, r_spike.OR_CI_hi, format_p_html(getP('LogSpikesPerHour')), ...
    fold_increase, pct_increase);

fprintf(fid,['The spike-seizure association attenuated with greater EEG-visit distance, although the effect was small '...
    '(OR=%.3f [%.3f-%.3f], %s): e.g., the model-predicted OR for spike rate was '], ...
    r_int_prox.OR, r_int_prox.OR_CI_lo, r_int_prox.OR_CI_hi, format_p_html(getP('LogSpikesPerHour:AbsLag_years')));


b_spike_val    = MMR.FE_M1.Beta(string(MMR.FE_M1.Term) == "LogSpikesPerHour");
    b_int_prox_val = MMR.FE_M1.Beta(string(MMR.FE_M1.Term) == "LogSpikesPerHour:AbsLag_years");
    or_at_lag      = exp(b_spike_val + b_int_prox_val .* [0.5, 2, 4]);
fprintf(fid,['%.2f at a 6-month lag, %.2f at 2 years, and %.2f at 4 years ' ...
    '(Fig. 4B). '], ...
    or_at_lag(1), or_at_lag(2), or_at_lag(3));


fprintf(fid,['Spike rates from EEGs obtained before versus after a clinic '...
    'visit were similarly associated with seizure occurrence  (interaction OR=%.3f [%.3f-%.3f], %s). '], ...
    r_int_dir.OR, r_int_dir.OR_CI_lo, r_int_dir.OR_CI_hi, format_p_html(getP('LogSpikesPerHour:VisitAfterEEG')));

fprintf(fid,'Visits occurring after the EEG had a lower baseline odds of seizure reporting (OR=%.2f [%.2f-%.2f], %s), consistent with gradual clinical improvement over time (Fig. S4). ', ...
    r_dir.OR, r_dir.OR_CI_lo, r_dir.OR_CI_hi, format_p_html(getP('VisitAfterEEG')));
fprintf(fid,'Compared with temporal lobe epilepsy, generalized epilepsy had a lower baseline odds of seizure reporting (OR=%.2f [%.2f-%.2f], %s), while frontal lobe epilepsy did not differ significantly (OR=%.2f [%.2f-%.2f], %s). ', ...
    r_general.OR, r_general.OR_CI_lo, r_general.OR_CI_hi, format_p_html(getP('EpiType3_cat_General')), ...
    r_frontal.OR, r_frontal.OR_CI_lo, r_frontal.OR_CI_hi, format_p_html(getP('EpiType3_cat_Frontal')));
fprintf(fid,[' Our secondary analysis also found that spike-seizure correlations were stronger '...
    'for clinic visits close in time to the EEG (Fig. S5). '])
fprintf(fid,['Together, these results confirm a positive spike rate–seizure association '...
    'and suggest it is strongest when the EEG is obtained close to the clinic visit, '...
    'consistent with spike rates tracking within-individual seizure burden over time.</p>\n']);

fprintf(fid,'<h2>Bootstrap diagnostics</h2>\n');
if isfield(MMR,'BootstrapConvergence')
    BC = MMR.BootstrapConvergence;
    fprintf(fid,'<p>Primary model (M1): %d/%d bootstrap iterations converged (%.1f%%).</p>\n', ...
        BC.M1_nConverged, BC.M1_nTotal, 100*BC.M1_nConverged/max(1,BC.M1_nTotal));
    fprintf(fid,'<p>Alternate model (M2): %d/%d bootstrap iterations converged (%.1f%%).</p>\n', ...
        BC.M2_nConverged, BC.M2_nTotal, 100*BC.M2_nConverged/max(1,BC.M2_nTotal));
    if BC.M1_nConverged/max(1,BC.M1_nTotal) < 0.95
        fprintf(fid,'<p><strong>Warning: M1 convergence rate is below 95%%. CIs may be unreliable.</strong></p>\n');
    end
    if BC.M2_nConverged/max(1,BC.M2_nTotal) < 0.95
        fprintf(fid,'<p><strong>Warning: M2 convergence rate is below 95%%. CIs may be unreliable.</strong></p>\n');
    end
else
    fprintf(fid,'<p>Bootstrap convergence information not available (nBoot=0).</p>\n');
end

%% Fig S5 legend
fprintf(fid,'<h2>Figure S5 legend</h2>\n');
fprintf(fid,['<p><strong>Fig. S5. The association between interictal spike rate and seizure frequency ' ...
    'is higher for clinic visits close in time to EEGs.</strong> ' ...
    '<strong>A:</strong> Distribution of the absolute time difference between clinic visits and EEG recordings, ' ...
    'taking the minimum in the case of multiple EEGs per patient. ' ...
    'Visits were stratified into short gap and long gap groups based on tertiles of the visit&ndash;EEG gap ' ...
    'distribution (lower third = short gap; upper third = long gap). ' ...
    'Shaded regions indicate the lower, middle, and upper thirds of the gap distribution, ' ...
    'with dashed vertical lines marking the tertile cutoffs (%.0f days and %.0f days). ' ...
    '<strong>B:</strong> Bootstrap distribution (%d iterations) of the difference in Spearman correlation ' ...
    'coefficients between interictal spike rate and seizure frequency for near versus far visit windows ' ...
    '(&Delta;&rho; = &rho;<sub>short gap</sub> &minus; &rho;<sub>long gap</sub>). ' ...
    'The spike&ndash;seizure correlation was stronger when clinic visits occurred closer in time to EEG acquisition ' ...
    '(N = %d patients with both short-gap and long-gap visits; ' ...
    '&rho;<sub>short gap</sub> = %.2f; &rho;<sub>long gap</sub> = %.2f; ' ...
    'observed [95%% CI] &Delta;&rho; = %.3f [%.2f&ndash;%.2f], one-sided p = %.3f).</p>\n'], ...
    NearFarStats.nearDays, NearFarStats.farDays, ...
    numel(NearFarStats.delta_boot), ...
    NearFarStats.nPatients, ...
    NearFarStats.rho_near, NearFarStats.rho_far, ...
    NearFarStats.delta_obs, NearFarStats.delta_ci_lo, NearFarStats.delta_ci_hi, ...
    NearFarStats.p_one_sided);

fprintf(fid,'</body></html>\n');
fclose(fid);
end

%% =====================================================================
%% REPORT STATUS RESOLUTION
%% =====================================================================

function ReportSlim = resolve_reported_spike_status(ReportForKeptSessions)
any_spikes    = string(ReportForKeptSessions.report_SPORADIC_EPILEPTIFORM_DISCHARGES);
isMainPresent = (any_spikes=="present");
isMainAbsent  = (any_spikes=="absent");

rawF = lower(strtrim(string(ReportForKeptSessions.jay_focal_epi)));
rawM = lower(strtrim(string(ReportForKeptSessions.jay_multifocal_epi)));
rawG = lower(strtrim(string(ReportForKeptSessions.jay_gen_epi)));

isF_P=rawF=="present"; isF_A=rawF=="absent";
isM_P=rawM=="present"; isM_A=rawM=="absent";
isG_P=rawG=="present"; isG_A=rawG=="absent";

presentJay_any = isF_P|isM_P|isG_P;
allJay_absent  = isF_A&isM_A&isG_A;
blankMain      = ~(isMainPresent|isMainAbsent);
blankJayAll    = ~(isF_P|isF_A)&~(isM_P|isM_A)&~(isG_P|isG_A);

if any((allJay_absent&isMainPresent)|(isMainAbsent&isF_P&isM_P&isG_P))
    error('Discordant spike presence between main and jay_* columns.');
end

repCombined = strings(height(ReportForKeptSessions),1);
repCombined(isMainPresent|presentJay_any) = "present";
repCombined(allJay_absent&blankMain)      = "absent";
repCombined(isMainAbsent&blankJayAll)     = "absent";
repCombined(repCombined=="")              = "unknown";

ReportSlim = ReportForKeptSessions(:,{'Patient','Session'});
ReportSlim.ReportStatus = categorical(repCombined,["absent","present","unknown"]);
end

%% =====================================================================
%% SPEARMAN FIGURE
%% =====================================================================

function [SpearmanResults, rs_all, p_all, n_all, rho_lo, rho_hi, subtype_ci] = ...
    spearman_plotting_function(PatientSpikeSz_All, PatientSpikeSz_Typed, ...
    canonical3, spearman_xLims, spearman_yLims, fig_out, freqFieldName, labelSuffix, nonZeroOnly)

nBoot = 5000; alpha = 0.05; fontL = 20;
COL_all   = [0.45 0.45 0.45];
COL_front = [0.93 0.69 0.13];
COL_temp  = [0.85 0.33 0.10];
COL_gen   = [0.00 0.45 0.74];
panelOrder = {'Frontal','Temporal','General'};
panelTitle = {'B. Frontal','C. Temporal','D. General'};

x_all = double(PatientSpikeSz_All.MeanSpikeRate_perHour);
y_all = double(PatientSpikeSz_All.(freqFieldName));
mask_all = isfinite(x_all) & isfinite(y_all);
if nonZeroOnly, mask_all = mask_all & (x_all>0) & (y_all>0); end
n_all = sum(mask_all);

if n_all>=3
    [rs_all,p_all] = corr(x_all(mask_all),y_all(mask_all),'Type','Spearman','Rows','complete');
    [~,rho_lo,rho_hi] = bootstrap_spearman_ci(x_all(mask_all),y_all(mask_all),nBoot,alpha);
else
    rs_all=NaN; p_all=NaN; rho_lo=NaN; rho_hi=NaN;
end

rowsOut = {};
subtype_ci = table(string(canonical3(:)), nan(numel(canonical3),1), nan(numel(canonical3),1), nan(numel(canonical3),1), ...
    'VariableNames',{'Group','rho','ci_lo','ci_hi'});
for ii = 1:numel(canonical3)
    g = canonical3(ii);
    mBase = (PatientSpikeSz_Typed.EpiType3==g);
    x = double(PatientSpikeSz_Typed.MeanSpikeRate_perHour(mBase));
    y = double(PatientSpikeSz_Typed.(freqFieldName)(mBase));
    mask = isfinite(x)&isfinite(y);
    if nonZeroOnly, mask=mask&(x>0)&(y>0); end
    n = sum(mask);
    if n>=3
        [rs,p] = corr(x(mask),y(mask),'Type','Spearman','Rows','complete');
        [rho,lo,hi] = bootstrap_spearman_ci(x(mask),y(mask),nBoot,alpha);
    else
        rs=NaN; p=NaN; rho=NaN; lo=NaN; hi=NaN;
    end
    subtype_ci.rho(ii)=rho; subtype_ci.ci_lo(ii)=lo; subtype_ci.ci_hi(ii)=hi;
    rowsOut(end+1,:) = {char(g),n,rs,p}; %#ok<AGROW>
end
SpearmanResults = cell2table(rowsOut,'VariableNames',{'Group','N','Spearman_r','p_raw'});
SpearmanResults.p_bonf = min(SpearmanResults.p_raw*height(SpearmanResults),1);

x_used = x_all(mask_all); y_used = y_all(mask_all);
minpos_rate = min(x_used(x_used>0)); if isempty(minpos_rate)||~isfinite(minpos_rate), minpos_rate=1e-6; end
minpos_sz   = min(y_used(y_used>0)); if isempty(minpos_sz)||~isfinite(minpos_sz), minpos_sz=1e-6; end
eps_rate = 0.5*minpos_rate; eps_sz = 0.5*minpos_sz;
xZero = log10(eps_sz); yZero = log10(eps_rate);

Tall = table(x_used, y_used, ...
    log10(x_used+(x_used<=0).*eps_rate), log10(y_used+(y_used<=0).*eps_sz), ...
    'VariableNames',{'SpikeRate_perHour','SzFreq','logSpikeRate','logSzFreq'});
isZeroSz  = (Tall.SzFreq==0); isZeroRate = (Tall.SpikeRate_perHour==0);
nonZero   = ~(isZeroSz|isZeroRate);

f2 = figure('Color','w','Position',[60 60 1200 900]);
tiledlayout(f2,2,2,'Padding','compact','TileSpacing','compact');

axA = nexttile(1); hold(axA,'on'); grid(axA,'on'); box(axA,'off');
xline(axA,xZero,':','Color',[0.4 0.4 0.4],'LineWidth',1.2);
yline(axA,yZero,':','Color',[0.4 0.4 0.4],'LineWidth',1.2);
scatter(axA,Tall.logSzFreq(nonZero),Tall.logSpikeRate(nonZero),14,COL_all,'filled','MarkerFaceAlpha',0.25);
plot(axA,Tall.logSzFreq(isZeroSz&~isZeroRate), Tall.logSpikeRate(isZeroSz&~isZeroRate),'*','Color',COL_all,'MarkerSize',7,'LineWidth',1);
plot(axA,Tall.logSzFreq(~isZeroSz&isZeroRate), Tall.logSpikeRate(~isZeroSz&isZeroRate),'*','Color',COL_all,'MarkerSize',8,'LineWidth',1);
plot(axA,Tall.logSzFreq(isZeroSz&isZeroRate),  Tall.logSpikeRate(isZeroSz&isZeroRate), '*','Color',COL_all,'MarkerSize',8,'LineWidth',1.2);
if n_all>=3
    X=[ones(n_all,1),Tall.logSzFreq]; b=X\Tall.logSpikeRate;
    xgrid=linspace(spearman_xLims(1),spearman_xLims(2),300)';
    plot(axA,xgrid,b(1)+b(2)*xgrid,'k-','LineWidth',2);
end
xlim(axA,spearman_xLims); ylim(axA,spearman_yLims);
xlabel(axA,'Seizures per month (log scale)','FontSize',fontL);
ylabel(axA,'Spikes per hour (log scale)','FontSize',fontL);
set_log10_ticks(axA,'x',eps_sz,spearman_xLims);
set_log10_ticks(axA,'y',eps_rate,spearman_yLims);
labs=string(axA.XTickLabel); [~,iMax]=max(axA.XTick); labs(iMax)=""; axA.XTickLabel=labs;
title(axA,sprintf('A. All epilepsy%s (N=%d)',labelSuffix,n_all),'FontSize',fontL,'FontWeight','bold');
text(axA,0.98,0.95,sprintf('\\rho=%.2f [%.2f-%.2f], %s',rs_all,rho_lo,rho_hi,p_label(p_all)),...
    'Units','normalized','HorizontalAlignment','right','VerticalAlignment','top','FontSize',fontL-2,'FontWeight','bold');
set(axA,'FontSize',fontL);

colMap = struct('Frontal',COL_front,'Temporal',COL_temp,'General',COL_gen);
for p = 1:3
    ax = nexttile(p+1); hold(ax,'on'); grid(ax,'on'); box(ax,'off');
    gStr = string(panelOrder{p});
    col  = colMap.(gStr);
    idx  = (string(PatientSpikeSz_Typed.EpiType3)==gStr) & ...
           isfinite(PatientSpikeSz_Typed.MeanSpikeRate_perHour) & ...
           isfinite(PatientSpikeSz_Typed.(freqFieldName));
    if nonZeroOnly
        idx = idx & (PatientSpikeSz_Typed.MeanSpikeRate_perHour>0) & ...
                    (PatientSpikeSz_Typed.(freqFieldName)>0);
    end
    if nnz(idx)==0, axis(ax,'off'); continue; end

    x_raw=double(PatientSpikeSz_Typed.(freqFieldName)(idx));
    y_raw=double(PatientSpikeSz_Typed.MeanSpikeRate_perHour(idx));
    logX=log10(x_raw+(x_raw<=0).*eps_sz);
    logY=log10(y_raw+(y_raw<=0).*eps_rate);
    isZx=(x_raw==0); isZy=(y_raw==0);

    xline(ax,xZero,':','Color',[0.4 0.4 0.4],'LineWidth',1.2);
    yline(ax,yZero,':','Color',[0.4 0.4 0.4],'LineWidth',1.2);
    scatter(ax,logX(~isZx&~isZy),logY(~isZx&~isZy),18,col,'filled','MarkerFaceAlpha',0.35);
    if any(isZx&~isZy), plot(ax,logX(isZx&~isZy),logY(isZx&~isZy),'*','Color',col,'MarkerSize',8,'LineWidth',1.1); end
    if any(~isZx&isZy), plot(ax,logX(~isZx&isZy),logY(~isZx&isZy),'*','Color',col,'MarkerSize',8,'LineWidth',1.1); end
    if any(isZx&isZy),  plot(ax,logX(isZx&isZy), logY(isZx&isZy), '*','Color',col,'MarkerSize',9,'LineWidth',1.2); end
    if nnz(~isZx&~isZy)>=3
        Xg=[ones(nnz(~isZx&~isZy),1),logX(~isZx&~isZy)]; bg=Xg\logY(~isZx&~isZy);
        xg=linspace(spearman_xLims(1),spearman_xLims(2),250)';
        plot(ax,xg,bg(1)+bg(2)*xg,'-','Color',col,'LineWidth',2);
    end
    xlim(ax,spearman_xLims); ylim(ax,spearman_yLims);
    xlabel(ax,'Seizures per month (log scale)','FontSize',fontL);
    ylabel(ax,'Spikes per hour (log scale)','FontSize',fontL);
    set_log10_ticks(ax,'x',eps_sz,spearman_xLims);
    set_log10_ticks(ax,'y',eps_rate,spearman_yLims);
    labs=string(ax.XTickLabel); [~,iMax]=max(ax.XTick); labs(iMax)=""; ax.XTickLabel=labs;

    row   = SpearmanResults(strcmp(string(SpearmanResults.Group),gStr),:);
    rowCI = subtype_ci(subtype_ci.Group==gStr,:);
    txt = sprintf('\\rho=%.2f [%.2f-%.2f], p_{bonf}%s', row.Spearman_r, rowCI.ci_lo, rowCI.ci_hi, ...
        string(regexprep(char(p_label(row.p_bonf)),'^p','')));
    title(ax,sprintf('%s%s (N=%d)',panelTitle{p},labelSuffix,nnz(idx)),'FontSize',fontL,'FontWeight','bold');
    text(ax,0.98,0.95,txt,'Units','normalized','HorizontalAlignment','right','VerticalAlignment','top','FontSize',fontL-3,'FontWeight','bold');
    set(ax,'FontSize',fontL);
end

if ~exist(fileparts(fig_out),'dir'), mkdir(fileparts(fig_out)); end
exportgraphics(f2, fig_out,'Resolution',300);
fprintf('Saved Spearman figure: %s\n', fig_out);
end

%% =====================================================================
%% SMALL UTILITIES
%% =====================================================================

function arr = json_to_string_array(s)
s = strtrim(string(s));
if s==""||s=="[]"||s=="<missing>", arr=strings(0,1); return; end
dec = jsondecode(char(s));
if iscell(dec)
    arr = strings(numel(dec),1);
    for k=1:numel(dec)
        x=dec{k};
        if ischar(x)||(isstring(x)&&isscalar(x)), arr(k)=string(x); end
    end
elseif ischar(dec)||(isstring(dec)&&isscalar(dec))
    arr = string(dec);
elseif isnumeric(dec)
    arr = string(dec(:));
else
    error('Unsupported JSON string-array type.');
end
arr = string(arr(:));
end

function arr = json_to_double_array(s)
s = strtrim(string(s));
if s==""||s=="[]"||s=="<missing>", arr=double([]); return; end
s   = regexprep(s,'null','NaN','ignorecase');
dec = jsondecode(char(s));
arr = double(dec(:));
end

function out = max_hasSz(x)
x=x(isfinite(x)); if isempty(x), out=NaN; else, out=max(x); end
end

function out = local_frac_hasSz1(hsVec)
hs=hsVec(isfinite(hsVec));
if isempty(hs), out=NaN; return; end
valid=(hs==0|hs==1);
if ~any(valid), out=NaN; else, out=nnz(hs(valid)==1)/nnz(valid); end
end

function out = local_min_omitnan(a)
a=a(isfinite(a)); if isempty(a), out=NaN; else, out=min(a); end
end

function out = local_first_nonmissing(s)
s=s(:); s=s(~ismissing(s)&strlength(s)>0);
if isempty(s), out=""; else, out=s(1); end
end

function ylog = to_log10_per_hour(x, eps_rate)
x=double(x); x(~isfinite(x)|x<=0)=eps_rate; ylog=log10(x);
end

function ylog = to_log10_per_month(f, eps_freq)
f=double(f); f(~isfinite(f)|f<=0)=eps_freq; ylog=log10(f);
end

function Yj = add_y_jitter_eps(Y, Y_ZERO, Y_LIMS, frac)
Yj=Y; mask=abs(Y-Y_ZERO)<1e-9;
if any(mask), amp=frac*diff(Y_LIMS); Yj(mask)=Yj(mask)+(rand(sum(mask),1)-0.5)*amp; end
end

function add_sigbar(ax, x1, x2, y, ptext)
tick=0.03*diff(ax.YLim);
plot(ax,[x1 x1 x2 x2],[y-tick,y,y,y-tick],'k-','LineWidth',1.3);
if ptext=="**"||ptext=="***", yOff=-0.012*diff(ax.YLim); else, yOff=0.003*diff(ax.YLim); end
text(ax,mean([x1 x2]),y+yOff,ptext,'HorizontalAlignment','center','VerticalAlignment','bottom','FontSize',20);
end

function pStr = p_label(p)
if isnan(p), pStr="p=NaN"; return; end
if p<0.001, pStr="p<0.001"; return; end
if p<0.01,  pStr=sprintf("p=%.2g",p); return; end
pStr=sprintf("p=%.2f",p);
end

function pStr = format_p_html(p)
if isnan(p), pStr='p = NaN'; return; end
if p<0.001, pStr='p &lt; 0.001'; return; end
if p<0.01
    s=sprintf('%.2g',p); if startsWith(s,'.'), s=['0' s]; end
    pStr=['p = ' s]; return;
end
pStr=sprintf('p = %.2f',p);
end

function set_log10_ticks(ax, whichAxis, eps_val, axisLims, maxPow)
if nargin<5||isempty(maxPow), maxPow=6; end
whichAxis=lower(whichAxis);
decades=10.^(0:maxPow); log_dec=log10(decades);
keep=(log_dec>=axisLims(1))&(log_dec<=axisLims(2));
ticks=log_dec(keep); labels=string(decades(keep));
log_eps=log10(eps_val);
if log_eps>=axisLims(1)&&log_eps<=axisLims(2)
    ticks=[log_eps;ticks(:)]; labels=["0";labels(:)];
end
if whichAxis=="x", ax.XTick=ticks; ax.XTickLabel=labels;
else,              ax.YTick=ticks; ax.YTickLabel=labels; end
end

function [med,lo,hi] = bootstrap_median_ci(x, nBoot, alpha)
x=double(x(:)); x=x(isfinite(x));
if isempty(x)||nBoot==0, med=median(x,'omitnan'); lo=NaN; hi=NaN; return; end
med=median(x,'omitnan'); n=numel(x);
bootM=nan(nBoot,1);
for b=1:nBoot, idx=randi(n,n,1); bootM(b)=median(x(idx)); end
lo=prctile(bootM,100*(alpha/2)); hi=prctile(bootM,100*(1-alpha/2));
end

function add_median_ci_overlay(ax, xpos, med_raw, lo_raw, hi_raw, eps_floor)
yMed=to_log10_per_hour(med_raw,eps_floor);
yLo =to_log10_per_hour(lo_raw, eps_floor);
yHi =to_log10_per_hour(hi_raw, eps_floor);
plot(ax,[xpos xpos],[yLo yHi],'k-','LineWidth',3);
plot(ax,xpos,yMed,'ko','MarkerFaceColor','k','MarkerSize',6);
end

function add_median_ci_overlay_month(ax, xpos, med_raw, lo_raw, hi_raw, eps_floor)
yMed=to_log10_per_month(med_raw,eps_floor);
yLo =to_log10_per_month(lo_raw, eps_floor);
yHi =to_log10_per_month(hi_raw, eps_floor);
plot(ax,[xpos xpos],[yLo yHi],'k-','LineWidth',3);
plot(ax,xpos,yMed,'ko','MarkerFaceColor','k','MarkerSize',6);
end

function [rho_hat,lo,hi] = bootstrap_spearman_ci(x, y, nBoot, alpha)
x=double(x(:)); y=double(y(:));
mask=isfinite(x)&isfinite(y); x=x(mask); y=y(mask); n=numel(x);
if n<3, rho_hat=NaN; lo=NaN; hi=NaN; return; end
rho_hat=corr(x,y,'Type','Spearman','Rows','complete');
rho_boot=nan(nBoot,1);
for b=1:nBoot, idx=randi(n,n,1); rho_boot(b)=corr(x(idx),y(idx),'Type','Spearman','Rows','complete'); end
lo=prctile(rho_boot,100*(alpha/2)); hi=prctile(rho_boot,100*(1-alpha/2));
end

function d = cliff_delta(x1, x2)
x1=x1(isfinite(x1(:))); x2=x2(isfinite(x2(:)));
n1=numel(x1); n2=numel(x2);
if n1==0||n2==0, d=NaN; return; end
[~,~,stats]=ranksum(x1,x2,'method','approx');
R1=stats.ranksum; U1=R1-n1*(n1+1)/2;
d=(2*U1/(n1*n2))-1;
end





function p = get_p_preferred(FE, BT, nm)
% Use bootstrap p if available, otherwise fall back to Wald p from FE table
p = FE.p(string(FE.Term)==nm);   % default
if ~isempty(BT) && ismember('Boot_p', BT.Properties.VariableNames)
    bt_row = BT(string(BT.Term)==nm, :);
    if ~isempty(bt_row)
        p = bt_row.Boot_p;
    end
end
end


%% =====================================================================
%% FLOW DIAGRAM
%% =====================================================================

function FigFlow = make_flowchart_figure(Views, MMR)

EC        = Views.ExclusionCounts;
n_subtype = numel(unique(MMR.ModelTable.Patient));
n_pairs   = height(MMR.ModelTable);

%% ---- Layout constants ----
FigFlow = figure('Color','w','Position',[100 100 820 820]);
ax = axes('Position',[0 0 1 1]);
axis(ax,'off'); hold(ax,'on');
xlim(ax,[0 1.15]); ylim(ax,[0 1]);

BOX_W     = 0.52;
BOX_H     = 0.08;
EXC_W     = 0.34;
EXC_H     = 0.075;
CX        = 0.46;
EXC_X_L   = CX + BOX_W/2;   % left edge of exclusion arrow start
EXC_X_R   = 0.76;            % left edge of exclusion box
FONT_MAIN = 15;
FONT_EXC  = 12;

COL_MAIN = [0.22 0.45 0.70];
COL_EXC  = [0.80 0.30 0.10];
COL_SUB  = [0.15 0.55 0.40];

%% ---- Helper: draw a rounded box with centered text ----
    function draw_box(cx, cy, w, h, txt, col, fsz)
        % Draw filled rounded-ish box using patch (compatible with older MATLAB)
        x0 = cx - w/2;
        y0 = cy - h/2;
        % Simple rectangle as patch (no rounding, but fully compatible)
        patch(ax, [x0, x0+w, x0+w, x0, x0], [y0, y0, y0+h, y0+h, y0], ...
            col, ...
            'FaceAlpha', 0.25, ...
            'EdgeColor', col, ...
            'LineWidth', 1.8);
        text(ax, cx, cy, txt, ...
            'HorizontalAlignment','center', ...
            'VerticalAlignment','middle', ...
            'FontSize',fsz, ...
            'FontName','Helvetica', ...
            'Color',[0.1 0.1 0.1], ...
            'Interpreter','none');
    end

%% ---- Helper: vertical arrow between two y positions ----
    %% ---- Helper: vertical arrow between two y positions ----
    function draw_arrow_down(cx, y_top, y_bot)
        plot(ax, [cx cx], [y_top y_bot], '-', ...
            'Color',[0.3 0.3 0.3], 'LineWidth',1.4);
        % Arrowhead pointing down
        hw = 0.012;  % half-width of arrowhead
        hl = 0.018;  % height of arrowhead
        fill(ax, [cx-hw, cx+hw, cx], [y_bot+hl, y_bot+hl, y_bot], ...
            [0.3 0.3 0.3], 'EdgeColor','none');
    end

%% ---- Helper: horizontal arrow from main column midpoint to exclusion box ----
    function draw_arrow_right(y_mid)
        x_start = CX;        % start at center of main column (on the down arrow)
        x_end   = EXC_X_R;
        plot(ax, [x_start x_end], [y_mid y_mid], '-', ...
            'Color',COL_EXC, 'LineWidth',1.2);
        % Arrowhead pointing right
        hw = 0.012;
        hl = 0.018;
        fill(ax, [x_end-hl, x_end-hl, x_end], [y_mid-hw, y_mid+hw, y_mid], ...
            COL_EXC, 'EdgeColor','none');
    end

%% ---- Y positions of main boxes ----
y1 = 0.90;   % total patients
y2 = 0.74;   % after outpatient/routine filter
y3 = 0.57;   % after epilepsy diagnosis filter
y4 = 0.40;   % final cohort (after sz freq filter)
y5 = 0.18;   % subtype + model box (combined)

%% ---- Midpoints between main boxes (where horizontal arrows branch off) ----
ym12 = (y1 + y2) / 2;
ym23 = (y2 + y3) / 2;
ym34 = (y3 + y4) / 2;

%% ---- Main boxes ----
draw_box(CX, y1, BOX_W, BOX_H, ...
    sprintf('All patients with EEG data\nN = %d', EC.nTotal), ...
    COL_MAIN, FONT_MAIN);

draw_box(CX, y2, BOX_W, BOX_H, ...
    sprintf('Outpatient routine EEG <=4 hours\nN = %d', EC.nAfterOutptRoutine), ...
    COL_MAIN, FONT_MAIN);

draw_box(CX, y3, BOX_W, BOX_H, ...
    sprintf('LLM-confirmed epilepsy diagnosis\nN = %d', ...
        EC.nAfterOutptRoutine - EC.nExcludedNoEpilepsy), ...
    COL_MAIN, FONT_MAIN);

draw_box(CX, y4, BOX_W, BOX_H, ...
    sprintf('Documented seizure frequency\nN = %d (primary cohort)', EC.nFinalCohort), ...
    COL_SUB, FONT_MAIN);

% Combined subtype + model box
draw_box(CX, y5, BOX_W, BOX_H*1.5, ...
    sprintf('Known epilepsy subtype\n(temporal, frontal, generalized)\nfor mixed effects model\nN = %d patients, %d EEG-visit pairs', ...
        n_subtype, n_pairs), ...
    COL_SUB, FONT_MAIN);

%% ---- Down arrows ----
draw_arrow_down(CX, y1-BOX_H/2, y2+BOX_H/2);
draw_arrow_down(CX, y2-BOX_H/2, y3+BOX_H/2);
draw_arrow_down(CX, y3-BOX_H/2, y4+BOX_H/2);
draw_arrow_down(CX, y4-BOX_H/2, y5+BOX_H*1.5/2);

%% ---- Horizontal exclusion arrows (branch off midpoint of each down arrow) ----
draw_arrow_right(ym12);
draw_arrow_right(ym23);
draw_arrow_right(ym34);

%% ---- Exclusion boxes ----
EXC_CX = EXC_X_R + EXC_W/2;   % center x of exclusion boxes

draw_box(EXC_CX, ym12, EXC_W, EXC_H, ...
    sprintf('Excluded: inpatient or\nambulatory EEG\nN = %d', ...
        EC.nTotal - EC.nAfterOutptRoutine), ...
    COL_EXC, FONT_EXC);

draw_box(EXC_CX, ym23, EXC_W, EXC_H, ...
    sprintf('Excluded: no epilepsy\ndiagnosis (NESD, uncertain,\nor unknown)\nN = %d', ...
        EC.nExcludedNoEpilepsy), ...
    COL_EXC, FONT_EXC);

draw_box(EXC_CX, ym34, EXC_W, EXC_H, ...
    sprintf('Excluded: no documented\nseizure frequency\nN = %d', ...
        EC.nExcludedNoSzFreq), ...
    COL_EXC, FONT_EXC);

%% ---- Title ----
text(ax, CX, 0.97, 'Study participant flow', ...
    'HorizontalAlignment','center', ...
    'VerticalAlignment','top', ...
    'FontSize',16, ...
    'FontWeight','bold', ...
    'FontName','Helvetica');

end


function GapTable = compute_visit_eeg_gaps(Vuniq, ReportForKeptSessions)
% Returns one row per visit with min |visit - EEG| in days

% ---- checks ----
require_cols(Vuniq, ["Patient","VisitDate"], "Vuniq");

if ismember("Patient", string(ReportForKeptSessions.Properties.VariableNames))
    pid = double(ReportForKeptSessions.Patient);
elseif ismember("patient_id", string(ReportForKeptSessions.Properties.VariableNames))
    pid = double(ReportForKeptSessions.patient_id);
else
    error("ReportForKeptSessions must have Patient or patient_id");
end

require_cols(ReportForKeptSessions, ["start_time_deid"], "ReportForKeptSessions");

% ---- EEG datetimes ----
EEG_raw = ReportForKeptSessions.start_time_deid;
if isdatetime(EEG_raw)
    EEG_dt = EEG_raw;
else
    EEG_dt = datetime(strtrim(string(EEG_raw)), ...
        'InputFormat',"yyyy-MM-dd'T'HH:mm:ss");
end

EEG_tbl = table(pid, EEG_dt, 'VariableNames', {'Patient','EEG_Date'});
EEG_tbl = EEG_tbl(~isnat(EEG_tbl.EEG_Date), :);

% ---- compute per-visit min gap ----
GapTable = Vuniq(:, {'Patient','VisitDate'});
GapTable.MinAbsGap_days = NaN(height(GapTable),1);

[gv, pidV] = findgroups(GapTable.Patient);

for k = 1:numel(pidV)
    p = pidV(k);
    idxV = find(gv == k);

    eegDates = EEG_tbl.EEG_Date(EEG_tbl.Patient == p);
    if isempty(eegDates)
        continue
    end

    for j = idxV'
        GapTable.MinAbsGap_days(j) = ...
            min(abs(days(eegDates - GapTable.VisitDate(j))));
    end
end

GapTable.MinAbsGap_years = GapTable.MinAbsGap_days / 365.25;
end




function V_f = restrict_visits_by_min_abs_gap(Vuniq, ReportForKeptSessions, minDays, maxDays)
% Keep visits for which min |VisitDate - EEG_Date| across that patient's EEGs is in [minDays, maxDays].
%
% minDays can be 0, maxDays can be Inf

needV = ["Patient","VisitDate","Freq_R1"];
missV = setdiff(needV, string(Vuniq.Properties.VariableNames));
if ~isempty(missV)
    error("Vuniq missing required columns: %s", strjoin(missV,", "));
end

% Accept either Patient or patient_id
if ismember("Patient", string(ReportForKeptSessions.Properties.VariableNames))
    pid = double(ReportForKeptSessions.Patient);
elseif ismember("patient_id", string(ReportForKeptSessions.Properties.VariableNames))
    pid = double(ReportForKeptSessions.patient_id);
else
    error("ReportForKeptSessions must contain Patient or patient_id.");
end

if ~ismember("start_time_deid", string(ReportForKeptSessions.Properties.VariableNames))
    error("ReportForKeptSessions missing start_time_deid.");
end

% Parse EEG datetimes
EEG_raw = ReportForKeptSessions.start_time_deid;
if isdatetime(EEG_raw)
    EEG_dt = EEG_raw;
else
    EEG_dt = datetime(strtrim(string(EEG_raw)), 'InputFormat',"yyyy-MM-dd'T'HH:mm:ss");
end

okEEG = ~isnat(EEG_dt) & isfinite(pid);
EEG_tbl = table(pid(okEEG), EEG_dt(okEEG), 'VariableNames', {'Patient','EEG_Date'});
if isempty(EEG_tbl)
    error("No valid EEG dates found (start_time_deid could not be parsed).");
end

V_f = Vuniq;
keep = false(height(V_f),1);

[gv, pidV] = findgroups(V_f.Patient);

for k = 1:numel(pidV)
    p = pidV(k);
    idxV = find(gv == k);
    vDates = V_f.VisitDate(idxV);

    eegDates = EEG_tbl.EEG_Date(EEG_tbl.Patient == p);
    if isempty(eegDates)
        continue
    end

    minAbsGap = inf(numel(vDates),1);
    for j = 1:numel(vDates)
        minAbsGap(j) = min(abs(days(eegDates - vDates(j))));
    end

    keep(idxV) = (minAbsGap >= minDays) & (minAbsGap <= maxDays);
end

fprintf('[Visit-EEG distance] Kept %d/%d visits with min|gap| in [%g, %g] days\n', ...
    nnz(keep), height(Vuniq), minDays, maxDays);

V_f = V_f(keep,:);
end

function NearFarStats = plot_delta_rho_histogram(Views, Vuniq, ReportForKeptSessions, ...
                                                 nearQ, farQ, nBoot, alpha, outPng)
% plot_delta_rho_histogram
% Plot (top) distribution of min Visit–EEG gaps with tercile shading + quantile cutoffs,
% and (bottom) bootstrap distribution of Δρ = ρ_near − ρ_far (Spearman),
% with Δρ x-axis centered around 0.
%
% Inputs:
%   nearQ, farQ : quantiles in [0,1], e.g., 0.333 and 0.667
%   nBoot       : bootstrap iterations (e.g., 5000)
%   alpha       : CI alpha (e.g., 0.05)
%   outPng      : output file path ("" or '' to skip saving)
%
% NOTE: This function DEFINES near/far by quantiles of MinAbsGap_days computed across
%       visits in Vuniq.

if nargin < 8, outPng = ""; end
if nargin < 4 || isempty(nearQ), nearQ = 0.333; end
if nargin < 5 || isempty(farQ),  farQ  = 0.667; end
if nargin < 6 || isempty(nBoot), nBoot = 5000; end
if nargin < 7 || isempty(alpha), alpha = 0.05; end

% ------------------ Base cohort patients (match main Spearman cohort) ------------------
if isfield(Views, 'PatientSpikeSz_All') && ~isempty(Views.PatientSpikeSz_All)
    basePatients = unique(double(Views.PatientSpikeSz_All.Patient));
else
    basePatients = unique(double(Views.PatientLevelSpikeRates.Patient));
end

SpikeTbl = Views.PatientLevelSpikeRates(:, {'Patient','MeanSpikeRate_perHour'});
SpikeTbl.Patient = double(SpikeTbl.Patient);
SpikeTbl = innerjoin(SpikeTbl, table(basePatients,'VariableNames',{'Patient'}), 'Keys','Patient');

% ------------------ Clinic visit counts per patient (non-NaN seizure freq) ------------------
V_base = innerjoin(Vuniq, ...
    table(basePatients,'VariableNames',{'Patient'}), ...
    'Keys','Patient');

% only visits with documented seizure frequency (including has Sz == 0)
V_base = V_base(isfinite(V_base.Freq_R1), :);

[gpV, pidV] = findgroups(V_base.Patient);
nVisitsPerPatient = splitapply(@numel, V_base.Freq_R1, gpV);

nPatients_total = numel(pidV);
nPatients_multi = nnz(nVisitsPerPatient >= 2);
pct_multi = 100 * nPatients_multi / max(1, nPatients_total);

fprintf(['[Visit counts] Patients with ≥2 clinic visits with documented seizure frequency: ' ...
         '%d/%d (%.1f%%)\n'], ...
         nPatients_multi, nPatients_total, pct_multi);


% ------------------ Get stats on num pts with multiple EEGs ---------
% Count EEGs per patient (each row is a patient-session EEG)
pid = double(ReportForKeptSessions.Patient);   % or patient_id if that's what you have
ses = double(ReportForKeptSessions.Session);   % or session_number

% count unique sessions per patient
[gp, pids] = findgroups(pid);
nEEG = splitapply(@(x) numel(unique(x)), ses, gp);

nPatients = numel(pids);
nMulti    = nnz(nEEG >= 2);
fprintf('Patients with >=2 EEGs: %d/%d (%.1f%%)\n', nMulti, nPatients, 100*nMulti/max(1,nPatients));


% ------------------ Compute min gaps for ALL visits ------------------
GapTable = compute_visit_eeg_gaps(Vuniq, ReportForKeptSessions);
gaps = double(GapTable.MinAbsGap_days);
gaps = gaps(isfinite(gaps));

if isempty(gaps)
    error("No finite MinAbsGap_days found.");
end

nearDays = quantile(gaps, nearQ);
farDays  = quantile(gaps, farQ);

% ------------------ Build NEAR/FAR visit subsets ------------------
V_near = restrict_visits_by_min_abs_gap(Vuniq, ReportForKeptSessions, 0, nearDays);
V_far  = restrict_visits_by_min_abs_gap(Vuniq, ReportForKeptSessions, farDays, Inf);

% ------------------ Patients with BOTH near and far visits ------------------
% restrict to base cohort (same patients used downstream)
Vn = innerjoin(V_near, table(basePatients,'VariableNames',{'Patient'}), 'Keys','Patient');
Vf = innerjoin(V_far,  table(basePatients,'VariableNames',{'Patient'}), 'Keys','Patient');

% require documented seizure frequency
Vn = Vn(isfinite(Vn.Freq_R1), :);
Vf = Vf(isfinite(Vf.Freq_R1), :);

pNear = unique(Vn.Patient);
pFar  = unique(Vf.Patient);

pBoth = intersect(pNear, pFar);

nTotal = numel(basePatients);
nBoth  = numel(pBoth);
pctBoth = 100 * nBoth / max(1, nTotal);

fprintf(['[Near/Far eligibility] Patients with ≥1 short-gap AND ≥1 long-gap visit ' ...
         '(documented seizure freq): %d/%d (%.1f%%)\n'], ...
         nBoth, nTotal, pctBoth);


% This takes the mean of visits in V_near to get Sz_near, and same for
% Sz_far
Sz_near = build_patient_seizure_metrics(V_near);  % Patient, MeanSzFreq, ...
Sz_far  = build_patient_seizure_metrics(V_far);

% Rename BEFORE join so names are guaranteed
Sz_near2 = Sz_near(:, {'Patient','MeanSzFreq'});
Sz_far2  = Sz_far(:,  {'Patient','MeanSzFreq'});

Sz_near2 = renamevars(Sz_near2, "MeanSzFreq", "MeanSzFreq_near");
Sz_far2  = renamevars(Sz_far2,  "MeanSzFreq", "MeanSzFreq_far");

% Join to spikes, require both near and far seizure metrics
J_near = innerjoin(SpikeTbl, Sz_near2, 'Keys','Patient');
J_far  = innerjoin(SpikeTbl, Sz_far2,  'Keys','Patient');
J = innerjoin(J_near, J_far(:,{'Patient','MeanSzFreq_far'}), 'Keys','Patient');

% Finite mask
mask = isfinite(J.MeanSpikeRate_perHour) & isfinite(J.MeanSzFreq_near) & isfinite(J.MeanSzFreq_far);
J = J(mask,:);

n = height(J);
if n < 3
    error("Not enough patients with BOTH near and far seizure metrics (n=%d).", n);
end

x  = double(J.MeanSpikeRate_perHour);
yN = double(J.MeanSzFreq_near);
yF = double(J.MeanSzFreq_far);

%% Are long gap visits later than short gap visits?
% ===== Prep (cohort + documented freq only) =====
Vn = innerjoin(V_near, table(basePatients,'VariableNames',{'Patient'}), 'Keys','Patient');
Vf = innerjoin(V_far,  table(basePatients,'VariableNames',{'Patient'}), 'Keys','Patient');

Vn = Vn(isfinite(Vn.Freq_R1), :);
Vf = Vf(isfinite(Vf.Freq_R1), :);

pBoth = intersect(unique(Vn.Patient), unique(Vf.Patient));

VnB = innerjoin(Vn, table(pBoth,'VariableNames',{'Patient'}), 'Keys','Patient');
VfB = innerjoin(Vf, table(pBoth,'VariableNames',{'Patient'}), 'Keys','Patient');

% ===== For each patient: median visit date in near vs far =====
[gn, pN] = findgroups(VnB.Patient);
nearDate_med = splitapply(@median, VnB.VisitDate, gn);

[gf, pF] = findgroups(VfB.Patient);
farDate_med  = splitapply(@median, VfB.VisitDate, gf);

% align rows by Patient (robust)
Tdate = innerjoin( ...
    table(double(pN), nearDate_med, 'VariableNames',{'Patient','NearDateMed'}), ...
    table(double(pF), farDate_med,  'VariableNames',{'Patient','FarDateMed'}), ...
    'Keys','Patient');

deltaDays = days(Tdate.FarDateMed - Tdate.NearDateMed);   % >0 means far is later

% Paired sign-rank test on within-patient date differences
if height(Tdate) >= 3
    p_time = signrank(deltaDays, 0, 'method','approx');  % H0: median delta = 0
else
    p_time = NaN;
end

fprintf(['[Chronology] Patients with both near+far: N=%d\n' ...
         '  Median(FarDateMed - NearDateMed) = %.1f days (IQR %.1f–%.1f), signrank p=%s\n'], ...
         height(Tdate), ...
         median(deltaDays,'omitnan'), prctile(deltaDays,25), prctile(deltaDays,75), ...
         p_label(p_time));

%% Do near visits have higher seizure frequencies
% ===== For each patient: mean seizure freq in near vs far (documented only) =====
nearFreq_mean = splitapply(@(x) mean(x,'omitnan'), VnB.Freq_R1, gn);
farFreq_mean  = splitapply(@(x) mean(x,'omitnan'), VfB.Freq_R1, gf);

Tfreq = innerjoin( ...
    table(double(pN), nearFreq_mean, 'VariableNames',{'Patient','NearMeanSz'}), ...
    table(double(pF),  farFreq_mean, 'VariableNames',{'Patient','FarMeanSz'}), ...
    'Keys','Patient');

% Paired sign-rank test: Far vs Near
deltaFreq = Tfreq.NearMeanSz - Tfreq.FarMeanSz;  % positive => near higher
if height(Tfreq) >= 3
    p_sz = signrank(Tfreq.NearMeanSz, Tfreq.FarMeanSz, 'method','approx');
else
    p_sz = NaN;
end

fprintf(['[Seizure freq] Patients with both near+far: N=%d\n' ...
         '  NearMeanSz median=%.2f, FarMeanSz median=%.2f; ' ...
         'median(Near-Far)=%.2f, signrank p=%s\n'], ...
         height(Tfreq), ...
         median(Tfreq.NearMeanSz,'omitnan'), ...
         median(Tfreq.FarMeanSz,'omitnan'), ...
         median(deltaFreq,'omitnan'), ...
         p_label(p_sz));

%% Main analysis
% ------------------ Observed correlations ------------------
rho_near = corr(x, yN, 'Type','Spearman', 'Rows','complete');
rho_far  = corr(x, yF, 'Type','Spearman', 'Rows','complete');
delta_obs = rho_near - rho_far;

% ------------------ Patient-level bootstrap ------------------
delta = nan(nBoot,1);
for b = 1:nBoot
    idx = randi(n, n, 1);
    rn = corr(x(idx), yN(idx), 'Type','Spearman', 'Rows','complete');
    rf = corr(x(idx), yF(idx), 'Type','Spearman', 'Rows','complete');
    delta(b) = rn - rf;
end

ci_lo = prctile(delta, 100*(alpha/2));
ci_hi = prctile(delta, 100*(1-alpha/2));
delta_med = median(delta,'omitnan');

p_one = mean(delta <= 0);                           % H1: near > far
p_two = 2*min(mean(delta <= 0), mean(delta >= 0));  % two-sided bootstrap p

% ------------------ Plot: gaps (top) + delta rho (bottom) ------------------

fig = figure('Color','w','Position',[120 80 950 780]);
tl = tiledlayout(fig, 2, 1, 'TileSpacing','compact', 'Padding','compact');

% ===================== TOP: Gap distribution + tercile shading =====================
ax1 = nexttile(tl,1); hold(ax1,'on'); box(ax1,'off'); grid(ax1,'on');

% Histogram first so y-lims are defined
hG = histogram(ax1, gaps, 60, 'EdgeColor','none');

xlabel(ax1, '|Visit - EEG| gap (days)','fontsize',20);
ylabel(ax1, 'Visit count','fontsize',20);

% Determine x-range for shading
xL = min(hG.BinEdges);
xU = max(hG.BinEdges);

% Clamp cutoffs to plotting range
nearX = max(min(nearDays, xU), xL);
farX  = max(min(farDays,  xU), xL);

% Use current y-limits
yl = get(ax1,'YLim');
yl_new = [yl(1) yl(1) + 1.3*(yl(2)-yl(1))];
set(ax1,'YLim', yl_new);

% Helper for shaded region [xa, xb]
makeShade = @(xa, xb, a) patch(ax1, ...
    [xa xb xb xa], [yl(1) yl(1) yl(2) yl(2)], ...
    [0 0 0], 'FaceAlpha', a, 'EdgeColor','none');

% Shade: Near / Middle / Far (slightly stronger on near/far)
pNear   = makeShade(xL,   nearX, 0.12);
pMiddle = makeShade(nearX, farX, 0.06);
pFar    = makeShade(farX,  xU,   0.12);

% Put shading behind bars
uistack([pNear pMiddle pFar], 'bottom');
uistack(hG, 'top');

% Cutoff lines
xline(ax1, nearDays, 'k--', 'LineWidth',2);
xline(ax1, farDays,  'k--', 'LineWidth',2);

% Title 
title(ax1, 'A. Visit–EEG gap distribution with lower and upper third cutoffs', ...
    'FontSize', 20, 'Interpreter','none');

% Region labels 
yl = get(ax1,'YLim');
yText = yl(2) * 0.99;

text(ax1, mean([xL nearX]), yText, sprintf("Short gap\n(lower third)"), ...
    'HorizontalAlignment','center', 'VerticalAlignment','top', ...
    'FontSize',20, 'Interpreter','tex');

text(ax1, mean([farX xU]), yText, sprintf("Long gap\n(upper third)"), ...
    'HorizontalAlignment','center', 'VerticalAlignment','top', ...
    'FontSize',20, 'Interpreter','tex');


% ===================== BOTTOM: Δρ distribution (centered at 0) =====================
ax2 = nexttile(tl,2); hold(ax2,'on'); box(ax2,'off'); grid(ax2,'on');

histogram(ax2, delta, 40, 'EdgeColor','none');
xline(ax2, 0,         'k--', 'LineWidth',2);
xline(ax2, delta_med, 'k-',  'LineWidth',2);
%xline(ax2, ci_lo,     'k:',  'LineWidth',2);
%xline(ax2, ci_hi,     'k:',  'LineWidth',2);

% symmetric x-lims around 0 (robust even if delta all positive)
maxAbs = max(abs(delta(isfinite(delta))));
if isempty(maxAbs) || ~isfinite(maxAbs) || maxAbs==0, maxAbs = 1e-3; end
pad = 0.08 * maxAbs;
xlim(ax2, [-maxAbs-pad, maxAbs+pad]);

xlabel(ax2, '\Delta\rho = \rho_{short gap} - \rho_{long gap}');
ylabel(ax2, 'Bootstrap count');

% Title text WITH interpreter explicitly set to avoid warnings
t2 = sprintf(['B. Distribution of differences in spike-seizure correlation\nbetween short and long visit-EEG gaps\n' ...
              '95%% CI [%.3f, %.3f], p = %.3g'], ...
              ci_lo, ci_hi, p_one);

title(ax2, t2, 'FontSize', 20, 'Interpreter','tex');

set([ax1 ax2],'FontSize',20);

if strlength(string(outPng)) > 0
    if ~exist(fileparts(outPng),'dir'), mkdir(fileparts(outPng)); end
    exportgraphics(fig, outPng, 'Resolution', 300);
end

fprintf(['\nFig S3 analysis:\n'...
    'N patients: %d\n'...
    'Median rho short gap: %1.2f\n'...
    'Median rho long gap: %1.2f\n'...
    'Median [95%% CI] difference in rho: %1.3f [%1.2f-%1.2f]\n'...
    'p = %1.4f.\n'],...
    n,rho_near,rho_far,delta_obs,ci_lo,ci_hi,p_one);

% ------------------ Bundle output ------------------
NearFarStats = struct();
NearFarStats.nPatients = n;

NearFarStats.nearQ = nearQ;
NearFarStats.farQ  = farQ;
NearFarStats.nearDays = nearDays;
NearFarStats.farDays  = farDays;

NearFarStats.rho_near = rho_near;
NearFarStats.rho_far  = rho_far;
NearFarStats.delta_obs = delta_obs;

NearFarStats.delta_boot = delta;
NearFarStats.delta_median = delta_med;
NearFarStats.delta_ci_lo = ci_lo;
NearFarStats.delta_ci_hi = ci_hi;

NearFarStats.p_one_sided = p_one;
NearFarStats.p_two_sided = p_two;

NearFarStats.tableUsed = J;
NearFarStats.gapsUsed = gaps;

end

function figH = make_eeg_duration_histogram(Views, outPath)
% Histogram comparing two EEG duration measures for the study cohort EEGs
% (Views.SessionsForFigures):
%   (1) File duration  — Duration_sec from the spike summary table
%   (2) Natus duration — duration_hms from the report, which clips out
%                        segments recorded prior to electrode connection
if nargin < 2, outPath = ''; end

FONT_SIZE = 20;
COL_FILE  = [0.22 0.45 0.70];   % file duration
COL_NATUS = [0.85 0.33 0.10];   % Natus server duration

%% --- File duration (minutes) ---
Sess = Views.SessionsForFigures;
require_cols(Sess, ["Patient","Session","Duration_sec"], "SessionsForFigures");
assert_unique_keys(Sess, "Patient", "Session", "SessionsForFigures");
FileDur = Sess(:, {'Patient','Session','Duration_sec'});
FileDur.FileMin = double(FileDur.Duration_sec) / 60;

%% --- Natus duration (minutes) ---
Rep = Views.ReportForKeptSessions;
require_cols(Rep, ["Patient","Session","duration_hms"], "ReportForKeptSessions");
assert_unique_keys(Rep, "Patient", "Session", "ReportForKeptSessions");
NatusDur = Rep(:, {'Patient','Session','duration_hms'});
NatusDur.NatusMin = minutes(NatusDur.duration_hms);   % duration array -> minutes

%% --- Join so both measures refer to the same EEGs ---
D = innerjoin(FileDur(:,{'Patient','Session','FileMin'}), ...
              NatusDur(:,{'Patient','Session','NatusMin'}), ...
              'Keys', {'Patient','Session'});
assert(height(D) == height(FileDur), ...
    'Join dropped EEGs: %d file rows vs %d matched (missing report row?).', ...
    height(FileDur), height(D));

file_min  = D.FileMin(isfinite(D.FileMin));
natus_min = D.NatusMin(isfinite(D.NatusMin));
assert(~isempty(file_min) && ~isempty(natus_min), 'No finite durations found.');

%% --- Common bins across both measures ---
binW  = 5;
allD  = [file_min; natus_min];
edges = 0 : binW : (ceil(max(allD)/binW)*binW);

figH = figure('Color','w','Position',[100 100 850 540]);
ax = axes(figH); hold(ax,'on'); box(ax,'off'); grid(ax,'on');

histogram(ax, file_min,  'BinEdges', edges, ...
    'FaceColor', COL_FILE,  'FaceAlpha', 0.5, 'EdgeColor','none', ...
    'DisplayName', sprintf('File duration (N=%d)', numel(file_min)));
histogram(ax, natus_min, 'BinEdges', edges, ...
    'FaceColor', COL_NATUS, 'FaceAlpha', 0.5, 'EdgeColor','none', ...
    'DisplayName', sprintf('Natus duration (N=%d)', numel(natus_min)));

med_file  = median(file_min);
med_natus = median(natus_min);
xline(ax, med_file,  '--', 'Color', COL_FILE,  'LineWidth', 2, 'HandleVisibility','off');
xline(ax, med_natus, '--', 'Color', COL_NATUS, 'LineWidth', 2, 'HandleVisibility','off');

xlabel(ax, 'EEG duration (minutes)', 'FontSize', FONT_SIZE);
ylabel(ax, 'Number of EEGs', 'FontSize', FONT_SIZE);
title(ax, sprintf('EEG durations in study cohort (N = %d EEGs)', height(D)), ...
    'FontSize', FONT_SIZE, 'FontWeight','bold');
legend(ax, 'Location','northeast', 'FontSize', FONT_SIZE-6);
set(ax, 'FontSize', FONT_SIZE);

%% --- Summary stats (durations + paired clipping) ---
qf = prctile(file_min,  [25 75]);
qn = prctile(natus_min, [25 75]);
paired = isfinite(D.FileMin) & isfinite(D.NatusMin);
clip   = D.FileMin(paired) - D.NatusMin(paired);   % how much Natus clips
fprintf(['File duration (min):  median=%.1f (IQR %.1f-%.1f)\n' ...
         'Natus duration (min): median=%.1f (IQR %.1f-%.1f)\n' ...
         'Clipped file-Natus (min): median=%.1f (IQR %.1f-%.1f), N=%d\n'], ...
    med_file,  qf(1), qf(2), ...
    med_natus, qn(1), qn(2), ...
    median(clip), prctile(clip,25), prctile(clip,75), nnz(paired));

%% --- Save ---
if strlength(string(outPath)) > 0
    save_fig(figH, outPath);
    fprintf('Saved EEG duration histogram: %s\n', outPath);
end
end

function DurCompare = add_duration_to_model(MMR, Views, alpha)
% Refit the primary model M1 with and without EEG duration (hours) as a
% covariate, and run a likelihood ratio test for whether it improves fit.
if nargin < 3, alpha = 0.05; end

%% --- Attach EEG duration (hours) to the model table ---
T = MMR.ModelTable;
require_cols(T, ["Patient","Session","HasSz_bin","LogSpikesPerHour", ...
    "AbsLag_years","VisitAfterEEG","EpiType3_cat","PatientID"], "MMR.ModelTable");

Dur = Views.SessionLevelSpikeRates(:, {'Patient','Session','Duration_sec'});
Dur.Duration_sec = double(Dur.Duration_sec);

nBefore = height(T);
T = innerjoin(T, Dur, 'Keys', {'Patient','Session'});
assert(height(T) == nBefore, ...
    'Duration join changed row count (%d -> %d); duplicate or missing session keys.', ...
    nBefore, height(T));

T.EEG_DurationHours = T.Duration_sec / 3600;

% Both models must be fit on identical rows for a valid LRT
keep = isfinite(T.EEG_DurationHours);
assert(all(keep), 'Non-finite EEG durations present in model table.');
T = T(keep, :);

%% --- Formulas: reduced = M1, full = M1 + EEG duration ---
formula_reduced = ['HasSz_bin ~ ' ...
    'LogSpikesPerHour * AbsLag_years + ' ...
    'LogSpikesPerHour * VisitAfterEEG + ' ...
    'EpiType3_cat + (1|PatientID)'];

formula_full = ['HasSz_bin ~ ' ...
    'LogSpikesPerHour * AbsLag_years + ' ...
    'LogSpikesPerHour * VisitAfterEEG + ' ...
    'EpiType3_cat + EEG_DurationHours + (1|PatientID)'];

glme_opts = {'Distribution','Binomial','Link','logit', ...
    'FitMethod','Laplace','CovariancePattern','Diagonal'};

%% --- Fit both models on identical rows ---
fprintf('\nFitting reduced model (M1, no duration)...\n');
mdl_reduced = fitglme(T, formula_reduced, glme_opts{:});
disp(mdl_reduced);

fprintf('\nFitting full model (M1 + EEG duration)...\n');
mdl_full = fitglme(T, formula_full, glme_opts{:});
disp(mdl_full);

%% --- Fixed effects table for the full model ---
[b, bn, s] = fixedEffects(mdl_full, 'Alpha', alpha);
FE_full = make_fe_table_logistic(bn, b, s);
fprintf('\nFull model fixed effects (with EEG duration):\n'); disp(FE_full);

%% --- Likelihood ratio test ---
fprintf('\n=== LRT: does EEG duration improve fit? ===\n');
lrt = compare(mdl_reduced, mdl_full);   % nested: full = reduced + EEG_DurationHours
disp(lrt);
lrt_p = lrt.pValue(2);

dur_row = FE_full(string(FE_full.Term)=="EEG_DurationHours", :);
fprintf(['\nEEG duration: OR=%.3f [%.3f-%.3f] per hour, p=%.4g\n' ...
    'LRT (1 df) reduced vs full: p=%.4g\n'], ...
    dur_row.OR, dur_row.OR_lo, dur_row.OR_hi, dur_row.p, lrt_p);

if lrt_p < alpha
    fprintf('EEG duration significantly improves model fit (p < %.2g).\n', alpha);
else
    fprintf('EEG duration does not significantly improve model fit (p >= %.2g).\n', alpha);
end

%% --- Bundle ---
DurCompare.ModelTable  = T;
DurCompare.mdl_reduced = mdl_reduced;
DurCompare.mdl_full    = mdl_full;
DurCompare.FE_full     = FE_full;
DurCompare.LRT         = lrt;
DurCompare.LRT_p       = lrt_p;
DurCompare.Duration_OR    = dur_row.OR;
DurCompare.Duration_OR_lo = dur_row.OR_lo;
DurCompare.Duration_OR_hi = dur_row.OR_hi;
DurCompare.Duration_p     = dur_row.p;
end

function S = apply_duration_source(S, R, durSource, durCol, countCol, deadtimeCsv, deadtimeCountCol)
% Overwrite the duration (and, for "deadtime", the spike count) used for spike
% rates, so that the chosen source propagates to spike rates, the routine <4h
% filter, and all downstream duration use.
%
%   "file"     : keep Duration_sec from the spike-counts file (full EDF span)
%   "natus"    : use duration_hms from the report (clips pre-connection segments)
%   "deadtime" : use the per-file split from count_spikes_deadtime.m --
%                count  = n_spikes_recording   (detections outside the dead window)
%                duration = EDF_Duration_sec - Deadtime_sec
%
% The "deadtime" denominator is EDF_Duration_sec - Deadtime_sec rather than
% Recorded_Duration_sec: the recording count was taken over exactly that span,
% so using it keeps numerator and denominator referring to the same window even
% for rows where the dead time was clamped to zero.

if nargin < 5, countCol         = ""; end
if nargin < 6, deadtimeCsv      = ""; end
if nargin < 7, deadtimeCountCol = ""; end

durSource = lower(string(durSource));

switch durSource
    case "file"
        fprintf('[Duration source] Using file duration (%s from spike counts).\n', durCol);
        return

    case "natus"
        require_cols(R, ["patient_id","session_number","duration_hms"], "ReportTable");

        % Parse duration_hms to seconds (accept either a duration array or HH:MM:SS text)
        dur_raw = R.duration_hms;
        if isduration(dur_raw)
            natus_sec = seconds(dur_raw);
        else
            natus_sec = seconds(duration(strtrim(string(dur_raw)), 'InputFormat','hh:mm:ss'));
        end

        % Unique (Patient, Session) -> Natus seconds map, with conflict check
        Map = table(double(R.patient_id), double(R.session_number), natus_sec, ...
            'VariableNames', {'Patient','Session','NatusDur_sec'});
        Map = unique(Map, 'rows');
        assert_unique_keys(Map, "Patient", "Session", "ReportTable duration map");

        % Look up per spike row (ismember preserves S's row order)
        Skey = [double(S.Patient), double(S.Session)];
        Mkey = [Map.Patient, Map.Session];
        [tf, loc] = ismember(Skey, Mkey, 'rows');

        newDur = nan(height(S),1);
        newDur(tf) = Map.NatusDur_sec(loc(tf));
        S.(durCol) = newDur;

        nMissing = nnz(~tf);
        fprintf(['[Duration source] Using Natus duration (duration_hms from report). ' ...
            '%d/%d spike rows lack a matching Natus duration and are set missing ' ...
            '(they will drop out at the routine filter).\n'], nMissing, height(S));

    case "deadtime"
        assert(strlength(string(countCol)) > 0, ...
            'DURATION_SOURCE="deadtime" requires countCol to be passed to apply_duration_source.');
        assert(strlength(string(deadtimeCsv)) > 0 && isfile(deadtimeCsv), ...
            'Dead-time CSV not found: %s', deadtimeCsv);
        assert(strlength(string(deadtimeCountCol)) == 0 || ...
            string(countCol) == string(deadtimeCountCol), ...
            ['countCol is "%s" but the dead-time file was built for "%s". ' ...
             'The dead-time split is threshold-specific; regenerate it at the ' ...
             'matching threshold or change countCol.'], ...
            string(countCol), string(deadtimeCountCol));

        D = readtable(deadtimeCsv, 'TextType','string', 'VariableNamingRule','preserve');
        require_cols(D, ["Patient","Session","EDF_Duration_sec","Deadtime_sec", ...
            "n_spikes_deadtime","n_spikes_recording","n_spikes_total"], "DeadtimeTable");

        nDead  = double(D.n_spikes_deadtime);
        nRec   = double(D.n_spikes_recording);
        nTot   = double(D.n_spikes_total);
        edfSec = double(D.EDF_Duration_sec);
        deadSec = double(D.Deadtime_sec);

        % The split must be exhaustive wherever it was computed at all
        okSplit = isfinite(nDead) & isfinite(nRec);
        assert(all(nDead(okSplit) + nRec(okSplit) == nTot(okSplit)), ...
            'Dead-time file: n_spikes_deadtime + n_spikes_recording ~= n_spikes_total in %d rows.', ...
            nnz(okSplit & (nDead + nRec ~= nTot)));
        assert(all(deadSec(okSplit) >= 0 & deadSec(okSplit) <= edfSec(okSplit)), ...
            'Dead-time file: Deadtime_sec outside [0, EDF_Duration_sec] in %d rows.', ...
            nnz(okSplit & ~(deadSec >= 0 & deadSec <= edfSec)));

        % Window the recording count was actually taken over
        recSec = edfSec - deadSec;

        usable = okSplit & isfinite(recSec) & recSec > 0;
        Map = table(double(D.Patient(usable)), double(D.Session(usable)), ...
            nRec(usable), recSec(usable), ...
            'VariableNames', {'Patient','Session','RecCount','RecDur_sec'});
        Map = unique(Map, 'rows');
        assert_unique_keys(Map, "Patient", "Session", "dead-time duration map");

        Skey = [double(S.Patient), double(S.Session)];
        Mkey = [Map.Patient, Map.Session];
        [tf, loc] = ismember(Skey, Mkey, 'rows');

        newCount = nan(height(S),1);
        newDur   = nan(height(S),1);
        newCount(tf) = Map.RecCount(loc(tf));
        newDur(tf)   = Map.RecDur_sec(loc(tf));

        % Both must be set or neither, or spike rate would silently mix sources
        assert(isequal(isfinite(newCount), isfinite(newDur)), ...
            'Dead-time join produced rows with a count but no duration (or vice versa).');

        S.(countCol) = newCount;
        S.(durCol)   = newDur;

        % Report what happened, including any rows the dead-time script flagged
        nFlagNoMatch = 0; nFlagNegative = 0;
        if ismember('Flag', D.Properties.VariableNames)
            flagStr = string(D.Flag);
            nFlagNoMatch  = nnz(contains(flagStr, "no_clinical_match"));
            nFlagNegative = nnz(contains(flagStr, "negative_deadtime"));
        end
        fprintf(['[Duration source] Using dead-time-corrected counts and durations ' ...
            '(%s -> n_spikes_recording, %s -> EDF - dead time).\n' ...
            '  %d/%d spike rows matched; %d set missing (they will drop out at the routine filter).\n' ...
            '  Dead-time file: %d rows with no clinical duration match, ' ...
            '%d rows with negative dead time clamped to zero.\n'], ...
            countCol, durCol, nnz(tf), height(S), nnz(~tf), ...
            nFlagNoMatch, nFlagNegative);

        assert(any(tf), 'No spike rows matched the dead-time file; check Patient/Session alignment.');

    otherwise
        error('DURATION_SOURCE must be "file", "natus", or "deadtime" (got "%s").', durSource);
end
end


function MRI = build_patient_mri_groups(ReportForKeptSessions)
% Collapse cnt_mri / cnt_mri_lesion to one MRI group per patient.
% Groups: "Normal", "MTS" (mesial temporal sclerosis),
%         "MCD"  (malformation of cortical/brain development).
% Patients with empty cnt_mri, or a lesion label outside those two, are dropped.
% Assumes values are consistent within patient (takes first non-empty).
require_cols(ReportForKeptSessions, ["Patient","cnt_mri","cnt_mri_lesion"], ...
    "ReportForKeptSessions");

pid = double(ReportForKeptSessions.Patient);
mri = lower(strtrim(string(ReportForKeptSessions.cnt_mri)));
les = lower(strtrim(string(ReportForKeptSessions.cnt_mri_lesion)));

blank = @(x) ismissing(x) | x=="" | x=="null" | x=="[null]" | x=="<missing>";
mri(blank(mri)) = "";
les(blank(les)) = "";

[uid, ~, g] = unique(pid, 'stable');
mri_one = strings(numel(uid),1);
les_one = strings(numel(uid),1);
for k = 1:numel(uid)
    mv = mri(g==k); mv = mv(strlength(mv)>0);
    lv = les(g==k); lv = lv(strlength(lv)>0);
    if ~isempty(mv), mri_one(k) = mv(1); end
    if ~isempty(lv), les_one(k) = lv(1); end
end

grp      = strings(numel(uid),1);
isLesion = contains(mri_one,"potential epileptogenic lesion");
grp(contains(mri_one,"normal"))                        = "Normal";
grp(isLesion & contains(les_one,"mesial temporal"))    = "MTS";
grp(isLesion & contains(les_one,"malformation"))       = "MCD";

keep = strlength(grp)>0;
MRI = table(uid(keep), categorical(grp(keep), ["Normal","MTS","MCD"]), ...
    'VariableNames', {'Patient','MRIGroup'});
fprintf('[MRI groups] Assigned %d/%d cohort patients to an MRI group.\n', ...
    nnz(keep), numel(uid));
end

function [FigMRI, MRIStats] = make_fig_mri_groups(Views, EPS_RATE, EPS_FREQ, nBoot, alpha, outPath)
% Five-panel MRI figure at the patient level, primary cohort:
%   A: spike rate across Normal/MTS/MCD  (Kruskal-Wallis + Bonferroni pairwise)
%   B: seizure frequency across the same three groups
%   C-E: spike rate vs seizure frequency correlation, one per group
%        (Spearman rho, bootstrap CI, Bonferroni p across the 3 groups)
if nargin < 2 || isempty(EPS_RATE), EPS_RATE = 30e-3; end
if nargin < 3 || isempty(EPS_FREQ), EPS_FREQ = 1e-3;  end
if nargin < 4 || isempty(nBoot),    nBoot    = 5000;  end
if nargin < 5 || isempty(alpha),    alpha    = 0.05;  end
if nargin < 6, outPath = ''; end

FONT       = 18;
grp_names  = ["Normal","MTS","MCD"];
grp_colors = [0.30 0.45 0.70; 0.85 0.33 0.10; 0.20 0.55 0.45];

%% --- Patient-level data: reuse primary cohort spike rate + sz freq ---
P   = Views.PatientSpikeSz_All(:, {'Patient','MeanSpikeRate_perHour','MeanSzFreq'});
MRI = build_patient_mri_groups(Views.ReportForKeptSessions);
D   = innerjoin(P, MRI, 'Keys', 'Patient');
D   = D(isfinite(D.MeanSpikeRate_perHour) & isfinite(D.MeanSzFreq), :);

nPer = arrayfun(@(nm) nnz(D.MRIGroup==nm), grp_names);
fprintf('[MRI figure] Normal N=%d, MTS N=%d, MCD N=%d (total %d)\n', ...
    nPer(1), nPer(2), nPer(3), height(D));

% Shared correlation-panel axis limits (comparable across groups)
xl_corr = [min(to_log10_per_month(D.MeanSzFreq,EPS_FREQ))-0.3, ...
           max(to_log10_per_month(D.MeanSzFreq,EPS_FREQ))+0.3];
yl_corr = [min(to_log10_per_hour(D.MeanSpikeRate_perHour,EPS_RATE))-0.3, ...
           max(to_log10_per_hour(D.MeanSpikeRate_perHour,EPS_RATE))+0.3];

FigMRI = figure('Color','w','Position',[60 60 1500 900]);
tiledlayout(FigMRI, 2, 3, 'TileSpacing','compact','Padding','compact');

%% --- Panel A: spike rate across groups ---
axA = nexttile(1);
sA = group_panel(axA, D.MeanSpikeRate_perHour, D.MRIGroup, EPS_RATE, ...
    'Spikes/hour (log scale)', 'A. Spike rate by MRI group');

%% --- Panel B: seizure frequency across groups ---
axB = nexttile(2);
sB = group_panel(axB, D.MeanSzFreq, D.MRIGroup, EPS_FREQ, ...
    'Seizures/month (log scale)', 'B. Seizure frequency by MRI group');

%% --- Tile 3: group-N summary ---
axS = nexttile(3); axis(axS,'off');
text(axS, 0.02, 0.95, sprintf(['MRI groups (patient level)\n\n' ...
    'Normal: N=%d\nMTS: N=%d\nMCD: N=%d\n\nTotal: N=%d'], ...
    nPer(1), nPer(2), nPer(3), height(D)), ...
    'Units','normalized','VerticalAlignment','top','FontSize',FONT);

%% --- Panels C-E: correlation per group (Bonferroni over 3 groups) ---
titles = ["C. Normal","D. MTS","E. MCD"];
corrStats = struct('Group',{},'rho',{},'p_raw',{},'p_bonf',{},'ci',{},'N',{});
for k = 1:3
    ax  = nexttile(3+k);
    sub = D(D.MRIGroup==grp_names(k), :);
    cs  = corr_panel(ax, sub.MeanSpikeRate_perHour, sub.MeanSzFreq, ...
        grp_colors(k,:), titles(k), 3);
    cs.Group = grp_names(k);
    corrStats(k) = cs;
end

%% --- Save + bundle ---
if strlength(string(outPath)) > 0
    save_fig(FigMRI, outPath);
    fprintf('Saved MRI figure: %s\n', outPath);
end
MRIStats.GroupN     = table(grp_names(:), nPer(:), 'VariableNames',{'Group','N'});
MRIStats.SpikeRate  = sA;
MRIStats.SzFreq     = sB;
MRIStats.Corr       = corrStats;

%% ================= nested panel helpers =================
    function stats = group_panel(ax, valAll, grpAll, epsFloor, ylab, ttl)
        hold(ax,'on'); box(ax,'off'); grid(ax,'on');
        Y  = to_log10_per_hour(valAll, epsFloor);   % generic log10 w/ eps floor
        yz = log10(epsFloor);
        yl = [min([yz; Y])-0.3, max(Y)+0.6];
        Yj = add_y_jitter_eps(Y, yz, yl, 0.02);

        boxchart(ax, grpAll, Yj, 'BoxFaceAlpha',0.25,'MarkerStyle','none');
        swarmchart(ax, grpAll, Yj, 18, 'filled','MarkerFaceAlpha',0.30);
        yline(ax, yz, ':', 'Color',[0.4 0.4 0.4], 'LineWidth',1.2);
        for k = 1:numel(grp_names)
            [m,lo,hi] = bootstrap_median_ci(valAll(grpAll==grp_names(k)), nBoot, alpha);
            add_median_ci_overlay(ax, k, m, lo, hi, epsFloor);
        end
        ylim(ax, yl);
        set_log10_ticks(ax, 'y', epsFloor, yl);
        ylabel(ax, ylab, 'FontSize', FONT);
        title(ax, ttl, 'FontSize', FONT, 'FontWeight','bold');

        p_kw = kruskalwallis(valAll, grpAll, 'off');
        pairs = [1 2; 1 3; 2 3];
        p_pair = nan(3,1);
        for i = 1:3
            xa = valAll(grpAll==grp_names(pairs(i,1)));
            xb = valAll(grpAll==grp_names(pairs(i,2)));
            if nnz(isfinite(xa))>=3 && nnz(isfinite(xb))>=3
                p_pair(i) = ranksum(xa, xb, 'method','approx');
            end
        end
        p_pair_bonf = min(p_pair*3, 1);
        yStep = 0.11*range(yl); y0 = yl(2) - 0.05*range(yl);
        for i = 1:3
            if isnan(p_pair_bonf(i)), continue; end
            add_sigbar(ax, pairs(i,1), pairs(i,2), y0-(i-1)*yStep, p_label(p_pair_bonf(i)));
        end

        lbl = string(ax.XTickLabel);
        for k = 1:numel(grp_names)
            lbl(lbl==grp_names(k)) = sprintf('%s (N=%d)', grp_names(k), nnz(grpAll==grp_names(k)));
        end
        ax.XTickLabel = lbl; ax.XTickLabelRotation = 20;
        set(ax,'FontSize',FONT);
        stats = struct('p_kw',p_kw,'pairs',pairs,'p_pair',p_pair,'p_pair_bonf',p_pair_bonf);
    end

    function stats = corr_panel(ax, xRate, ySz, col, ttl, pBonfMult)
        hold(ax,'on'); box(ax,'off'); grid(ax,'on');
        n = numel(xRate);
        if n >= 3
            [rho,p_raw] = corr(xRate, ySz, 'Type','Spearman','Rows','complete');
            [~,lo,hi]   = bootstrap_spearman_ci(xRate, ySz, nBoot, alpha);
        else
            rho=NaN; p_raw=NaN; lo=NaN; hi=NaN;
        end
        p_bonf = min(p_raw*pBonfMult, 1);

        lx = to_log10_per_month(ySz,  EPS_FREQ);   % x = seizures/month
        ly = to_log10_per_hour (xRate, EPS_RATE);  % y = spikes/hour
        scatter(ax, lx, ly, 26, col, 'filled', 'MarkerFaceAlpha',0.5);

        nz = (xRate>0) & (ySz>0);
        if nnz(nz) >= 3
            b  = [ones(nnz(nz),1), to_log10_per_month(ySz(nz),EPS_FREQ)] \ ...
                 to_log10_per_hour(xRate(nz),EPS_RATE);
            xg = linspace(xl_corr(1), xl_corr(2), 100)';
            plot(ax, xg, b(1)+b(2)*xg, '-', 'Color',col, 'LineWidth',2);
        end

        xlim(ax, xl_corr); ylim(ax, yl_corr);
        set_log10_ticks(ax, 'x', EPS_FREQ, xl_corr);
        set_log10_ticks(ax, 'y', EPS_RATE, yl_corr);
        xlabel(ax, 'Seizures/month (log scale)', 'FontSize', FONT);
        ylabel(ax, 'Spikes/hour (log scale)',    'FontSize', FONT);
        title(ax, sprintf('%s (N=%d)', ttl, n), 'FontSize', FONT, 'FontWeight','bold');
        if isfinite(rho)
            text(ax, 0.97, 0.95, sprintf('\\rho=%.2f [%.2f-%.2f]\np_{bonf}%s', ...
                rho, lo, hi, regexprep(char(p_label(p_bonf)),'^p','')), ...
                'Units','normalized','HorizontalAlignment','right', ...
                'VerticalAlignment','top','FontSize',FONT-4,'FontWeight','bold');
        end
        set(ax,'FontSize',FONT);
        stats = struct('Group',"",'rho',rho,'p_raw',p_raw,'p_bonf',p_bonf,'ci',[lo hi],'N',n);
    end
end

function [FigBias, BiasStats] = make_fig_report_bias(Views, SzFreqPerPatient, ...
    EPS_RATE, EPS_FREQ, nBoot, alpha, outPath)
% Compares EEGs WITH an available spike report (present/absent) vs those
% WITHOUT (unknown). EEG (session) level.
%   A: spike rate            (Wilcoxon rank-sum)
%   B: seizure frequency     (patient mean assigned to each EEG; rank-sum)
%   C: epilepsy type (Gen/Temporal/Frontal) 2x3 chi-square, drawn as a
%      row-normalized confusion matrix
if nargin < 3 || isempty(EPS_RATE), EPS_RATE = 30e-3; end
if nargin < 4 || isempty(EPS_FREQ), EPS_FREQ = 1e-3;  end
if nargin < 5 || isempty(nBoot),    nBoot    = 5000;  end
if nargin < 6 || isempty(alpha),    alpha    = 0.05;  end
if nargin < 7, outPath = ''; end

FONT       = 18;
grp_names  = ["Report available","No report"];
ylab_short = ["Report avail.","No report"];
canon3     = ["General","Temporal","Frontal"];

%% --- Assemble EEG-level table: report status + spike rate + patient sz freq/type ---
RS = resolve_reported_spike_status(Views.ReportForKeptSessions);   % Patient,Session,ReportStatus
E  = innerjoin(RS, Views.SessionLevelSpikeRates(:,{'Patient','Session','SpikesPerHour'}), ...
    'Keys', {'Patient','Session'});
E  = innerjoin(E, SzFreqPerPatient(:,{'Patient','MeanSzFreq'}), 'Keys','Patient');
E  = innerjoin(E, Views.PatientLevelSpikeRates(:,{'Patient','EpiType3'}), 'Keys','Patient');
assert_unique_keys(E, "Patient", "Session", "report-bias EEG table");

hasRep        = (E.ReportStatus=="present") | (E.ReportStatus=="absent");
E.Grp         = categorical(repmat(grp_names(2),height(E),1), grp_names);
E.Grp(hasRep) = grp_names(1);

fprintf('[Report bias] %d EEGs with a report, %d without (of %d cohort EEGs)\n', ...
    nnz(hasRep), nnz(~hasRep), height(E));

FigBias = figure('Color','w','Position',[60 60 1550 520]);
tiledlayout(FigBias, 1, 3, 'TileSpacing','compact','Padding','compact');

%% --- Panel A: spike rate ---
axA = nexttile(1);
sA = ranksum_panel(axA, E.SpikesPerHour, E.Grp, EPS_RATE, @add_median_ci_overlay, ...
    'Spikes/hour (log scale)', 'A. Spike rate');

%% --- Panel B: seizure frequency ---
axB = nexttile(2);
sB = ranksum_panel(axB, E.MeanSzFreq, E.Grp, EPS_FREQ, @add_median_ci_overlay_month, ...
    'Seizures/month (log scale)', 'B. Seizure frequency');

%% --- Panel C: epilepsy type confusion matrix + 2x3 chi-square ---
axC   = nexttile(3);
typed = ismember(string(E.EpiType3), canon3) & ~ismissing(E.EpiType3);
Et    = E(typed,:);
O = zeros(2,3);
for r = 1:2
    for c = 1:3
        O(r,c) = nnz(Et.Grp==grp_names(r) & string(Et.EpiType3)==canon3(c));
    end
end
[chi2, p_chi2, df, Eexp] = chi2_contingency(O);
rowProp = O ./ max(sum(O,2),1);   % row-normalized: type distribution within each group

imagesc(axC, rowProp); hold(axC,'on');
colormap(axC, flipud(gray)); clim(axC,[0 1]);
for r = 1:2
    for c = 1:3
        txtcol = [0 0 0]; if rowProp(r,c) > 0.6, txtcol = [1 1 1]; end
        text(axC, c, r, sprintf('%d\n(%.0f%%)', O(r,c), 100*rowProp(r,c)), ...
            'HorizontalAlignment','center','VerticalAlignment','middle', ...
            'FontSize',FONT-3,'Color',txtcol);
    end
end
axC.XTick = 1:3; axC.XTickLabel = canon3;
axC.YTick = 1:2; axC.YTickLabel = ylab_short;
xlabel(axC,'Epilepsy type','FontSize',FONT);
warnstr = ''; if any(Eexp(:) < 5), warnstr = ' (expected<5)'; end
title(axC, sprintf('C. Epilepsy type\n\\chi^2(%d)=%.1f, %s%s', ...
    df, chi2, char(p_label(p_chi2)), warnstr), 'FontSize',FONT,'FontWeight','bold');
set(axC,'FontSize',FONT); box(axC,'on'); axis(axC,'tight');

%% --- Save + bundle ---
if strlength(string(outPath)) > 0
    save_fig(FigBias, outPath);
    fprintf('Saved report-bias figure: %s\n', outPath);
end
BiasStats.nWithReport = nnz(hasRep);
BiasStats.nNoReport   = nnz(~hasRep);
BiasStats.SpikeRate   = sA;
BiasStats.SzFreq      = sB;
BiasStats.EpiType     = struct('Counts',O,'Expected',Eexp,'chi2',chi2,'df',df, ...
    'p',p_chi2,'RowLabels',grp_names,'ColLabels',canon3);

%% ================= nested helper =================
    function stats = ranksum_panel(ax, val, grp, epsFloor, ciFn, ylab, ttl)
        hold(ax,'on'); box(ax,'off'); grid(ax,'on');
        x1 = val(grp==grp_names(1)); x1 = x1(isfinite(x1));
        x2 = val(grp==grp_names(2)); x2 = x2(isfinite(x2));
        if ~isempty(x1) && ~isempty(x2)
            p = ranksum(x1, x2, 'method','approx');
            d = cliff_delta(x1, x2);
        else
            p = NaN; d = NaN;
        end
        Y  = to_log10_per_hour(val, epsFloor);   % generic log10 with eps floor
        yz = log10(epsFloor);
        yl = [min([yz; Y])-0.3, max(Y)+0.7];
        Yj = add_y_jitter_eps(Y, yz, yl, 0.02);
        boxchart(ax, grp, Yj, 'BoxFaceAlpha',0.25,'MarkerStyle','none');
        swarmchart(ax, grp, Yj, 14, 'filled','MarkerFaceAlpha',0.20);
        yline(ax, yz, ':', 'Color',[0.4 0.4 0.4],'LineWidth',1.2);
        for k = 1:2
            [m,lo,hi] = bootstrap_median_ci(val(grp==grp_names(k)), nBoot, alpha);
            ciFn(ax, k, m, lo, hi, epsFloor);
        end
        ylim(ax, yl);
        set_log10_ticks(ax, 'y', epsFloor, yl);
        ylabel(ax, ylab, 'FontSize', FONT);
        if isfinite(p), add_sigbar(ax, 1, 2, yl(2)-0.10*range(yl), p_label(p)); end
        title(ax, ttl, 'FontSize', FONT, 'FontWeight','bold');
        lbl = string(ax.XTickLabel);
        for k = 1:2
            lbl(lbl==grp_names(k)) = sprintf('%s (N=%d)', grp_names(k), nnz(grp==grp_names(k)));
        end
        ax.XTickLabel = lbl; ax.XTickLabelRotation = 15;
        set(ax,'FontSize',FONT);
        stats = struct('p_ranksum',p,'cliff_delta',d,'n1',numel(x1),'n2',numel(x2));
    end
end

function [chi2, p, df, E] = chi2_contingency(O)
% Pearson chi-square test of independence for contingency table O.
rowSum = sum(O,2); colSum = sum(O,1); N = sum(O(:));
E    = rowSum * colSum / N;                 % expected counts under independence
chi2 = sum(((O - E).^2) ./ E, 'all');
df   = (size(O,1)-1) * (size(O,2)-1);
p    = 1 - chi2cdf(chi2, df);
end

function [SpearmanResults, rs_all, p_all, n_all, rho_lo, rho_hi, subtype_ci, trimInfo] = ...
    spearman_trim_top_spikers(Views, canonical3, spearman_xLims, spearman_yLims, ...
        fig_out, trimFrac, nonZeroOnly)
% Sensitivity version of the Spearman figure with the top trimFrac of highest
% spike-rate patients removed (reviewer check: is the association driven by
% very-high-density patients?). Threshold is set on the primary all-epilepsy
% cohort and applied consistently to both the All and subtype-typed tables.
if nargin < 6 || isempty(trimFrac),    trimFrac    = 0.10; end
if nargin < 7 || isempty(nonZeroOnly), nonZeroOnly = true; end   % matches Fig S1

All   = Views.PatientSpikeSz_All;
Typed = Views.PatientSpikeSz_Typed;

% Cutoff from the all-epilepsy cohort; keep the bottom (1 - trimFrac)
thr       = prctile(All.MeanSpikeRate_perHour, 100*(1-trimFrac));
keepAll   = All.MeanSpikeRate_perHour   <= thr;
keepTyped = Typed.MeanSpikeRate_perHour <= thr;

fprintf(['[Trim top %.0f%%] Spike-rate cutoff = %.2f spikes/hour. ' ...
    'Removed %d/%d all-epilepsy and %d/%d subtype-typed patients.\n'], ...
    100*trimFrac, thr, nnz(~keepAll), height(All), nnz(~keepTyped), height(Typed));

suffix = sprintf(' (top %.0f%% spikers removed)', 100*trimFrac);
[SpearmanResults, rs_all, p_all, n_all, rho_lo, rho_hi, subtype_ci] = ...
    spearman_plotting_function(All(keepAll,:), Typed(keepTyped,:), canonical3, ...
        spearman_xLims, spearman_yLims, fig_out, 'MeanSzFreq', suffix, nonZeroOnly);

trimInfo = struct('trimFrac',trimFrac,'threshold',thr, ...
    'nRemovedAll',nnz(~keepAll),'nRemovedTyped',nnz(~keepTyped), ...
    'nKeptAll',nnz(keepAll),'nKeptTyped',nnz(keepTyped));
end

%% =====================================================================
%% GENERALIZED-EPILEPSY SUBTYPE SPEARMAN FIGURE
%% =====================================================================
% Paste these three functions as LOCAL functions inside
% run_spike_sz_pipeline_clean.m (they call bootstrap_spearman_ci,
% set_log10_ticks, p_label, and save_fig, which are local to that file).
%
% Call it after Views is built, e.g. just after the Fig S1 trim block:
%
%   figGenSub_out = '../output/FigGenSubtypes.png';
%   [GenSubtypeResults, FigGenSub] = make_fig_generalized_subtypes( ...
%       Views, spearman_xLims, spearman_yLims, figGenSub_out, 10, nBoot, alpha, false);
%   disp(GenSubtypeResults);

function [GenSubtypeResults, FigGenSub] = make_fig_generalized_subtypes(Views, ...
    spearman_xLims, spearman_yLims, fig_out, minN, nBoot, alpha, nonZeroOnly)
% Spike rate vs seizure frequency, stratified by generalized-epilepsy subtype.
%
% Cohort: primary-cohort patients with EpiType3 == "General". Subtypes come
% from EpilepsySpecific, canonicalized by keyword (see
% canonicalize_generalized_subtype). Subtypes with fewer than minN patients
% are pooled into "Other generalized"; if that pool is still under minN it is
% dropped from the panels but retained in the pooled "All generalized" panel.
%
% Panel A is all generalized patients; each remaining panel is one subtype.
% Subtype p-values are Bonferroni-adjusted across the plotted subtypes only.
%
% Returns a results table (pooled row first, p_bonf = NaN for that row).

if nargin < 2 || isempty(spearman_xLims), spearman_xLims = [-3.5 4]; end
if nargin < 3 || isempty(spearman_yLims), spearman_yLims = [-1.5 3];  end
if nargin < 4, fig_out = ''; end
if nargin < 5 || isempty(minN),        minN        = 10;    end
if nargin < 6 || isempty(nBoot),       nBoot       = 5000;  end
if nargin < 7 || isempty(alpha),       alpha       = 0.05;  end
if nargin < 8 || isempty(nonZeroOnly), nonZeroOnly = false; end

FONT = 20;
COL  = [0.00 0.45 0.74;
        0.85 0.33 0.10;
        0.93 0.69 0.13;
        0.49 0.18 0.56;
        0.47 0.67 0.19;
        0.30 0.75 0.93;
        0.64 0.08 0.18];
COL_ALL = [0.45 0.45 0.45];

%% ---------------- Build the generalized-only patient table ----------------
require_cols(Views.PatientLevelSpikeRates, ...
    ["Patient","EpiType3","EpilepsySpecific"], "PatientLevelSpikeRates");
require_cols(Views.PatientSpikeSz_All, ...
    ["Patient","MeanSpikeRate_perHour","MeanSzFreq"], "PatientSpikeSz_All");

PL = Views.PatientLevelSpikeRates(:, {'Patient','EpiType3','EpilepsySpecific'});
D  = innerjoin(Views.PatientSpikeSz_All(:, ...
        {'Patient','MeanSpikeRate_perHour','MeanSzFreq'}), PL, 'Keys','Patient');

D = D(~ismissing(D.EpiType3) & string(D.EpiType3) == "General", :);
D = D(isfinite(D.MeanSpikeRate_perHour) & isfinite(D.MeanSzFreq), :);
if nonZeroOnly
    D = D(D.MeanSpikeRate_perHour > 0 & D.MeanSzFreq > 0, :);
end

assert(~isempty(D), ...
    'No generalized-epilepsy patients with both a spike rate and a seizure frequency.');
assert(numel(unique(D.Patient)) == height(D), ...
    'Duplicate Patient rows in the generalized-subtype table (%d rows, %d patients).', ...
    height(D), numel(unique(D.Patient)));

D.Subtype = canonicalize_generalized_subtype(D.EpilepsySpecific);
assert(all(strlength(D.Subtype) > 0), 'Empty subtype label produced for %d patients.', ...
    nnz(strlength(D.Subtype) == 0));

%% ---------------- Inventory of raw -> canonical labels ----------------
% Printed so the keyword mapping can be checked against the actual vocabulary
% of epilepsy_specific before the figure is used for anything.
rawLab = strtrim(string(D.EpilepsySpecific));
rawLab(ismissing(rawLab) | rawLab == "") = "<blank>";
[gr, rawU] = findgroups(rawLab);
canonU     = splitapply(@(s) s(1), D.Subtype, gr);
nRaw       = splitapply(@numel, D.Subtype, gr);
fprintf('\n[Generalized subtypes] epilepsy_specific -> canonical label (N=%d patients)\n', height(D));
for i = 1:numel(rawU)
    fprintf('  %-45s -> %-26s N=%d\n', rawU(i), canonU(i), nRaw(i));
end

%% ---------------- Pool rare subtypes, then drop panels under minN ----------------
[g0, lab0] = findgroups(D.Subtype);
n0    = splitapply(@numel, D.Subtype, g0);
small = lab0(n0 < minN);
D.Subtype(ismember(D.Subtype, small)) = "Other generalized";

[g1, lab1] = findgroups(D.Subtype);
n1   = splitapply(@numel, D.Subtype, g1);
keep = n1 >= minN;
labs = lab1(keep);
nPer = n1(keep);

[~, ord] = sort(nPer, 'descend');
labs = labs(ord);
nPer = nPer(ord);

% Keep the catch-all buckets at the end regardless of size
isLast = ismember(labs, ["Other generalized","Unspecified generalized"]);
labs   = [labs(~isLast); labs(isLast)];
nPer   = [nPer(~isLast); nPer(isLast)];
nG     = numel(labs);

assert(nG >= 1, ...
    ['No generalized subtype reaches minN=%d patients (largest group N=%d). ' ...
     'Lower minN or widen the keyword mapping.'], minN, max(n1));

fprintf('[Generalized subtypes] %d subtype panels at minN=%d; %d/%d patients shown in panels\n', ...
    nG, minN, sum(nPer), height(D));

%% ---------------- Correlations ----------------
[rho_all, lo_all, hi_all, p_all] = spearman_with_ci( ...
    D.MeanSpikeRate_perHour, D.MeanSzFreq, nBoot, alpha);

rho = nan(nG,1); lo = nan(nG,1); hi = nan(nG,1); p_raw = nan(nG,1);
for k = 1:nG
    m = (D.Subtype == labs(k));
    [rho(k), lo(k), hi(k), p_raw(k)] = spearman_with_ci( ...
        D.MeanSpikeRate_perHour(m), D.MeanSzFreq(m), nBoot, alpha);
end
p_bonf = min(p_raw * nG, 1);

GenSubtypeResults = table( ...
    ["All generalized"; labs(:)], ...
    [height(D); nPer(:)], ...
    [rho_all; rho], [lo_all; lo], [hi_all; hi], ...
    [p_all; p_raw], [NaN; p_bonf], ...
    'VariableNames', {'Group','N','Spearman_rho','ci_lo','ci_hi','p_raw','p_bonf'});

%% ---------------- Zero handling / log-scale floors ----------------
% Same convention as spearman_plotting_function: zeros are pinned half a
% decade-step below the smallest positive value and drawn as asterisks.
minpos_rate = min(D.MeanSpikeRate_perHour(D.MeanSpikeRate_perHour > 0));
minpos_sz   = min(D.MeanSzFreq(D.MeanSzFreq > 0));
if isempty(minpos_rate) || ~isfinite(minpos_rate), minpos_rate = 1e-6; end
if isempty(minpos_sz)   || ~isfinite(minpos_sz),   minpos_sz   = 1e-6; end
eps_rate = 0.5 * minpos_rate;
eps_sz   = 0.5 * minpos_sz;
xZero    = log10(eps_sz);
yZero    = log10(eps_rate);

%% ---------------- Figure ----------------
nPanels = nG + 1;
nCols   = min(3, nPanels);
nRows   = ceil(nPanels / nCols);

FigGenSub = figure('Color','w','Position',[60 60 min(1800, 600*nCols) 460*nRows]);
tiledlayout(FigGenSub, nRows, nCols, 'Padding','compact', 'TileSpacing','compact');

% Panel A: all generalized
axA = nexttile(1);
draw_corr_panel(axA, D.MeanSzFreq, D.MeanSpikeRate_perHour, COL_ALL, ...
    sprintf('A. All generalized (N=%d)', height(D)), ...
    sprintf('\\rho=%.2f [%.2f-%.2f], %s', rho_all, lo_all, hi_all, p_label(p_all)));

% Panels B..: one per subtype
for k = 1:nG
    ax  = nexttile(k+1);
    m   = (D.Subtype == labs(k));
    col = COL(mod(k-1, size(COL,1)) + 1, :);
    txt = sprintf('\\rho=%.2f [%.2f-%.2f], p_{bonf}%s', rho(k), lo(k), hi(k), ...
        string(regexprep(char(p_label(p_bonf(k))), '^p', '')));
    draw_corr_panel(ax, D.MeanSzFreq(m), D.MeanSpikeRate_perHour(m), col, ...
        sprintf('%s. %s (N=%d)', char('A'+k), labs(k), nPer(k)), txt);
end

if strlength(string(fig_out)) > 0
    save_fig(FigGenSub, fig_out);
    fprintf('Saved generalized-subtype figure: %s\n', fig_out);
end

%% ================= nested plotting helper =================
    function draw_corr_panel(ax, xSz, ySpike, col, ttl, annotTxt)
        hold(ax,'on'); grid(ax,'on'); box(ax,'off');

        logX = log10(xSz    + (xSz    <= 0) .* eps_sz);
        logY = log10(ySpike + (ySpike <= 0) .* eps_rate);
        isZx = (xSz == 0);
        isZy = (ySpike == 0);

        xline(ax, xZero, ':', 'Color',[0.4 0.4 0.4], 'LineWidth',1.2);
        yline(ax, yZero, ':', 'Color',[0.4 0.4 0.4], 'LineWidth',1.2);

        scatter(ax, logX(~isZx & ~isZy), logY(~isZx & ~isZy), 20, col, ...
            'filled', 'MarkerFaceAlpha',0.35);
        if any(isZx & ~isZy)
            plot(ax, logX(isZx & ~isZy), logY(isZx & ~isZy), '*', ...
                'Color',col, 'MarkerSize',8, 'LineWidth',1.1);
        end
        if any(~isZx & isZy)
            plot(ax, logX(~isZx & isZy), logY(~isZx & isZy), '*', ...
                'Color',col, 'MarkerSize',8, 'LineWidth',1.1);
        end
        if any(isZx & isZy)
            plot(ax, logX(isZx & isZy), logY(isZx & isZy), '*', ...
                'Color',col, 'MarkerSize',9, 'LineWidth',1.2);
        end

        % Fit line on doubly non-zero points only, as in the main Spearman figure
        nz = ~isZx & ~isZy;
        if nnz(nz) >= 3
            b  = [ones(nnz(nz),1), logX(nz)] \ logY(nz);
            xg = linspace(spearman_xLims(1), spearman_xLims(2), 250)';
            plot(ax, xg, b(1) + b(2)*xg, '-', 'Color',col, 'LineWidth',2);
        end

        xlim(ax, spearman_xLims); ylim(ax, spearman_yLims);
        xlabel(ax, 'Seizures per month (log scale)', 'FontSize',FONT);
        ylabel(ax, 'Spikes per hour (log scale)',    'FontSize',FONT);
        set_log10_ticks(ax, 'x', eps_sz,   spearman_xLims);
        set_log10_ticks(ax, 'y', eps_rate, spearman_yLims);
        labsX = string(ax.XTickLabel);
        [~, iMax] = max(ax.XTick);
        labsX(iMax) = "";
        ax.XTickLabel = labsX;

        title(ax, ttl, 'FontSize',FONT, 'FontWeight','bold');
        text(ax, 0.98, 0.95, annotTxt, 'Units','normalized', ...
            'HorizontalAlignment','right', 'VerticalAlignment','top', ...
            'FontSize',FONT-4, 'FontWeight','bold');
        set(ax, 'FontSize', FONT);
    end
end


function [rho, lo, hi, p] = spearman_with_ci(xRate, ySz, nBoot, alpha)
% Spearman rho between spike rate and seizure frequency, with a
% patient-level bootstrap CI. Returns NaNs if fewer than 3 usable patients.
xRate = double(xRate(:));
ySz   = double(ySz(:));
mask  = isfinite(xRate) & isfinite(ySz);
xRate = xRate(mask); ySz = ySz(mask);
if numel(xRate) < 3
    rho = NaN; lo = NaN; hi = NaN; p = NaN;
    return
end
[rho, p]   = corr(xRate, ySz, 'Type','Spearman', 'Rows','complete');
[~, lo, hi] = bootstrap_spearman_ci(xRate, ySz, nBoot, alpha);
end


function lab = canonicalize_generalized_subtype(specRaw)
% Map free-text epilepsy_specific values to canonical generalized-syndrome
% labels. Anything unrecognized keeps its original text, so unmapped
% vocabulary shows up in the printed inventory rather than being silently
% swallowed into an "other" bucket.
s   = lower(strtrim(string(specRaw)));
lab = strings(numel(s), 1);

wordHit = @(t, w) ~isempty(regexp(t, ['\<' w '\>'], 'once'));

for i = 1:numel(s)
    t = s(i);
    if ismissing(t) || t == "" || t == "null" || t == "[null]" || t == "<missing>" || ...
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
    elseif contains(t,"tonic-clonic alone") || contains(t,"tonic clonic alone") || wordHit(t,"gtca")
        lab(i) = "GTC alone";
    elseif contains(t,"absence")
        lab(i) = "Absence, other";
    elseif contains(t,"tonic-clonic") || contains(t,"tonic clonic")
        lab(i) = "GTC alone";
    else
        lab(i) = strtrim(string(specRaw(i)));
    end
end
end