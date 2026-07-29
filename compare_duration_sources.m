function Cmp = compare_duration_sources(opts)
%COMPARE_DURATION_SOURCES  Agreement between two DURATION_SOURCE definitions
%of spike rate, as used by run_spike_sz_pipeline_clean.
%
%   Cmp = compare_duration_sources;                  % file vs deadtime
%   Cmp = compare_duration_sources(source="natus");  % natus vs deadtime
%
% Standalone: does not call into the pipeline, but reproduces its spike-rate
% definitions exactly.
%
%   file      rate = (ALL detections)              / EDF span
%   natus     rate = (ALL detections)              / duration_hms
%   deadtime  rate = (detections outside the dead window) / (EDF span - dead time)
%
% Dead-time is always the reference (x axis, denominator of the ratio).
%
% ---------------------------------------------------------------------
% WHY "file" AND "deadtime" DISAGREE
%
% Let  f = Deadtime_sec / EDF_Duration_sec      (share of the file that is dead)
%      q = n_spikes_deadtime / n_spikes_total   (share of detections in it)
%
% Then, exactly:
%
%       rate_file / rate_deadtime  =  (1 - f) / (1 - q)
%
% so the two agree if and only if q == f. Two distinct regimes are possible:
%   q > f   the dead window is ENRICHED for detections; the file rate is
%           inflated by junk picked up on disconnected electrodes
%   q < f   the dead window is QUIET; dead-time correction is mostly removing
%           denominator, and every file rate shifts down by a similar factor
% Panel C separates them. Panel D shows which half of the ratio dominates.
%
% ---------------------------------------------------------------------
% IS THE DEAD WINDOW REAL? (panels E and F)
%
% duration_hms is meant to clip pre-connection segments, so
%
%       clip = EDF_Duration_sec - duration_hms
%
% is an INDEPENDENT estimate of the same leading dead window. Panel E plots
% Deadtime_sec against it. Agreement means two unrelated methods converge on
% the same window and the correction can be trusted. Systematic overshoot
% (Deadtime_sec > clip) means the dead-time detector is flagging more than
% pre-connection time, in which case every dead-time rate is inflated by a
% too-small denominator. Panel F shows the two implied dead fractions as
% distributions.
%
% ---------------------------------------------------------------------
% FIGURES (2 x 3, saved to outDir)
%   A  rate vs rate, log-log, identity line, Spearman + Lin's CCC
%   B  Bland-Altman on log10 rate (bias and limits back-transformed to a ratio)
%   C  file mode:  detection share q vs duration share f, with identity line
%      natus mode: duration_hms vs (EDF span - dead time)
%   D  numerator and denominator contributions to the log ratio, vs dead fraction
%   E  Deadtime_sec vs the Natus-implied clip, with identity line
%   F  distribution of both implied dead fractions
%
% Bland-Altman is done on log10 because spike rates span several orders of
% magnitude; a raw-difference BA plot on these data is dominated by the few
% highest-rate EEGs and its limits of agreement are not interpretable.

arguments
    opts.source             (1,1) string  {mustBeMember(opts.source,["file","natus"])} = "file"
    opts.spikeCounts        (1,1) string  = "../data/spike_counts.csv"
    opts.report             (1,1) string  = "../data/clinical_data_deidentified.csv"
    opts.deadtime           (1,1) string  = "../data/spike_counts_deadtime_0p46.csv"
    opts.countCol           (1,1) string  = "count_0_46"
    opts.outDir             (1,1) string  = "../output"
    opts.maxRoutineHours    (1,1) double  = 4
    opts.restrictOutpatient (1,1) logical = true
    opts.restrictRoutine    (1,1) logical = true   % <=maxRoutineHours under BOTH sources
    opts.validateDeadtime   (1,1) logical = true   % panels E and F
    opts.figName            (1,1) string  = ""
end

FONT   = 13;
isFile = (opts.source == "file");
if strlength(opts.figName) == 0
    opts.figName = sprintf("duration_source_agreement_%s_vs_deadtime.png", opts.source);
end
if ~exist(opts.outDir,'dir'), mkdir(opts.outDir); end

%% ===================== 1. LOAD =====================
S = readtable(opts.spikeCounts,'TextType','string','VariableNamingRule','preserve');
require_cols(S, ["Patient","Session",opts.countCol,"Duration_sec"], "Spikes");

D = readtable(opts.deadtime,'TextType','string','VariableNamingRule','preserve');
require_cols(D, ["Patient","Session","EDF_Duration_sec","Deadtime_sec", ...
                 "n_spikes_deadtime","n_spikes_recording","n_spikes_total"], "Deadtime");

needNatus  = opts.validateDeadtime || ~isFile;
needReport = opts.restrictOutpatient || needNatus;
if needReport
    R = readtable(opts.report,'TextType','string','VariableNamingRule','preserve');
    reqR = ["patient_id","session_number"];
    if opts.restrictOutpatient
        reqR = [reqR, "acquired_on","report_PATIENT_CLASS","jay_in_or_out"];
    end
    if needNatus, reqR = [reqR, "duration_hms"]; end
    require_cols(R, reqR, "Report");
end

Base = table(double(S.Patient), double(S.Session), ...
    double(S.(opts.countCol)), double(S.Duration_sec), ...
    'VariableNames',{'Patient','Session','N_total_file','EDF_sec_file'});
assert_unique_keys(Base, "Base spike counts");

%% ===================== 2. DEAD-TIME TABLE =====================
Dead = table(double(D.Patient), double(D.Session), ...
    double(D.n_spikes_total), double(D.n_spikes_recording), ...
    double(D.n_spikes_deadtime), double(D.EDF_Duration_sec), double(D.Deadtime_sec), ...
    'VariableNames',{'Patient','Session','N_total_dt','N_rec','N_dead', ...
                     'EDF_sec','Dead_sec'});

ok = isfinite(Dead.N_dead) & isfinite(Dead.N_rec) & isfinite(Dead.N_total_dt);
assert(all(Dead.N_dead(ok) + Dead.N_rec(ok) == Dead.N_total_dt(ok)), ...
    'Dead-time file: spike split is not exhaustive in %d rows.', ...
    nnz(ok & (Dead.N_dead + Dead.N_rec ~= Dead.N_total_dt)));
assert(all(Dead.Dead_sec(ok) >= 0 & Dead.Dead_sec(ok) <= Dead.EDF_sec(ok)), ...
    'Dead-time file: Deadtime_sec outside [0, EDF_Duration_sec].');

Dead.T_rec_sec = Dead.EDF_sec - Dead.Dead_sec;
nDeadRaw   = height(Dead);
nDeadDropped = nnz(~(ok & isfinite(Dead.T_rec_sec) & Dead.T_rec_sec > 0));
Dead = Dead(ok & isfinite(Dead.T_rec_sec) & Dead.T_rec_sec > 0, :);
Dead = unique(Dead, 'rows');
assert_unique_keys(Dead, "dead-time map");

%% ===================== 3. NATUS DURATIONS =====================
% Needed as the comparator in natus mode, and as the independent dead-window
% estimate for panels E and F in either mode.
Natus = table('Size',[0 3], 'VariableTypes',{'double','double','double'}, ...
    'VariableNames',{'Patient','Session','T_natus_sec'});
if needNatus
    dur_raw = R.duration_hms;
    if isduration(dur_raw)
        natus_sec = seconds(dur_raw);
    else
        natus_sec = seconds(duration(strtrim(string(dur_raw)),'InputFormat','hh:mm:ss'));
    end
    Natus = table(double(R.patient_id), double(R.session_number), natus_sec, ...
        'VariableNames',{'Patient','Session','T_natus_sec'});
    nNatusRaw = height(Natus);
    Natus = Natus(isfinite(Natus.T_natus_sec) & Natus.T_natus_sec > 0, :);
    Natus = unique(Natus, 'rows');
    assert_unique_keys(Natus, "Natus duration map");
end

%% ===================== 4. COMPARATOR DURATION =====================
if isFile
    Cmpr = Base(:, {'Patient','Session'});
    Cmpr.T_cmp_sec = Base.EDF_sec_file;
    Cmpr = Cmpr(isfinite(Cmpr.T_cmp_sec) & Cmpr.T_cmp_sec > 0, :);
    nCmpRaw = height(Base);
else
    Cmpr = Natus;
    Cmpr.Properties.VariableNames{'T_natus_sec'} = 'T_cmp_sec';
    nCmpRaw = nNatusRaw;
end
assert_unique_keys(Cmpr, "comparator duration map");

%% ===================== 5. JOIN =====================
J = innerjoin(Base, Cmpr, 'Keys',{'Patient','Session'});
J = innerjoin(J,    Dead, 'Keys',{'Patient','Session'});
assert(~isempty(J), 'No EEGs have both a %s duration and a dead-time row.', opts.source);
assert_unique_keys(J, "joined comparison table");

% Left-join the Natus duration so a missing one costs panel E a point rather
% than dropping the EEG from the whole comparison.
if isFile && opts.validateDeadtime
    J = outerjoin(J, Natus, 'Keys',{'Patient','Session'}, 'Type','left','MergeKeys',true);
elseif ~isFile
    J.T_natus_sec = J.T_cmp_sec;
end

% Both files must come from the same detections at the same threshold,
% otherwise this compares two detectors rather than two duration definitions.
assert(all(J.N_total_dt == J.N_total_file), ...
    ['Total spike counts disagree between %s and the dead-time file in %d/%d EEGs. ' ...
     'The two files were built at different thresholds or from different runs.'], ...
    opts.countCol, nnz(J.N_total_dt ~= J.N_total_file), height(J));

edfMismatch = abs(J.EDF_sec - J.EDF_sec_file) > 1;   % 1 s tolerance
assert(~any(edfMismatch), ...
    'EDF_Duration_sec disagrees with Duration_sec by >1 s in %d EEGs.', nnz(edfMismatch));

%% ===================== 6. FILTERS =====================
maxSec = opts.maxRoutineHours * 3600;
routC  = J.T_cmp_sec <= maxSec;
routD  = J.T_rec_sec <= maxSec;

FilterDiscord = struct( ...
    'nRoutine_cmp_only',  nnz(routC & ~routD), ...
    'nRoutine_dead_only', nnz(routD & ~routC), ...
    'nRoutine_both',      nnz(routC &  routD), ...
    'nJoined',            height(J));

% T_rec <= T_edf by construction, so in file mode the routine set under
% dead-time must be a superset of the routine set under file.
if isFile
    assert(FilterDiscord.nRoutine_cmp_only == 0, ...
        ['%d EEGs are routine under the EDF span but not after dead-time removal. ' ...
         'That is impossible unless Deadtime_sec is negative somewhere.'], ...
        FilterDiscord.nRoutine_cmp_only);
end

if opts.restrictOutpatient
    acq   = lower(strtrim(string(R.acquired_on)));
    cls   = lower(strtrim(string(R.report_PATIENT_CLASS)));
    jay   = lower(strtrim(string(R.jay_in_or_out)));
    isOut = contains(acq,"spe") | contains(acq,"radnor") | (cls=="outpatient") | (jay=="out");
    OutKeys = unique(table(double(R.patient_id(isOut)), double(R.session_number(isOut)), ...
        'VariableNames',{'Patient','Session'}), 'rows');
    assert(~isempty(OutKeys), 'No outpatient sessions identified.');
    J = innerjoin(J, OutKeys, 'Keys',{'Patient','Session'});
    routC = J.T_cmp_sec <= maxSec;
    routD = J.T_rec_sec <= maxSec;
end

if opts.restrictRoutine
    J = J(routC & routD, :);
end
assert(height(J) >= 10, 'Only %d EEGs survive the filters; nothing to plot.', height(J));

%% ===================== 7. RATES =====================
J.Rate_cmp  = J.N_total_file ./ J.T_cmp_sec * 3600;
J.Rate_dead = J.N_rec        ./ J.T_rec_sec * 3600;
J.DeadFrac  = J.Dead_sec ./ J.EDF_sec;                             % f
J.DetFrac   = J.N_dead ./ max(J.N_total_file, 1);                  % q (0 if no spikes)

both = isfinite(J.Rate_cmp) & isfinite(J.Rate_dead);
pos  = both & J.Rate_cmp > 0 & J.Rate_dead > 0;   % log axes need strictly positive

ZeroCounts = struct( ...
    'nZero_cmp_only',  nnz(both & J.Rate_cmp  == 0 & J.Rate_dead > 0), ...
    'nZero_dead_only', nnz(both & J.Rate_dead == 0 & J.Rate_cmp  > 0), ...
    'nZero_both',      nnz(both & J.Rate_cmp  == 0 & J.Rate_dead == 0), ...
    'nPositiveBoth',   nnz(pos));

x = J.Rate_dead(pos);       % reference
y = J.Rate_cmp(pos);        % comparator
lx = log10(x); ly = log10(y);

%% ===================== 8. AGREEMENT STATISTICS =====================
[rho_s,  p_s]   = corr(x, y, 'Type','Spearman');
[r_log,  p_log] = corr(lx, ly, 'Type','Pearson');
ccc_log = lins_ccc(lx, ly);

d  = ly - lx;                       % log10 ratio, comparator / dead-time
mu = (ly + lx) / 2;
bias   = mean(d);
sd_d   = std(d);
loa_lo = bias - 1.96*sd_d;
loa_hi = bias + 1.96*sd_d;

[rho_prop, p_prop] = corr(mu, d, 'Type','Spearman');

num_part = log10(J.N_total_file(pos) ./ J.N_rec(pos));   % >= 0 always
den_part = log10(J.T_cmp_sec(pos)    ./ J.T_rec_sec(pos));
assert(max(abs(d - (num_part - den_part))) < 1e-9, ...
    'Log-ratio decomposition does not reconstruct the observed difference.');

if isFile
    f = J.DeadFrac(pos); q = J.DetFrac(pos);
    assert(max(abs(d - (log10(1-f) - log10(1-q)))) < 1e-9, ...
        'The (1-f)/(1-q) identity does not hold; f or q is computed wrong.');
    Enrich = struct('f', f, 'q', q, ...
        'nEnriched',    nnz(q > f), 'fracEnriched', nnz(q > f)/numel(f), ...
        'nDepleted',    nnz(q < f), ...
        'nNoDetInDead', nnz(q == 0), ...
        'median_f',     median(f), 'median_q', median(q), ...
        'nAnyDeadtime', nnz(f > 0));
else
    Enrich = struct('note', "not computed in natus mode");
end

%% ===================== 9. DEAD-WINDOW VALIDATION (panels E, F) =====================
% clip = EDF span - duration_hms is an independent estimate of the same
% leading dead window, from the Natus header rather than from the signal.
doValidate = opts.validateDeadtime && ismember('T_natus_sec', J.Properties.VariableNames);
if doValidate
    clip_sec = J.EDF_sec - J.T_natus_sec;
    vOK  = isfinite(clip_sec) & isfinite(J.Dead_sec);
    cs   = clip_sec(vOK);
    ds   = J.Dead_sec(vOK);
    resid = ds - cs;                              % >0 means dead-time flags more

    [rho_v, p_v] = corr(cs, ds, 'Type','Spearman');
    Validation = struct( ...
        'n',              nnz(vOK), ...
        'nMissingNatus',  nnz(~vOK), ...
        'nNegativeClip',  nnz(cs < 0), ...
        'rho',            rho_v, 'p', p_v, ...
        'median_clip_min',   median(cs)/60, ...
        'median_dead_min',   median(ds)/60, ...
        'median_resid_min',  median(resid)/60, ...
        'iqr_resid_min',     [prctile(resid,25) prctile(resid,75)]/60, ...
        'fracOvershoot',     mean(resid > 60), ...   % >1 min more than the clip
        'clipFrac',          cs ./ J.EDF_sec(vOK), ...
        'deadFrac',          ds ./ J.EDF_sec(vOK));
else
    Validation = struct('note', "not computed (validateDeadtime off or no Natus durations)");
end

%% ===================== 10. FIGURE =====================
cmpLabel = char(opts.source);
fig = figure('Color','w','Position',[40 40 1700 900]);
tiledlayout(fig, 2, 3, 'TileSpacing','compact','Padding','compact');
COL = [0.22 0.45 0.70];

% ---- A: rate vs rate ----
axA = nexttile(1); hold(axA,'on'); box(axA,'off'); grid(axA,'on');
lims = [min([lx;ly]) max([lx;ly])] + [-0.15 0.15];
plot(axA, lims, lims, 'k--', 'LineWidth',1.4);
scatter(axA, lx, ly, 14, COL, 'filled', 'MarkerFaceAlpha',0.28);
xlim(axA, lims); ylim(axA, lims); axis(axA,'square');
xlabel(axA, 'Dead-time corrected (spikes/h, log_{10})', 'FontSize',FONT);
ylabel(axA, sprintf('%s (spikes/h, log_{10})', cmpLabel), 'FontSize',FONT);
title(axA, sprintf('A. Rate agreement (N=%d EEGs)', numel(x)), ...
    'FontSize',FONT, 'FontWeight','bold');
text(axA, 0.03, 0.97, sprintf('\\rho_{s}=%.3f\nr_{log}=%.3f\nCCC_{log}=%.3f', ...
    rho_s, r_log, ccc_log), 'Units','normalized', ...
    'VerticalAlignment','top', 'FontSize',FONT-2);
set(axA,'FontSize',FONT);

% ---- B: Bland-Altman on log10 ----
axB = nexttile(2); hold(axB,'on'); box(axB,'off'); grid(axB,'on');
scatter(axB, mu, d, 14, COL, 'filled', 'MarkerFaceAlpha',0.28);
yline(axB, bias,   'r-',  'LineWidth',2);
yline(axB, loa_lo, 'r--', 'LineWidth',1.6);
yline(axB, loa_hi, 'r--', 'LineWidth',1.6);
yline(axB, 0,      'k:',  'LineWidth',1.4);
xlabel(axB, 'Mean of log_{10} rates', 'FontSize',FONT);
ylabel(axB, sprintf('\\Deltalog_{10} (%s - dead-time)', cmpLabel), 'FontSize',FONT);
title(axB, 'B. Bland-Altman, log_{10} scale', 'FontSize',FONT, 'FontWeight','bold');
text(axB, 0.03, 0.97, sprintf(['bias=%.3f (\\times%.2f)\n95%% LoA %.3f to %.3f' ...
    ' (\\times%.2f to \\times%.2f)'], bias, 10^bias, loa_lo, loa_hi, 10^loa_lo, 10^loa_hi), ...
    'Units','normalized','VerticalAlignment','top','FontSize',FONT-2);
set(axB,'FontSize',FONT);

% ---- C: the mechanism ----
axC = nexttile(3); hold(axC,'on'); box(axC,'off'); grid(axC,'on');
if isFile
    lim3 = [0 max([Enrich.f; Enrich.q])*1.05 + eps];
    plot(axC, lim3, lim3, 'k--', 'LineWidth',1.4);
    scatter(axC, Enrich.f, Enrich.q, 14, [0.49 0.18 0.56], 'filled', ...
        'MarkerFaceAlpha',0.28);
    xlim(axC, lim3); ylim(axC, lim3); axis(axC,'square');
    xlabel(axC, 'f = dead time / EDF span', 'FontSize',FONT);
    ylabel(axC, 'q = detections in dead window / all', 'FontSize',FONT);
    title(axC, 'C. Enriched or quiet?', 'FontSize',FONT, 'FontWeight','bold');
    text(axC, 0.03, 0.97, sprintf(['enriched q>f: %d (%.1f%%)\ndepleted q<f: %d\n' ...
        'zero detections in dead window: %d\nmedian f=%.3f, median q=%.3f'], ...
        Enrich.nEnriched, 100*Enrich.fracEnriched, Enrich.nDepleted, ...
        Enrich.nNoDetInDead, Enrich.median_f, Enrich.median_q), ...
        'Units','normalized','VerticalAlignment','top','FontSize',FONT-3);
else
    tc = J.T_cmp_sec(pos)/60;  tr = J.T_rec_sec(pos)/60;
    lim3 = [0 max([tc;tr])*1.05];
    plot(axC, lim3, lim3, 'k--', 'LineWidth',1.4);
    scatter(axC, tr, tc, 14, [0.85 0.33 0.10], 'filled', 'MarkerFaceAlpha',0.28);
    xlim(axC, lim3); ylim(axC, lim3); axis(axC,'square');
    xlabel(axC, 'EDF span - dead time (min)', 'FontSize',FONT);
    ylabel(axC, 'Natus duration\_hms (min)', 'FontSize',FONT);
    title(axC, 'C. Do the denominators agree?', 'FontSize',FONT, 'FontWeight','bold');
    text(axC, 0.03, 0.97, sprintf('median diff = %.1f min\nIQR %.1f to %.1f', ...
        median(tc-tr), prctile(tc-tr,25), prctile(tc-tr,75)), ...
        'Units','normalized','VerticalAlignment','top','FontSize',FONT-2);
end
set(axC,'FontSize',FONT);

% ---- D: numerator vs denominator contribution ----
axD = nexttile(4); hold(axD,'on'); box(axD,'off'); grid(axD,'on');
scatter(axD, J.DeadFrac(pos), num_part, 14, [0.49 0.18 0.56], 'filled', ...
    'MarkerFaceAlpha',0.28, 'DisplayName','numerator: log_{10}(N_{total}/N_{rec})');
scatter(axD, J.DeadFrac(pos), den_part, 14, [0.10 0.55 0.55], 'filled', ...
    'MarkerFaceAlpha',0.28, 'DisplayName','denominator: log_{10}(T_{cmp}/T_{rec})');
yline(axD, 0, 'k:', 'LineWidth',1.4);
xlabel(axD, 'f = dead time / EDF span', 'FontSize',FONT);
ylabel(axD, 'Contribution magnitude', 'FontSize',FONT);
title(axD, 'D. Which half drives the difference?', 'FontSize',FONT, 'FontWeight','bold');
legend(axD, 'Location','northwest', 'FontSize',FONT-4);
text(axD, 0.97, 0.03, sprintf(['median numerator = %.3f\nmedian denominator = %.3f\n' ...
    'net median \\Deltalog_{10} = %.3f'], median(num_part), median(den_part), median(d)), ...
    'Units','normalized','HorizontalAlignment','right','VerticalAlignment','bottom', ...
    'FontSize',FONT-2);
set(axD,'FontSize',FONT);

% ---- E: dead window vs the Natus-implied clip ----
axE = nexttile(5); hold(axE,'on'); box(axE,'off'); grid(axE,'on');
if doValidate
    csMin = cs/60; dsMin = ds/60;
    limE = [min([csMin; dsMin; 0]) max([csMin; dsMin])*1.05];
    plot(axE, limE, limE, 'k--', 'LineWidth',1.4);
    scatter(axE, csMin, dsMin, 14, [0.85 0.33 0.10], 'filled', 'MarkerFaceAlpha',0.28);
    xlim(axE, limE); ylim(axE, limE); axis(axE,'square');
    xlabel(axE, 'EDF span - duration\_hms (min)', 'FontSize',FONT);
    ylabel(axE, 'Deadtime\_sec (min)', 'FontSize',FONT);
    title(axE, 'E. Two estimates of the same window', 'FontSize',FONT, 'FontWeight','bold');
    text(axE, 0.03, 0.97, sprintf(['\\rho_{s}=%.3f, %s\nmedian clip=%.1f, dead=%.1f min\n' ...
        'median residual=%.1f min\n>1 min overshoot: %.1f%%\nnegative clips: %d'], ...
        Validation.rho, p_label(Validation.p), Validation.median_clip_min, ...
        Validation.median_dead_min, Validation.median_resid_min, ...
        100*Validation.fracOvershoot, Validation.nNegativeClip), ...
        'Units','normalized','VerticalAlignment','top','FontSize',FONT-3);
else
    axis(axE,'off');
    text(axE, 0.5, 0.5, 'E. Validation not computed', 'HorizontalAlignment','center', ...
        'FontSize',FONT);
end
set(axE,'FontSize',FONT);

% ---- F: implied dead fractions ----
axF = nexttile(6); hold(axF,'on'); box(axF,'off'); grid(axF,'on');
if doValidate
    edgesF = 0:0.02:1;
    histogram(axF, Validation.deadFrac, 'BinEdges',edgesF, 'FaceColor',[0.85 0.33 0.10], ...
        'FaceAlpha',0.5, 'EdgeColor','none', 'DisplayName','Deadtime\_sec / EDF');
    histogram(axF, Validation.clipFrac, 'BinEdges',edgesF, 'FaceColor',[0.22 0.45 0.70], ...
        'FaceAlpha',0.5, 'EdgeColor','none', 'DisplayName','(EDF - duration\_hms) / EDF');
    xline(axF, median(Validation.deadFrac), '--', 'Color',[0.85 0.33 0.10], ...
        'LineWidth',2, 'HandleVisibility','off');
    xline(axF, median(Validation.clipFrac), '--', 'Color',[0.22 0.45 0.70], ...
        'LineWidth',2, 'HandleVisibility','off');
    xlabel(axF, 'Implied dead fraction of the EDF span', 'FontSize',FONT);
    ylabel(axF, 'Number of EEGs', 'FontSize',FONT);
    title(axF, 'F. How much of the file is being removed?', ...
        'FontSize',FONT, 'FontWeight','bold');
    legend(axF, 'Location','northeast', 'FontSize',FONT-4);
    text(axF, 0.97, 0.55, sprintf('median dead=%.3f\nmedian clip=%.3f', ...
        median(Validation.deadFrac), median(Validation.clipFrac)), ...
        'Units','normalized','HorizontalAlignment','right','FontSize',FONT-3);
else
    histogram(axF, J.DeadFrac(pos), 'BinEdges',0:0.02:1, 'FaceColor',[0.85 0.33 0.10], ...
        'FaceAlpha',0.6, 'EdgeColor','none');
    xline(axF, median(J.DeadFrac(pos)), 'k--', 'LineWidth',2);
    xlabel(axF, 'f = dead time / EDF span', 'FontSize',FONT);
    ylabel(axF, 'Number of EEGs', 'FontSize',FONT);
    title(axF, 'F. Dead fraction', 'FontSize',FONT, 'FontWeight','bold');
end
set(axF,'FontSize',FONT);

outPng = fullfile(opts.outDir, opts.figName);
exportgraphics(fig, outPng, 'Resolution',300);

%% ===================== 11. CONSOLE SUMMARY =====================
fprintf('\n=== %s vs dead-time spike rates ===\n', cmpLabel);
fprintf('EEGs joined %d (of %d spike rows, %d comparator rows, %d dead-time rows; %d dead-time rows unusable)\n', ...
    FilterDiscord.nJoined, height(Base), nCmpRaw, nDeadRaw, nDeadDropped);
fprintf('Routine <=%.0f h under both %d; %s only %d; dead-time only %d\n', ...
    opts.maxRoutineHours, FilterDiscord.nRoutine_both, cmpLabel, ...
    FilterDiscord.nRoutine_cmp_only, FilterDiscord.nRoutine_dead_only);
fprintf('Analysed %d EEGs positive under both (zero under %s only %d, dead-time only %d, both %d)\n', ...
    ZeroCounts.nPositiveBoth, cmpLabel, ZeroCounts.nZero_cmp_only, ...
    ZeroCounts.nZero_dead_only, ZeroCounts.nZero_both);
fprintf('Spearman %.3f (%s); Pearson on log10 %.3f (%s); Lin CCC on log10 %.3f\n', ...
    rho_s, p_label(p_s), r_log, p_label(p_log), ccc_log);
fprintf('Bias %.3f log10 units: %s is %.2fx the dead-time rate on average\n', ...
    bias, cmpLabel, 10^bias);
fprintf('95%% limits of agreement %.3f to %.3f (ratio %.2fx to %.2fx)\n', ...
    loa_lo, loa_hi, 10^loa_lo, 10^loa_hi);
fprintf('Median contributions: numerator %.3f, denominator %.3f, net %.3f\n', ...
    median(num_part), median(den_part), median(d));
if isFile
    fprintf('Dead window: enriched (q>f) %d, depleted (q<f) %d, zero detections %d, of %d\n', ...
        Enrich.nEnriched, Enrich.nDepleted, Enrich.nNoDetInDead, numel(Enrich.f));
end
if doValidate
    fprintf('\n--- Dead-window validation against duration_hms (N=%d, %d missing) ---\n', ...
        Validation.n, Validation.nMissingNatus);
    fprintf('Spearman(clip, Deadtime_sec) = %.3f (%s)\n', Validation.rho, p_label(Validation.p));
    fprintf('Median clip %.1f min vs median Deadtime_sec %.1f min\n', ...
        Validation.median_clip_min, Validation.median_dead_min);
    fprintf('Median residual (dead - clip) %.1f min (IQR %.1f to %.1f)\n', ...
        Validation.median_resid_min, Validation.iqr_resid_min(1), Validation.iqr_resid_min(2));
    fprintf('Dead-time overshoots the clip by >1 min in %.1f%% of EEGs; %d negative clips\n', ...
        100*Validation.fracOvershoot, Validation.nNegativeClip);
    fprintf('Median implied dead fraction: %.3f (dead-time) vs %.3f (Natus clip)\n', ...
        median(Validation.deadFrac), median(Validation.clipFrac));
end
fprintf('Saved: %s\n', outPng);

%% ===================== 12. RETURN =====================
Cmp = struct( ...
    'Source',         opts.source, ...
    'nAnalysed',      numel(x), ...
    'Spearman',       rho_s,   'Spearman_p', p_s, ...
    'Pearson_log',    r_log,   'Pearson_log_p', p_log, ...
    'CCC_log',        ccc_log, ...
    'BA_bias_log10',  bias,    'BA_bias_ratio', 10^bias, ...
    'BA_loa_log10',   [loa_lo loa_hi], 'BA_loa_ratio', [10^loa_lo 10^loa_hi], ...
    'BA_sd_log10',    sd_d, ...
    'PropBias_rho',   rho_prop, 'PropBias_p', p_prop, ...
    'FigurePath',     string(outPng));

% Assigned after construction: struct() collapses when handed an empty struct.
Cmp.Table                    = J;
Cmp.NumeratorContribution    = num_part;
Cmp.DenominatorContribution  = den_part;
Cmp.Enrichment               = Enrich;
Cmp.Validation               = Validation;
Cmp.ZeroCounts               = ZeroCounts;
Cmp.FilterDiscord            = FilterDiscord;
Cmp.Figure                   = fig;
end


%% #####################################################################
%% ##  LOCAL UTILITIES
%% #####################################################################

function require_cols(T, cols, name)
missing = setdiff(string(cols), string(T.Properties.VariableNames));
assert(isempty(missing), '%s is missing required columns: %s', ...
    name, strjoin(missing, ", "));
end


function assert_unique_keys(T, name)
key = string(T.Patient) + "|" + string(T.Session);
assert(numel(unique(key)) == numel(key), ...
    '%s has duplicated (Patient, Session) keys.', name);
end


function c = lins_ccc(x, y)
%LINS_CCC  Concordance correlation coefficient: agreement about the identity
% line, not just correlation. Penalises the systematic offset that Pearson and
% Spearman are blind to.
x = x(:); y = y(:);
mx = mean(x); my = mean(y);
vx = var(x,1); vy = var(y,1);
sxy = mean((x-mx).*(y-my));
c = 2*sxy / (vx + vy + (mx-my)^2);
end


function s = p_label(p)
if isnan(p),  s = "p=NaN";               return; end
if p < 0.001, s = "p<0.001";             return; end
if p < 0.01,  s = sprintf("p=%.2g", p);  return; end
s = sprintf("p=%.2f", p);
end
