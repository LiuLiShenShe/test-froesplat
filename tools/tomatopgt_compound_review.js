(() => {
  const panel = document.querySelector('#review');
  const rows = JSON.parse(document.querySelector('#review-data').textContent);
  const $ = (selector) => panel.querySelector(selector);
  const statuses = {strong_suggestion: '优先核验', review_suggestion: '待核验', ambiguous_assignment: '有歧义',
    no_candidate: '没有候选', registration_uncertain: '配准不可靠', competing_or_merged: '冲突或合并',
    no_reliable_match: '没有可靠对应'};
  const key = (r) => [r.scene, r.reference_scope, r.official_point_instance, r.candidate_id].join('/');
  const storageKey = `tpg-review:${panel.dataset.version}`;
  let saved = {};
  try { saved = JSON.parse(localStorage.getItem(storageKey) || '{}'); } catch (_) { /* Export remains available. */ }
  let selected = [];
  let cursor = 0;
  [...new Set(rows.map((r) => r.scene))].forEach((scene) => $('#scene').add(new Option(scene, scene)));

  function note(message) { $('#save-status').textContent = message; }
  function persist() {
    try { localStorage.setItem(storageKey, JSON.stringify(saved)); }
    catch (_) { note('浏览器无法保存到本地，请在关闭页面前导出 CSV。'); }
  }
  function display() {
    const row = selected[cursor];
    $('#item').hidden = !row;
    $('#position').textContent = `${selected.length ? cursor + 1 : 0} / ${selected.length} · 已核验 ${Object.keys(saved).length}`;
    $('#prev').disabled = cursor <= 0;
    $('#next').disabled = cursor >= selected.length - 1;
    updateStats();
    if (!row) return;
    const current = saved[key(row)];
    $('#organ').textContent = `${row.scene} · ${row.reference_scope === 'cotyledon' ? '子叶' : '复叶'} ${row.official_id || `点标签 ${row.official_point_instance}`} → ${row.candidate_id || '未匹配'}`;
    $('#detail').textContent = `${statuses[row.status]} · 重叠 ${row.overlap_score || '—'} · 官方覆盖 ${row.reference_coverage || '—'} · 预测覆盖 ${row.prediction_coverage || '—'}`;
    $('#alternates').textContent = row.alternate_candidates ? `备选：${row.alternate_candidates}` : '';
    const base = `${row.scene}/`;
    for (const [id, property] of [['overlay', 'overlay'], ['rgb', 'reference_rgb'], ['mesh', 'model_viewer'], ['ply', 'model_ply']]) {
      const element = $(`#${id}`);
      const source = (element.tagName === 'IMG' && row[`${property}_image`]) || (row[property] && base + row[property]);
      const attr = element.tagName === 'IMG' ? 'src' : 'href';
      element.hidden = !source;
      if (source) element.setAttribute(attr, source);
      else element.removeAttribute(attr);
    }
    $('#scan-page').href = `${base}index.html`;
    const scaleReview = row.metric_pair_qc === 'scale_review';
    $('#dimensions').textContent = `模型 / 参考（cm）：直线叶长 ${row.model_chord_cm || '—'} / ${row.official_chord_cm || '—'}，横向叶宽 ${row.model_transverse_width_cm || '—'} / ${row.official_transverse_width_cm || '—'}` +
      (scaleReview ? `。配准尺度比 ${row.alignment_only_scale}，先复核标定，暂不计入误差。` : '');
    $('#accept').disabled = !row.candidate_id;
    $('#verify-chord').disabled = !row.model_chord_cm || scaleReview;
    $('#verify-width').disabled = !row.model_transverse_width_cm || scaleReview;
    $('#verify-chord').checked = current?.reviewed_chord_pair || false;
    $('#verify-width').checked = current?.reviewed_width_pair || false;
    $('#clear').disabled = !current;
    note(current ? ({confirm:'已确认对应', reject:'已否决对应', unresolved:'已记录无法判定'}[current.decision]) : '未核验');
  }
  function filter() {
    const scan = $('#scene').value;
    const status = $('#status').value;
    const scope = $('#scope').value;
    selected = rows.filter((r) => (!scan || scan === r.scene) &&
      (!scope || (String(r.metric_scan).toLowerCase() === 'true' && r.reference_scope === 'compound_leaf')) && (!status || status === r.status ||
      (status === 'suggestions' && ['strong_suggestion', 'review_suggestion'].includes(r.status))));
    cursor = 0;
    display();
  }
  function updateStats() {
    const traits = [['叶长', 'reviewed_chord_pair', 'model_chord_cm', 'official_chord_cm'],
      ['叶宽', 'reviewed_width_pair', 'model_transverse_width_cm', 'official_transverse_width_cm']];
    $('#error-summary').textContent = traits.map(([label, flag, pred, ref]) => {
      const errors = rows.filter((r) => r.metric_pair_qc === 'eligible_for_review' && saved[key(r)]?.decision === 'confirm' && saved[key(r)][flag] && r[pred] && r[ref])
        .map((r) => Number(r[pred]) - Number(r[ref]));
      const n = errors.length;
      return n ? `${label} n=${n} · MAE ${(errors.reduce((a, x) => a + Math.abs(x), 0) / n).toFixed(3)} cm · RMSE ${Math.sqrt(errors.reduce((a, x) => a + x*x, 0) / n).toFixed(3)} cm` : `${label} n=0`;
    }).join(' | ');
  }
  function decide(decision) {
    const row = selected[cursor];
    if (!row || (decision === 'confirm' && !row.candidate_id)) return;
    saved[key(row)] = {decision, reviewed_at: new Date().toISOString(),
      reviewed_chord_pair: decision === 'confirm' && !$('#verify-chord').disabled && $('#verify-chord').checked,
      reviewed_width_pair: decision === 'confirm' && !$('#verify-width').disabled && $('#verify-width').checked};
    if (cursor < selected.length - 1) cursor++;
    display();
    persist();
  }
  function exportDecisions() {
    const fields = ['scene','reference_scope','official_point_instance','official_id','candidate_id','status','decision',
      'human_confirmed','reviewed_at','reviewed_chord_pair','reviewed_width_pair','official_chord_cm','model_chord_cm',
      'official_transverse_width_cm','model_transverse_width_cm'];
    const quote = (x) => `"${String(x ?? '').replaceAll('"', '""')}"`;
    const records = rows.flatMap((r) => {
      const s = saved[key(r)];
      if (!s) return [];
      const result = {...r, ...s, human_confirmed:s.decision === 'confirm'};
      return [fields.map((f) => quote(result[f])).join(',')];
    });
    if (!records.length) { note('还没有可导出的核验记录。'); return; }
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob(['\ufeff', [fields.join(','), ...records].join('\r\n')], {type:'text/csv'}));
    a.download = 'reviewed_tomatopgt_compound_matches.csv';
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 5000);
  }
  $('#scene').addEventListener('change', filter);
  $('#scope').addEventListener('change', filter);
  $('#status').addEventListener('change', filter);
  $('#prev').onclick = () => { if (cursor > 0) cursor--; display(); };
  $('#next').onclick = () => { if (cursor < selected.length - 1) cursor++; display(); };
  $('#accept').onclick = () => decide('confirm');
  $('#reject').onclick = () => decide('reject');
  $('#defer').onclick = () => decide('unresolved');
  $('#clear').onclick = () => { delete saved[key(selected[cursor])]; display(); persist(); };
  $('#export').onclick = exportDecisions;
  filter();
  panel.classList.remove('preview-only');
  $('#preview-note').hidden = true;
})();
