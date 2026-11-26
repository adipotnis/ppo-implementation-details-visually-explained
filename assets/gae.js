// Interactive GAE: one env, one rollout of M steps, stepped through PPO's backward GAE loop.
(function () {
  const M = 8;
  const rewards = [0, 0, 1, 0, 3, 0, 0, 1];
  const values  = [1.2, 1.4, 1.6, 1.0, 2.5, 0.8, 0.9, 1.1];
  const dones   = [0, 0, 0, 0, 0, 1, 0, 0];  // dones[t] = 1: obs t is the first obs of a new episode
  const nextValue = 1.0;                      // V(next_obs), from section 1
  const nextDone = 0;
  const epColor = t => (t < 5 ? '#1D9E75' : '#D85A30');

  const $ = id => document.getElementById(id);
  const gIn = $('g-in'), lIn = $('l-in');
  let k = 0;     // how many loop iterations have run (M + 1 = returns computed too)
  let sel = 0;   // column whose advantage breakdown is shown

  const fmt = x => { const s = (Math.abs(x) < 0.005 ? 0 : x).toFixed(2); return s === '-0.00' ? '0.00' : s; };
  const sub = s => `<sub>${s}</sub>`;

  function compute(g, l) {
    const nnt = [], nv = [], delta = [], adv = [], lastIn = [];
    let last = 0;
    for (let t = M - 1; t >= 0; t--) {
      nnt[t] = t === M - 1 ? 1 - nextDone : 1 - dones[t + 1];
      nv[t] = t === M - 1 ? nextValue : values[t + 1];
      delta[t] = rewards[t] + g * nv[t] * nnt[t] - values[t];
      lastIn[t] = last;
      adv[t] = last = delta[t] + g * l * nnt[t] * last;
    }
    const ret = adv.map((a, t) => a + values[t]);
    return { nnt, nv, delta, adv, ret, lastIn };
  }

  // Weight of delta[t + j] inside adv[t]: (γλ)^j, zeroed once an episode boundary is crossed.
  function weights(t, g, l, nnt) {
    const w = new Array(M).fill(0);
    let c = 1;
    for (let j = t; j < M; j++) {
      w[j] = c;
      c *= g * l * nnt[j];
      if (c === 0) break;
    }
    return w;
  }

  const curT = () => (k >= 1 && k <= M ? M - k : null);

  function tint(x, max) {
    if (max === 0) return 'transparent';
    const a = Math.min(Math.abs(x) / max, 1) * 0.45;
    return x >= 0 ? `rgba(29,158,117,${a})` : `rgba(216,90,48,${a})`;
  }

  function render() {
    const g = +gIn.value, l = +lIn.value;
    $('g-out').textContent = g.toFixed(2);
    $('l-out').textContent = l.toFixed(2);
    const R = compute(g, l);
    const t0 = curT();
    const w = weights(sel, g, l, R.nnt);
    const isDone = t => (k > M) || (k >= 1 && t >= M - k);  // advantage[t] computed yet?
    const dMax = Math.max(...R.delta.map(Math.abs));
    const aMax = Math.max(...R.adv.map(Math.abs));

    const colCls = t => {
      let c = '';
      if (t === t0) c += ' cur';
      if (t0 !== null && t === t0 + 1) c += ' src';
      if (t === sel) c += ' sel';
      return c;
    };

    let h = '';
    // header
    h += `<div class="gl"></div>`;
    for (let t = 0; t < M; t++) {
      h += `<button type="button" class="gh${dones[t] ? ' done' : ''}${colCls(t)}" data-t="${t}" style="--ep:${epColor(t)}" aria-label="Show breakdown of A at t = ${t}">t = ${t}</button>`;
    }
    h += `<div class="gh nextcol${t0 === M - 1 ? ' src' : ''}" style="--ep:${epColor(M)}">next</div>`;

    const row = (label, cell, nextCell) => {
      h += `<div class="gl">${label}</div>`;
      for (let t = 0; t < M; t++) h += cell(t);
      h += nextCell;
    };
    const c = (t, inner, style = '', extra = '') => `<div class="gc${colCls(t)}${extra}" data-t="${t}" style="${style}">${inner}</div>`;
    const nx = (inner, extra = '') => `<div class="gc nextcol${t0 === M - 1 && inner ? ' src' : ''}${extra}">${inner}</div>`;

    row('rewards[t]', t => c(t, rewards[t]), nx(''));
    row('dones[t]', t => c(t, dones[t], '', dones[t] ? ' flag' : ''), nx(`${nextDone}<small>next_done</small>`));
    row('values[t]', t => c(t, fmt(values[t])), nx(`${fmt(nextValue)}<small>next_value</small>`, ' nv'));
    row('delta δ<sub>t</sub>', t => c(t, fmt(R.delta[t]), `background:${tint(R.delta[t], dMax)}`), nx(''));
    row('advantages A<sub>t</sub>', t => isDone(t)
      ? c(t, fmt(R.adv[t]), `background:${tint(R.adv[t], aMax)}`)
      : c(t, '·', '', ' empty'), nx(''));
    row('returns', t => k > M ? c(t, fmt(R.ret[t])) : c(t, '·', '', ' empty'), nx(''));
    row(`weight in A${sub(sel)}`, t => c(t,
      w[t] ? `<span class="wbar" style="height:${Math.max(2, w[t] * 22)}px"></span>${fmt(w[t])}` : (t < sel ? '' : '0'),
      '', ' wcell'), nx(''));

    $('gae-grid').innerHTML = h;
    $('gae-grid').querySelectorAll('[data-t]').forEach(el => el.addEventListener('click', () => {
      sel = +el.dataset.t;
      render();
    }));

    // Loop readout + code highlight
    let lines = [];
    let s;
    if (k === 0) {
      s = `<div>Press <span class="key">Step</span> to run the loop. It starts at the <span class="key">last</span> step (t = ${M - 1}) and walks backward, because each A${sub('t')} needs A${sub('t+1')}.</div>`;
      lines = [1, 2];
    } else if (t0 !== null) {
      const t = t0, last = t === M - 1;
      lines = last ? [3, 4, 5, 6, 10, 11] : [3, 7, 8, 9, 10, 11];
      s = `<div><span class="key">t = ${t}</span>${last ? ' (last step of the segment)' : ''}: ` +
        (last
          ? `nextnonterminal = 1 − next_done = ${R.nnt[t]}, &nbsp;nextvalues = next_value = ${fmt(R.nv[t])}`
          : `nextnonterminal = 1 − dones[${t + 1}] = ${R.nnt[t]}, &nbsp;nextvalues = values[${t + 1}] = ${fmt(R.nv[t])}`) + `</div>`;
      s += `<div>delta = ${rewards[t]} + ${g.toFixed(2)}·${fmt(R.nv[t])}·${R.nnt[t]} − ${fmt(values[t])} = <span class="key">${fmt(R.delta[t])}</span></div>`;
      s += `<div>A${sub(t)} = ${fmt(R.delta[t])} + ${(g * l).toFixed(2)}·${R.nnt[t]}·${fmt(R.lastIn[t])} = <span class="key">${fmt(R.adv[t])}</span> &nbsp;<span class="env">(γλ = ${(g * l).toFixed(2)}, lastgaelam = ${fmt(R.lastIn[t])})</span></div>`;
      if (!R.nnt[t]) s += `<div class="masked">dones[${t + 1}] = 1: obs ${t + 1} starts a new episode, so both the bootstrap and the chain from A${sub(t + 1)} are cut.</div>`;
    } else {
      lines = [12];
      s = `<div><span class="key">returns = advantages + values</span>. These are the value-function targets (the TD(λ) return). All ${M} advantages came from one backward pass.</div>`;
    }
    $('gae-out').innerHTML = s;
    document.querySelectorAll('#gae-code .ln').forEach(el => el.classList.toggle('on', lines.includes(+el.dataset.l)));
    $('gae-ph').textContent = k === 0 ? 'Not started' : k <= M ? `Iteration ${k} of ${M}` : 'Done';
    $('gae-step').disabled = k > M;

    renderInspect(R, w);
  }

  function renderInspect(R, w) {
    const g = +gIn.value, l = +lIn.value;
    const terms = [];
    let end = sel;
    for (let j = sel; j < M && w[j]; j++) { terms.push(`${fmt(w[j])}·δ${sub(j)}`); end = j; }
    let why;
    if (end < M - 1 && R.nnt[end] === 0) why = `stops after δ${sub(end)}: obs ${end + 1} starts a new episode`;
    else if (end < M - 1) why = `is just δ${sub(sel)}: with λ = 0, GAE is the one-step TD error`;
    else why = `runs to the end of the segment; the last δ bootstraps from next_value`;
    $('gae-w').innerHTML =
      `<div><span class="key">What A${sub(sel)} is made of</span> (click any column): ` +
      `A${sub(sel)} = ${terms.join(' + ')} = <span class="key">${fmt(R.adv[sel])}</span></div>` +
      `<div class="env">The weights are (γλ)<sup>j</sup> = ${(g * l).toFixed(2)}<sup>j</sup>. The sum ${why}.</div>`;
  }

  $('gae-step').addEventListener('click', () => { if (k <= M) { k++; if (k <= M) sel = M - k; render(); } });
  $('gae-all').addEventListener('click', () => { k = M + 1; render(); });
  $('gae-reset').addEventListener('click', () => { k = 0; sel = 0; render(); });
  gIn.addEventListener('input', render);
  lIn.addEventListener('input', render);

  render();
})();
