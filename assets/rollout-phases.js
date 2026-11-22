// Interactive grid: 3 parallel envs x 4 rollout phases of M steps each.
// Shows how next_obs and next_done link one rollout phase to the next.
(function () {
  const E = 3;            // number of parallel environments
  const M = 6;            // steps per rollout phase
  const P = 4;            // number of rollout phases
  // Step index at which each episode starts, per env.
  const starts = [[0, 9, 18], [0, 4, 12], [0]];
  const pal = ['#1D9E75', '#D85A30', '#7F77DD', '#BA7517'];

  const ep = (e, t) => starts[e].filter(s => s <= t).length - 1;
  const done = (e, t) => t > 0 && starts[e].includes(t);

  const grid = document.getElementById('grid');
  const out = document.getElementById('out');
  const ph = document.getElementById('ph');
  let j = 0;

  function render() {
    let h = '<div class="row"><span class="lab"></span>';
    for (let p = 0; p <= P; p++) {
      const n = p < P ? M : 1;
      h += `<div class="hd blk" data-p="${p}" style="width:${n * 21 - 2}px">${p < P ? 'phase ' + (p + 1) : 'end'}</div>`;
    }
    h += '</div>';
    for (let e = 0; e < E; e++) {
      h += `<div class="row"><span class="lab">env ${e}</span>`;
      for (let p = 0; p <= P; p++) {
        const n = p < P ? M : 1;
        h += `<div class="blk" data-p="${p}">`;
        for (let k = 0; k < n; k++) {
          const t = p * M + k;
          const d = done(e, t);
          h += `<div class="cell${d ? ' done' : ''}" data-t="${t}" title="env ${e}, step ${t}, done=${d ? 1 : 0}" style="background:${pal[ep(e, t) % pal.length]}"></div>`;
        }
        h += '</div>';
      }
      h += '</div>';
    }
    grid.innerHTML = h;
    update();
  }

  function update() {
    const a = j * M, b = a + M - 1, nx = (j + 1) * M;

    grid.querySelectorAll('.blk').forEach(el => {
      const p = +el.dataset.p;
      el.style.opacity = (p === j || p === j + 1) ? 1 : 0.25;
    });
    grid.querySelectorAll('.cell[data-t]').forEach(c => {
      const t = +c.dataset.t;
      c.classList.toggle('nx', t === nx);
      c.classList.toggle('init', j > 0 && t === a);
      c.style.opacity = (t > nx && +c.parentElement.dataset.p === j + 1) ? 0.25 : 1;
    });

    ph.textContent = `Rollout phase ${j + 1} of ${P}`;

    let s = `<div><span class="key">data</span> = steps ${a}–${b} (M = ${M} per env). ` +
      (j > 0 ? `<code>data[0]</code> is the <code>next_obs</code> left over from phase ${j}.`
             : `<code>data[0]</code> is the reset observation.`) + `</div>`;
    s += `<div><span class="key">next_obs</span> = step ${nx} &nbsp;·&nbsp; <span class="key">next_done</span> = [${[...Array(E).keys()].map(e => done(e, nx) ? 1 : 0).join(', ')}]</div>`;
    for (let e = 0; e < E; e++) {
      s += `<div class="env">env ${e}: ` + (done(e, nx)
        ? `<span class="masked">next_done = 1</span>, so <code>next_obs</code> starts a new episode and the bootstrap is masked: (1 − 1)·V(next_obs) = 0`
        : `next_done = 0, so the episode is still going and the return bootstraps from V(next_obs)`) + `</div>`;
    }
    s += j < P - 1
      ? `<div style="margin-top:6px">In phase ${j + 2}, the step-${nx} observation becomes <code>data[0]</code>, and stepping simply continues.</div>`
      : `<div style="margin-top:6px">env 2 has been in one episode the whole time. PPO still trained on it using fixed M-step segments.</div>`;
    out.innerHTML = s;
  }

  const go = d => { j = (j + d + P) % P; update(); };
  document.getElementById('prev').addEventListener('click', () => go(-1));
  document.getElementById('next').addEventListener('click', () => go(1));
  document.getElementById('rollout-figure').addEventListener('keydown', ev => {
    if (ev.key === 'ArrowLeft') { go(-1); ev.preventDefault(); }
    if (ev.key === 'ArrowRight') { go(1); ev.preventDefault(); }
  });

  render();
})();
