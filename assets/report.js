/* Shared Mol* viewer helpers for Boltz-2 nanobody reports.
   Requires molstar.js (window.molstar) to be loaded first. */
(function () {
  const VIEWERS = {};
  const OPTS = {
    layoutIsExpanded: false,
    layoutShowControls: true,
    layoutShowRemoteState: false,
    layoutShowSequence: true,
    layoutShowLog: false,
    layoutShowLeftPanel: true,
    viewportShowExpand: false,
    viewportShowSelectionMode: false,
    viewportShowAnimation: false,
    viewportBackgroundColor: '#ffffff'
  };

  function diag(msg, isErr) {
    const el = document.getElementById('diag');
    if (el) {
      el.textContent += (isErr ? ' [ERR] ' : ' ') + msg + ' |';
      if (isErr) el.style.color = '#b3261e';
    }
    console.log('[boltz-report] ' + msg);
  }

  function host(name) {
    const id = 'viewer-' + name;
    let box = document.getElementById(id);
    if (!box) {
      const parent = document.getElementById('molstar-viewer');
      if (!parent) return null;
      parent.innerHTML = '';
      box = document.createElement('div');
      box.id = id;
      box.style.width = '100%';
      box.style.height = '100%';
      parent.appendChild(box);
    }
    return box;
  }

  async function ensureViewer(key) {
    if (VIEWERS[key]) return VIEWERS[key];
    if (!window.molstar || !molstar.Viewer) {
      throw new Error('Mol* library is not available');
    }
    const box = host(key);
    if (!box) throw new Error('viewer host not found');
    const v = await molstar.Viewer.create(box, OPTS);
    VIEWERS[key] = v;
    return v;
  }

  async function buildWithTheme(v, cif, themeName) {
    const plugin = v.plugin;
    await plugin.clear();
    const data = await plugin.builders.data.rawData({ data: cif, label: 'boltz-model' });
    const traj = await plugin.builders.structure.parseTrajectory(data, 'mmcif');
    try {
      await plugin.builders.structure.hierarchy.applyPreset(traj, 'default', {
        representationPresetParams: { theme: { globalName: themeName } }
      });
      diag('loaded with theme ' + themeName);
      return true;
    } catch (e) {
      diag('theme preset failed: ' + e.message, true);
      await v.loadStructureFromData(cif, 'mmcif', { dataLabel: 'boltz-model' });
      return false;
    }
  }

  // M-18: 로딩을 직렬화해 빠른 탭 전환 시 뷰어가 중복 생성/겹쳐 로드되지 않게 한다
  let loadChain = Promise.resolve();
  function load(key, cif, themeName) {
    const task = loadChain.then(async () => {
      const v = await ensureViewer(key);
      await buildWithTheme(v, cif, themeName || 'plddt-confidence');
      try { v.plugin.managers.camera.reset(); } catch (e) { /* ignore */ }
      VIEWERS[key].__loadedFor = key;
      return v;
    });
    // M-11(정밀): 실패를 삼키고 resolve 하면 호출부가 성공으로 오인해 같은 탭
    // 재시도가 막힌다. 직렬화 체인은 실패와 무관하게 진행시키되, 호출자에게는
    // 거부(reject)된 Promise 를 돌려준다.
    loadChain = task.catch(() => {});
    return task.catch((e) => {
      diag('load failed: ' + e.message, true);
      throw e;
    });
  }

  async function setTheme(key, themeName) {
    const v = VIEWERS[key];
    if (!v) {
      throw new Error('viewer not ready');
    }
    const state = v.plugin.state;
    const data = state.data;
    let n = 0;
    const cells = data.cells && data.cells.values ? Array.from(data.cells.values()) : [];
    for (const cell of cells) {
      const t = cell.transform;
      if (!t || !t.params || t.params.colorTheme === undefined) continue;
      const params = Object.assign({}, t.params, { colorTheme: { name: themeName, params: {} } });
      await state.updateTransform(data, t.ref, params);
      n++;
    }
    if (!n) {
      throw new Error('no representations found for theme update');
    }
    diag('setTheme ' + themeName + ' -> ' + n + ' representations');
    return n;
  }

  async function resetCamera(key) {
    const v = VIEWERS[key];
    if (v) { try { v.plugin.managers.camera.reset(); } catch (e) { /* ignore */ } }
  }

  window.RB = { diag, load, setTheme, ensureViewer, resetCamera, VIEWERS };
})();
