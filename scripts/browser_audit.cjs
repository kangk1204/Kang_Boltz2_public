#!/usr/bin/env node
'use strict';

const fs = require('fs');
const fsp = fs.promises;
const os = require('os');
const path = require('path');
const http = require('http');
const net = require('net');
const crypto = require('crypto');
const { spawn } = require('child_process');
const WebSocket = require('/usr/share/nodejs/ws');

const CHROME = '/usr/bin/google-chrome';

function parseArgs(argv) {
  const out = {};
  for (let i = 2; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--root') out.root = argv[++i];
    else if (arg === '--out') out.out = argv[++i];
    else if (arg === '--help' || arg === '-h') out.help = true;
    else throw new Error(`unknown argument: ${arg}`);
  }
  return out;
}

function usage() {
  return [
    'Usage: node scripts/browser_audit.cjs --root <reports-root> --out <evidence-dir>',
    '',
    'Expected report paths under --root:',
    '  single/index.html',
    '  batch/index.html',
    '  target/index.html  (optional target-only batch report)',
  ].join('\n');
}

function stamp() {
  return new Date().toISOString();
}

function sha256(buf) {
  return crypto.createHash('sha256').update(buf).digest('hex');
}

async function fileHash(file) {
  return sha256(await fsp.readFile(file));
}

function contentType(file) {
  const ext = path.extname(file).toLowerCase();
  if (ext === '.html') return 'text/html; charset=utf-8';
  if (ext === '.js') return 'text/javascript; charset=utf-8';
  if (ext === '.css') return 'text/css; charset=utf-8';
  if (ext === '.png') return 'image/png';
  if (ext === '.jpg' || ext === '.jpeg') return 'image/jpeg';
  if (ext === '.svg') return 'image/svg+xml';
  if (ext === '.json') return 'application/json; charset=utf-8';
  if (ext === '.cif' || ext === '.bcif') return 'text/plain; charset=utf-8';
  return 'application/octet-stream';
}

function within(child, root) {
  const rel = path.relative(root, child);
  return rel === '' || (!!rel && !rel.startsWith('..') && !path.isAbsolute(rel));
}

async function listenServer(root) {
  const server = http.createServer(async (req, res) => {
    try {
      const url = new URL(req.url, 'http://127.0.0.1/');
      let decoded = decodeURIComponent(url.pathname);
      if (decoded.endsWith('/')) decoded += 'index.html';
      const file = path.resolve(root, `.${decoded}`);
      if (!within(file, root)) {
        res.writeHead(403);
        res.end('forbidden');
        return;
      }
      const st = await fsp.stat(file);
      if (!st.isFile()) {
        res.writeHead(404);
        res.end('not found');
        return;
      }
      res.writeHead(200, {
        'content-type': contentType(file),
        'cache-control': 'no-store',
      });
      fs.createReadStream(file).pipe(res);
    } catch {
      res.writeHead(404);
      res.end('not found');
    }
  });
  await new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', resolve);
  });
  const { port } = server.address();
  return { server, origin: `http://127.0.0.1:${port}` };
}

function freePort() {
  return new Promise((resolve, reject) => {
    const srv = net.createServer();
    srv.once('error', reject);
    srv.listen(0, '127.0.0.1', () => {
      const { port } = srv.address();
      srv.close(() => resolve(port));
    });
  });
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function getJson(url, timeoutMs = 5000) {
  const deadline = Date.now() + timeoutMs;
  let lastErr;
  while (Date.now() < deadline) {
    try {
      return await new Promise((resolve, reject) => {
        const req = http.get(url, res => {
          let body = '';
          res.setEncoding('utf8');
          res.on('data', chunk => { body += chunk; });
          res.on('end', () => {
            if (res.statusCode < 200 || res.statusCode >= 300) {
              reject(new Error(`HTTP ${res.statusCode}: ${body.slice(0, 120)}`));
              return;
            }
            try { resolve(JSON.parse(body)); } catch (err) { reject(err); }
          });
        });
        req.once('error', reject);
        req.setTimeout(1000, () => req.destroy(new Error('timeout')));
      });
    } catch (err) {
      lastErr = err;
      await sleep(100);
    }
  }
  throw lastErr || new Error(`timed out reading ${url}`);
}

async function waitForDevTools(port, proc, timeoutMs = 10000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (proc.exitCode !== null) throw new Error(`Chrome exited early with code ${proc.exitCode}`);
    try {
      return await getJson(`http://127.0.0.1:${port}/json/version`, 1000);
    } catch {
      await sleep(100);
    }
  }
  throw new Error('timed out waiting for Chrome DevTools');
}

async function launchChrome(userDataDir, extraArgs = []) {
  const port = await freePort();
  const args = [
    '--headless=new',
    '--disable-dev-shm-usage',
    '--enable-unsafe-swiftshader',
    '--ignore-gpu-blocklist',
    '--hide-scrollbars',
    '--window-size=1440,1000',
    `--remote-debugging-port=${port}`,
    `--user-data-dir=${userDataDir}`,
    'about:blank',
    ...extraArgs,
  ];
  const proc = spawn(CHROME, args, { stdio: ['ignore', 'ignore', 'pipe'] });
  let stderr = '';
  proc.stderr.on('data', chunk => {
    stderr += chunk.toString();
    if (stderr.length > 20000) stderr = stderr.slice(-20000);
  });
  try {
    const version = await waitForDevTools(port, proc);
    return { proc, port, version, args, stderr: () => stderr };
  } catch (err) {
    proc.kill('SIGTERM');
    await sleep(500);
    if (proc.exitCode === null) proc.kill('SIGKILL');
    err.stderr = stderr;
    throw err;
  }
}

async function launchChromeWithFallback(userDataDir) {
  try {
    return await launchChrome(userDataDir);
  } catch (err) {
    const text = String(err.stderr || err.message || '');
    if (!/sandbox|zygote|No usable sandbox/i.test(text)) throw err;
    await rmrf(userDataDir);
    await fsp.mkdir(userDataDir, { recursive: true });
    const retry = await launchChrome(userDataDir, ['--no-sandbox']);
    retry.usedNoSandbox = true;
    retry.firstLaunchError = text.slice(-2000);
    return retry;
  }
}

class CDP {
  constructor(wsUrl) {
    this.wsUrl = wsUrl;
    this.ws = null;
    this.nextId = 1;
    this.pending = new Map();
    this.listeners = new Map();
  }

  async connect() {
    this.ws = new WebSocket(this.wsUrl);
    await new Promise((resolve, reject) => {
      this.ws.once('open', resolve);
      this.ws.once('error', reject);
    });
    this.ws.on('message', data => this._message(data));
  }

  _message(data) {
    const msg = JSON.parse(data.toString());
    if (msg.id && this.pending.has(msg.id)) {
      const { resolve, reject } = this.pending.get(msg.id);
      this.pending.delete(msg.id);
      if (msg.error) reject(new Error(`${msg.error.message}${msg.error.data ? `: ${msg.error.data}` : ''}`));
      else resolve(msg.result || {});
      return;
    }
    if (msg.method && this.listeners.has(msg.method)) {
      for (const fn of this.listeners.get(msg.method)) fn(msg.params || {});
    }
  }

  on(method, fn) {
    if (!this.listeners.has(method)) this.listeners.set(method, []);
    this.listeners.get(method).push(fn);
  }

  send(method, params = {}) {
    const id = this.nextId++;
    this.ws.send(JSON.stringify({ id, method, params }));
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      setTimeout(() => {
        if (!this.pending.has(id)) return;
        this.pending.delete(id);
        reject(new Error(`CDP timeout: ${method}`));
      }, 30000);
    });
  }

  close() {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) this.ws.close();
  }
}

async function openPage(browserPort, browserWs, url) {
  const browser = new CDP(browserWs);
  await browser.connect();
  const created = await browser.send('Target.createTarget', { url: 'about:blank' });
  browser.close();
  const targets = await getJson(`http://127.0.0.1:${browserPort}/json/list`);
  const target = targets.find(t => t.id === created.targetId);
  if (!target || !target.webSocketDebuggerUrl) throw new Error('could not find page websocket');
  const page = new CDP(target.webSocketDebuggerUrl);
  await page.connect();
  const errors = [];
  page.on('Runtime.exceptionThrown', e => {
    errors.push({
      type: 'exception',
      text: e.exceptionDetails && (e.exceptionDetails.text || e.exceptionDetails.exception?.description),
      timestamp: stamp(),
    });
  });
  page.on('Runtime.consoleAPICalled', e => {
    if (e.type === 'error') {
      errors.push({
        type: 'console.error',
        text: (e.args || []).map(a => a.value || a.description || '').join(' '),
        timestamp: stamp(),
      });
    }
  });
  page.on('Log.entryAdded', e => {
    if (e.entry && e.entry.level === 'error' && e.entry.source !== 'network') {
      errors.push({
        type: 'log.error',
        text: e.entry.text,
        source: e.entry.source,
        timestamp: stamp(),
      });
    }
  });
  await page.send('Page.enable');
  await page.send('Runtime.enable');
  await page.send('Log.enable');
  await page.send('Emulation.setDeviceMetricsOverride', {
    width: 1440,
    height: 1000,
    deviceScaleFactor: 1,
    mobile: false,
  });
  let loaded = false;
  page.on('Page.loadEventFired', () => { loaded = true; });
  await page.send('Page.navigate', { url });
  await waitFor(() => loaded, 20000, `load event for ${url}`);
  return { page, errors };
}

async function evalValue(page, expression, awaitPromise = true) {
  const result = await page.send('Runtime.evaluate', {
    expression,
    awaitPromise,
    returnByValue: true,
    userGesture: true,
  });
  if (result.exceptionDetails) {
    throw new Error(result.exceptionDetails.exception?.description || result.exceptionDetails.text || 'evaluation failed');
  }
  return result.result ? result.result.value : undefined;
}

async function waitFor(fn, timeoutMs, label) {
  const deadline = Date.now() + timeoutMs;
  let last = '';
  while (Date.now() < deadline) {
    try {
      const value = await fn();
      if (value) return value;
      last = String(value);
    } catch (err) {
      last = err.message;
    }
    await sleep(250);
  }
  throw new Error(`timed out waiting for ${label}${last ? ` (${last})` : ''}`);
}

async function waitForExpr(page, expression, timeoutMs, label) {
  return waitFor(() => evalValue(page, expression), timeoutMs, label);
}

function ok(name, details = {}) {
  return { ...details, name, status: 'pass', timestamp: stamp() };
}

function skip(name, reason) {
  return { name, status: 'skip', timestamp: stamp(), reason };
}

function fail(name, err, details = {}) {
  return { ...details, name, status: 'fail', timestamp: stamp(), error: String(err && err.message ? err.message : err) };
}

async function screenshot(page, outFile) {
  const shot = await page.send('Page.captureScreenshot', { format: 'png', captureBeyondViewport: true });
  await fsp.writeFile(outFile, Buffer.from(shot.data, 'base64'));
  return { path: outFile, sha256: await fileHash(outFile), bytes: (await fsp.stat(outFile)).size };
}

function canvasProbeExpr() {
  return `(() => {
    const canvases = Array.from(document.querySelectorAll('canvas'));
    return canvases.map((c, i) => {
      let dataUrlLength = null;
      let dataUrlHash = null;
      try {
        const data = c.toDataURL('image/png');
        dataUrlLength = data.length;
        dataUrlHash = data.slice(0, 128) + ':' + data.slice(-128);
      } catch (e) {
        dataUrlHash = 'unreadable:' + e.message;
      }
      const r = c.getBoundingClientRect();
      return { index: i, width: c.width, height: c.height, clientWidth: Math.round(r.width), clientHeight: Math.round(r.height), dataUrlLength, dataUrlHash };
    });
  })()`;
}

async function assertMolLoaded(page, label) {
  await waitForExpr(page, `(() => !!(window.RB && RB.VIEWERS && RB.VIEWERS.main && document.querySelector('#molstar-viewer canvas')))()`, 45000, `${label} Mol* viewer`);
  const canvases = await evalValue(page, canvasProbeExpr());
  const visible = (canvases || []).filter(c => c.clientWidth > 50 && c.clientHeight > 50);
  if (!visible.length) throw new Error(`${label}: no visible Mol* canvas`);
  const rendered = visible.some(c => c.dataUrlLength && c.dataUrlLength > 1500);
  if (!rendered) throw new Error(`${label}: visible canvas did not produce a readable nontrivial PNG`);
  return { canvases: visible };
}

function relUrl(origin, rel) {
  return `${origin}/${rel.split(path.sep).map(encodeURIComponent).join('/')}`;
}

async function auditSingle(ctx, rel) {
  const checks = [];
  const url = relUrl(ctx.origin, rel);
  const { page, errors } = await openPage(ctx.chrome.port, ctx.chrome.version.webSocketDebuggerUrl, url);
  try {
    await waitForExpr(page, `document.readyState === 'complete'`, 15000, 'single document ready');
    const initial = await assertMolLoaded(page, 'single initial');
    checks.push(ok('single initial Mol* load', initial));

    const tabCount = await evalValue(page, `document.querySelectorAll('.model-tabs [role="tab"], .model-tabs .tab').length`);
    if (tabCount < 1) throw new Error('single report has no model tabs');
    await evalValue(page, `(() => { const t = document.querySelector('.model-tabs [role="tab"], .model-tabs .tab'); t.focus(); t.dispatchEvent(new MouseEvent('click', { bubbles: true })); return document.activeElement === t; })()`);
    await page.send('Input.dispatchKeyEvent', { type: 'keyDown', key: 'Enter', code: 'Enter', windowsVirtualKeyCode: 13 });
    await page.send('Input.dispatchKeyEvent', { type: 'keyUp', key: 'Enter', code: 'Enter', windowsVirtualKeyCode: 13 });
    if (tabCount > 1) {
      await page.send('Input.dispatchKeyEvent', { type: 'keyDown', key: 'ArrowRight', code: 'ArrowRight', windowsVirtualKeyCode: 39 });
      await page.send('Input.dispatchKeyEvent', { type: 'keyUp', key: 'ArrowRight', code: 'ArrowRight', windowsVirtualKeyCode: 39 });
      await waitForExpr(page, `document.activeElement && document.activeElement.matches('.model-tabs [role="tab"], .model-tabs .tab')`, 5000, 'single ArrowRight tab focus');
    }
    const selected = await evalValue(page, `(() => {
      const active = document.querySelector('.model-tabs [aria-selected="true"]');
      return active ? { key: active.dataset.key, text: active.textContent.trim() } : null;
    })()`);
    checks.push(ok('single model tabs keyboard Enter/right selection', { tabCount, selected }));

    const overlayRoundTrip = await evalValue(page, `(async () => {
      const keys = Array.from(document.querySelectorAll('.model-tabs [data-key]')).map(t => t.dataset.key);
      const key = keys.find(k => STRUCTS && STRUCTS['overlay_' + k]);
      if (!key) return { skipped: true, reason: 'no overlay structure embedded in this single report' };
      await showModel(key);
      await showOverlay();
      const overlayLoaded = (LOADED === 'overlay_' + key);
      await showModel(key);
      return {
        skipped: false,
        key,
        overlayLoaded,
        returnedToModel: LOADED === key,
        activeKey: document.querySelector('.model-tabs [aria-selected="true"]')?.dataset.key || null,
        loaded: LOADED
      };
    })()`);
    if (overlayRoundTrip.skipped) checks.push(skip('single overlay load/return', overlayRoundTrip.reason));
    else if (!overlayRoundTrip.overlayLoaded || !overlayRoundTrip.returnedToModel || overlayRoundTrip.activeKey !== overlayRoundTrip.key) {
      throw new Error(`single overlay round-trip failed: ${JSON.stringify(overlayRoundTrip)}`);
    } else {
      checks.push(ok('single overlay load/return', overlayRoundTrip));
    }

    const fastTransition = await evalValue(page, `(async () => {
      const tabs = Array.from(document.querySelectorAll('.model-tabs [data-key]'));
      const keys = tabs.map(t => t.dataset.key);
      const start = keys.find(k => STRUCTS && STRUCTS['overlay_' + k]);
      const last = keys[keys.length - 1];
      if (!start || !last || keys.length < 2) {
        return { skipped: true, reason: 'requires at least two model tabs and one overlay structure' };
      }
      await showModel(start);
      const overlayPromise = showOverlay();
      const modelPromise = showModel(last);
      const settled = await Promise.allSettled([overlayPromise, modelPromise]);
      await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
      const activeKey = document.querySelector('.model-tabs [aria-selected="true"]')?.dataset.key || null;
      return {
        skipped: false,
        start,
        requestedLast: last,
        activeKey,
        loaded: LOADED,
        current: CURRENT,
        settled: settled.map(s => ({ status: s.status, reason: s.reason ? String(s.reason.message || s.reason) : undefined }))
      };
    })()`);
    if (fastTransition.skipped) checks.push(skip('single overlay-to-tab fast transition', fastTransition.reason));
    else if (fastTransition.activeKey !== fastTransition.requestedLast || fastTransition.loaded !== fastTransition.requestedLast || fastTransition.current !== fastTransition.requestedLast) {
      throw new Error(`single fast transition settled on wrong model: ${JSON.stringify(fastTransition)}`);
    } else {
      checks.push(ok('single overlay-to-tab fast transition', fastTransition));
    }

    const recovery = await evalValue(page, `(async () => {
      const keys = Array.from(document.querySelectorAll('.model-tabs [data-key]')).map(t => t.dataset.key);
      const key = keys.find(k => STRUCTS['overlay_' + k]);
      const other = keys.find(k => k !== key);
      if (!key || !other) return { skipped: true, reason: 'requires two models and an overlay' };
      const cases = [];
      for (const kind of ['model', 'overlay']) {
        await showModel(key);
        const plugin = RB.VIEWERS.main.plugin;
        const builder = plugin.builders.structure;
        const original = builder.parseTrajectory;
        let injected = false;
        try {
          builder.parseTrajectory = async () => {
            injected = true;
            throw new Error('browser audit: injected trajectory failure');
          };
          if (kind === 'model') await showModel(other);
          else await showOverlay();
        } finally {
          builder.parseTrajectory = original;
        }
        const cleared = plugin.managers.structure.hierarchy.current.structures.length === 0;
        await showModel(key);
        const recovered = LOADED === key && plugin.managers.structure.hierarchy.current.structures.length > 0;
        cases.push({ kind, injected, cleared, recovered });
      }
      return { skipped: false, cases };
    })()`);
    if (recovery.skipped) checks.push(skip('single failed model/overlay load recovery', recovery.reason));
    else if (recovery.cases.some(c => !c.injected || !c.cleared || !c.recovered)) {
      throw new Error(`single failed load did not recover: ${JSON.stringify(recovery)}`);
    } else {
      checks.push(ok('single failed model/overlay load recovery', recovery));
    }

    const themes = await evalValue(page, `(async () => {
      const out = [];
      for (const name of ['chain-id', 'plddt-confidence', 'uncertainty']) {
        try { await RB.setTheme('main', name); out.push({ name, ok: true }); }
        catch (e) { out.push({ name, ok: false, error: e.message }); }
      }
      return out;
    })()`);
    const badTheme = themes.find(t => !t.ok);
    if (badTheme) throw new Error(`theme change failed: ${badTheme.name}: ${badTheme.error}`);
    checks.push(ok('single theme changes', { themes }));

    const shot = await screenshot(page, path.join(ctx.out, 'single.png'));
    checks.push(ok('single readable screenshot', { screenshot: shot }));
    if (errors.length) throw new Error(`single page emitted ${errors.length} JS errors`);
    checks.push(ok('single no JS errors'));
  } catch (err) {
    checks.push(fail('single audit', err, { jsErrors: errors }));
  } finally {
    page.close();
  }
  return { name: 'single', rel, url, checks, jsErrors: errors };
}

async function auditBatch(ctx, rel, kind) {
  const checks = [];
  const url = relUrl(ctx.origin, rel);
  const { page, errors } = await openPage(ctx.chrome.port, ctx.chrome.version.webSocketDebuggerUrl, url);
  try {
    await waitForExpr(page, `document.readyState === 'complete' && document.querySelectorAll('#tbody tr[data-job]').length > 0`, 15000, `${kind} rows`);
    const initial = await evalValue(page, `(() => ({ rows: ROWS.length, cols: COLS.map(c => c.key), jobs: Object.keys(JOBS).length }))()`);
    checks.push(ok(`${kind} table rendered`, initial));

    const sort = await evalValue(page, `(async () => {
      const btn = document.querySelector('#thead [data-sort-key]');
      if (!btn) return { ok: false, error: 'no sortable control' };
      const th = btn.closest('th');
      const key = btn.dataset.sortKey;
      const before = th.getAttribute('aria-sort');
      const beforeOrder = Array.from(document.querySelectorAll('#tbody tr[data-job]')).map(r => r.dataset.job);
      btn.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true, view: window }));
      await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
      const firstOrder = Array.from(document.querySelectorAll('#tbody tr[data-job]')).map(r => r.dataset.job);
      // The first direction may already match the displayed order. Toggle once
      // more before requiring a visible change for distinct job names.
      if (beforeOrder.length > 1 && beforeOrder.join('\\u0000') === firstOrder.join('\\u0000')) {
        const toggle = document.querySelector('#thead [data-sort-key="' + CSS.escape(key) + '"]');
        toggle.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true, view: window }));
        await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
      }
      const nextBtn = document.querySelector('#thead [data-sort-key="' + CSS.escape(key) + '"]');
      const nextTh = nextBtn ? nextBtn.closest('th') : null;
      const after = nextTh ? nextTh.getAttribute('aria-sort') : null;
      const sorted = document.querySelector('#thead [aria-sort="ascending"], #thead [aria-sort="descending"]');
      const effectiveAfter = after || (sorted ? sorted.getAttribute('aria-sort') : null);
      const afterOrder = Array.from(document.querySelectorAll('#tbody tr[data-job]')).map(r => r.dataset.job);
      const orderComparable = beforeOrder.length > 1;
      const orderChanged = !orderComparable || beforeOrder.join('\\u0000') !== afterOrder.join('\\u0000');
      return {
        ok: (effectiveAfter === 'ascending' || effectiveAfter === 'descending') && orderChanged,
        key,
        before,
        after,
        effectiveAfter,
        sortedKey: sorted ? (sorted.querySelector('[data-sort-key]')?.dataset.sortKey || null) : null,
        sortedAria: sorted ? sorted.getAttribute('aria-sort') : null,
        beforeOrder,
        firstOrder,
        afterOrder,
        orderComparable,
        orderChanged,
        headText: document.querySelector('#thead')?.textContent?.trim() || '',
        hasSetSort: typeof setSort === 'function'
      };
    })()`);
    if (!sort.ok) throw new Error(sort.error || `aria-sort did not update: ${JSON.stringify(sort)}`);
    checks.push(ok(`${kind} sort button aria-sort`, sort));

    const filter = await evalValue(page, `(() => {
      const input = document.querySelector('#batch-search');
      const first = ROWS[0] && ROWS[0].job;
      if (!input || !first) return { ok: false, error: 'missing search input or rows' };
      input.value = first;
      input.dispatchEvent(new Event('input', { bubbles: true }));
      const visible = document.querySelectorAll('#tbody tr[data-job]').length;
      const count = document.querySelector('#count')?.textContent || '';
      input.value = '';
      input.dispatchEvent(new Event('input', { bubbles: true }));
      return { ok: visible >= 1 && visible <= ROWS.length, query: first, visible, count };
    })()`);
    if (!filter.ok) throw new Error(filter.error || 'search filter did not change visible rows');
    checks.push(ok(`${kind} search filter`, filter));

    const failed = await evalValue(page, `(async () => {
      const row = ROWS.find(r => r.status && r.status !== 'ok');
      if (!row) return { skipped: true, reason: 'no failed/incomplete rows in this report' };
      await openJob(row.job);
      await new Promise(resolve => setTimeout(resolve, 250));
      const detail = document.querySelector('#detail-kpis')?.textContent || '';
      const missing = getComputedStyle(document.querySelector('#viewer-missing')).display !== 'none';
      const diag = document.querySelector('#diag')?.textContent || '';
      return { skipped: false, job: row.job, jobStatus: row.status, missing, hasDetail: /detail|불완전|미완료|오류|error|failed/i.test(detail), diag };
    })()`);
    if (failed.skipped) checks.push(skip(`${kind} failed row diagnostic`, failed.reason));
    else if (!failed.missing || !failed.hasDetail) throw new Error(`failed row did not expose diagnostic detail for ${failed.job}`);
    else checks.push(ok(`${kind} failed row detail opens diagnostic`, failed));

    const valid = await evalValue(page, `(async () => {
      const row = ROWS.find(r => r.status === 'ok' && JOBS[r.job] && JOBS[r.job].cif);
      if (!row) return { skipped: true, reason: 'no valid embedded-CIF rows in this report' };
      await openJob(row.job);
      return { skipped: false, job: row.job };
    })()`);
    if (valid.skipped) {
      checks.push(skip(`${kind} valid row Mol* load`, valid.reason));
    } else {
      const mol = await assertMolLoaded(page, `${kind} valid row ${valid.job}`);
      checks.push(ok(`${kind} valid row Mol* load`, { ...valid, ...mol }));
      const shot = await screenshot(page, path.join(ctx.out, `${kind}.png`));
      checks.push(ok(`${kind} readable screenshot`, { screenshot: shot }));
      await evalValue(page, `backToTable()`);
      const back = await evalValue(page, `getComputedStyle(document.querySelector('#table-view')).display !== 'none' && getComputedStyle(document.querySelector('#detail')).display === 'none'`);
      if (!back) throw new Error('back navigation did not return to table view');
      checks.push(ok(`${kind} back navigation`));
    }

    if (kind === 'target') {
      const target = await evalValue(page, `(() => {
        const cols = COLS.map(c => ({ key: c.key, label: c.label }));
        const forbidden = cols.filter(c => /iptm_nb|ipsae|nb_plddt|cdr3_plddt|iface_pae|n_contacts/.test(c.key));
        const headerText = document.querySelector('#thead')?.textContent || '';
        const forbiddenHeader = /ipTM\\s*\\(nb\\|ag\\)|CDR3|nb\\s*pLDDT|iface PAE|컨택트/.test(headerText);
        return { ok: forbidden.length === 0 && !forbiddenHeader, cols, forbidden, headerText, forbiddenHeader };
      })()`);
      if (!target.ok) throw new Error(`target-only batch exposes binder-specific columns: ${JSON.stringify(target.forbidden)}`);
      checks.push(ok('target-only batch hides binder columns', target));
    }

    if (errors.length) throw new Error(`${kind} page emitted ${errors.length} JS errors`);
    checks.push(ok(`${kind} no JS errors`));
  } catch (err) {
    checks.push(fail(`${kind} audit`, err, { jsErrors: errors }));
  } finally {
    page.close();
  }
  return { name: kind, rel, url, checks, jsErrors: errors };
}

async function discover(root) {
  const candidates = [
    { name: 'single', rel: path.join('single', 'index.html'), required: true },
    { name: 'batch', rel: path.join('batch', 'index.html'), required: true },
    { name: 'target', rel: path.join('target', 'index.html'), required: false },
  ];
  const pages = [];
  for (const c of candidates) {
    const file = path.join(root, c.rel);
    if (fs.existsSync(file)) {
      pages.push({ ...c, file, sha256: await fileHash(file) });
    } else if (c.required) {
      pages.push({ ...c, missing: true, file });
    }
  }
  return pages;
}

async function rmrf(p) {
  if (!p) return;
  await fsp.rm(p, { recursive: true, force: true });
}

async function main() {
  const args = parseArgs(process.argv);
  if (args.help) {
    console.log(usage());
    return 0;
  }
  if (!args.root || !args.out) throw new Error(`--root and --out are required\n${usage()}`);
  const root = path.resolve(args.root);
  const out = path.resolve(args.out);
  if (!fs.existsSync(root) || !fs.statSync(root).isDirectory()) {
    console.log(JSON.stringify({
      status: 'ready',
      timestamp: stamp(),
      message: 'browser audit harness is installed; provide --root after reports are generated',
      root,
      out,
    }, null, 2));
    return 2;
  }
  await fsp.mkdir(out, { recursive: true });

  let server;
  let chrome;
  let profile;
  const started = stamp();
  const evidence = {
    started,
    root,
    out,
    chrome: { path: CHROME },
    sourceHashes: {},
    reports: [],
    status: 'fail',
  };

  try {
    if (!fs.existsSync(CHROME)) throw new Error(`${CHROME} not found`);
    evidence.sourceHashes.script = await fileHash(__filename);
    for (const asset of ['assets/report.js', 'assets/molstar.js', 'assets/molstar.css']) {
      const file = path.resolve(__dirname, '..', asset);
      if (fs.existsSync(file)) evidence.sourceHashes[asset] = await fileHash(file);
    }

    const pages = await discover(root);
    evidence.discovered = pages;
    const missingRequired = pages.filter(p => p.missing && p.required);
    if (missingRequired.length) {
      throw new Error(`missing required report(s): ${missingRequired.map(p => p.rel).join(', ')}`);
    }

    const served = await listenServer(root);
    server = served.server;
    evidence.server = { origin: served.origin, bind: '127.0.0.1' };
    profile = await fsp.mkdtemp(path.join(os.tmpdir(), 'boltz-browser-audit-'));
    chrome = await launchChromeWithFallback(profile);
    evidence.chrome.version = chrome.version.Browser;
    evidence.chrome.protocolVersion = chrome.version['Protocol-Version'];
    evidence.chrome.usedNoSandbox = !!chrome.usedNoSandbox;
    evidence.chrome.args = chrome.args;
    if (chrome.firstLaunchError) evidence.chrome.firstLaunchError = chrome.firstLaunchError;

    const ctx = { origin: served.origin, out, chrome };
    for (const page of pages) {
      if (page.missing) continue;
      if (page.name === 'single') evidence.reports.push(await auditSingle(ctx, page.rel));
      else evidence.reports.push(await auditBatch(ctx, page.rel, page.name));
    }
    const failed = evidence.reports.flatMap(r => r.checks.filter(c => c.status === 'fail'));
    evidence.status = failed.length ? 'fail' : 'pass';
    evidence.finished = stamp();
    evidence.chrome.stderrTail = chrome.stderr().slice(-4000);
    const outJson = path.join(out, 'browser_audit.json');
    await fsp.writeFile(outJson, JSON.stringify(evidence, null, 2));
    console.log(JSON.stringify({ status: evidence.status, evidence: outJson, screenshots: evidence.reports.flatMap(r => r.checks.map(c => c.screenshot).filter(Boolean)) }, null, 2));
    return failed.length ? 1 : 0;
  } catch (err) {
    evidence.status = 'fail';
    evidence.finished = stamp();
    evidence.error = String(err && err.stack ? err.stack : err);
    if (chrome) evidence.chrome.stderrTail = chrome.stderr().slice(-4000);
    await fsp.mkdir(out, { recursive: true });
    const outJson = path.join(out, 'browser_audit.json');
    await fsp.writeFile(outJson, JSON.stringify(evidence, null, 2));
    console.error(JSON.stringify({ status: 'fail', evidence: outJson, error: err.message }, null, 2));
    return 1;
  } finally {
    if (chrome && chrome.proc && chrome.proc.exitCode === null) {
      chrome.proc.kill('SIGTERM');
      await sleep(500);
      if (chrome.proc.exitCode === null) chrome.proc.kill('SIGKILL');
    }
    if (server) await new Promise(resolve => server.close(resolve));
    await rmrf(profile);
  }
}

main().then(code => process.exit(code)).catch(err => {
  console.error(err.stack || err.message || String(err));
  process.exit(1);
});
