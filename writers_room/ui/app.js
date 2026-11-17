/**
 * The Writers' Room — Client Application
 * ========================================
 * Handles WebSocket streaming, screenplay rendering, dialect sliders,
 * and the critic rejection log.
 */

// ── Constants ─────────────────────────────────────────────────────────

const isFile = window.location.protocol === 'file:' || !window.location.host;
const API_BASE = isFile ? 'http://localhost:8000' : window.location.origin;
const WS_PROTOCOL = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
const WS_HOST = isFile ? 'localhost:8000' : window.location.host;
const WS_BASE  = `${WS_PROTOCOL}//${WS_HOST}`;

const CHAR_EMOJIS = {
  merc_vance:    '⚓',
  dr_lena_soros: '🧬',
  tommy_k:       '💻',
};

const CHAR_COLORS = {
  merc_vance:    '#e07a5f',
  dr_lena_soros: '#81b29a',
  tommy_k:       '#f2cc8f',
};

// ── State ─────────────────────────────────────────────────────────────

let currentSceneId = null;
let ws = null;
let totalBeats = 0;
let currentBeats = 0;
let lineCount = 0;
let wordCount = 0;
let characters = [];
let rejLogCleared = false;

// ── Init ──────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', async () => {
  // Load available characters
  try {
    const resp = await fetch(`${API_BASE}/api/characters`);
    if (resp.ok) {
      characters = await resp.json();
      renderCharCheckboxes();
    }
  } catch (e) {
    console.warn('Could not load characters:', e);
    // Fallback: hardcode the three known characters
    characters = [
      { character_id: 'merc_vance',    display_name: 'Merc Vance',     archetype: 'Cynical Dockside Smuggler' },
      { character_id: 'dr_lena_soros', display_name: 'Dr. Lena Soros',  archetype: 'Cold Corporate Geneticist' },
      { character_id: 'tommy_k',      display_name: 'Tommy K',         archetype: 'Hyperactive Teenage Street Hacker' },
    ];
    renderCharCheckboxes();
  }

  // Beats slider
  const slider = document.getElementById('maxBeats');
  const display = document.getElementById('maxBeatsVal');
  slider.addEventListener('input', () => { display.textContent = slider.value; });
});

// ── Character checkboxes ──────────────────────────────────────────────

function renderCharCheckboxes() {
  const container = document.getElementById('charCheckboxes');
  container.innerHTML = '';
  characters.forEach(c => {
    const lbl = document.createElement('label');
    const inp = document.createElement('input');
    inp.type = 'checkbox';
    inp.value = c.character_id;
    inp.checked = true;
    const emoji = CHAR_EMOJIS[c.character_id] || '🎭';
    lbl.appendChild(inp);
    lbl.appendChild(document.createTextNode(` ${emoji} ${c.display_name}`));
    lbl.style.color = CHAR_COLORS[c.character_id] || '#e0e0e0';
    container.appendChild(lbl);
  });
}

// ── Start Scene ───────────────────────────────────────────────────────

async function startScene() {
  const premise  = document.getElementById('premise').value.trim();
  const location = document.getElementById('location').value.trim();
  const maxBeats = parseInt(document.getElementById('maxBeats').value, 10);

  if (!premise) { toast('Please enter a scene premise.', 'warn'); return; }
  if (!location) { toast('Please enter a location.', 'warn'); return; }

  const selectedChars = [];
  document.querySelectorAll('#charCheckboxes input:checked').forEach(cb => {
    selectedChars.push(cb.value);
  });
  if (selectedChars.length < 2) { toast('Select at least 2 characters.', 'warn'); return; }

  // Disable the button
  const btn = document.getElementById('btnGenerate');
  btn.disabled = true;
  btn.textContent = '⏳ Starting…';

  try {
    const resp = await fetch(`${API_BASE}/api/scene/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        premise,
        location,
        characters: selectedChars,
        max_beats: maxBeats,
      }),
    });

    if (!resp.ok) {
      const err = await resp.json();
      toast(`Error: ${err.detail || resp.statusText}`, 'error');
      btn.disabled = false;
      btn.textContent = '🎬 Generate Scene';
      return;
    }

    const data = await resp.json();
    currentSceneId = data.scene_id;
    totalBeats = data.max_beats;
    currentBeats = 0;
    lineCount = 0;
    wordCount = 0;

    // Reset the screenplay pane
    const body = document.getElementById('screenplayBody');
    body.innerHTML = '';

    // Add scene heading
    const heading = document.createElement('div');
    heading.className = 'beat-marker';
    heading.textContent = normalizeLocation(location);
    body.appendChild(heading);

    if (premise) {
      const premiseEl = document.createElement('div');
      premiseEl.style.cssText = 'color: var(--text-muted); font-style: italic; margin: 8px 15%; text-align: center;';
      premiseEl.textContent = premise;
      body.appendChild(premiseEl);
    }

    // Show dialect sliders
    buildDialectSliders(selectedChars);

    // Update progress
    updateProgress(0, totalBeats, 'Generating…');
    updateStats();

    // Connect WebSocket
    connectWebSocket(currentSceneId);

    btn.textContent = '⏳ Generating…';
    toast('Scene started!', 'success');

  } catch (e) {
    toast(`Failed to start: ${e.message}`, 'error');
    btn.disabled = false;
    btn.textContent = '🎬 Generate Scene';
  }
}

// ── WebSocket ─────────────────────────────────────────────────────────

function connectWebSocket(sceneId) {
  if (ws) { ws.close(); }

  ws = new WebSocket(`${WS_BASE}/ws/scene/${sceneId}`);
  const dot = document.getElementById('connDot');
  const label = document.getElementById('connLabel');

  ws.onopen = () => {
    dot.classList.add('connected');
    label.textContent = 'Connected';
  };

  ws.onclose = () => {
    dot.classList.remove('connected');
    label.textContent = 'Disconnected';
  };

  ws.onerror = () => {
    dot.classList.remove('connected');
    label.textContent = 'Error';
  };

  ws.onmessage = (event) => {
    try {
      const msg = JSON.parse(event.data);
      handleEvent(msg);
    } catch (e) {
      console.warn('Bad WS message:', event.data);
    }
  };
}

// ── Event Handlers ────────────────────────────────────────────────────

function handleEvent(msg) {
  const { type, data } = msg;

  switch (type) {
    case 'beat':
      handleBeat(data);
      break;
    case 'line':
      handleLine(data);
      break;
    case 'line_accepted':
      handleLineAccepted(data);
      break;
    case 'lint_pass':
      handleLintPass(data);
      break;
    case 'lint_fail':
      handleLintFail(data);
      break;
    case 'critic_pass':
      handleCriticPass(data);
      break;
    case 'critic_fail':
      handleCriticFail(data);
      break;
    case 'retry':
      handleRetry(data);
      break;
    case 'complete':
      handleComplete(data);
      break;
    case 'error':
      handleError(data);
      break;
    case 'heartbeat':
      break;
    default:
      console.log('Unknown event:', type, data);
  }
}

function handleBeat(data) {
  currentBeats++;
  updateProgress(currentBeats, totalBeats, `Beat ${currentBeats}/${totalBeats}`);

  const body = document.getElementById('screenplayBody');
  const marker = document.createElement('div');
  marker.className = 'beat-marker';
  marker.textContent = `— beat ${data.beat_number || currentBeats} —`;
  body.appendChild(marker);

  // Show typing indicator
  showTypingIndicator(data.next_speaker);
  autoScroll();
}

function handleLine(data) {
  removeTypingIndicator();

  const body = document.getElementById('screenplayBody');
  const charId = data.character_id || 'unknown';
  const displayName = getDisplayName(charId);
  const emoji = CHAR_EMOJIS[charId] || '🎭';

  const block = document.createElement('div');
  block.className = `dialogue-block char-${charId}`;
  block.id = `line-${data.beat_number}-${Date.now()}`;

  // Character heading
  const heading = document.createElement('div');
  heading.className = 'character-heading';
  heading.innerHTML = `<span class="char-emoji">${emoji}</span> ${escapeHtml(displayName.toUpperCase())}`;
  block.appendChild(heading);

  // Stage direction
  if (data.stage_direction) {
    const dir = document.createElement('div');
    dir.className = 'stage-direction';
    let direction = data.stage_direction.trim();
    if (!direction.startsWith('(')) direction = `(${direction}`;
    if (!direction.endsWith(')')) direction = `${direction})`;
    dir.textContent = direction;
    block.appendChild(dir);
  }

  // Dialogue text
  const text = document.createElement('div');
  text.className = 'dialogue-text';
  text.textContent = data.spoken_text || '…';
  block.appendChild(text);

  body.appendChild(block);

  // Update stats
  lineCount++;
  wordCount += (data.spoken_text || '').split(/\s+/).filter(Boolean).length;
  updateStats();
  autoScroll();
}

function handleLineAccepted(data) {
  // Flash the last dialogue block green briefly
  const blocks = document.querySelectorAll('.dialogue-block');
  if (blocks.length > 0) {
    const last = blocks[blocks.length - 1];
    last.classList.add('flash-pass');
    setTimeout(() => last.classList.remove('flash-pass'), 800);
  }
}

function handleLintPass(data) {
  addRejectionEntry({
    type: 'LINT PASS',
    pass: true,
    violations: [],
  });
}

function handleLintFail(data) {
  addRejectionEntry({
    type: 'LINT FAIL',
    pass: false,
    violations: data.violations || [],
  });

  // Flash the last block red
  const blocks = document.querySelectorAll('.dialogue-block');
  if (blocks.length > 0) {
    const last = blocks[blocks.length - 1];
    last.classList.add('flash-fail');
    setTimeout(() => last.classList.remove('flash-fail'), 1200);
  }
}

function handleCriticPass(data) {
  const score = data.similarity_score != null ? data.similarity_score.toFixed(2) : '?';
  addRejectionEntry({
    type: `CRITIC PASS (${score})`,
    pass: true,
    violations: data.feedback ? [data.feedback] : [],
  });
}

function handleCriticFail(data) {
  const score = data.similarity_score != null ? data.similarity_score.toFixed(2) : '?';
  addRejectionEntry({
    type: `CRITIC FAIL (${score})`,
    pass: false,
    violations: data.feedback ? [data.feedback] : [],
  });
}

function handleRetry(data) {
  const body = document.getElementById('screenplayBody');
  const badge = document.createElement('div');
  badge.className = 'retry-badge';
  badge.textContent = `↻ Retry ${data.retry_count} — ${(data.violations || []).join('; ').substring(0, 80)}`;
  body.appendChild(badge);

  showTypingIndicator();
  autoScroll();
}

function handleComplete(data) {
  removeTypingIndicator();
  updateProgress(totalBeats, totalBeats, 'Complete!');

  const body = document.getElementById('screenplayBody');
  const banner = document.createElement('div');
  banner.className = 'completion-banner';
  banner.textContent = '✦ SCENE COMPLETE ✦';
  body.appendChild(banner);

  // Enable export and re-enable generate
  document.getElementById('btnExport').disabled = false;
  const btn = document.getElementById('btnGenerate');
  btn.disabled = false;
  btn.textContent = '🎬 Generate Scene';

  toast('Scene generation complete!', 'success');
  autoScroll();
}

function handleError(data) {
  removeTypingIndicator();
  toast(`Error: ${data.error || 'Unknown error'}`, 'error');

  const btn = document.getElementById('btnGenerate');
  btn.disabled = false;
  btn.textContent = '🎬 Generate Scene';
}

// ── Dialect Sliders ───────────────────────────────────────────────────

function buildDialectSliders(charIds) {
  const panel = document.getElementById('dialectPanel');
  const container = document.getElementById('dialectSliders');
  container.innerHTML = '';
  panel.classList.add('active');

  charIds.forEach(cid => {
    const displayName = getDisplayName(cid);
    const emoji = CHAR_EMOJIS[cid] || '🎭';

    const group = document.createElement('div');
    group.className = `dialect-char-group ${cid}`;

    group.innerHTML = `
      <h4>${emoji} ${escapeHtml(displayName)}</h4>
      ${makeSlider(cid, 'slang_density', 'Slang Density', 0, 100, 50, '%')}
      ${makeSlider(cid, 'emotional_hostility', 'Hostility', 0, 100, 30, '%')}
      ${makeSlider(cid, 'formality', 'Formality', 0, 100, 50, '%')}
      ${makeSlider(cid, 'sentence_length', 'Sentence Len ±', -5, 5, 0, '')}
    `;

    container.appendChild(group);
  });

  // Attach listeners
  container.querySelectorAll('input[type="range"]').forEach(inp => {
    inp.addEventListener('change', () => {
      const cid = inp.dataset.character;
      const mod = inp.dataset.modifier;
      const val = parseFloat(inp.value);
      inp.nextElementSibling.textContent = val + (inp.dataset.unit || '');
      updateDialect(cid, mod, val);
    });
    inp.addEventListener('input', () => {
      inp.nextElementSibling.textContent = inp.value + (inp.dataset.unit || '');
    });
  });
}

function makeSlider(cid, modifier, label, min, max, defaultVal, unit) {
  return `
    <label>${label}</label>
    <div class="slider-row">
      <input type="range" min="${min}" max="${max}" value="${defaultVal}"
             data-character="${cid}" data-modifier="${modifier}" data-unit="${unit}" />
      <span class="slider-value">${defaultVal}${unit}</span>
    </div>
  `;
}

async function updateDialect(characterId, modifier, value) {
  if (!currentSceneId) return;
  try {
    const resp = await fetch(`${API_BASE}/api/scene/${currentSceneId}/dialect`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ character_id: characterId, modifier, value }),
    });
    if (resp.ok) {
      toast(`${modifier} updated for ${getDisplayName(characterId)}`, 'success');
    }
  } catch (e) {
    console.warn('Dialect update failed:', e);
  }
}

// ── Export ─────────────────────────────────────────────────────────────

async function exportScreenplay() {
  if (!currentSceneId) return;
  try {
    const resp = await fetch(`${API_BASE}/api/scene/${currentSceneId}/screenplay`);
    if (!resp.ok) { toast('Export failed.', 'error'); return; }
    const data = await resp.json();
    const text = data.screenplay || '';

    const blob = new Blob([text], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `writers-room-${currentSceneId}.fountain`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    toast('Screenplay exported!', 'success');
  } catch (e) {
    toast(`Export failed: ${e.message}`, 'error');
  }
}

// ── Rejection Log ─────────────────────────────────────────────────────

function addRejectionEntry({ type, pass, violations }) {
  const log = document.getElementById('rejectionLog');

  // Clear the placeholder on first entry
  if (!rejLogCleared) {
    log.innerHTML = '';
    rejLogCleared = true;
  }

  const entry = document.createElement('div');
  entry.className = `rejection-entry ${pass ? 'pass' : ''}`;

  const typeSpan = `<span class="rej-type">${escapeHtml(type)}</span>`;
  let violationText = '';
  if (violations.length > 0) {
    violationText = `<span class="rej-line">${escapeHtml(violations.join(' | ').substring(0, 200))}</span>`;
  }

  entry.innerHTML = `${typeSpan}${violationText}`;
  log.appendChild(entry);

  // Auto-scroll the log
  log.scrollTop = log.scrollHeight;
}

// ── Typing Indicator ──────────────────────────────────────────────────

function showTypingIndicator(speakerId) {
  removeTypingIndicator();
  const body = document.getElementById('screenplayBody');
  const indicator = document.createElement('div');
  indicator.className = 'typing-indicator';
  indicator.id = 'typingIndicator';

  const emoji = speakerId ? (CHAR_EMOJIS[speakerId] || '🎭') : '✍️';
  const name = speakerId ? getDisplayName(speakerId) : 'Agent';

  indicator.innerHTML = `
    ${emoji} ${escapeHtml(name)} is composing…
    <span class="dot"></span><span class="dot"></span><span class="dot"></span>
  `;
  body.appendChild(indicator);
  autoScroll();
}

function removeTypingIndicator() {
  const el = document.getElementById('typingIndicator');
  if (el) el.remove();
}

// ── Progress ──────────────────────────────────────────────────────────

function updateProgress(current, total, label) {
  const pct = total > 0 ? Math.min(100, (current / total) * 100) : 0;
  document.getElementById('progressFill').style.width = `${pct}%`;
  document.getElementById('progressLabel').textContent = label;
}

function updateStats() {
  document.getElementById('statLines').textContent = `Lines: ${lineCount}`;
  document.getElementById('statWords').textContent = `Words: ${wordCount}`;
  document.getElementById('statBeats').textContent = `Beats: ${currentBeats} / ${totalBeats}`;
}

// ── Helpers ───────────────────────────────────────────────────────────

function getDisplayName(charId) {
  const c = characters.find(ch => ch.character_id === charId);
  if (c) return c.display_name;
  return charId.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
}

function normalizeLocation(loc) {
  const upper = loc.toUpperCase().trim();
  if (upper.startsWith('INT.') || upper.startsWith('EXT.')) return upper;
  return `INT. ${upper} - DAY`;
}

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

function autoScroll() {
  const body = document.getElementById('screenplayBody');
  requestAnimationFrame(() => {
    body.scrollTop = body.scrollHeight;
  });
}

// ── Toast Notifications ───────────────────────────────────────────────

function toast(message, type = 'info') {
  const container = document.getElementById('toastContainer');
  const el = document.createElement('div');
  el.className = 'toast';

  if (type === 'error')   el.style.borderColor = '#ef5350';
  if (type === 'success') el.style.borderColor = '#4caf50';
  if (type === 'warn')    el.style.borderColor = '#f2cc8f';

  el.textContent = message;
  container.appendChild(el);

  setTimeout(() => {
    if (el.parentNode) el.parentNode.removeChild(el);
  }, 3000);
}
