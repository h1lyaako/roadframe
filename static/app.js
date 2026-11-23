const $ = id => document.getElementById(id);
const video = $('video');
const icons = {
  play: '<path d="m8 5 12 7-12 7z" fill="currentColor" stroke="none"/>',
  pause: '<path d="M9 5v14m6-14v14" stroke-width="2.5"/>',
  sound: '<path d="M4 10v4h4l5 4V6l-5 4zm12-1a4 4 0 0 1 0 6m2-8a7 7 0 0 1 0 10"/>',
  muted: '<path d="M4 10v4h4l5 4V6l-5 4zm12 0 4 4m0-4-4 4"/>',
};
let job, selected = null, busy = false, revision = 0, exporting = false;
let archive = [], failures = [], manualTime = null, previewing = false;
let userNavigated = false, archiveRun = 0, imageQueue = Promise.resolve();
const videoName = /\.(mp4|mov|webm|mkv|avi)$/i;
const order = new Intl.Collator('ru', { numeric: true, sensitivity: 'base' });

function icon(id, name, label) {
  $(id).innerHTML = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" aria-hidden="true">' + icons[name] + '</svg>';
  $(id).setAttribute('aria-label', label);
  $(id).title = label;
}

function clock(seconds) {
  const value = Math.max(0, Math.floor(seconds || 0));
  const parts = [Math.floor(value / 60) % 60, value % 60];
  if (value >= 3600) parts.unshift(Math.floor(value / 3600));
  return parts.map(part => String(part).padStart(2, '0')).join(':');
}

function preciseTime(seconds) {
  const hundredths = Math.round(seconds * 100);
  const fraction = String(hundredths % 100).padStart(2, '0');
  return clock(hundredths / 100) + ',' + fraction;
}

function notify(message, error = false) {
  const box = $('notice');
  box.hidden = !message;
  box.textContent = message;
  box.classList.toggle('error', error);
}

async function json(response) {
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || 'Не удалось выполнить запрос.');
  return data;
}

async function analyzeFile(file, part, index, total) {
  const body = file ? new FormData() : null;
  if (body) body.append('video', file);
  const query = new URLSearchParams({mode: $('mode').value});
  if (!file) { query.set('demo', 'collision'); query.set('part', part); }
  const endpoint = '/api/jobs?' + query;
  const { id } = await json(await fetch(endpoint, { method: 'POST', body, headers: $('mode').value === 'codex' ? {'X-Roadframe-Codex':'1'} : {} }));
  while (true) {
    const result = await json(await fetch('/api/jobs/' + id, { cache: 'no-store' }));
    if (result.status === 'error') throw new Error(result.error);
    if (result.status === 'ready') return result;
    $('progress-label').textContent = (index + 1) + ' из ' + total + ' · ' + (file?.name || result.name) + ' · ' + result.phase;
    $('progress').value = (index + result.progress / 100) / total * 100;
    await new Promise(resolve => setTimeout(resolve, 450));
  }
}

async function start(files, folderName = '') {
  if (busy) return;
  const demo = !files;
  const pathOf = file => file.webkitRelativePath || file.name;
  const inputs = demo ? [1, 2] : Array.from(files)
    .filter(file => videoName.test(file.name)).sort((a, b) => order.compare(pathOf(a), pathOf(b)));
  if (!inputs.length) return notify('В выбранной папке нет поддерживаемых видео.', true);
  if (inputs.length > 100) return notify('В одном архиве можно открыть до 100 видео. Выберите часть файлов.', true);
  if (!demo && inputs.reduce((sum, file) => sum + file.size, 0) > 20 * 1024 ** 3) {
    return notify('Архив превышает 20 ГБ. Выберите часть файлов.', true);
  }
  busy = true;
  $('mode').disabled = true;
  $('upload').disabled = $('demo').disabled = $('open-folder').disabled = true;
  $('progress-box').hidden = false;
  notify('');
  archive = [];
  failures = [];
  const run = ++archiveRun;
  imageQueue = Promise.resolve();
  manualTime = null;
  previewing = false;
  userNavigated = false;
  job = null;
  selected = null;
  revision++;
  video.pause();
  video.removeAttribute('src');
  video.load();
  $('workspace').hidden = false;
  $('workspace').classList.remove('has-video');
  $('empty').hidden = true;
  $('archive').hidden = false;
  $('credit').hidden = !demo;
  $('subtitle').textContent = demo ? 'Пример ДТП' : (folderName || 'Выбранные записи');
  switchTab('events');
  renderArchive();
  renderEvents();
  const started = performance.now();
  $('elapsed').textContent = '00:00';
  const timer = setInterval(() => { $('elapsed').textContent = clock((performance.now() - started) / 1000); }, 1000);
  try {
    for (const [index, input] of inputs.entries()) {
      const name = demo ? ['Санкт-Петербург', 'Трёхгорный'][index] : pathOf(input);
      $('progress-label').textContent = (index + 1) + ' из ' + inputs.length + ' · Читаем ' + name;
      $('progress').value = index / inputs.length * 100;
      try {
        if (!demo && input.size > 1024 ** 3) throw new Error('Файл больше 1 ГБ.');
        const result = await analyzeFile(demo ? null : input, input, index, inputs.length);
        result.path = demo ? result.name : (folderName ? name.slice(folderName.length + 1) : name);
        result.events.sort((a, b) => a.time - b.time);
        archive.push(result);
        if (!job || (!userNavigated && !selected && result.events.length)) show(result, result.events[0]);
        if (result.events.length) {
          imageQueue = imageQueue.then(() => eventImages(result, run));
        }
      } catch (error) {
        failures.push({ name, message: error.message });
      }
      renderArchive();
      renderEvents();
    }
    if (!archive.length) {
      notify('Не удалось открыть ни одной записи. Подробности в списке файлов.', true);
    }
  } finally {
    clearInterval(timer);
    $('progress-box').hidden = true;
    $('upload').disabled = $('demo').disabled = $('open-folder').disabled = false;
    busy = false;
    renderEvents();
    $('mode').disabled = false;
  }
}

function renderArchive() {
  $('archive-info').textContent = 'С отметками: ' + archive.filter(record => record.events.length).length + ' из ' + archive.length + ' видео';
  $('file-count').textContent = archive.length;
  $('files').replaceChildren(...archive.map(record => {
    const button = document.createElement('button');
    button.className = 'file-segment';
    button.setAttribute('aria-pressed', String(record === job));
    button.title = record.path;
    button.onclick = () => {
      userNavigated = true;
      show(record, selected && job === record ? selected : record.events[0]);
    };
    const label = document.createElement('strong');
    label.textContent = record.path;
    const position = document.createElement('span');
    position.textContent = clock(record.duration) + ' · ' + (record.events.length ? 'Отметок: ' + record.events.length : 'Без отметок');
    const icon = document.createElement('span');
    icon.className = 'file-icon';
    icon.setAttribute('aria-hidden', 'true');
    icon.innerHTML = '<svg viewBox="0 0 24 24"><rect x="3.5" y="5.5" width="17" height="13" rx="2"/><path d="m10 9 5 3-5 3z"/></svg>';
    const copy = document.createElement('span');
    copy.className = 'file-copy';
    copy.append(label, position);
    button.append(icon, copy);
    return button;
  }));
  $('file-errors').hidden = !failures.length;
  $('error-count').textContent = 'Пропущено файлов: ' + failures.length;
  $('error-list').replaceChildren(...failures.map(({name, message}) => {
    const item = document.createElement('li');
    item.textContent = name + ' · ' + message;
    return item;
  }));
}

function switchTab(name) {
  for (const tab of ['events', 'files']) {
    const active = tab === name;
    $(tab + '-tab').setAttribute('aria-selected', String(active));
    $(tab + '-tab').tabIndex = active ? 0 : -1;
    $(tab + '-panel').hidden = !active;
  }
}

function show(result, event = null) {
  if (job === result) {
    select(event);
    return;
  }
  job = result;
  selected = event;
  manualTime = null;
  previewing = false;
  $('clip-settings').open = false;
  const token = ++revision;
  $('workspace').hidden = false;
  $('workspace').classList.add('has-video');
  $('empty').hidden = true;
  $('filename').textContent = job.path;
  $('filename').title = job.path;
  $('file-info').textContent = 'Длительность ' + preciseTime(job.duration);
  $('seek').max = job.duration;
  $('seek').value = 0;
  $('ticks').replaceChildren(...Array.from({ length: 5 }, (_, index) => {
    const tick = document.createElement('span');
    tick.textContent = (job.duration < 8 ? preciseTime : clock)(job.duration * index / 4);
    return tick;
  }));
  renderArchive();
  renderEvents();
  renderMarkers();
  const url = '/api/jobs/' + job.id + '/video';
  video.pause();
  video.onloadedmetadata = () => {
    if (token !== revision) return;
    if (previewing) seek(selection()[0]);
    else if (selected) seek(selected.time);
    else sync();
  };
  video.src = url;
  thumbnails(url, job.duration, token);
  renderClip();
  sync(event?.time ?? 0);
}

function renderEvents() {
  const focused = archive.flatMap(record => record.events).find(event => event.button === document.activeElement);
  const rows = archive.flatMap(record => record.events
    .map(event => ({record, event})));
  $('events').replaceChildren();
  $('count').textContent = String(rows.length);
  if (!rows.length) {
    const p = document.createElement('p');
    p.className = 'empty-events';
    p.textContent = busy ? 'Идёт поиск эпизодов…' : 'Эпизоды не найдены. Записи доступны во вкладке «Все видео».';
    $('events').append(p);
  }
  for (const {record, event} of rows) {
    const button = document.createElement('button');
    button.className = 'event';
    button.setAttribute('aria-pressed', String(event === selected));
    const time = document.createElement('time');
    time.textContent = preciseTime(event.time);
    const description = document.createElement('span');
    description.className = 'event-description';
    const name = document.createElement('strong');
    name.textContent = record.path;
    name.title = record.path;
    const source = document.createElement('span');
    source.textContent = event.signal;
    description.append(name, source);
    const still = document.createElement('span');
    still.className = 'event-still';
    still.setAttribute('aria-hidden', 'true');
    const img = document.createElement('img');
    img.alt = '';
    if (event.still) img.src = event.still;
    still.append(img);
    event.image = img;
    button.append(still, description, time);
    button.onclick = () => {
      userNavigated = true;
      show(record, event);
    };
    event.button = button;
    $('events').append(button);
  }
  focused?.button?.focus({preventScroll: true});
}

function renderMarkers() {
  $('markers').replaceChildren();
  for (const item of job.events) {
    const marker = document.createElement('button');
    marker.className = 'marker';
    marker.style.left = item.time / job.duration * 100 + '%';
    marker.title = preciseTime(item.time) + ' · ' + item.signal;
    marker.setAttribute('aria-label', marker.title);
    marker.onclick = () => { userNavigated = true; select(item); };
    $('markers').append(marker);
  }
}

function select(item) {
  selected?.button?.setAttribute('aria-pressed', 'false');
  selected = item;
  manualTime = null;
  previewing = false;
  selected?.button?.setAttribute('aria-pressed', 'true');
  video.pause();
  if (item) seek(item.time);
  else sync();
  renderClip();
}

function seek(seconds, manual = false) {
  if (!job || !Number.isFinite(video.duration)) return;
  if (manual) {
    userNavigated = true;
    if (previewing) video.pause();
    previewing = false;
    renderClip();
  }
  video.currentTime = Math.max(0, Math.min(video.duration, seconds));
  sync(video.currentTime);
}

function selection() {
  const anchor = selected?.time ?? manualTime;
  if (anchor == null) return null;
  const before = Math.min(60, Math.max(0, Number($('before').value) || 0));
  const after = Math.min(60, Math.max(1, Number($('after').value) || 1));
  const end = Math.min(job.duration, anchor + after);
  const start = Math.min(Math.max(0, anchor - before), Math.max(0, end - 1));
  return [start, end].map(value => Math.round(value * 1000) / 1000);
}

function renderClip() {
  if (!job) return;
  const range = selection();
  const anchor = selected?.time ?? manualTime;
  $('clip-actions').hidden = $('clip-settings').hidden = $('clip-range').hidden = $('range-key').hidden = !range;
  $('mark-key').hidden = !job.events.length;
  $('manual').hidden = !!range;
  $('export').disabled = exporting;
  $('export-label').textContent = exporting ? 'Сохраняем…' : selected && selected.status !== 'visible_consequences' ? 'Скачать фрагмент ДТП' : 'Скачать фрагмент';
  $('preview-clip').textContent = previewing ? 'Остановить просмотр' : 'Смотреть фрагмент';
  if (!range) {
    $('clip-title').textContent = 'В этом видео нет отметок';
    $('range-label').textContent = 'Можно посмотреть запись и отметить момент вручную.';
    return;
  }
  const [start, end] = range;
  const title = selected?.signal ?? 'Ваш момент';
  $('clip-title').textContent = title + ' · ' + preciseTime(anchor);
  $('range-label').textContent = 'Фрагмент ' + preciseTime(start) + ' – ' + preciseTime(end);
  $('clip-range').style.left = (start / job.duration * 100) + '%';
  $('clip-range').style.width = ((end - start) / job.duration * 100) + '%';
}

function sync(seconds = video.currentTime) {
  if (!job) return;
  $('clock').textContent = preciseTime(seconds) + ' / ' + preciseTime(job.duration);
  $('needle').style.left = Math.min(100, seconds / job.duration * 100) + '%';
  $('seek').value = seconds;
  $('seek').setAttribute('aria-valuetext', preciseTime(seconds));
}

function mediaEvent(element, type, action) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => finish(new Error('Видео недоступно')), 6000);
    const ok = () => finish();
    const fail = () => finish(new Error('Видео недоступно'));
    function finish(error) {
      clearTimeout(timer);
      element.removeEventListener(type, ok);
      element.removeEventListener('error', fail);
      if (error) reject(error); else resolve();
    }
    element.addEventListener(type, ok, { once: true });
    element.addEventListener('error', fail, { once: true });
    action();
  });
}

async function thumbnails(url, duration, token) {
  const film = $('frames');
  film.replaceChildren();
  const reader = document.createElement('video');
  reader.muted = true;
  reader.preload = 'auto';
  try {
    await mediaEvent(reader, 'loadeddata', () => { reader.src = url; });
    for (let i = 0; i < 7 && token === revision; i++) {
      await mediaEvent(reader, 'seeked', () => { reader.currentTime = duration * (i + .5) / 7; });
      if (token !== revision) break;
      const frame = document.createElement('canvas');
      frame.width = 160;
      frame.height = 90;
      frame.getContext('2d').drawImage(reader, 0, 0, 160, 90);
      film.append(frame);
    }
  } catch {
    // Timeline seeking and event markers still work without thumbnails.
  } finally {
    reader.removeAttribute('src');
    reader.load();
  }
}

async function eventImages(record, run) {
  if (run !== archiveRun) return;
  const reader = document.createElement('video');
  reader.muted = true;
  try {
    await mediaEvent(reader, 'loadeddata', () => { reader.src = '/api/jobs/' + record.id + '/video'; });
    const canvas = document.createElement('canvas');
    canvas.width = 144;
    canvas.height = Math.max(1, Math.round(144 * reader.videoHeight / reader.videoWidth));
    for (const event of record.events) {
      if (run !== archiveRun) break;
      const time = Math.min(event.time, Math.max(0, reader.duration - .05));
      if (Math.abs(reader.currentTime - time) > .001) {
        await mediaEvent(reader, 'seeked', () => { reader.currentTime = time; });
      }
      canvas.getContext('2d').drawImage(reader, 0, 0, canvas.width, canvas.height);
      event.still = canvas.toDataURL('image/jpeg', .78);
      if (event.image) event.image.src = event.still;
    }
  } catch {
    // Local timestamps and exports remain available if a thumbnail cannot be decoded.
  } finally {
    reader.removeAttribute('src');
    reader.load();
  }
}

for (const tab of ['events', 'files']) {
  const button = $(tab + '-tab');
  button.onclick = () => switchTab(tab);
  button.onkeydown = event => {
    if (!['ArrowRight', 'ArrowLeft', 'Home', 'End'].includes(event.key)) return;
    event.preventDefault();
    const next = event.key === 'Home' ? 'events' : event.key === 'End' ? 'files' : tab === 'files' ? 'events' : 'files';
    switchTab(next);
    $(next + '-tab').focus();
  };
}

function markCursor() {
  if (!job || !Number.isFinite(video.duration)) return;
  userNavigated = true;
  video.pause();
  previewing = false;
  selected?.button?.setAttribute('aria-pressed', 'false');
  selected = null;
  manualTime = Math.min(job.duration, Math.max(0, video.currentTime));
  $('clip-settings').open = false;
  renderClip();
  $('preview-clip').focus();
}
$('manual').onclick = $('use-cursor').onclick = markCursor;

$('preview-clip').onclick = async () => {
  if (!job || !selection()) return;
  userNavigated = true;
  if (previewing) {
    video.pause();
    previewing = false;
    renderClip();
    return;
  }
  previewing = true;
  renderClip();
  seek(selection()[0]);
  const token = revision;
  try { await video.play(); } catch (error) {
    if (token !== revision || error.name === 'AbortError') return;
    previewing = false;
    renderClip();
    notify('Браузер не смог воспроизвести фрагмент.', true);
  }
};

$('upload').onclick = () => $('file').click();
$('demo').onclick = () => start();
$('open-folder').onclick = () => $('folder').click();
$('file').onchange = () => {
  if ($('file').files.length) start($('file').files);
  $('file').value = '';
};
$('folder').onchange = () => {
  const files = $('folder').files;
  if (files.length) start(files, files[0].webkitRelativePath.split('/')[0]);
  $('folder').value = '';
};
document.addEventListener('dragover', event => {
  if (!event.dataTransfer.types.includes('Files')) return;
  event.preventDefault();
  document.body.classList.add('dragging');
});
document.addEventListener('dragleave', event => {
  if (!event.relatedTarget) document.body.classList.remove('dragging');
});
document.addEventListener('drop', event => {
  document.body.classList.remove('dragging');
  if (!event.dataTransfer.files.length) return;
  event.preventDefault();
  start(event.dataTransfer.files);
});
$('seek').oninput = () => seek(Number($('seek').value), true);
$('play').onclick = () => {
  userNavigated = true;
  if (video.paused) video.play().catch(() => notify('Браузер не смог воспроизвести запись.', true));
  else video.pause();
};
$('back').onclick = () => seek(video.currentTime - 5, true);
$('forward').onclick = () => seek(video.currentTime + 5, true);
$('mute').onclick = () => { video.muted = !video.muted; };
$('full').onclick = () => video.requestFullscreen?.().catch(() => notify('Полноэкранный режим недоступен.', true));
video.onplay = () => icon('play', 'pause', 'Пауза');
video.onpause = () => {
  icon('play', 'play', 'Воспроизвести');
  if (video.paused && previewing) {
    previewing = false;
    renderClip();
  }
};
video.onvolumechange = () => icon('mute', video.muted ? 'muted' : 'sound', video.muted ? 'Включить звук' : 'Выключить звук');
video.ontimeupdate = () => {
  if (previewing && video.currentTime >= selection()[1]) {
    const end = selection()[1];
    previewing = false;
    video.pause();
    seek(end);
    renderClip();
  }
  sync();
};
video.onerror = () => { if (job) notify('Не удалось воспроизвести запись.', true); };
icon('play', 'play', 'Воспроизвести');
icon('mute', 'sound', 'Выключить звук');
for (const id of ['before', 'after']) $(id).oninput = () => {
  if (previewing) video.pause();
  renderClip();
};
function updateMode() {
  const mode = $('mode').value;
  $('mode-hint').textContent = mode === 'codex'
    ? 'Кадры отправляются в OpenAI через ваш аккаунт Codex.'
    : mode === 'ru' ? 'Поиск на компьютере' : 'Обработка на компьютере';
}
$('mode').onchange = () => {
  try { localStorage.setItem('roadframe-mode', $('mode').value); } catch {}
  updateMode();
};
let preferredMode = new URLSearchParams(location.search).get('mode');
if (!preferredMode) {
  try { preferredMode = localStorage.getItem('roadframe-mode'); } catch {}
}
if (['motion','ru','codex'].includes(preferredMode)) $('mode').value = preferredMode;
updateMode();

$('export').onclick = async () => {
  if (!job || exporting || !selection()) return;
  const source = job;
  const anchor = selected?.time ?? manualTime;
  const [start, end] = selection();
  const kind = selected ? (selected.status === 'visible_consequences' ? 'последствия' : 'ДТП') : 'фрагмент';
  exporting = true;
  renderClip();
  notify('');
  try {
    const response = await fetch('/api/jobs/' + source.id + '/clip?' + new URLSearchParams({ start, end }));
    if (!response.ok) await json(response);
    const url = URL.createObjectURL(await response.blob());
    const link = document.createElement('a');
    link.href = url;
    link.download = source.name.replace(/\.[^.]+$/, '') + '-' + kind + '-' + preciseTime(anchor).replaceAll(':', '-') + '.mp4';
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 30000);
  } catch (error) {
    notify(error.message, true);
  } finally {
    exporting = false;
    renderClip();
  }
};
