const $ = id => document.getElementById(id);
const video = $('video');
const icons = {
  play: '<path d="m8 5 12 7-12 7z" fill="currentColor" stroke="none"/>',
  pause: '<path d="M9 5v14m6-14v14" stroke-width="2.5"/>',
  sound: '<path d="M4 10v4h4l5 4V6l-5 4zm12-1a4 4 0 0 1 0 6m2-8a7 7 0 0 1 0 10"/>',
  muted: '<path d="M4 10v4h4l5 4V6l-5 4zm12 0 4 4m0-4-4 4"/>',
};
let job, selected = null, busy = false, revision = 0, exporting = false;
let archive = [], failures = [], duration = 0;
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
  const tenths = Math.round(seconds * 10);
  return clock(tenths / 10) + ',' + String(tenths % 10);
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
  const endpoint = '/api/jobs' + (file ? '' : '?demo=1&part=' + part);
  const { id } = await json(await fetch(endpoint, { method: 'POST', body }));
  while (true) {
    const result = await json(await fetch('/api/jobs/' + id, { cache: 'no-store' }));
    if (result.status === 'error') throw new Error(result.error);
    if (result.status === 'ready') return result;
    $('progress-label').textContent = (index + 1) + ' из ' + total + ' · ' + result.phase;
    $('progress').value = (index + result.progress / 100) / total * 100;
    await new Promise(resolve => setTimeout(resolve, 450));
  }
}

async function start(files, folderName = '') {
  if (busy) return;
  const demo = !files;
  const pathOf = file => file.webkitRelativePath || file.name;
  const inputs = demo ? [1, 2, 3] : Array.from(files)
    .filter(file => videoName.test(file.name)).sort((a, b) => order.compare(pathOf(a), pathOf(b)));
  if (!inputs.length) return notify('В выбранной папке нет поддерживаемых видео.', true);
  if (inputs.length > 100) return notify('В одном архиве можно открыть до 100 видео. Выберите часть файлов.', true);
  if (!demo && inputs.reduce((sum, file) => sum + file.size, 0) > 20 * 1024 ** 3) {
    return notify('Архив превышает 20 ГБ. Выберите часть файлов.', true);
  }
  busy = true;
  $('upload').disabled = $('demo').disabled = $('open-folder').disabled = true;
  $('progress-box').hidden = false;
  notify('');
  archive = [];
  failures = [];
  duration = 0;
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
  $('archive-name').textContent = demo ? 'Зимняя поездка в Козе' : (folderName || 'Выбранные записи');
  $('subtitle').textContent = demo ? 'Демонстрационный архив' : 'Архив регистратора';
  renderArchive();
  try {
    for (const [index, input] of inputs.entries()) {
      const name = demo ? 'Козе, часть ' + input : pathOf(input);
      $('progress-label').textContent = (index + 1) + ' из ' + inputs.length + ' · Читаем ' + name;
      $('progress').value = index / inputs.length * 100;
      try {
        if (!demo && input.size > 1024 ** 3) throw new Error('Файл больше 1 ГБ.');
        const result = await analyzeFile(demo ? null : input, input, index, inputs.length);
        result.path = demo ? result.name : (folderName ? name.slice(folderName.length + 1) : name);
        result.offset = duration;
        duration += result.duration;
        archive.push(result);
        if (!job) show(result, result.events[0]);
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
    $('progress-box').hidden = true;
    $('upload').disabled = $('demo').disabled = $('open-folder').disabled = false;
    busy = false;
  }
}

function renderArchive() {
  $('archive-info').textContent = archive.length + ' ' + fileWord(archive.length) + ' · ' + clock(duration);
  $('files').replaceChildren(...archive.map(record => {
    const button = document.createElement('button');
    button.className = 'file-segment';
    button.setAttribute('aria-pressed', String(record === job));
    button.title = record.path;
    button.onclick = () => show(record);
    const label = document.createElement('strong');
    label.textContent = record.path;
    const position = document.createElement('span');
    position.textContent = clock(record.offset) + ' – ' + clock(record.offset + record.duration);
    const icon = document.createElement('span');
    icon.className = 'file-icon';
    icon.setAttribute('aria-hidden', 'true');
    icon.innerHTML = '<svg viewBox="0 0 24 24"><rect x="3.5" y="5.5" width="17" height="13" rx="2"/><path d="m10 9 5 3-5 3z"/></svg>';
    const copy = document.createElement('span');
    copy.className = 'file-copy';
    copy.append(label, position);
    button.append(icon, copy);
    if (record.events.length) {
      const count = document.createElement('span');
      count.className = 'file-count';
      count.setAttribute('aria-label', 'Отметок: ' + record.events.length);
      count.textContent = record.events.length;
      button.append(count);
    }
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

function fileWord(count) {
  if (count % 100 >= 11 && count % 100 <= 14) return 'файлов';
  if (count % 10 === 1) return 'файл';
  if (count % 10 >= 2 && count % 10 <= 4) return 'файла';
  return 'файлов';
}

function show(result, event = null, autoplay = false) {
  if (job === result) {
    select(event);
    return;
  }
  job = result;
  selected = event;
  const token = ++revision;
  $('workspace').hidden = false;
  $('workspace').classList.add('has-video');
  $('empty').hidden = true;
  $('filename').textContent = job.path;
  $('filename').title = job.path;
  $('file-info').textContent = 'Файл ' + (archive.indexOf(job) + 1) + ' · ' + clock(job.duration);
  $('seek').max = job.duration;
  $('seek').value = 0;
  $('ticks').replaceChildren(...Array.from({ length: 5 }, (_, index) => {
    const tick = document.createElement('span');
    tick.textContent = clock(job.duration * index / 4);
    return tick;
  }));
  renderArchive();
  renderEvents();
  renderMarkers();
  const url = '/api/jobs/' + job.id + '/video';
  video.pause();
  video.onloadedmetadata = () => {
    if (token !== revision) return;
    if (selected) seek(selected.time);
    else sync();
    if (autoplay) video.play().catch(() => notify('Браузер не смог воспроизвести запись.', true));
  };
  video.src = url;
  thumbnails(url, job.duration, token);
  sync(event?.time ?? 0);
}

function renderEvents() {
  const rows = archive.flatMap(record => [...record.events].sort((a, b) => a.time - b.time)
    .map(event => ({record, event})));
  $('events').replaceChildren();
  $('count').textContent = String(rows.length);
  if (!rows.length) {
    const p = document.createElement('p');
    p.className = 'empty-events';
    p.textContent = 'Признаков столкновения не найдено. Запись можно проверить вручную.';
    $('events').append(p);
  }
  for (const {record, event} of rows) {
    const button = document.createElement('button');
    button.className = 'event';
    button.setAttribute('aria-pressed', String(event === selected));
    const time = document.createElement('time');
    time.textContent = clock(record.offset + event.time);
    const description = document.createElement('span');
    description.className = 'event-description';
    const name = document.createElement('strong');
    name.textContent = event.signal;
    const source = document.createElement('span');
    source.textContent = record.path + ' · ' + preciseTime(event.time);
    source.title = record.path;
    description.append(name, source);
    const arrow = document.createElement('span');
    arrow.textContent = '›';
    arrow.className = 'chevron';
    arrow.setAttribute('aria-hidden', 'true');
    button.append(time, description, arrow);
    button.onclick = () => show(record, event);
    event.button = button;
    $('events').append(button);
  }
}

function renderMarkers() {
  $('markers').replaceChildren();
  for (const item of job.events) {
    const marker = document.createElement('button');
    marker.className = 'marker';
    marker.style.left = item.time / job.duration * 100 + '%';
    marker.title = preciseTime(item.time) + ' · ' + item.signal;
    marker.setAttribute('aria-label', marker.title);
    marker.onclick = () => select(item);
    $('markers').append(marker);
  }
}

function select(item) {
  selected?.button?.setAttribute('aria-pressed', 'false');
  selected = item;
  selected?.button?.setAttribute('aria-pressed', 'true');
  video.pause();
  if (item) seek(item.time);
  else sync();
}

function seek(seconds, manual = false) {
  if (!job || !Number.isFinite(video.duration)) return;
  if (manual && selected) {
    selected.button.setAttribute('aria-pressed', 'false');
    selected = null;
  }
  video.currentTime = Math.max(0, Math.min(video.duration, seconds));
  sync(video.currentTime);
}

function selection(seconds) {
  const anchor = selected?.time ?? Math.min(job.duration, Math.max(0, seconds));
  const before = Math.min(60, Math.max(0, Number($('before').value) || 0));
  const after = Math.min(60, Math.max(1, Number($('after').value) || 1));
  const end = Math.min(job.duration, anchor + after);
  return [Math.min(Math.max(0, anchor - before), Math.max(0, end - 1)), end];
}

function sync(seconds = video.currentTime) {
  if (!job) return;
  $('clock').textContent = clock(seconds) + ' / ' + clock(job.duration);
  $('needle').style.left = Math.min(100, seconds / job.duration * 100) + '%';
  $('seek').value = seconds;
  $('seek').setAttribute('aria-valuetext', clock(seconds));
  const [start, end] = selection(seconds);
  $('clip-range').style.left = (start / job.duration * 100) + '%';
  $('clip-range').style.width = ((end - start) / job.duration * 100) + '%';
  $('range-label').textContent = clock(start) + ' — ' + clock(end) + ' · ' + Math.round(end - start) + ' сек.';
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
$('play').onclick = () => video.paused ? video.play().catch(() => notify('Браузер не смог воспроизвести запись.', true)) : video.pause();
$('back').onclick = () => seek(video.currentTime - 5, true);
$('forward').onclick = () => seek(video.currentTime + 5, true);
$('mute').onclick = () => { video.muted = !video.muted; };
$('full').onclick = () => video.requestFullscreen?.().catch(() => notify('Полноэкранный режим недоступен.', true));
video.onplay = () => icon('play', 'pause', 'Пауза');
video.onpause = () => icon('play', 'play', 'Воспроизвести');
video.onended = () => {
  video.onpause();
  const next = job && archive[archive.indexOf(job) + 1];
  if (next) show(next, null, true);
};
video.onvolumechange = () => icon('mute', video.muted ? 'muted' : 'sound', video.muted ? 'Включить звук' : 'Выключить звук');
video.ontimeupdate = () => sync();
video.onerror = () => { if (job) notify('Не удалось воспроизвести запись.', true); };
icon('play', 'play', 'Воспроизвести');
icon('mute', 'sound', 'Выключить звук');
for (const id of ['before', 'after']) $(id).oninput = () => sync();
$('export').onclick = async () => {
  if (!job || exporting) return;
  exporting = true;
  const button = $('export');
  const original = button.innerHTML;
  button.disabled = true;
  button.textContent = 'Готовим отрывок…';
  notify('');
  try {
    const source = job;
    const [start, end] = selection(video.currentTime);
    const response = await fetch('/api/jobs/' + source.id + '/clip?' + new URLSearchParams({ start, end }));
    if (!response.ok) await json(response);
    const url = URL.createObjectURL(await response.blob());
    const link = document.createElement('a');
    link.href = url;
    link.download = source.name.replace(/\.[^.]+$/, '') + '-' + clock(start).replaceAll(':', '-') + '.mp4';
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 30000);
  } catch (error) {
    notify(error.message, true);
  } finally {
    button.innerHTML = original;
    button.disabled = false;
    exporting = false;
  }
};
