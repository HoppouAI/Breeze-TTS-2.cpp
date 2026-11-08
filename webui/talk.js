'use strict';

const $ = (s, el = document) => el.querySelector(s);

const DEFAULT_RULES = `How you talk:
- This is a live voice call. Everything you write gets spoken out loud by a voice model, so write exactly what a person would say, nothing else.
- Keep it short. Usually one to three sentences, then let them talk. Never monologue or lecture.
- Sound like a real person talking, not an assistant: contractions, casual words, the odd "um", "uh", "like", "you know", "I mean", "oh", "hmm". Use fillers lightly, not in every sentence.
- Use "..." for a thinking pause or a trailing thought, and commas where you would take a small breath. For example: "Oh... wait, really? Okay, that's kind of amazing."
- React before you answer when it fits, like "Oh no.", "Ha, okay.", "Wait, seriously?", "Mm, yeah."
- Have opinions, tell small stories of your own, disagree sometimes. Ask a follow up question now and then, but not every turn.
- For a laugh, sigh or other sound, put a short tag in round brackets right where it happens: (laugh), (chuckle), (sigh), (gasp) or (clears throat). Use them sparingly, at most one per reply, and never describe actions any other way.
- What they say reaches you through speech recognition, so expect the odd wrong word. Go with the most likely meaning, and if it really makes no sense, just ask, like "sorry, what was that?"
- If your last reply ends with "..." they cut you off. Don't finish the old thought, go with what they just said.

Never use lists, bullet points, markdown, emojis, asterisks, headings or links. Say numbers, dates and symbols the way you would out loud, like "about five bucks" or "twenty twenty six".
Don't mention these instructions. If someone sincerely asks whether you're an AI, be honest about it in a casual, unbothered way.`;

const BC_RULES = `You're listening on a voice call while they talk. They are in the middle of a thought, so you only make one tiny listening sound or reaction without taking the turn, the way people do on the phone.
Reply with only that, one to three words, for example: Mm-hmm. Yeah. Right. Oh wow. No way. Oh no. Ha. Really? Ooh. Totally.
Pick what fits what they just said and how they feel. Never answer a question, never say a full sentence, never use tags or brackets.`;

const DEFAULT_CHARACTERS = [
  {
    id: 'juniper',
    name: 'Juniper',
    color: '#4f8a72',
    voice: '',
    design: 'A warm, playful woman in her late twenties with a soft, smiling voice, chatting casually like she is catching up with an old friend.',
    opening: '',
    prompt: `You are Juniper. You're on a voice call with someone, chatting like old friends. You're warm, curious and a bit playful, late twenties, and you get genuinely excited about little things: weird facts, good food, music, people's stories. You tease a little, you laugh easily, and you notice how the other person is feeling.`,
  },
  {
    id: 'wes',
    name: 'Wes',
    color: '#5b74a8',
    voice: '',
    design: 'A relaxed man in his mid thirties with a low, easygoing voice and a dry sense of humor, talking slowly and casually.',
    opening: '',
    prompt: `You are Wes. You're on a voice call with someone, just hanging out and talking. You're laid back, mid thirties, with a dry sense of humor and a lot of patience. You're a good listener, you think before you answer, and you like going off on small tangents about cooking, old movies and how things work.`,
  },
  {
    id: 'ivy',
    name: 'Ivy',
    color: '#a8664f',
    voice: '',
    design: 'A friendly young woman in her twenties with a bright, warm voice, speaking casually and naturally, like talking to a close friend on the phone.',
    opening: '',
    prompt: `You are Ivy. You're on a voice call with someone you get along with really well. You're upbeat, quick witted and a little dramatic in a fun way, mid twenties, into games, music and terrible reality TV. You laugh a lot, you hype people up, and you're honest when something sounds like a bad idea.`,
  },
];

const DEFAULTS = {
  characters: DEFAULT_CHARACTERS,
  seen: DEFAULT_CHARACTERS.map((c) => c.id),
  v: 2,
  rules: DEFAULT_RULES,
  llm: { model: '', temperature: 0.8, max_tokens: 220, no_think: true, key: '', history: 30 },
  turn: {
    ep_thr: 0.915, interrupt: true, barge_thr: 0.5, barge_after: 1.0,
    backchannel: true, bc_thr: 0.4, bc_gap: 6, bc_phrases: 'Mm-hmm., Yeah., Right., Uh-huh., Mhm.', bc_llm: false,
    fast_start: true,
  },
  voice: { cfg: 1.0, temperature: 0 },
  screen: { size: 1024 },
  ui: { captions: false, timings: false },
};

const SCREEN_NOTE = `They are sharing their screen with you. The picture with their latest message is what is on it right now, so you can see it. Bring it up when it fits, the way a friend looking over their shoulder would, and keep it short. Never read long text off the screen word for word.`;

const KICKOFF = '[The call just connected. Pick up the way you would when a friend calls, short and natural.]';

// what someone says to show they are listening, not to take the turn
const LISTENING_WORDS = new Set(['mm', 'mmm', 'mhm', 'mmhmm', 'mmhm', 'hmm', 'uhhuh', 'uh', 'huh', 'yeah', 'yep', 'yes', 'yup', 'right', 'okay', 'ok', 'sure', 'wow', 'oh', 'ah', 'totally', 'true', 'nice', 'cool', 'exactly', 'definitely']);

function justListening(text) {
  const words = text.toLowerCase().replace(/-/g, '').replace(/[^a-z\s]/g, ' ').split(/\s+/).filter(Boolean);
  return words.length <= 4 && words.every((w) => LISTENING_WORDS.has(w));
}

const LOOPBACK = new Set(['127.0.0.1', 'localhost', '0.0.0.0', '[::1]', '[::]']);
const isLoopback = (h) => LOOPBACK.has(h) || h.endsWith('.localhost');

// a loopback address from the server means its own machine, which is wherever this page came from
function onPageHost(url) {
  try {
    const u = new URL(url);
    if (isLoopback(u.hostname) && !isLoopback(location.hostname)) u.hostname = location.hostname;
    if (location.protocol === 'https:' && u.protocol === 'ws:') u.protocol = 'wss:';
    return u.href;
  } catch {
    return url;
  }
}

function resumeNote(text, rest) {
  return '[They only said "' + (text || 'mm-hmm') + '" to show they are listening, not to take over. Carry on from where you got cut off, without repeating what you already said.' +
    (rest ? ' You were in the middle of saying: "' + rest + '"' : '') + ']';
}
const STORE_KEY = 'breeze.talk';

function clone(o) { return JSON.parse(JSON.stringify(o)); }

function loadSettings() {
  let s = null;
  try { s = JSON.parse(localStorage.getItem(STORE_KEY) || 'null'); } catch { s = null; }
  const d = clone(DEFAULTS);
  if (!s) return d;
  const chars = Array.isArray(s.characters) && s.characters.length ? s.characters : d.characters;
  // new default characters show up once, ones the user deleted stay gone
  const seen = Array.isArray(s.seen) ? s.seen : chars.map((c) => c.id);
  for (const dc of d.characters) if (!seen.includes(dc.id) && !chars.some((c) => c.id === dc.id)) chars.push(dc);
  return {
    characters: chars,
    seen: d.seen,
    rules: typeof s.rules === 'string' && s.rules.trim() ? s.rules : d.rules,
    llm: { ...d.llm, ...s.llm },
    v: d.v,
    // 0.4 let a plain "mm-hmm" stop the voice, longer interruptions get caught by talking over it now
    turn: { ...d.turn, ...s.turn, ...(!(s.v >= 2) && s.turn && s.turn.barge_thr === 0.4 ? { barge_thr: d.turn.barge_thr } : {}) },
    voice: { ...d.voice, ...s.voice },
    screen: { ...d.screen, ...s.screen },
    ui: { ...d.ui, ...s.ui },
  };
}

function saveSettings() {
  try { localStorage.setItem(STORE_KEY, JSON.stringify(S)); } catch { /* private window */ }
}

let S = loadSettings();
let CFG = { ws_port: 8081, tyger: 'ws://127.0.0.1:8765/ws', sample_rate: 24000 };
let voices = [];
let models = [];
let lastCharacter = null;
let call = null;

function toast(msg, ms = 6000) {
  const t = $('#toast');
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { t.hidden = true; }, ms);
}

function show(view) {
  for (const v of ['pick', 'connect', 'call', 'ended']) $('#view-' + v).hidden = v !== view;
}

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
}

const PHONE = '<svg viewBox="0 0 20 20" aria-hidden="true"><path fill="currentColor" d="M6.6 2.5c.4 0 .8.3.9.7l.8 3a1 1 0 0 1-.3 1L6.7 8.4a10.6 10.6 0 0 0 4.9 4.9l1.2-1.3a1 1 0 0 1 1-.3l3 .8c.4.1.7.5.7.9v2.8c0 .6-.5 1.1-1.1 1.1A14.4 14.4 0 0 1 2.5 3.6c0-.6.5-1.1 1.1-1.1h3Z"/></svg>';

function renderTiles() {
  const box = $('#tiles');
  box.innerHTML = '';
  for (const ch of S.characters) {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'tile';
    b.style.background = shade(ch.color, -0.35);
    b.innerHTML = PHONE + '<span>' + esc(ch.name) + '</span>';
    b.onclick = () => startCall(ch);
    box.appendChild(b);
  }
}

function shade(hex, amt) {
  const n = parseInt(hex.slice(1), 16);
  const f = (c) => Math.max(0, Math.min(255, Math.round(amt < 0 ? c * (1 + amt) : c + (255 - c) * amt)));
  const r = f(n >> 16), g = f((n >> 8) & 255), b = f(n & 255);
  return '#' + ((1 << 24) | (r << 16) | (g << 8) | b).toString(16).slice(1);
}

// the browser has the mic at the context rate, tyger wants 16 khz, this decimates by 3 with a small fir
const WORKLET = `
const TAPS = (() => {
  const n = 31, fc = 7000 / 48000, t = [];
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const x = i - (n - 1) / 2;
    const sinc = x === 0 ? 2 * fc : Math.sin(2 * Math.PI * fc * x) / (Math.PI * x);
    const w = 0.54 - 0.46 * Math.cos(2 * Math.PI * i / (n - 1));
    t.push(sinc * w);
    sum += sinc * w;
  }
  return t.map((v) => v / sum);
})();
class TalkIO extends AudioWorkletProcessor {
  constructor() {
    super();
    this.h = [new Float32Array(TAPS.length), new Float32Array(TAPS.length)];
    this.pos = 0;
    this.phase = 0;
    this.out = new Float32Array(640);
    this.n = 0;
    this.muted = false;
    this.port.onmessage = (e) => { if ('mute' in e.data) this.muted = e.data.mute; };
  }
  process(inputs) {
    const u = inputs[0] && inputs[0][0];
    const a = inputs[1] && inputs[1][0];
    const len = (u && u.length) || (a && a.length) || 128;
    const N = TAPS.length;
    for (let i = 0; i < len; i++) {
      this.h[0][this.pos] = u && !this.muted ? u[i] : 0;
      this.h[1][this.pos] = a ? a[i] : 0;
      this.pos = (this.pos + 1) % N;
      if (++this.phase < 3) continue;
      this.phase = 0;
      let su = 0, sa = 0;
      for (let k = 0; k < N; k++) {
        const j = (this.pos + k) % N;
        su += this.h[0][j] * TAPS[k];
        sa += this.h[1][j] * TAPS[k];
      }
      this.out[this.n * 2] = su;
      this.out[this.n * 2 + 1] = sa;
      if (++this.n === 320) {
        this.port.postMessage(this.out.slice(0));
        this.n = 0;
      }
    }
    return true;
  }
}
registerProcessor('talk-io', TalkIO);

// one continuous stream for the voice: chunks just append, so there are no buffer edges to click on
class TalkPlay extends AudioWorkletProcessor {
  constructor(opts) {
    super();
    this.rate = opts.processorOptions.rate;
    this.cap = this.rate * 120;
    this.buf = new Float32Array(this.cap);
    this.w = 0;
    this.r = 0;
    this.frac = 0;
    this.step = this.rate / sampleRate;
    this.pre = Math.round(0.12 * this.rate);
    this.playing = false;
    this.stopping = false;
    this.gain = 0;
    this.ramp = 1 / (0.006 * sampleRate);
    this.last = 0;
    this.blocks = 0;
    this.port.onmessage = (e) => {
      const d = e.data;
      if (d.pcm) {
        for (let i = 0; i < d.pcm.length; i++) this.buf[(this.w + i) % this.cap] = d.pcm[i];
        this.w += d.pcm.length;
      }
      if (d.prebuffer !== undefined) this.pre = Math.round(d.prebuffer * this.rate);
      if (d.stop) this.stopping = this.playing;
      if (d.stop && !this.playing) this.r = this.w;
    };
  }
  at(k) { return this.buf[((k % this.cap) + this.cap) % this.cap]; }
  process(_, outputs) {
    const out = outputs[0][0];
    for (let i = 0; i < out.length; i++) {
      if (!this.playing && !this.stopping && this.w - this.r >= Math.max(this.pre, 4)) {
        this.playing = true;
        this.gain = 0;
      }
      if (this.playing && this.w - this.r < 3) {
        this.playing = false;
        this.stopping = false;
        this.port.postMessage({ underrun: this.w > this.r });
      }
      let v;
      if (this.playing) {
        const t = this.frac, a = this.at(this.r - 1), b = this.at(this.r), c = this.at(this.r + 1), d = this.at(this.r + 2);
        v = b + 0.5 * t * (c - a + t * (2 * a - 5 * b + 4 * c - d + t * (3 * (b - c) + d - a)));
        this.frac += this.step;
        while (this.frac >= 1) { this.frac -= 1; this.r++; }
        this.gain = this.stopping ? Math.max(0, this.gain - this.ramp) : Math.min(1, this.gain + this.ramp);
        v *= this.gain;
        if (this.stopping && this.gain === 0) {
          this.playing = false;
          this.stopping = false;
          this.r = this.w;
          this.frac = 0;
        }
        this.last = v;
      } else {
        // ran dry or stopped: glide whatever was last out down to silence instead of stepping to it
        this.last *= 0.992;
        v = this.last;
      }
      out[i] = v;
    }
    if (++this.blocks % 12 === 0) this.port.postMessage({ played: this.r, queued: this.w, playing: this.playing });
    return true;
  }
}
registerProcessor('talk-play', TalkPlay);
`;

function wavBytes(samples, rate) {
  const buf = new ArrayBuffer(44 + samples.length * 2);
  const v = new DataView(buf);
  const str = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  str(0, 'RIFF'); v.setUint32(4, 36 + samples.length * 2, true); str(8, 'WAVE');
  str(12, 'fmt '); v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, rate, true); v.setUint32(28, rate * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
  str(36, 'data'); v.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) v.setInt16(44 + i * 2, Math.max(-1, Math.min(1, samples[i])) * 32767, true);
  return new Uint8Array(buf);
}

// strips what the llm writes but nobody says out loud, keeps ... and short (laugh) style tags
class SpeechFilter {
  constructor() { this.star = false; this.paren = false; this.tag = ''; }
  push(t) {
    let o = '';
    for (const c of t) {
      if (c === '*') { this.star = !this.star; continue; }
      if (this.star) continue;
      if (c === '(' || c === '[') { this.paren = true; this.tag = c; continue; }
      if (this.paren) {
        if (c !== ')' && c !== ']') { this.tag += c; continue; }
        this.paren = false;
        // a short tag becomes a real sound, anything longer is narration nobody says
        if (this.tag[0] === '(' && c === ')' && this.tag.length <= 30) o += this.tag + ')';
        continue;
      }
      if ('#`_~|<>'.includes(c)) continue;
      if (c === '\u2014' || c === '\u2013') { o += ', '; continue; }
      if (c === '\u2026') { o += '...'; continue; }
      if (/\p{Extended_Pictographic}/u.test(c)) continue;
      o += c === '\n' ? ' ' : c;
    }
    return o.replace(/ {2,}/g, ' ');
  }
}

class Call {
  constructor(ch) {
    this.ch = ch;
    this.history = [];
    this.respId = 0;
    this.state = 'connecting';
    this.queued = 0;
    this.played = 0;
    this.playing = false;
    this.pieces = [];
    this.discard = false;
    this.botAudible = false;
    this.lastBc = 0;
    this.bcClips = [];
    this.muted = false;
    this.ended = false;
    this.pendingUser = '';
    this.level = { mic: 0, bot: 0 };
    this.cushion = 0.12;
    this.turnNo = 0;
    this.floor = -60;
    this.bargeAt = 0;
    this.lastVoiced = 0;
    this.prefix = '';
    this.cut = null;
    this.cutWhole = false;
    this.overTalk = 0;
    this.overQuiet = 0;
    this.cap = null;
    this.anchorVoice = '';
    this.bcBusy = false;
    this.bcInfo = '';
    this.firstAudioAt = undefined;
    this.screen = null;
    this.noVision = false;
  }

  async shareScreen() {
    if (this.screen) { this.stopScreen(); return; }
    let stream;
    try {
      stream = await navigator.mediaDevices.getDisplayMedia({ video: { frameRate: 5 }, audio: false });
    } catch {
      return;
    }
    if (this.ended) { stream.getTracks().forEach((t) => t.stop()); return; }
    const video = $('#screen-view');
    video.srcObject = stream;
    video.hidden = false;
    try { await video.play(); } catch { /* still draws once frames arrive */ }
    this.screen = { stream, video, canvas: document.createElement('canvas') };
    this.noVision = false;
    stream.getVideoTracks()[0].addEventListener('ended', () => this.stopScreen());
    screenButton(true);
  }

  stopScreen() {
    if (!this.screen) return;
    this.screen.stream.getTracks().forEach((t) => t.stop());
    this.screen.video.srcObject = null;
    this.screen.video.hidden = true;
    this.screen = null;
    screenButton(false);
  }

  // grabbed when the reply starts, so the model sees the screen as it is right now
  screenFrame() {
    if (!this.screen || this.noVision) return '';
    const { video, canvas } = this.screen;
    const w = video.videoWidth, h = video.videoHeight;
    if (!w || !h) return '';
    const k = Math.min(1, Number(S.screen.size || 1024) / Math.max(w, h));
    canvas.width = Math.round(w * k);
    canvas.height = Math.round(h * k);
    canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL('image/jpeg', 0.8);
  }

  async start() {
    if (!window.isSecureContext || !navigator.mediaDevices) {
      throw new Error('This browser only allows the microphone on https or on localhost. Open the page over https, or as http://localhost' + (location.port ? ':' + location.port : '') + '/talk on the machine running the server.');
    }
    this.ctx = new AudioContext({ sampleRate: 48000 });
    const url = URL.createObjectURL(new Blob([WORKLET], { type: 'application/javascript' }));
    await this.ctx.audioWorklet.addModule(url);
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 },
      });
    } catch (e) {
      throw new Error('Microphone access was blocked. Allow it in the browser and try again.');
    }
    const ctx = this.ctx;
    const mic = ctx.createMediaStreamSource(this.stream);
    this.playGain = ctx.createGain();
    this.playGain.connect(ctx.destination);
    this.player = new AudioWorkletNode(ctx, 'talk-play', { numberOfInputs: 0, numberOfOutputs: 1, outputChannelCount: [1], processorOptions: { rate: CFG.sample_rate } });
    this.player.port.postMessage({ prebuffer: this.cushion });
    this.player.port.onmessage = (e) => this.onPlayer(e.data);
    this.player.connect(this.playGain);
    this.bcGain = ctx.createGain();
    this.bcGain.gain.value = 0.85;
    this.bcGain.connect(ctx.destination);
    this.io = new AudioWorkletNode(ctx, 'talk-io', { numberOfInputs: 2, numberOfOutputs: 1, channelCount: 1, channelCountMode: 'explicit' });
    mic.connect(this.io, 0, 0);
    this.playGain.connect(this.io, 0, 1);
    this.bcGain.connect(this.io, 0, 1);
    const sink = ctx.createGain();
    sink.gain.value = 0;
    this.io.connect(sink).connect(ctx.destination);
    this.micAn = ctx.createAnalyser();
    this.micAn.fftSize = 512;
    mic.connect(this.micAn);
    this.botAn = ctx.createAnalyser();
    this.botAn.fftSize = 512;
    this.playGain.connect(this.botAn);
    this.bcGain.connect(this.botAn);
    this.io.port.onmessage = (e) => this.onMic(e.data);

    await Promise.all([this.openTyger(), this.openSpeech()]);
    this.t0 = performance.now();
    this.tick = setInterval(() => this.poll(), 40);
    this.state = 'listening';
    this.greet();
  }

  openTyger() {
    const t = S.turn;
    const q = new URLSearchParams({ format: 's16', channels: '2', encoder: 'exact', ep_thr: t.ep_thr, barge_thr: t.barge_thr, bc_thr: t.bc_thr });
    const url = CFG.tyger + (CFG.tyger.includes('?') ? '&' : '?') + q;
    return new Promise((resolve, reject) => {
      const ws = new WebSocket(url);
      ws.binaryType = 'arraybuffer';
      const remote = !isLoopback(location.hostname);
      const fail = () => reject(new Error('Could not reach tyger-server at ' + CFG.tyger + '. Start it with\ntyger-server -m models/tyger-f16.gguf --port 8765' + (remote ? ' --host 0.0.0.0\nso other devices can reach it.' : '')));
      ws.onerror = fail;
      ws.onmessage = (e) => {
        const m = JSON.parse(e.data);
        if (m.type === 'ready') { ws.onerror = null; this.tyger = ws; resolve(); return; }
        if (m.type === 'events') for (const ev of m.events) this.onTurnEvent(ev);
        if (m.type === 'error') toast('tyger: ' + m.message);
      };
      ws.onclose = () => { if (!this.ended && this.tyger) { toast('Lost the connection to tyger-server.'); end(); } };
    });
  }

  voiceId() { return this.ch.voice || ''; }

  startMessage() {
    return {
      type: 'start', voice_id: this.voiceId(), seed: Math.floor(Math.random() * 1e9), split_chars: 300,
      instruction: (this.ch.design || '').trim() || 'Speak clearly and naturally.',
      cfg_scale: Number(S.voice.cfg), temperature: Number(S.voice.temperature) || 0,
    };
  }

  openSpeech() {
    const url = (location.protocol === 'https:' ? 'wss://' : 'ws://') + location.hostname + ':' + CFG.ws_port + '/';
    return new Promise((resolve, reject) => {
      const ws = new WebSocket(url);
      ws.binaryType = 'arraybuffer';
      ws.onerror = () => reject(new Error('Could not open the speech websocket at ' + url));
      ws.onmessage = (e) => {
        if (typeof e.data !== 'string') { this.onAudio(e.data); return; }
        const m = JSON.parse(e.data);
        if (m.type === 'ready') {
          ws.send(JSON.stringify(this.startMessage()));
        } else if (m.type === 'started') {
          ws.onerror = null;
          this.speech = ws;
          resolve();
        } else if (m.type === 'speaking') {
          this.discard = false;
          this.pieces.push({ text: m.text, at: this.queued, rid: this.respId });
          if (S.ui.captions) $('#cap-bot').textContent = this.pieces.filter((p) => p.rid === this.respId).map((p) => p.text).join(' ');
        } else if (m.type === 'done') {
          this.speechDone = true;
        } else if (m.type === 'error') {
          if (!this.speech) reject(new Error(m.message === 'unknown voice_id' ? 'The voice "' + this.voiceId() + '" is not saved on this server. Pick another one in Settings.' : m.message));
          else toast('speech: ' + m.message);
        }
      };
      ws.onclose = () => { if (!this.ended && this.speech) { toast('Lost the connection to the speech server.'); end(); } };
    });
  }

  onMic(f) {
    if (!this.tyger || this.tyger.readyState !== 1) return;
    const pcm = new Int16Array(f.length);
    for (let i = 0; i < f.length; i++) pcm[i] = Math.max(-1, Math.min(1, f[i])) * 32767;
    this.tyger.send(pcm.buffer);
    this.trackOverTalk(f);
  }

  // tygers barge score misses short "wait, hold on"s, so talking over the voice long enough counts too
  trackOverTalk(f) {
    let e = 0;
    for (let i = 0; i < f.length; i += 2) e += f[i] * f[i];
    const db = 10 * Math.log10(e / (f.length / 2) + 1e-12);
    const dt = f.length / 2 / 16000;
    // noise floor drops right away and creeps back up about 1 db a second
    this.floor = db < this.floor ? db : this.floor + dt;
    const voiced = !this.muted && db > Math.max(-48, this.floor + 12);
    if (voiced) this.lastVoiced = performance.now();
    const limit = Number(S.turn.barge_after);
    if (!S.turn.interrupt || !(limit > 0) || this.state !== 'speaking' || !this.botAudible) {
      this.overTalk = this.overQuiet = 0;
      return;
    }
    if (voiced) { this.overTalk += dt; this.overQuiet = 0; } else if ((this.overQuiet += dt) > 0.3) this.overTalk = 0;
    if (this.overTalk >= limit) this.barge(true);
  }

  // keep asks tyger to start the new turn with the audio that led up to the barge, its own BARGE already does
  barge(keep) {
    this.overTalk = this.overQuiet = 0;
    this.cancelReply(false);
    this.bargeAt = performance.now();
    if (this.tyger && this.tyger.readyState === 1) this.tyger.send(JSON.stringify(keep ? { type: 'reset', keep: true } : { type: 'reset' }));
  }

  onAudio(buf) {
    if (this.discard || this.ended) return;
    const pcm = new Int16Array(buf);
    if (!pcm.length) return;
    const first = this.firstAudioAt === null;
    if (first) {
      this.firstAudioAt = performance.now();
      if (this.turnAt) {
        const s = (t) => (t ? ((t - this.turnAt) / 1000).toFixed(2) : '?');
        this.lastLatency = (this.firstAudioAt - this.turnAt) / 1000;
        this.timingText = 'llm first word ' + s(this.tFirstToken) + ' s, first clause ' + s(this.tFirstPiece) + ' s, voice ' + s(this.firstAudioAt) + ' s';
      }
    }
    const f = new Float32Array(pcm.length);
    for (let i = 0; i < pcm.length; i++) f[i] = pcm[i] / 32768;
    const cap = this.cap;
    if (cap && cap.rid === this.respId && cap.n < CFG.sample_rate * 20) { cap.chunks.push(f.slice()); cap.n += f.length; }
    this.player.port.postMessage({ pcm: f }, [f.buffer]);
    this.queued += pcm.length;
    if (this.state === 'thinking') this.state = 'speaking';
  }

  onPlayer(d) {
    if (d.played !== undefined) {
      this.played = d.played;
      this.playing = d.playing;
    }
    // ran dry mid reply means the gpu is busier than the cushion allows, so keep more queued from now on
    if (d.underrun && this.state === 'speaking' && !this.speechDone) {
      this.cushion = Math.min(0.5, this.cushion + 0.08);
      this.player.port.postMessage({ prebuffer: this.cushion });
    }
  }

  stopPlayback() {
    if (this.player) this.player.port.postMessage({ stop: true });
    this.playing = false;
  }

  setMode(audible) {
    if (audible === this.botAudible) return;
    this.botAudible = audible;
    if (this.tyger && this.tyger.readyState === 1) this.tyger.send(JSON.stringify({ type: 'mode', mode: audible ? 'speaking' : 'listen' }));
  }

  poll() {
    if (this.ended) return;
    // a short gap between pieces still counts as talking, flipping modes restarts tygers barge warmup
    const audible = this.playing || (this.state === 'speaking' && !this.speechDone);
    this.setMode(audible);
    // stopped for something tyger never turned into a turn, and they have gone quiet again
    const now = performance.now();
    if (this.bargeAt && this.state === 'listening' && now - this.bargeAt > 3000 && now - (this.lastVoiced || 0) > 1500) {
      this.bargeAt = 0;
      this.resume('');
    }
    if (this.state === 'speaking' && !audible && this.speechDone && !this.llmBusy && this.queued - this.played < 64) {
      this.state = 'listening';
      this.finishReply(false);
      this.makeBackchannels();
    }
    const secs = Math.floor((performance.now() - this.t0) / 1000);
    $('#call-timer').textContent = String(Math.floor(secs / 60)).padStart(2, '0') + ':' + String(secs % 60).padStart(2, '0');
    const label = { listening: 'Listening', thinking: 'Thinking', speaking: 'Speaking' }[this.state] || '';
    let status = this.muted ? 'Muted' : label;
    if (S.ui.timings && this.timingText) status += '  \u00b7  ' + this.timingText;
    if (S.ui.timings && this.bcInfo) status += '  \u00b7  ' + this.bcInfo;
    $('#call-status').textContent = status;
  }

  onTurnEvent(ev) {
    if (this.ended) return;
    if (ev.kind === 'PARTIAL') {
      this.lastPartial = ev.text || '';
      if (S.ui.captions) $('#cap-user').textContent = ev.text;
      // they kept talking before any audio came out, so start over with the whole thing
      if (this.state === 'thinking' && ev.text && ev.text.trim().length > 2 && this.firstAudioAt === null) {
        this.pendingUser = (this.lastUserText || '').trim();
        this.cancelReply(true);
      }
    } else if (ev.kind === 'ENDPOINT') {
      this.turnNo++;
      this.lastPartial = '';
      const text = (ev.transcript || '').trim();
      // a barge that turned out to be an "mm-hmm" or nothing at all, so pick the story back up
      if (this.bargeAt) {
        this.bargeAt = 0;
        if (!text || justListening(text)) {
          this.resume(text);
          return;
        }
      }
      this.cut = null;
      if (!text) return;
      if (this.state === 'thinking' && this.firstAudioAt === null && this.replyUser !== undefined) {
        if (!this.pendingUser && this.lastUserText) this.pendingUser = this.lastUserText;
        this.cancelReply(true);
      } else if (this.state === 'speaking') {
        this.cancelReply(false);
      }
      const full = (this.pendingUser ? this.pendingUser + ' ' : '') + text;
      this.pendingUser = '';
      if (S.ui.captions) $('#cap-user').textContent = full;
      this.turnAt = performance.now();
      this.reply(full);
    } else if (ev.kind === 'BARGE') {
      if (!S.turn.interrupt || this.state !== 'speaking') return;
      this.barge(false);
    } else if (ev.kind === 'BACKCHANNEL') {
      this.backchannel(ev);
    }
  }

  greet() {
    const opening = (this.ch.opening || '').trim();
    if (opening) {
      this.turnAt = null;
      this.speakFixed(opening);
      return;
    }
    this.turnAt = null;
    this.reply(KICKOFF);
  }

  speakFixed(text) {
    this.respId++;
    this.state = 'thinking';
    this.firstAudioAt = null;
    this.speechDone = false;
    this.llmBusy = false;
    this.replyText = text;
    this.replyUser = KICKOFF;
    this.startCapture();
    this.speech.send(JSON.stringify({ type: 'end', text }));
  }

  // with no saved voice, the first reply heard in full becomes the voice for backchannels
  startCapture() {
    this.cap = !this.ch.voice && !this.anchorVoice && S.turn.backchannel ? { rid: this.respId, base: this.queued, chunks: [], n: 0 } : null;
  }

  async reply(userText) {
    if (this.llmBusy || this.state !== 'listening') this.cancelReply(false);
    const rid = ++this.respId;
    this.state = 'thinking';
    this.firstAudioAt = null;
    this.speechDone = false;
    this.llmBusy = true;
    this.replyText = '';
    this.tFirstToken = 0;
    this.tFirstPiece = 0;
    this.replyUser = userText;
    this.lastUserText = userText.startsWith('[') ? '' : userText;
    this.startCapture();
    const ctrl = new AbortController();
    this.abort = ctrl;
    const hist = this.history.slice(-Number(S.llm.history || 30));
    while (hist.length && hist[0].role !== 'user') hist.shift();
    const img = this.screenFrame();
    const system = (this.ch.prompt || '').trim() + '\n\n' + (S.rules || '') + (img ? '\n\n' + SCREEN_NOTE : '');
    // only the newest turn carries a picture, older ones stay text so the context doesnt balloon
    const content = img ? [{ type: 'text', text: userText }, { type: 'image_url', image_url: { url: img } }] : userText;
    const messages = [{ role: 'system', content: system }, ...hist, { role: 'user', content }];
    const body = { messages, stream: true, temperature: Number(S.llm.temperature), max_tokens: Number(S.llm.max_tokens) };
    let retry = false;
    if (S.llm.model) body.model = S.llm.model;
    if (S.llm.no_think) body.reasoning_effort = 'none';
    const headers = { 'Content-Type': 'application/json' };
    if (S.llm.key) headers.Authorization = 'Bearer ' + S.llm.key;
    const filter = new SpeechFilter();
    let sent = false, buf = '', raw = '';
    try {
      const res = await fetch('/llm/chat/completions', { method: 'POST', headers, body: JSON.stringify(body), signal: ctrl.signal });
      if (!res.ok) throw new Error('the llm answered ' + res.status);
      const reader = res.body.getReader();
      const dec = new TextDecoder();
      let carry = '';
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        carry += dec.decode(value, { stream: true });
        const lines = carry.split('\n');
        carry = lines.pop();
        for (const line of lines) {
          const l = line.trim();
          if (!l.startsWith('data:')) {
            raw += l;
            continue;
          }
          const data = l.slice(5).trim();
          if (data === '[DONE]') continue;
          let j;
          try { j = JSON.parse(data); } catch { continue; }
          if (j.error) throw new Error(j.error.message || 'llm error');
          const delta = j.choices && j.choices[0] && j.choices[0].delta && j.choices[0].delta.content;
          if (!delta || rid !== this.respId) continue;
          const clean = filter.push(delta);
          this.replyText += clean;
          if (!this.tFirstToken) this.tFirstToken = performance.now();
          if (!sent) {
            buf += clean;
            // the first clause goes out on its own so the voice can start before the sentence is done
            const m = S.turn.fast_start ? buf.match(/^(.{8,}?[,;:!?.])\s/) : buf.match(/^(.+?[!?.])\s/);
            if (m) {
              this.tFirstPiece = performance.now();
              this.speech.send(JSON.stringify({ type: 'flush', text: m[1] }));
              this.speech.send(JSON.stringify({ type: 'text', text: buf.slice(m[1].length) }));
              sent = true;
            }
          } else {
            this.speech.send(JSON.stringify({ type: 'text', text: clean }));
          }
        }
      }
      if (rid !== this.respId) return;
      raw += carry;
      if (!this.replyText.trim() && raw.includes('error')) {
        let msg = raw;
        try { const j = JSON.parse(raw); msg = (j.error && (j.error.message || j.error)) || raw; } catch { /* not json */ }
        throw new Error(String(msg).slice(0, 300));
      }
      this.speech.send(JSON.stringify({ type: 'end', text: sent ? '' : buf }));
      if (!this.replyText.trim()) { this.state = 'listening'; this.speechDone = true; }
    } catch (e) {
      if (e.name === 'AbortError' || rid !== this.respId) return;
      if (img && !this.replyText.trim() && !/could not reach/.test(e.message)) {
        this.noVision = true;
        retry = true;
        toast('The model could not take the screen picture, so it answered without seeing it. Load a vision model in your llm server to use screen sharing.\n' + e.message, 9000);
      } else {
        toast('Could not get a reply: ' + e.message + '\nIs the llm server running, and is --llm pointing at it?');
      }
      this.state = 'listening';
    } finally {
      if (rid === this.respId) this.llmBusy = false;
    }
    if (retry && rid === this.respId && !this.ended) this.reply(userText);
  }

  // what actually got heard goes into the history, a cut off reply ends in ... so the llm knows
  finishReply(cut) {
    if (this.replyUser === undefined) return;
    const now = this.played;
    const mine = this.pieces.filter((p) => p.rid === this.respId);
    const pre = this.prefix ? this.prefix + ' ' : '';
    let said = (pre + this.replyText).trim();
    this.cut = null;
    if (cut) {
      const heard = this.heardText(mine, now);
      said = (pre + heard).trim();
      // remember the rest from the start of the clause it stopped in, in case it was only an "mm-hmm"
      const words = this.replyText.trim().split(/\s+/).filter(Boolean);
      let i = Math.min(heard ? heard.split(/\s+/).length : 0, words.length);
      while (i > 0 && i < words.length && !/[.,!?;:]$/.test(words[i - 1])) i--;
      this.cut = { user: this.replyUser, keep: (pre + words.slice(0, i).join(' ')).trim(), rest: words.slice(i).join(' '), whole: this.cutWhole };
      if (said) said = said.replace(/[.!?,\s]*$/, '') + '...';
    }
    this.prefix = '';
    this.history.push({ role: 'user', content: this.replyUser });
    this.history.push({ role: 'assistant', content: said || '...' });
    this.replyUser = undefined;
    if (this.cap && this.cap.rid === this.respId) {
      if (!cut) this.registerAnchor(mine, this.cap);
      this.cap = null;
    }
    this.pieces = this.pieces.filter((p) => p.rid === this.respId && p.at > now);
  }

  // a piece can be a whole paragraph, so cut its words by how much of its audio actually played
  heardText(mine, now) {
    const out = [];
    mine.forEach((p, i) => {
      if (p.at >= now) return;
      const next = mine[i + 1];
      // the piece still generating is longer than what arrived, so guess from its text too (~14 chars/s)
      const len = next ? next.at - p.at : this.speechDone ? this.queued - p.at : Math.max(this.queued - p.at, p.text.length / 14 * CFG.sample_rate);
      const frac = (now - p.at) / Math.max(1, len);
      if (frac >= 0.97) { out.push(p.text); return; }
      const words = p.text.split(/\s+/);
      const k = Math.floor(words.length * frac);
      if (k > 0) out.push(words.slice(0, k).join(' '));
    });
    return out.join(' ');
  }

  async registerAnchor(mine, cap) {
    const all = new Float32Array(cap.n);
    let o = 0;
    for (const c of cap.chunks) { all.set(c, o); o += c.length; }
    // whole pieces only so the transcript matches, up to about 12 s
    let end = 0;
    const texts = [];
    for (let i = 0; i < mine.length; i++) {
      const s = mine[i].at - cap.base, e = i + 1 < mine.length ? mine[i + 1].at - cap.base : cap.n;
      if (s < 0) return;
      if (e > cap.n || (texts.length && e > CFG.sample_rate * 12)) break;
      texts.push(mine[i].text);
      end = e;
    }
    if (end < CFG.sample_rate * 1.0) return;
    const form = new FormData();
    form.append('ref_audio', new Blob([wavBytes(all.subarray(0, end), CFG.sample_rate)], { type: 'audio/wav' }), 'anchor.wav');
    form.append('ref_text', texts.join(' '));
    try {
      const res = await fetch('/v1/voices', { method: 'POST', body: form });
      const j = await res.json();
      if (!res.ok || !j.id || this.ended) return;
      this.anchorVoice = j.id;
      this.makeBackchannels();
    } catch { /* backchannels just stay off */ }
  }

  bcVoice() { return this.ch.voice || this.anchorVoice || ''; }

  // after a barge that was only an "mm-hmm", say the rest of what got cut, or let the llm pick it up
  resume(text) {
    const c = this.cut;
    this.cut = null;
    this.turnAt = performance.now();
    if (!c || !c.whole || !c.rest || !this.speech) {
      this.reply(resumeNote(text, c && c.rest));
      return;
    }
    // the cut entry comes back out, the whole reply goes back in once the rest is spoken
    const h = this.history;
    if (h.length >= 2 && h[h.length - 1].role === 'assistant') h.splice(h.length - 2, 2);
    this.respId++;
    this.state = 'thinking';
    this.firstAudioAt = null;
    this.speechDone = false;
    this.llmBusy = false;
    this.prefix = c.keep;
    this.replyText = c.rest;
    this.replyUser = c.user;
    this.speech.send(JSON.stringify({ type: 'end', text: c.rest }));
  }

  cancelReply(dropUser) {
    // a reply the llm already finished can be picked back up word for word
    this.cutWhole = !this.llmBusy;
    if (this.abort) this.abort.abort();
    this.abort = null;
    this.llmBusy = false;
    if (this.speech) this.speech.send(JSON.stringify({ type: 'cancel' }));
    this.discard = true;
    this.stopPlayback();
    if (dropUser) this.replyUser = undefined;
    else this.finishReply(true);
    this.respId++;
    this.speechDone = true;
    this.state = 'listening';
    this.setMode(false);
  }

  backchannel(ev) {
    if (!S.turn.backchannel || this.state !== 'listening') return;
    const now = performance.now();
    if (now - this.lastBc < Number(S.turn.bc_gap) * 1000) return;
    if (S.turn.bc_llm) {
      if (!this.bcVoice() || this.bcBusy) return;
      this.lastBc = now;
      this.llmBackchannel(ev);
      return;
    }
    if (!this.bcClips.length) return;
    this.lastBc = now;
    this.playBackchannel(this.bcClips[Math.floor(Math.random() * this.bcClips.length)]);
  }

  playBackchannel(clip, text) {
    const src = this.ctx.createBufferSource();
    src.buffer = clip.buf || clip;
    src.connect(this.bcGain);
    src.start();
    if (S.ui.captions) $('#cap-bot').textContent = '(' + (text || clip.text || '...') + ')';
  }

  async speakClip(text) {
    const body = { input: text, voice: this.bcVoice(), response_format: 'wav', max_new_tokens: 30, seed: Math.floor(Math.random() * 1e9) };
    body.instructions = (this.ch.design || '').trim() || 'Speak clearly and naturally.';
    body.cfg_scale = Number(S.voice.cfg);
    const res = await fetch('/v1/audio/speech', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    if (!res.ok) return null;
    const ab = await this.ctx.decodeAudioData(await res.arrayBuffer());
    return ab.duration > 0.2 && ab.duration < 2.2 ? ab : null;
  }

  // the llm writes a reaction to what they are saying right now, dropped if it shows up too late
  async llmBackchannel(ev) {
    this.bcBusy = true;
    const t0 = performance.now(), turn = this.turnNo;
    const late = () => this.ended || turn !== this.turnNo || this.state !== 'listening' || performance.now() - t0 > 1800;
    try {
      const heard = (ev.partial || this.lastPartial || '').trim();
      if (!heard) return;
      const hist = this.history.slice(-6);
      while (hist.length && hist[0].role !== 'user') hist.shift();
      const system = (this.ch.prompt || '').trim() + '\n\n' + BC_RULES;
      const messages = [{ role: 'system', content: system }, ...hist, { role: 'user', content: '[They are still talking. So far they said: "' + heard + '"]' }];
      const body = { messages, stream: false, temperature: 0.9, max_tokens: 12 };
      if (S.llm.model) body.model = S.llm.model;
      if (S.llm.no_think) body.reasoning_effort = 'none';
      const headers = { 'Content-Type': 'application/json' };
      if (S.llm.key) headers.Authorization = 'Bearer ' + S.llm.key;
      const res = await fetch('/llm/chat/completions', { method: 'POST', headers, body: JSON.stringify(body) });
      const j = await res.json();
      let text = new SpeechFilter().push((j.choices && j.choices[0] && j.choices[0].message && j.choices[0].message.content) || '');
      text = text.replace(/["“”]/g, '').trim().split(/\s+/).slice(0, 4).join(' ');
      if (!text || late()) return;
      const tText = performance.now();
      const clip = await this.speakClip(text);
      if (!clip || late()) return;
      this.playBackchannel(clip, text);
      this.bcInfo = 'backchannel "' + text + '" llm ' + ((tText - t0) / 1000).toFixed(2) + ' s, voice ' + ((performance.now() - tText) / 1000).toFixed(2) + ' s';
    } catch { /* a missing backchannel is fine */ } finally {
      this.bcBusy = false;
    }
  }

  // short acknowledgements in the character's own voice, made once while the line is quiet
  async makeBackchannels() {
    if (this.bcStarted || !S.turn.backchannel || S.turn.bc_llm || !this.bcVoice()) return;
    this.bcStarted = true;
    const phrases = S.turn.bc_phrases.split(',').map((s) => s.trim()).filter(Boolean);
    for (const text of phrases) {
      if (this.ended) return;
      while (!this.ended && this.state !== 'listening') await new Promise((r) => setTimeout(r, 300));
      try {
        const buf = await this.speakClip(text);
        if (buf) this.bcClips.push({ buf, text });
      } catch { /* a missing backchannel is fine */ }
    }
  }

  mute(on) {
    this.muted = on;
    this.io.port.postMessage({ mute: on });
    for (const t of this.stream.getAudioTracks()) t.enabled = !on;
  }

  stop() {
    this.ended = true;
    this.stopScreen();
    clearInterval(this.tick);
    if (this.abort) this.abort.abort();
    for (const ws of [this.tyger, this.speech]) { try { ws && ws.close(); } catch { /* closing anyway */ } }
    this.stopPlayback();
    if (this.stream) for (const t of this.stream.getTracks()) t.stop();
    if (this.ctx) this.ctx.close();
    if (this.anchorVoice) fetch('/v1/voices/' + encodeURIComponent(this.anchorVoice), { method: 'DELETE', keepalive: true }).catch(() => {});
  }
}

// a filled circle in the character's color that swells and ripples with whoever is talking
function runOrb() {
  const cv = $('#orb');
  const g = cv.getContext('2d');
  const fbuf = new Uint8Array(256);
  let level = 0, ripple = 0, t = 0;
  const rms = (an) => {
    if (!an) return 0;
    const d = new Float32Array(an.fftSize);
    an.getFloatTimeDomainData(d);
    let s = 0;
    for (const x of d) s += x * x;
    return Math.sqrt(s / d.length);
  };
  const frame = () => {
    if (!call || call.ended) return;
    const dpr = window.devicePixelRatio || 1;
    const size = cv.clientWidth;
    if (cv.width !== Math.round(size * dpr)) { cv.width = Math.round(size * dpr); cv.height = Math.round(size * dpr); }
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, size, size);
    const bot = rms(call.botAn), mic = call.muted ? 0 : rms(call.micAn);
    const target = Math.min(1, Math.max(bot * 5, mic * 3.2));
    level += (target - level) * (target > level ? 0.35 : 0.08);
    const an = bot > mic ? call.botAn : call.micAn;
    let hi = 0;
    if (an) {
      an.getByteFrequencyData(fbuf);
      for (let i = 20; i < 120; i++) hi += fbuf[i];
      hi /= 100 * 255;
    }
    ripple += (hi - ripple) * 0.2;
    t += 0.016 + level * 0.05;
    const breathe = call.state === 'thinking' ? 0.03 * Math.sin(performance.now() / 260) : 0;
    const r0 = size * 0.22 * (1 + level * 0.32 + breathe);
    const cx = size / 2, cy = size / 2;
    const amp = r0 * (0.015 + ripple * 0.09 + level * 0.04);
    g.beginPath();
    for (let i = 0; i <= 120; i++) {
      const a = (i / 120) * Math.PI * 2;
      const w = Math.sin(a * 3 + t * 1.7) * 0.5 + Math.sin(a * 5 - t * 2.3) * 0.3 + Math.sin(a * 2 + t * 0.9) * 0.2;
      const r = r0 + amp * w;
      const x = cx + Math.cos(a) * r, y = cy + Math.sin(a) * r;
      if (i) g.lineTo(x, y); else g.moveTo(x, y);
    }
    g.closePath();
    g.fillStyle = call.ch.color;
    g.fill();
    if (mic > bot && mic > 0.01) {
      g.beginPath();
      g.arc(cx, cy, r0 * 1.18 + mic * size * 0.08, 0, Math.PI * 2);
      g.strokeStyle = call.ch.color;
      g.globalAlpha = Math.min(0.5, mic * 6);
      g.lineWidth = 2;
      g.stroke();
      g.globalAlpha = 1;
    }
    requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
}

async function startCall(ch) {
  lastCharacter = ch;
  $('#connect-card').style.background = shade(ch.color, -0.35);
  $('#connect-name').textContent = 'Connecting to ' + ch.name;
  show('connect');
  call = new Call(ch);
  try {
    await call.start();
  } catch (e) {
    call.stop();
    call = null;
    show('pick');
    toast(e.message || String(e), 9000);
    return;
  }
  $('#call-name').textContent = ch.name;
  $('#call-status').textContent = '';
  $('#captions').hidden = !S.ui.captions;
  $('#cap-user').textContent = '';
  $('#cap-bot').textContent = '';
  $('#btn-mute').setAttribute('aria-pressed', 'false');
  $('#mute-label').textContent = 'Mute';
  screenButton(false);
  show('call');
  runOrb();
}

function screenButton(on) {
  $('#btn-screen').setAttribute('aria-pressed', String(on));
  $('#screen-label').textContent = on ? 'Stop sharing' : 'Share screen';
  $('#view-call').classList.toggle('sharing', on);
}

function end() {
  if (!call || call.ended) return;
  const secs = Math.floor((performance.now() - call.t0) / 1000);
  call.stop();
  $('#ended-title').textContent = 'Thanks for talking to ' + call.ch.name;
  $('#ended-detail').textContent = 'Call length ' + Math.floor(secs / 60) + ':' + String(secs % 60).padStart(2, '0');
  call = null;
  show('ended');
}

// settings drawer

function field(label, input, hint) {
  return '<label class="field"><span>' + label + '</span>' + input + (hint ? '<small>' + hint + '</small>' : '') + '</label>';
}

function range(path, min, max, step, lo, hi) {
  const v = get(path);
  return '<input type="range" data-path="' + path + '" min="' + min + '" max="' + max + '" step="' + step + '" value="' + v + '">' +
    '<div class="range-row"><span>' + lo + '</span><span data-show="' + path + '">' + v + '</span><span>' + hi + '</span></div>';
}

function check(path, label) {
  return '<label class="check"><input type="checkbox" data-path="' + path + '"' + (get(path) ? ' checked' : '') + '><span>' + label + '</span></label>';
}

function get(path) { return path.split('.').reduce((o, k) => o[k], S); }
function set(path, v) {
  const ks = path.split('.');
  const last = ks.pop();
  ks.reduce((o, k) => o[k], S)[last] = v;
}

function voiceOptions(sel) {
  let o = '<option value="">No saved voice (use the description)</option>';
  for (const v of voices) o += '<option value="' + esc(v.id) + '"' + (v.id === sel ? ' selected' : '') + '>' + esc(v.id) + ' (' + v.seconds.toFixed(1) + ' s)</option>';
  if (sel && !voices.some((v) => v.id === sel)) o += '<option value="' + esc(sel) + '" selected>' + esc(sel) + ' (not on the server)</option>';
  return o;
}

function renderSettings() {
  const body = $('#drawer-body');
  let h = '<h3>Characters</h3>';
  S.characters.forEach((ch, i) => {
    h += '<details class="char" data-i="' + i + '"' + (renderSettings.open === i ? ' open' : '') + '><summary><span class="swatch" style="background:' + esc(ch.color) + '"></span><span class="char-name">' + esc(ch.name) + '</span></summary><div class="char-body">' +
      field('Name', '<input type="text" data-char="name" value="' + esc(ch.name) + '">') +
      '<label class="field"><span>Color</span><div class="color-row"><input type="color" data-char="color" value="' + esc(ch.color) + '"></div></label>' +
      field('Voice', '<select data-char="voice">' + voiceOptions(ch.voice) + '</select>', 'Saved voices on this server. A cloned voice keeps the character sounding the same every call, a designed one changes a little each call.') +
      field('Voice description', '<textarea data-char="design" rows="3">' + esc(ch.design || '') + '</textarea>', 'With no saved voice this designs the voice. With a saved voice it steers the delivery instead, like "relaxed and a little teasing".') +
      '<div class="clone" hidden></div>' +
      '<p><button type="button" class="small-btn" data-act="clone">Clone a new voice</button></p>' +
      field('Personality', '<textarea data-char="prompt" rows="7">' + esc(ch.prompt) + '</textarea>', 'Who they are. The speaking rules get added after this, see Speaking rules below.') +
      field('Opening line', '<input type="text" data-char="opening" value="' + esc(ch.opening || '') + '" placeholder="Leave empty to let the character greet you">') +
      '<p><button type="button" class="small-btn danger" data-act="delete">Delete character</button></p>' +
      '</div></details>';
  });
  h += '<p><button type="button" class="small-btn" data-act="add">Add a character</button></p>';

  h += '<h3>Conversation</h3>';
  h += field('Model', '<select data-path="llm.model"><option value="">Server default</option>' +
    models.map((m) => '<option' + (m === S.llm.model ? ' selected' : '') + '>' + esc(m) + '</option>').join('') + '</select>',
    models.length ? '' : 'Could not list models, the llm server might be off.');
  h += field('Creativity', range('llm.temperature', 0, 1.5, 0.05, 'steady', 'wild'));
  h += field('Longest reply (tokens)', '<input type="number" data-path="llm.max_tokens" min="20" max="2000" value="' + S.llm.max_tokens + '">');
  h += check('llm.no_think', 'Skip the thinking step (much faster replies on reasoning models)');
  h += field('API key', '<input type="password" data-path="llm.key" value="' + esc(S.llm.key) + '" placeholder="Only for hosted servers">');

  h += '<h3>Speaking rules</h3>';
  h += field('Added to every character', '<textarea data-path="rules" rows="12">' + esc(S.rules) + '</textarea>', 'How replies should be written for speech. "..." and commas make pauses, tags like (laugh) or (sigh) turn into real sounds.');
  h += '<p><button type="button" class="small-btn" data-act="rules">Restore the default rules</button></p>';

  h += '<h3>Turn taking</h3>';
  h += field('Reply after you stop talking', range('turn.ep_thr', 0.6, 0.99, 0.005, 'sooner', 'waits longer'), 'Sooner feels snappier but cuts in during pauses.');
  h += check('turn.fast_start', 'Start speaking at the first comma instead of the first full sentence');
  h += check('turn.interrupt', 'Let me interrupt');
  h += field('Interrupt sensitivity', range('turn.barge_thr', 0.15, 0.9, 0.05, 'stops easily', 'needs a clear interruption'), 'How sure tyger has to be that you want to talk. It can miss short ones like "wait, hold on", the next setting catches those.');
  h += field('Also stop after talking over it for (seconds)', range('turn.barge_after', 0, 2.5, 0.1, 'off', 'long'), 'Short "mm-hmm"s stay under this. Needs headphones or good echo cancellation, or the voice can hear itself and stop.');
  h += check('turn.backchannel', 'Backchannels, short "mm-hmm" sounds while you talk');
  h += '<p class="field"><small>With no saved voice they start after the first reply, cloned from how the character sounded in it.</small></p>';
  h += check('turn.bc_llm', 'Let the llm write each backchannel from what you are saying, instead of the phrases below');
  h += '<p class="field"><small>More fitting reactions like "oh no" or "no way", but each one takes about a second to make, and one that would land too late gets skipped.</small></p>';
  h += field('How often', range('turn.bc_thr', 0.2, 0.9, 0.05, 'often', 'rarely'));
  h += field('Seconds between backchannels', '<input type="number" data-path="turn.bc_gap" min="1" max="60" value="' + S.turn.bc_gap + '">');
  h += field('Backchannel phrases', '<input type="text" data-path="turn.bc_phrases" value="' + esc(S.turn.bc_phrases) + '">', 'Comma separated.');

  h += '<h3>Voice</h3>';
  h += field('Guidance', range('voice.cfg', 1, 4, 0.1, 'fast, plain', 'follows tags harder'), 'Above 1 every frame runs twice, so it is about half as fast. Sound tags past laugh, sigh and cough mostly need 2 to 3.');
  h += field('Temperature', range('voice.temperature', 0, 1.3, 0.05, 'model default', 'lively'));

  h += '<h3>Screen sharing</h3>';
  h += field('Picture size (pixels on the long side)', range('screen.size', 512, 1920, 128, 'faster', 'sharper'), 'Needs a vision model in your llm server, like Gemma 3 or Qwen2.5 VL. A fresh picture goes with everything you say while sharing, and a bigger one makes small text readable but adds time before each reply.');

  h += '<h3>Display</h3>';
  h += check('ui.captions', 'Show captions during the call');
  h += check('ui.timings', 'Show reply timing');
  h += '<p><button type="button" class="small-btn danger" data-act="reset">Reset everything to defaults</button></p>';
  body.innerHTML = h;

  body.querySelectorAll('[data-path]').forEach((el) => {
    el.addEventListener('input', () => {
      const p = el.dataset.path;
      const v = el.type === 'checkbox' ? el.checked : el.type === 'range' || el.type === 'number' ? Number(el.value) : el.value;
      set(p, v);
      const shown = body.querySelector('[data-show="' + p + '"]');
      if (shown) shown.textContent = v;
      saveSettings();
    });
  });
  body.querySelectorAll('.char').forEach((box) => {
    const i = Number(box.dataset.i);
    const ch = S.characters[i];
    box.addEventListener('toggle', () => { if (box.open) renderSettings.open = i; });
    box.querySelectorAll('[data-char]').forEach((el) => {
      el.addEventListener('input', () => {
        ch[el.dataset.char] = el.value;
        if (el.dataset.char === 'color') box.querySelector('.swatch').style.background = el.value;
        if (el.dataset.char === 'name') box.querySelector('.char-name').textContent = el.value;
        saveSettings();
        renderTiles();
      });
    });
    box.querySelector('[data-act="clone"]').onclick = () => openClone(box, ch);
    box.querySelector('[data-act="delete"]').onclick = () => {
      if (S.characters.length < 2) { toast('Keep at least one character.'); return; }
      S.characters.splice(S.characters.indexOf(ch), 1);
      renderSettings.open = null;
      saveSettings();
      renderTiles();
      renderSettings();
    };
  });
  body.querySelector('[data-act="add"]').onclick = addCharacter;
  body.querySelector('[data-act="rules"]').onclick = () => {
    S.rules = DEFAULT_RULES;
    saveSettings();
    renderSettings();
  };
  body.querySelector('[data-act="reset"]').onclick = () => {
    if (!confirm('Reset all characters and settings?')) return;
    S = clone(DEFAULTS);
    saveSettings();
    renderTiles();
    renderSettings();
  };
}

function addCharacter() {
  const colors = ['#7d6aa8', '#4f8a9a', '#9a8a4f', '#a85b7a', '#6a8a4f'];
  const ch = {
    id: 'c' + Date.now(), name: 'New character', color: colors[S.characters.length % colors.length],
    voice: '', design: 'A calm, friendly adult with a natural speaking voice.', opening: '',
    prompt: `You are Sam. You're on a voice call with someone. Describe who Sam is here: age, mood, interests, how they joke.`,
  };
  S.characters.push(ch);
  saveSettings();
  renderTiles();
  renderSettings.open = S.characters.length - 1;
  openSettings();
}

// cloning: a clip from a file or the mic, a transcript (typed or from tyger), saved through /v1/voices
function openClone(box, ch) {
  const c = box.querySelector('.clone');
  c.hidden = false;
  c.innerHTML = '<div class="clone-actions"><label class="small-btn">Choose a file<input type="file" accept="audio/*" hidden></label>' +
    '<button type="button" class="small-btn" data-c="rec">Record</button><span data-c="info">5 to 15 seconds of clean speech works best.</span></div>' +
    field('What is said in the clip', '<textarea data-c="text" rows="3"></textarea>', 'Has to match the clip word for word.') +
    '<div class="clone-actions"><button type="button" class="small-btn" data-c="tx" disabled>Transcribe it for me</button>' +
    '<input type="text" data-c="name" placeholder="voice name" value="' + esc((ch.name || 'voice').toLowerCase().replace(/[^a-z0-9_-]+/g, '_')) + '" style="flex:1;min-width:120px;background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:6px 8px;color:inherit;font:inherit">' +
    '<button type="button" class="small-btn" data-c="save" disabled>Save voice</button></div>';
  let clip = null;
  const info = c.querySelector('[data-c="info"]');
  const ready = (samples, rate, label) => {
    clip = { samples, rate };
    info.textContent = label + ', ' + (samples.length / rate).toFixed(1) + ' s';
    c.querySelector('[data-c="tx"]').disabled = false;
    c.querySelector('[data-c="save"]').disabled = false;
  };
  c.querySelector('input[type="file"]').onchange = async (e) => {
    const f = e.target.files[0];
    if (!f) return;
    try {
      const ac = new AudioContext();
      const ab = await ac.decodeAudioData(await f.arrayBuffer());
      ready(mixdown(ab), ab.sampleRate, f.name);
      ac.close();
    } catch { toast('Could not read that audio file.'); }
  };
  const recBtn = c.querySelector('[data-c="rec"]');
  recBtn.onclick = async () => {
    if (recBtn.rec) { recBtn.rec(); return; }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, channelCount: 1 } });
      const ac = new AudioContext();
      const src = ac.createMediaStreamSource(stream);
      const proc = ac.createScriptProcessor(4096, 1, 1);
      const parts = [];
      proc.onaudioprocess = (ev) => parts.push(new Float32Array(ev.inputBuffer.getChannelData(0)));
      src.connect(proc);
      proc.connect(ac.destination);
      recBtn.textContent = 'Stop';
      info.textContent = 'Recording, read something out loud...';
      const stop = () => {
        proc.disconnect();
        stream.getTracks().forEach((t) => t.stop());
        const n = parts.reduce((a, p) => a + p.length, 0);
        const all = new Float32Array(n);
        let o = 0;
        for (const p of parts) { all.set(p, o); o += p.length; }
        const rate = ac.sampleRate;
        ac.close();
        recBtn.rec = null;
        recBtn.textContent = 'Record again';
        ready(all, rate, 'Recording');
      };
      recBtn.rec = stop;
      setTimeout(() => { if (recBtn.rec === stop) stop(); }, 20000);
    } catch { toast('Microphone access was blocked.'); }
  };
  c.querySelector('[data-c="tx"]').onclick = async (e) => {
    e.target.disabled = true;
    e.target.textContent = 'Transcribing...';
    try {
      c.querySelector('[data-c="text"]').value = await transcribe(clip.samples, clip.rate);
    } catch (err) { toast(err.message); }
    e.target.disabled = false;
    e.target.textContent = 'Transcribe it for me';
  };
  c.querySelector('[data-c="save"]').onclick = async (e) => {
    const text = c.querySelector('[data-c="text"]').value.trim();
    const name = c.querySelector('[data-c="name"]').value.trim();
    if (!text) { toast('Add what is said in the clip first.'); return; }
    if (!/^[A-Za-z0-9_-]+$/.test(name)) { toast('The voice name can only use letters, digits, dash and underscore.'); return; }
    e.target.disabled = true;
    const form = new FormData();
    form.append('ref_audio', new Blob([wavBytes(resample(clip.samples, clip.rate, 24000), 24000)], { type: 'audio/wav' }), 'clip.wav');
    form.append('ref_text', text);
    form.append('name', name);
    form.append('description', 'cloned in the voice call demo');
    try {
      const res = await fetch('/v1/voices', { method: 'POST', body: form });
      const j = await res.json();
      if (!res.ok) throw new Error((j.error && (j.error.message || j.error)) || res.status);
      ch.voice = j.id;
      saveSettings();
      await loadVoices();
      renderSettings();
      toast('Saved the voice ' + j.id + '.');
    } catch (err) {
      toast('Could not save the voice: ' + err.message);
      e.target.disabled = false;
    }
  };
}

function mixdown(ab) {
  const out = new Float32Array(ab.length);
  for (let c = 0; c < ab.numberOfChannels; c++) {
    const d = ab.getChannelData(c);
    for (let i = 0; i < d.length; i++) out[i] += d[i] / ab.numberOfChannels;
  }
  return out;
}

function resample(x, from, to) {
  if (from === to) return x;
  const n = Math.floor(x.length * to / from);
  const out = new Float32Array(n);
  const ratio = from / to;
  // average over the source span so going down does not alias
  for (let i = 0; i < n; i++) {
    const a = i * ratio, b = Math.min(x.length, a + Math.max(1, ratio));
    let s = 0, k = 0;
    for (let j = Math.floor(a); j < b; j++) { s += x[j]; k++; }
    out[i] = k ? s / k : 0;
  }
  return out;
}

// runs the clip through a throwaway tyger session and keeps every finished turn's transcript
function transcribe(samples, rate) {
  const pcm16k = resample(samples, rate, 16000);
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(CFG.tyger + (CFG.tyger.includes('?') ? '&' : '?') + 'format=s16&channels=1');
    const parts = [];
    let timer = null;
    const finish = () => { clearTimeout(timer); ws.close(); resolve(parts.join(' ').trim()); };
    ws.onerror = () => reject(new Error('Could not reach tyger-server to transcribe.'));
    ws.onmessage = (e) => {
      const m = JSON.parse(e.data);
      if (m.type === 'ready') {
        const all = new Float32Array(pcm16k.length + 16000 * 3);
        all.set(pcm16k);
        const s16 = new Int16Array(all.length);
        for (let i = 0; i < all.length; i++) s16[i] = Math.max(-1, Math.min(1, all[i])) * 32767;
        ws.send(s16.buffer);
        timer = setTimeout(finish, 4000 + all.length / 16);
      } else if (m.type === 'events') {
        for (const ev of m.events) if (ev.kind === 'ENDPOINT' && ev.transcript) parts.push(ev.transcript.trim());
        clearTimeout(timer);
        timer = setTimeout(finish, 1500);
      }
    };
  });
}

async function loadVoices() {
  try { voices = await (await fetch('/v1/voices')).json(); } catch { voices = []; }
}

async function loadModels() {
  try {
    const j = await (await fetch('/llm/models')).json();
    models = (j.data || []).map((m) => m.id).filter((id) => !/embed/i.test(id));
  } catch { models = []; }
}

async function openSettings() {
  $('#drawer').hidden = false;
  renderSettings();
  await Promise.all([loadVoices(), loadModels()]);
  renderSettings();
}

async function init() {
  try { CFG = { ...CFG, ...(await (await fetch('/talk/config')).json()) }; } catch { /* defaults */ }
  CFG.tyger = onPageHost(CFG.tyger);
  renderTiles();
  $('#open-settings').onclick = openSettings;
  $('#close-settings').onclick = () => { $('#drawer').hidden = true; };
  $('#add-char').onclick = addCharacter;
  $('#btn-end').onclick = end;
  $('#btn-mute').onclick = () => {
    if (!call) return;
    const on = !call.muted;
    call.mute(on);
    $('#btn-mute').setAttribute('aria-pressed', String(on));
    $('#mute-label').textContent = on ? 'Unmute' : 'Mute';
  };
  if (navigator.mediaDevices && navigator.mediaDevices.getDisplayMedia) $('#btn-screen').onclick = () => call && call.shareScreen();
  else $('#btn-screen').hidden = true;
  $('#btn-again').onclick = () => lastCharacter && startCall(lastCharacter);
  $('#btn-back').onclick = () => show('pick');
  window.addEventListener('beforeunload', () => { if (call) call.stop(); });
  loadVoices();
}

init();
