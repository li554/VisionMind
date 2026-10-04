/* VisionMind Agent 对话前端 — DSH 视觉复刻交互层 v2
 * 事件来自 QWebChannel bridge(Python 侧 corecoder agent 事件流)
 * 形态对齐 DSH 真实 UI: 正文与思考/工具行按到达顺序平铺交错,
 * 行 = 图标 + 标题 + 2px 圆点分隔符 + 摘要(文件名带下划线)。
 */
"use strict";

const $ = (sel) => document.querySelector(sel);
const chat = $("#chat");
const scroll = $("#chat-scroll");
const input = $("#input");
const sendBtn = $("#send-btn");
const modelBadge = $("#model-badge");
const jumpBtn = $("#jump-bottom");
let welcome = $("#welcome");
const historyBtn = $("#history-btn");
const sessionDrawer = $("#session-drawer");
const sessionList = $("#session-list");
const pendingBox = $("#pending-imgs");
const ctxChip = $("#ctx-chip");
const attachBtn = $("#attach-btn");
const fileInput = $("#file-input");
const sessionTitle = $("#session-title");
const newChatBtn = $("#new-chat-btn");
const cmdChipRow = $("#cmd-chip-row");
const cmdChipLabel = $("#cmd-chip");

let bridge = null;
let assistantEl = null;     // 当前助手轮容器
let mdEl = null;            // 当前正文段
let mdBuf = "";             // 当前段原始 markdown
let reasoningEl = null;     // 当前思考折叠行
let reasoningBuf = "";
let reasoningStartTs = 0;
let reasoningTick = null;

marked.setOptions({ breaks: true, gfm: true });

let welcomeTemplate = null;
window.addEventListener("DOMContentLoaded", () => {
  const w = document.getElementById("welcome");
  if (w) welcomeTemplate = w.cloneNode(true);
});

/* ---------------- 工具函数 ---------------- */

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

function escapeHtml(s) {
  return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;")
    .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

/* 路径样 token 加下划线(DSH 文件链接形态) */
function richSummary(text) {
  const esc = escapeHtml(text);
  return esc.replace(
    /([A-Za-z]:)?[\w.\- \\\/]{2,}[\w.\-]+\.[A-Za-z]{1,6}/g,
    (m) => (m.includes("\\") || m.includes("/")) ? `<span class="file-link">${m}</span>` : m
  );
}

function nearBottom() {
  return scroll.scrollHeight - scroll.scrollTop - scroll.clientHeight < 90;
}

function scrollToBottom(force) {
  if (force || nearBottom()) scroll.scrollTop = scroll.scrollHeight;
}

scroll.addEventListener("scroll", () => {
  jumpBtn.classList.toggle("show", !nearBottom());
});
jumpBtn.addEventListener("click", () => {
  scroll.scrollTop = scroll.scrollHeight;
});

function hideWelcome() {
  if (welcome) welcome.remove();
}

/* ---------------- 会话标题(顶部固定) ---------------- */

let titleSet = false;  // 是否已由首条用户消息生成标题

function setSessionTitle(text) {
  const t = (text || "").replace(/\s+/g, " ").trim();
  if (!t) {
    sessionTitle.textContent = "新会话";
    sessionTitle.title = "新会话";
    return;
  }
  sessionTitle.textContent = t.length > 40 ? t.slice(0, 40) + "…" : t;
  sessionTitle.title = t;
}

function maybeSetTitleFromMessage(text) {
  if (titleSet) return;
  const t = (text || "").trim();
  if (!t) return;
  setSessionTitle(t);
  titleSet = true;
}

function showWelcome() {
  let w = document.getElementById("welcome");
  if (!w) {
    w = welcomeTemplate.cloneNode(true);
    chat.appendChild(w);
    bindQuickChips();
  }
  welcome = w;
}

function bindQuickChips() {
  document.querySelectorAll(".quick-chip").forEach((chip) => {
    if (chip.dataset.bound) return;
    chip.dataset.bound = "1";
    chip.addEventListener("click", () => {
      input.value = chip.dataset.prompt || "";
      syncSend();
      send();
    });
  });
}

function fmtDuration(ms) {
  const s = Math.round(ms / 100) / 10;
  return s < 60 ? `${s}s` : `${Math.floor(s / 60)}m${Math.round(s % 60)}s`;
}

/* 行领头的类型图标(DSH: 思考 ✳ / 终端 ❯□ / 写 ✎ / 默认 ✦) */
function toolIcon(name) {
  const n = (name || "").toLowerCase();
  if (/(bash|pwsh|powershell|cmd|shell|terminal)/.test(n)) {
    return '<span class="term-box">&gt;_</span>';
  }
  if (/(write|edit|replace|annotat|draw|create|add|set_|delete|remove)/.test(n)) return "✎";
  if (/(read|view|search|glob|grep|list|get_|find|show)/.test(n)) return "▤";
  if (/(batch|run|execute|one_click|process)/.test(n)) return "❯";
  return "✦";
}

function copyText(text) {
  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(text).catch(() => fallbackCopy(text));
  } else {
    fallbackCopy(text);
  }
}

function fallbackCopy(text) {
  const ta = el("textarea");
  ta.value = text;
  ta.style.position = "fixed";
  ta.style.opacity = "0";
  document.body.appendChild(ta);
  ta.select();
  try { document.execCommand("copy"); } catch (_) {}
  ta.remove();
}

/* ---------------- 消息行 ---------------- */

function addUserRow(htmlText, rawText) {
  hideWelcome();
  const row = el("div", "msg-user-row");
  const bubble = el("div", "msg-user");
  bubble.innerHTML = htmlText;
  row.appendChild(bubble);

  // 首条用户消息(含历史回放的首条) → 用指令生成会话标题
  maybeSetTitleFromMessage(rawText || bubble.textContent);

  const actions = el("div", "msg-actions");
  const copyBtn = el("button", "msg-act", "⧉ 复制");
  copyBtn.addEventListener("click", () => copyText(rawText || bubble.textContent));
  actions.appendChild(copyBtn);
  row.appendChild(actions);

  chat.appendChild(row);
  scrollToBottom(true);
}

function addSystemRow(text) {
  chat.appendChild(el("div", "msg-system", text));
  scrollToBottom(true);
}

/* ---------------- 助手轮:正文段与行交错平铺 ---------------- */

function assistantStarted() {
  hideWelcome();
  assistantEl = el("div", "msg-assistant");
  assistantEl._segments = [];
  mdEl = null;
  mdBuf = "";

  const actions = el("div", "msg-actions");
  const copyBtn = el("button", "msg-act", "⧉ 复制");
  copyBtn.addEventListener("click", () => {
    copyText((assistantEl._segments || []).join("\n\n"));
  });
  actions.appendChild(copyBtn);
  assistantEl.appendChild(actions);

  chat.appendChild(assistantEl);
  scrollToBottom(true);
}

/* 正文段:首个 token 创建;插入行之后再来 token 则新起一段 */
function ensureMd() {
  if (mdEl) return mdEl;
  hideWelcome();
  mdEl = el("div", "md");
  mdBuf = "";
  // actions 行保持在容器末尾之外的动作区:插到第一个位置后面不优雅,
  // 直接 append 会排在 actions 后面 → 用 insertBefore 放到 actions 之前
  const actions = assistantEl.querySelector(".msg-actions");
  if (actions) {
    assistantEl.insertBefore(mdEl, actions);
  } else {
    assistantEl.appendChild(mdEl);
  }
  assistantEl._segments.push("");
  return mdEl;
}

function renderMd() {
  if (!mdEl) return;
  mdEl.innerHTML = marked.parse(mdBuf);
  scrollToBottom();
}

function assistantToken(tok) {
  if (!assistantEl) assistantStarted();
  const seg = ensureMd();
  mdBuf += tok;
  assistantEl._segments[assistantEl._segments.length - 1] = mdBuf;
  renderMd();
}

function assistantFinished() {
  finishReasoning();
  if (mdEl) renderMd();
  if (assistantEl) {
    const hasText = (assistantEl._segments || []).some((s) => s.trim());
    const hasRows = assistantEl.querySelector(".fold-row, .io-card, .tool-sub");
    if (!hasText && !hasRows) assistantEl.remove();
    else if (!hasText) {
      const a = assistantEl.querySelector(".msg-actions");
      if (a) a.remove();
    }
  }
  assistantEl = null;
  mdEl = null;
  mdBuf = "";
  scrollToBottom();
}

/* ---------------- 思考折叠行(✳ / 展开态 ⌄) ---------------- */

function ensureReasoning() {
  if (reasoningEl) return reasoningEl;
  hideWelcome();
  reasoningBuf = "";
  const row = el("div", "fold-row");
  row.dataset.state = "running";
  row.dataset.open = "false";

  const head = el("div", "fold-head");
  const icon = el("span", "fold-icon", "✳");
  const title = el("span", "fold-title", "思考");
  const sep = el("span", "fold-sep");
  const summary = el("span", "fold-summary");
  summary.textContent = "…";
  head.append(icon, title, sep, summary);
  const body = el("div", "fold-body");
  row.append(head, body);

  head.addEventListener("click", () => {
    const open = row.dataset.open === "true";
    row.dataset.open = open ? "false" : "true";
    icon.textContent = open ? "✳" : "⌄";  // 展开态图标位换箭头(DSH 形态)
  });
  reasoningStartTs = Date.now();
  reasoningTick = setInterval(() => {
    title.textContent = `思考 · ${fmtDuration(Date.now() - reasoningStartTs)}`;
  }, 500);

  reasoningEl = { row, title, summary, body, icon };
  // 思考行总是新起一段之前插入(属于" upcoming 正文"的前置)
  insertRow(row);
  scrollToBottom(true);
  return reasoningEl;
}

function reasoningToken(tok) {
  const r = ensureReasoning();
  reasoningBuf += tok;
  const lines = reasoningBuf.split("\n").map((s) => s.trim()).filter(Boolean);
  r.summary.textContent = lines.length ? lines[lines.length - 1] : "…";
  r.summary.innerHTML = richSummary(r.summary.textContent);
  r.body.textContent = reasoningBuf;
  if (r.row.dataset.open === "true") r.body.scrollTop = r.body.scrollHeight;
  scrollToBottom();
}

function finishReasoning() {
  if (!reasoningEl) return;
  clearInterval(reasoningTick);
  const r = reasoningEl;
  r.row.dataset.state = "idle";
  r.title.textContent = `思考 · ${fmtDuration(Date.now() - reasoningStartTs)}`;
  reasoningEl = null;
  scrollToBottom();
}

/* ---------------- 工具折叠行 ---------------- */

function insertRow(row) {
  // 行插入到当前正文段之后(actions 之前):正文在前面,行随到达顺序追加
  const actions = assistantEl ? assistantEl.querySelector(".msg-actions") : null;
  if (assistantEl && actions) {
    assistantEl.insertBefore(row, actions);
  } else {
    chat.appendChild(row);
  }
  // 行之后的 token 新起一段正文(DSH 时间序交错形态)
  mdEl = null;
  mdBuf = "";
}

function toolAdded(id, name, summary, payloadJson) {
  hideWelcome();
  let payload = {};
  try { payload = JSON.parse(payloadJson); } catch (_) {}

  const row = el("div", "fold-row");
  row.dataset.state = payload.status === "success" ? "done" : (payload.status || "running");
  row.dataset.open = "false";
  row.dataset.toolId = id;
  row._inputText = payload.input || "";

  const head = el("div", "fold-head");
  const icon = el("span", "fold-icon");
  icon.innerHTML = toolIcon(name);
  head.appendChild(icon);
  head.appendChild(el("span", "fold-title", name));
  head.appendChild(el("span", "fold-sep"));
  const summaryEl = el("span", "fold-summary");
  summaryEl.innerHTML = richSummary(summary && summary !== name ? summary : name);
  head.appendChild(summaryEl);
  const suffix = el("span", "fold-suffix", payload.duration || "");
  head.appendChild(suffix);
  const body = buildBody(row, payload);
  row.append(head, body);

  head.addEventListener("click", () => {
    const open = row.dataset.open === "true";
    row.dataset.open = open ? "false" : "true";
    body.style.display = ""; // 交回 CSS 控制
  });

  insertRow(row);
  scrollToBottom();
}

/* 输入参数美化:合法 JSON 缩进展开,其余原样(DSH IN 段形态) */
function prettyInput(text) {
  const t = (text || "").trim();
  if (!t) return "";
  if (t.startsWith("{") || t.startsWith("[")) {
    try {
      return JSON.stringify(JSON.parse(t), null, 2);
    } catch (_) {}
  }
  // main_window 的 args_str 形如 k='v', k2=v — 逐参数换行提升可读性
  return t.replace(/, (?=[A-Za-z_][\w]*=)/g, ",\n");
}

function buildBody(row, payload) {
  const body = el("div", "fold-body tool-body");
  // 工具卡与上方调用行左缘对齐(DSH ToolRow ioCard margin-left 4px),
  // 不继承 reasoning 的 24px 缩进
  body.style.paddingLeft = "2px";
  body.style.whiteSpace = "normal";
  const frag = document.createDocumentFragment();

  if (payload.steps && payload.steps.length) {
    const sub = el("div", "tool-sub");
    const marks = { done: "✓", active: "⟳", pending: "○" };
    for (const s of payload.steps) {
      const st = el("div", "tool-step");
      st.dataset.status = s.status || "pending";
      st.appendChild(el("span", "step-mark", marks[s.status] || "○"));
      st.appendChild(el("span", "", s.label || ""));
      sub.appendChild(st);
    }
    frag.appendChild(sub);
  }

  const hasInput = !!(payload.input && payload.input.trim());
  const hasOutput = !!(payload.output && payload.output.trim());
  if (hasInput || hasOutput) {
    const card = el("div", "io-card");
    if (hasInput) {
      const sec = el("div", "io-section");
      sec.appendChild(el("span", "io-label", "IN"));
      sec.appendChild(el("div", "io-content", prettyInput(payload.input)));
      card.appendChild(sec);
    }
    if (hasOutput) {
      const sec = el("div", "io-section io-out");
      sec.appendChild(el("span", "io-label", "OUT"));
      const out = el("div", "io-content", payload.output);
      if (row.dataset.state === "error") out.dataset.error = "true";
      sec.appendChild(out);
      card.appendChild(sec);
    }
    frag.appendChild(card);
  }
  if (!frag.childNodes.length) {
    frag.appendChild(el("div", "io-empty", "（无详情）"));
  }
  body.appendChild(frag);
  return body;
}

function toolUpdated(id, status, output) {
  const row = chat.querySelector(`[data-tool-id="${id}"]`);
  if (!row) return;
  row.dataset.state = status === "success" ? "done" : status;
  const suffix = row.querySelector(".fold-suffix");
  if (suffix && status === "success" && !suffix.textContent) {
    suffix.textContent = "✓";
    suffix.classList.add("ok");
  }
  if (output && output.trim()) {
    const old = row.querySelector(".fold-body");
    const fresh = buildBody(row, { input: row._inputText || "", output });
    if (row.dataset.open === "true") fresh.style.display = "block";
    row.replaceChild(fresh, old);
  }
}

/* ---------------- 引用注入(「添加到对话」) ---------------- */

function referenceAdded(text) {
  const cur = input.value.trimEnd();
  input.value = cur ? cur + " " + text : text;
  autoGrow(); syncSend();
  input.focus();
  input.setSelectionRange(input.value.length, input.value.length);
}

/* ---------------- ask_user 提问卡 ---------------- */

function askUserRequested(reqId, questionsJson) {
  hideWelcome();
  let questions = [];
  try { questions = JSON.parse(questionsJson) || []; } catch (_) {}

  const card = el("div", "ask-card");
  card.appendChild(el("div", "ask-title", "❓ Agent 提问"));

  const answers = [];
  const inputs = [];

  for (const q of questions) {
    const qid = q.id || String(answers.length);
    const block = el("div", "ask-q");
    if (q.header) block.appendChild(el("div", "ask-q-header", q.header));
    block.appendChild(el("div", "ask-q-text", q.question || ""));

    const multi = !!q.multi_select;
    const selected = new Set();
    const optWrap = el("div", "ask-options");

    for (const opt of q.options || []) {
      const btn = el("button", "ask-opt", opt);
      btn.addEventListener("click", () => {
        if (multi) {
          if (selected.has(opt)) selected.delete(opt);
          else selected.add(opt);
          btn.classList.toggle("sel", selected.has(opt));
        } else {
          selected.clear();
          optWrap.querySelectorAll(".ask-opt").forEach((b) => b.classList.remove("sel"));
          selected.add(opt);
          btn.classList.add("sel");
        }
        custom.value = "";
        syncSubmit();
      });
      optWrap.appendChild(btn);
    }
    if (optWrap.childNodes.length) block.appendChild(optWrap);

    const custom = el("input", "ask-custom");
    custom.placeholder = "或输入自定义回答…";
    custom.addEventListener("input", () => {
      if (custom.value.trim()) {
        selected.clear();
        optWrap.querySelectorAll(".ask-opt").forEach((b) => b.classList.remove("sel"));
      }
      syncSubmit();
    });
    block.appendChild(custom);
    card.appendChild(block);

    inputs.push({ qid, selected, custom });
  }

  const row = el("div", "approval-row");
  const submit = el("button", "approval-btn approval-allow", "提交回答");
  const status = el("span", "approval-status");
  status.style.display = "none";
  row.append(submit, status);
  card.appendChild(row);

  function syncSubmit() {
    submit.disabled = !inputs.some(
      (x) => x.selected.size > 0 || x.custom.value.trim()
    );
  }
  submit.disabled = true;
  submit.addEventListener("click", () => {
    for (const x of inputs) {
      if (!(x.selected.size > 0 || x.custom.value.trim())) return;
    }
    const payload = inputs.map((x) => ({
      id: x.qid,
      selected: [...x.selected],
      ...(x.custom.value.trim() ? { custom: x.custom.value.trim() } : {}),
    }));
    card.classList.add("answered");
    submit.disabled = true;
    card.querySelectorAll(".ask-opt, .ask-custom").forEach((e) => (e.disabled = true));
    const digest = payload
      .map((a) => [...a.selected, a.custom || ""].filter(Boolean).join(" / "))
      .join("；");
    status.textContent = "已回答: " + digest;
    status.style.display = "";
    bridge.answerQuestion(reqId, JSON.stringify({ answers: payload }));
  });

  insertRow(card);
  scrollToBottom(true);
}

/* ---------------- 计划审批卡 ---------------- */

function planApprovalRequested() {
  const card = el("div", "plan-card");
  card.appendChild(el("div", "plan-card-title", "📋 执行计划已生成"));
  card.appendChild(el("div", "plan-card-text",
    "计划模式调研完成。执行将恢复全部工具并开始实施；继续修改则保持计划模式。"));
  const row = el("div", "approval-row");
  const run = el("button", "approval-btn approval-allow", "▶ 执行计划");
  const edit = el("button", "approval-btn plan-edit-btn", "✎ 继续修改");
  const status = el("span", "approval-status");
  status.style.display = "none";
  row.append(run, edit, status);
  card.appendChild(row);

  function decide(execute) {
    if (card.classList.contains("answered")) return;
    card.classList.add("answered");
    run.disabled = edit.disabled = true;
    status.textContent = execute ? "✅ 已批准，开始执行" : "✎ 继续修改（保持计划模式）";
    status.style.display = "";
    bridge.planApprove(execute);
  }
  run.addEventListener("click", () => decide(true));
  edit.addEventListener("click", () => decide(false));

  chat.appendChild(card);
  scrollToBottom(true);
}

/* ---------------- 工具审批卡 ---------------- */

function approvalRequested(reqId, tool, reason) {  const card = el("div", "approval-card");
  card.appendChild(el("div", "approval-title", "🔐 Agent 申请解锁 " + (tool || "bash")));
  card.appendChild(el("div", "approval-reason", reason || "（未提供理由）"));
  const row = el("div", "approval-row");
  const allow = el("button", "approval-btn approval-allow", "允许本次执行");
  const deny = el("button", "approval-btn approval-deny", "拒绝");
  const status = el("span", "approval-status");
  status.style.display = "none";
  row.append(allow, deny, status);
  card.appendChild(row);

  let answered = false;
  function decide(ok) {
    if (answered) return;
    answered = true;
    allow.disabled = deny.disabled = true;
    status.textContent = ok ? "✅ 已允许（仅本次任务）" : "⛔ 已拒绝";
    status.style.display = "";
    bridge.approvalRespond(reqId, ok);
  }
  allow.addEventListener("click", () => decide(true));
  deny.addEventListener("click", () => decide(false));

  insertRow(card);
  scrollToBottom(true);
}

/* ---------------- 附件(粘贴/上传/拖拽图片) ---------------- */

let pendingImages = [];

function addPendingImage(dataUrl) {
  if (!dataUrl.startsWith("data:image/")) return;
  if (pendingImages.length >= 6) return;
  pendingImages.push(dataUrl);
  renderPending();
  syncSend();
}

function renderPending() {
  pendingBox.innerHTML = "";
  pendingBox.hidden = pendingImages.length === 0;
  pendingImages.forEach((url, idx) => {
    const chip = el("div", "pending-chip");
    const img = el("img");
    img.src = url;
    chip.appendChild(img);
    const del = el("span", "pending-del", "×");
    del.title = "移除";
    del.addEventListener("click", () => {
      pendingImages.splice(idx, 1);
      renderPending();
      syncSend();
    });
    chip.appendChild(del);
    pendingBox.appendChild(chip);
  });
}

async function fileToDataUrl(file) {
  // 大图降采样到最长边 1024,控制 base64 体积;小图保留原格式
  const isPng = file.type === "image/png";
  if (isPng && file.size < 1.5 * 1024 * 1024) {
    return await readAsDataUrl(file);
  }
  try {
    const bitmap = await createImageBitmap(file);
    const maxSide = 1024;
    const scale = Math.min(1, maxSide / Math.max(bitmap.width, bitmap.height));
    const canvas = document.createElement("canvas");
    canvas.width = Math.round(bitmap.width * scale);
    canvas.height = Math.round(bitmap.height * scale);
    canvas.getContext("2d").drawImage(bitmap, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL("image/jpeg", 0.85);
  } catch (_) {
    return await readAsDataUrl(file);
  }
}

function readAsDataUrl(file) {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(r.result);
    r.onerror = reject;
    r.readAsDataURL(file);
  });
}

async function addFiles(files) {
  for (const f of files) {
    if (f.type.startsWith("image/")) {
      addPendingImage(await fileToDataUrl(f));
    }
  }
}

attachBtn.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => {
  addFiles([...fileInput.files]);
  fileInput.value = "";
});

input.addEventListener("paste", (e) => {
  const files = [...(e.clipboardData?.files || [])];
  if (files.length) {
    e.preventDefault();
    addFiles(files);
  }
});

const composerCard = document.querySelector(".composer-card");
["dragover", "dragenter"].forEach((ev) =>
  composerCard.addEventListener(ev, (e) => e.preventDefault())
);
composerCard.addEventListener("drop", (e) => {
  e.preventDefault();
  addFiles([...(e.dataTransfer?.files || [])]);
});

/* ---------------- 上下文用量圆环 + 悬浮小卡(仅图标上方) ---------------- */

let ctxStats = null;
let ctxCard = null;

function renderCtx(stats) {
  const pct = Math.min(100, Math.round(stats.percentage || 0));
  const fmt = (n) => "~" + Math.round(n / 1000) + "K";
  ctxChip.hidden = false;
  ctxChip.style.background =
    `conic-gradient(var(--info) ${pct}%, var(--border-l2) 0)`;
  ctxChip.title = `上下文已用 ${pct}%  ${fmt(stats.total)} / ${fmt(stats.max)}`;

  if (ctxCard) ctxCard.remove();
  ctxCard = el("div", "ctx-card");
  const head = el("div", "ctx-head");
  head.appendChild(el("span", "ctx-head-label", "上下文已用 " + pct + "%"));
  head.appendChild(el("span", "ctx-head-total",
    `${fmt(stats.total)} / ${fmt(stats.max)}`));
  const bar = el("div", "ctx-bar");
  const mk = (cls, ratio) => {
    const s = el("span", "ctx-seg " + cls);
    s.style.width = Math.max(0, Math.min(100, ratio)) + "%";
    bar.appendChild(s);
  };
  const max = Math.max(stats.max || 1, 1);
  mk("seg-sys", (stats.system_prompt / max) * 100);
  mk("seg-tools", (stats.tools / max) * 100);
  mk("seg-msg", (stats.messages / max) * 100);
  ctxCard.appendChild(bar);
  const rows = [
    ["seg-sys", "系统提示词", stats.system_prompt],
    ["seg-tools", "工具", stats.tools],
    ["seg-msg", "对话消息", stats.messages],
  ];
  for (const [cls, label, val] of rows) {
    const row = el("div", "ctx-row");
    row.appendChild(el("span", "ctx-dot " + cls));
    row.appendChild(el("span", "ctx-row-label", label));
    row.appendChild(el("span", "ctx-row-val", fmt(val)));
    ctxCard.appendChild(row);
  }
  document.getElementById("app").appendChild(ctxCard);
  ctxCard.style.display = "none";
  positionCtxCard();
}

/* 小卡固定 240px 宽,右缘对齐圆环图标,悬浮在图标上方 */
function positionCtxCard() {
  if (!ctxCard) return;
  const r = ctxChip.getBoundingClientRect();
  ctxCard.style.left = "auto";
  ctxCard.style.right = Math.max(8, window.innerWidth - r.right) + "px";
  ctxCard.style.bottom = (window.innerHeight - r.top + 10) + "px";
}

ctxChip.addEventListener("mouseenter", () => {
  if (ctxCard) { positionCtxCard(); ctxCard.style.display = "flex"; }
});
ctxChip.addEventListener("mouseleave", () => {
  if (ctxCard) ctxCard.style.display = "none";
});

/* ---------------- 模型切换面板(DSH 形态:覆盖输入区上方) ---------------- */

const modelPanel = $("#model-panel");
const modelPanelList = $("#model-panel-list");

function positionModelPanel() {
  const r = modelBadge.getBoundingClientRect();
  modelPanel.style.left = "auto";
  modelPanel.style.right = Math.max(8, window.innerWidth - r.right) + "px";
  modelPanel.style.bottom = (window.innerHeight - r.top + 10) + "px";
}

modelBadge.addEventListener("click", () => {
  if (!modelPanel.hidden) { modelPanel.hidden = true; return; }
  if (!bridge) return;
  bridge.getModels((r) => {
    let data = [];
    try { data = JSON.parse(r); } catch (_) {}
    if (!data.length) return;
    modelPanelList.innerHTML = "";
    for (const prov of data) {
      modelPanelList.appendChild(el("div", "model-group", prov.name));
      for (const m of prov.models) {
        const row = el("div", "model-row" +
          ((prov.active && m === prov.active_model) ? " active" : ""));
        row.appendChild(el("span", "model-row-name", m));
        if (prov.active && m === prov.active_model) {
          row.appendChild(el("span", "model-row-check", "✓"));
        }
        row.addEventListener("click", () => {
          modelPanel.hidden = true;
          bridge.selectModel(prov.pid, m);
        });
        modelPanelList.appendChild(row);
      }
    }
    modelPanel.hidden = false;
    positionModelPanel();
  });
});

$("#model-back").addEventListener("click", () => { modelPanel.hidden = true; });

function closeModelMenu() {
  modelPanel.hidden = true;
}

/* ---------------- 执行状态:发送按钮变运行中方块(DSH 交互) ---------------- */

let running = false;

function execStatus(text) {
  running = !!text;
  sendBtn.classList.toggle("running", running);
  sendBtn.textContent = running ? "■" : "↑";
  sendBtn.title = running ? "停止执行" : "发送";
  sendBtn.disabled = running ? false
    : (!input.value.trim() && !pendingImages.length && !activeCmd);
  if (running) scrollToBottom();
}

/* ---------------- 输入区 ---------------- */

function autoGrow() {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 220) + "px";
}

function syncSend() {
  if (running) return;
  sendBtn.disabled = !input.value.trim() && !pendingImages.length && !activeCmd;
}

function closeMenus() {
  closeSuggest(); closeCmdMenu(); closeModelMenu();
}

/* 点击指令 chip 整体移除(原子删除) */
cmdChipRow.addEventListener("click", () => {
  if (activeCmd) {
    setCmdChip(null);
    input.focus();
  }
});

input.addEventListener("input", () => { autoGrow(); syncSend(); onInputChanged(); });
input.addEventListener("keydown", onInputKeydown);
sendBtn.addEventListener("click", () => {
  if (running) { bridge && bridge.stopClicked(); return; }
  send();
});

function send() {
  let text = input.value.trim();
  if ((!text && !pendingImages.length && !activeCmd) || !bridge || running) return;
  closeMenus();
  modelPanel.hidden = true;
  // 指令 chip → 重构为 "<cmd> [目标]" 文本(避免 chip 部分被当作普通文本)
  if (activeCmd) {
    const re = new RegExp("^" + activeCmd.replace(/[.*+?^${}()|[\]\\]/g, "\\$&") + "(?:\\s|$)");
    text = text.replace(re, "").trim();
    text = text ? activeCmd + " " + text : activeCmd;
  }
  if (pendingImages.length) {
    bridge.sendMessagePayload(JSON.stringify({
      text, images: [...pendingImages],
    }));
    pendingImages = [];
    renderPending();
  } else {
    bridge.sendMessage(text);
  }
  input.value = "";
  autoGrow();
  setCmdChip(null);
}

document.querySelectorAll(".quick-chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    input.value = chip.dataset.prompt || "";
    syncSend();
    send();
  });
});

/* ---------------- 轮次结束:清扫未收到结果的工具行 ---------------- */

function sweepStuckRows() {
  document.querySelectorAll('.fold-row[data-state="running"]').forEach((row) => {
    row.dataset.state = "error";
    const summary = row.querySelector(".fold-summary");
    if (summary && !summary.dataset.marked) {
      summary.dataset.marked = "1";
      summary.textContent += "  ⚠ 未收到执行结果";
    }
    const suffix = row.querySelector(".fold-suffix");
    if (suffix) { suffix.textContent = "✗"; suffix.style.color = "var(--error)"; }
  });
}

/* ---------------- / 命令菜单 ---------------- */

const COMMANDS = [
  { cmd: "/plan", desc: "计划模式：只读调研并产出执行计划，审批后执行" },
  { cmd: "/compact", desc: "手动压缩上下文：旧消息摘要，保留最近 10 条" },
];

/* 所有 / 指令(/plan、/compact 等)均作为整体高亮 chip，不作为普通文本混入输入框：
 * 添加(菜单选择)与删除(点击 chip/输入框行首退格)均按整体处理。
 * 发送时由 send() 重构为 "<cmd> [目标]" 文本交给后端。 */
let activeCmd = null;  // null | "/plan" | "/compact" ...

function setCmdChip(cmd) {
  activeCmd = cmd || null;
  cmdChipRow.hidden = !activeCmd;
  cmdChipLabel.textContent = activeCmd || "";
  input.placeholder = activeCmd === "/plan"
    ? "输入计划目标，回车发送"
    : (activeCmd ? "输入内容或直接发送指令" : "给智能体发消息");
  syncSend();
}

let cmdMenu = null;
let cmdIndex = 0;
let atMenu = null;
let atIndex = 0;
let atCandidates = [];

function buildMenu(cls) {
  const menu = el("div", cls);
  document.getElementById("app").appendChild(menu);
  return menu;
}

function positionMenu(menu) {
  const card = document.querySelector(".composer-card");
  if (!card) return;
  const r = card.getBoundingClientRect();
  menu.style.left = r.left + "px";
  menu.style.bottom = (window.innerHeight - r.top + 6) + "px";
  menu.style.width = r.width + "px";
}

function openCmdMenu() {
  closeSuggest();
  if (!cmdMenu) cmdMenu = buildMenu("cmd-menu");
  cmdIndex = 0;
  renderCmdMenu();
  cmdMenu.style.display = "block";
  positionMenu(cmdMenu);
}

function renderCmdMenu() {
  if (!cmdMenu) return;
  cmdMenu.innerHTML = "";
  COMMANDS.forEach((c, i) => {
    const item = el("div", "menu-item" + (i === cmdIndex ? " sel" : ""));
    item.appendChild(el("span", "menu-cmd", c.cmd));
    item.appendChild(el("span", "menu-desc", c.desc));
    item.addEventListener("click", () => pickCommand(c));
    cmdMenu.appendChild(item);
  });
}

function pickCommand(c) {
  closeCmdMenu();
  // 所有指令统一为整体 chip：选中已激活的指令 → 关闭；选中其他指令 → 切换
  setCmdChip(activeCmd === c.cmd ? null : c.cmd);
  input.value = "";
  autoGrow();
  input.focus();
}

function closeCmdMenu() {
  if (cmdMenu) cmdMenu.style.display = "none";
}

/* ---------------- @ 引用候选(经典侧栏同源) ---------------- */

const AT_TEMPLATES = [
  "@原图:当前图", "@图:当前图索引", "@局部:当前图标注", "@渲染图:当前图",
  "@渲染局部:当前图标注", "@ROI", "@类别:", "@难样本", "@搜索:", "@报告:",
];

function openAtMenu() {
  closeCmdMenu();
  if (!atMenu) atMenu = buildMenu("at-menu");
  atIndex = 0;
  renderAtMenu();
  atMenu.style.display = "block";
  positionMenu(atMenu);
}

function renderAtMenu() {
  if (!atMenu) return;
  atMenu.innerHTML = "";
  atCandidates.slice(0, 8).forEach((c, i) => {
    const item = el("div", "menu-item" + (i === atIndex ? " sel" : ""));
    item.appendChild(el("span", "menu-cmd", c));
    item.addEventListener("click", () => pickAt(c));
    atMenu.appendChild(item);
  });
}

function pickAt(cand) {
  const pos = input.selectionStart;
  const text = input.value;
  const idx = text.lastIndexOf("@", pos - 1);
  if (idx >= 0) {
    input.value = text.slice(0, idx) + cand + text.slice(pos);
    input.setSelectionRange(idx + cand.length, idx + cand.length);
  }
  closeSuggest();
  input.focus();
  autoGrow(); syncSend();
}

function closeSuggest() {
  if (atMenu) atMenu.style.display = "none";
}

function atCandidatesFor(prefix) {
  if (!prefix) return AT_TEMPLATES.slice();
  const matched = AT_TEMPLATES.filter((t) => t.includes(prefix));
  if (matched.length) return matched;
  // 不匹配语法模板 → 文件名搜索(桥同步调用,registry 内有上限)
  try {
    const kw = prefix.startsWith("文件:") ? prefix.slice(3) : prefix;
    const files = JSON.parse(bridge.searchFiles(kw));
    return files.map((p) => "@文件:" + p);
  } catch (_) {
    return [];
  }
}

function onInputChanged() {
  const pos = input.selectionStart;
  const text = input.value;

  // "/" 仅行首触发命令菜单
  if (text.startsWith("/") && !text.includes(" ")) {
    openCmdMenu();
    return;
  }
  closeCmdMenu();

  // "@" 触发引用候选(取光标前最近的 @)
  const atIdx = text.lastIndexOf("@", pos - 1);
  if (atIdx >= 0) {
    const prefix = text.slice(atIdx + 1, pos);
    if (prefix.length <= 64 && !prefix.includes(" ")) {
      atCandidates = atCandidatesFor(prefix);
      if (atCandidates.length) { openAtMenu(); return; }
    }
  }
  closeSuggest();
}

function onInputKeydown(e) {
  // 指令 chip 整体删除：光标位于输入框行首时退格即移除整个 chip
  if (e.key === "Backspace" && activeCmd &&
      input.selectionStart === 0 && input.selectionEnd === 0) {
    e.preventDefault();
    setCmdChip(null);
    return;
  }
  // 命令菜单键盘导航
  if (cmdMenu && cmdMenu.style.display === "block") {
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      const n = COMMANDS.length;
      cmdIndex = (cmdIndex + (e.key === "ArrowDown" ? 1 : -1) + n) % n;
      renderCmdMenu();
      return;
    }
    if (e.key === "Enter" || e.key === "Tab") {
      e.preventDefault();
      pickCommand(COMMANDS[cmdIndex]);
      return;
    }
    if (e.key === "Escape") { closeCmdMenu(); return; }
    return;
  }
  // @ 候选键盘导航
  if (atMenu && atMenu.style.display === "block") {
    const n = Math.min(atCandidates.length, 8);
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      atIndex = (atIndex + (e.key === "ArrowDown" ? 1 : -1) + n) % n;
      renderAtMenu();
      return;
    }
    if (e.key === "Enter" || e.key === "Tab") {
      e.preventDefault();
      if (atCandidates[atIndex]) pickAt(atCandidates[atIndex]);
      return;
    }
    if (e.key === "Escape") { closeSuggest(); return; }
    return;
  }
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    send();
  }
}

/* ---------------- 历史会话抽屉 ---------------- */

function renderSessionList(sessions) {
  sessionList.innerHTML = "";
  if (!sessions.length) {
    sessionList.appendChild(el("div", "session-empty", "暂无历史会话"));
    return;
  }
  for (const s of sessions) {
    const card = el("div", "session-card");
    const head = el("div", "session-card-head");
    head.appendChild(el("span", "session-time", s.time || ""));
    const del = el("span", "session-del", "×");
    del.title = "删除会话";
    head.appendChild(del);
    card.appendChild(head);
    card.appendChild(el("div", "session-preview", s.preview || "(无预览)"));

    card.addEventListener("click", () => {
      if (s.id) bridge.openSession(s.id);
      sessionDrawer.hidden = true;
    });
    del.addEventListener("click", (e) => {
      e.stopPropagation();
      if (s.id) bridge.deleteSession(s.id);
      card.remove();
      if (!sessionList.querySelector(".session-card")) {
        renderSessionList([]);
      }
    });
    sessionList.appendChild(card);
  }
}

historyBtn.addEventListener("click", () => {
  if (!sessionDrawer.hidden) { sessionDrawer.hidden = true; return; }
  closeMenus();
  sessionDrawer.hidden = false;
  sessionList.innerHTML = '<div class="session-empty">加载中…</div>';
  bridge.getSessions((r) => {
    let data = [];
    try { data = JSON.parse(r); } catch (_) {}
    renderSessionList(data);
  });
});
$("#session-close").addEventListener("click", () => { sessionDrawer.hidden = true; });
$("#session-new").addEventListener("click", () => {
  bridge.newSession();
  sessionDrawer.hidden = true;
});

/* 顶部带圈加号：新建对话（清空对话区并重置 Agent 历史，标题回到「新会话」） */
newChatBtn.addEventListener("click", () => {
  if (!bridge) return;
  sessionDrawer.hidden = true;
  closeMenus();
  bridge.newSession();
});

/* ---------------- 主题/模型 ---------------- */

function applyState(jsonText) {
  try {
    const st = JSON.parse(jsonText);
    if (st.theme) document.body.dataset.theme = st.theme;
    if (st.model !== undefined) {
      const t = document.querySelector("#model-text");
      if (t) t.textContent = st.model;
    }
  } catch (_) {}
}

/* ---------------- QWebChannel 初始化 ---------------- */

window.addEventListener("DOMContentLoaded", () => {
  if (typeof qt === "undefined") {
    addSystemRow("未检测到 Qt WebChannel（请在应用内打开）。");
    return;
  }
  new QWebChannel(qt.webChannelTransport, (channel) => {
    bridge = channel.objects.bridge;
    bridge.userAdded.connect(addUserRow);
    bridge.systemAdded.connect(addSystemRow);
    bridge.assistantStarted.connect(assistantStarted);
    bridge.assistantToken.connect(assistantToken);
    bridge.assistantFinished.connect(assistantFinished);
    bridge.reasoningToken.connect(reasoningToken);
    bridge.reasoningFinished.connect(finishReasoning);
    bridge.toolAdded.connect(toolAdded);
    bridge.toolUpdated.connect(toolUpdated);
    bridge.execStatus.connect(execStatus);
    bridge.approvalRequested.connect(approvalRequested);
    bridge.askUserRequested.connect(askUserRequested);
    bridge.referenceAdded.connect(referenceAdded);
    bridge.planApprovalRequested.connect(planApprovalRequested);
    bridge.turnFinished.connect(sweepStuckRows);
    bridge.historyCleared.connect((mode) => {
      try {
        chat.innerHTML = "";
        assistantEl = null; mdEl = null; mdBuf = "";
        titleSet = false;
        setSessionTitle("新会话");
        showWelcome();
      } catch (e) {
        window.__hcErr = String(e.message || e);
      }
    });
    bridge.contextStats.connect((s) => {
      try { renderCtx(JSON.parse(s)); } catch (_) {}
    });
    bridge.statePayload.connect(applyState);
    bridge.frontendReady();
    bridge.requestInit();
    ctxChip.hidden = false;  // 圆环一开始就显示(0%,统计推送后更新)
    input.focus();
  });
});
