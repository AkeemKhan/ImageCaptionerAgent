"use strict";

const IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp"];
const UPLOAD_BATCH = 20;
const TRAY_PREVIEWS = 12;

const TOOL_LABELS = {
  list_images: "Listed images",
  read_png_metadata: "Read metadata",
  caption_from_png_info: "PNG info caption",
  caption_with_vision: "Anthropic caption (no PNG info)",
  describe_image: "Anthropic caption",
  ENHANCE_CAPTION: "Enhanced caption",
  BLIP_CAPTIONER: "BLIP caption",
  caption_image: "Captioned with the local model",
  write_caption: "Saved caption",
  flip_image: "Created flipped copy",
  mirror_caption: "Mirrored left/right in caption",
  caption_folder: "Captioned folder",
};

// Steps that change what the captions panel shows.
const OUTPUT_TOOLS = ["caption_from_png_info", "caption_with_vision", "write_caption", "flip_image", "ENHANCE_CAPTION"];

const $ = (id) => document.getElementById(id);
const els = {
  app: document.querySelector(".app"),
  sessions: $("sessions"),
  title: $("chat-title"),
  messages: $("messages"),
  composer: $("composer"),
  input: $("input"),
  send: $("send"),
  tray: $("tray"),
  fileInput: $("file-input"),
  folderInput: $("folder-input"),
  outputs: $("outputs"),
  outputCount: $("output-count"),
  download: $("download"),
  dropOverlay: $("drop-overlay"),
  chat: $("chat"),
  modelLabel: $("model-label"),
};

const state = {
  sessionId: null,
  staged: [], // { file, url }
  skipped: 0, // non-image files ignored in the current selection
  busy: false,
  pendingChoice: null,
  assistantEl: null, // bubble receiving streamed text
  assistantText: "",
  activity: null, // current group of tool steps
};

// ---------- helpers ----------

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let detail = response.statusText;
    try {
      detail = (await response.json()).detail || detail;
    } catch {
      /* not JSON */
    }
    throw new Error(typeof detail === "string" ? detail : "Request failed");
  }
  return response;
}

function isImageName(name) {
  const lower = name.toLowerCase();
  return IMAGE_EXTENSIONS.some((ext) => lower.endsWith(ext));
}

function scrollToEnd() {
  els.messages.scrollTop = els.messages.scrollHeight;
}

function escapeHtml(text) {
  return text.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

/** Minimal Markdown to HTML. Input is escaped first, so the output is safe to insert. */
function renderMarkdown(source) {
  const inline = (text) =>
    text
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/(^|[\s(])\*([^*\s][^*]*)\*/g, "$1<em>$2</em>")
      .replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');

  const html = [];
  let list = null;
  let paragraph = [];
  const flushParagraph = () => {
    if (paragraph.length) html.push(`<p>${paragraph.join("<br>")}</p>`);
    paragraph = [];
  };
  const closeList = () => {
    if (list) html.push(`</${list}>`);
    list = null;
  };

  const blocks = escapeHtml(source).split(/```/);
  blocks.forEach((block, index) => {
    if (index % 2 === 1) {
      flushParagraph();
      closeList();
      html.push(`<pre><code>${block.replace(/^[\w-]*\n/, "")}</code></pre>`);
      return;
    }
    for (const line of block.split("\n")) {
      const heading = line.match(/^#{1,6}\s+(.*)/);
      const bullet = line.match(/^\s*[-*]\s+(.*)/);
      const numbered = line.match(/^\s*\d+[.)]\s+(.*)/);
      if (heading) {
        flushParagraph();
        closeList();
        html.push(`<h3>${inline(heading[1])}</h3>`);
      } else if (bullet || numbered) {
        flushParagraph();
        const kind = bullet ? "ul" : "ol";
        if (list !== kind) {
          closeList();
          html.push(`<${kind}>`);
          list = kind;
        }
        html.push(`<li>${inline((bullet || numbered)[1])}</li>`);
      } else if (!line.trim()) {
        flushParagraph();
        closeList();
      } else {
        closeList();
        paragraph.push(inline(line));
      }
    }
  });
  flushParagraph();
  closeList();
  return html.join("");
}

// ---------- transcript rendering ----------

function showEmptyState() {
  els.messages.replaceChildren();
  const empty = el("div", "empty");
  empty.append(
    el("h2", "", "Caption your images"),
    el(
      "p",
      "",
      "Drop, paste or attach images or a folder. Each image gets a .txt caption with the same name, " +
        "taken from its embedded generation prompt or written by the local vision model."
    )
  );
  els.messages.append(empty);
}

function addMessageNode(node) {
  els.messages.querySelector(".empty")?.remove();
  els.messages.append(node);
  scrollToEnd();
}

function addUserMessage(text, files) {
  state.activity = null;
  const node = el("div", "msg msg-user");
  if (text) node.append(el("div", "", text));
  if (files && files.length) {
    const label = files.length === 1 ? files[0] : `${files.length} images: ${files.slice(0, 3).join(", ")}${files.length > 3 ? ", …" : ""}`;
    node.append(el("div", "attached", `Attached ${label}`));
  }
  addMessageNode(node);
}

function addAssistantText(text) {
  state.activity = null;
  if (!state.assistantEl) {
    state.assistantEl = el("div", "msg msg-assistant");
    state.assistantText = "";
    addMessageNode(state.assistantEl);
  }
  state.assistantText += text;
  state.assistantEl.innerHTML = renderMarkdown(state.assistantText);
  scrollToEnd();
}

function endAssistantText() {
  state.assistantEl = null;
  state.assistantText = "";
}

function addNotice(text, isError = false) {
  state.activity = null;
  addMessageNode(el("div", isError ? "error-msg" : "notice", text));
}

function stepLabel(name, input) {
  const target = (input && (input.path || input.image_path || input.directory)) || "";
  const base = String(target).split("/").pop();
  return `${TOOL_LABELS[name] || name}${base ? ` · ${base}` : ""}`;
}

function ensureActivity() {
  if (state.activity) return state.activity;
  endAssistantText();
  const details = el("details", "activity");
  const summary = el("summary");
  const list = el("ul");
  details.append(summary, list);
  addMessageNode(details);
  state.activity = { details, summary, list, steps: new Map() };
  return state.activity;
}

function updateActivitySummary(activity) {
  const steps = [...activity.steps.values()];
  const running = steps.filter((s) => s.status === "running");
  const failed = steps.filter((s) => s.status === "failed").length;
  const done = steps.length - running.length;
  activity.summary.replaceChildren();
  if (running.length) {
    activity.summary.append(el("span", "spinner"), el("span", "", `${running[0].label} (${done} of ${steps.length} done)`));
  } else {
    const noun = steps.length === 1 ? "step" : "steps";
    activity.summary.append(el("span", "", failed ? `${steps.length} ${noun}, ${failed} failed` : `${steps.length} ${noun} completed`));
  }
}

function setStep(id, name, input, status, summary) {
  const activity = ensureActivity();
  let step = activity.steps.get(id);
  if (!step) {
    const row = el("li");
    const mark = el("span");
    const label = el("span", "step-label");
    row.append(mark, label);
    activity.list.append(row);
    step = { row, mark, labelEl: label };
    activity.steps.set(id, step);
  }
  // A step can finish under a different name than it started with: the
  // captioning pipeline starts with PNG info and may end with the vision model.
  step.label = stepLabel(name, input);
  step.labelEl.textContent = step.label;
  step.status = status;
  step.mark.className = status === "running" ? "spinner" : status === "ok" ? "step-ok" : "step-failed";
  step.mark.textContent = status === "running" ? "" : status === "ok" ? "✓" : "✕";
  if (summary) step.row.title = summary;
  updateActivitySummary(activity);
  scrollToEnd();
}

function addChoice(choice, active) {
  state.activity = null;
  endAssistantText();
  const card = el("div", "choice");
  card.dataset.choiceId = choice.id;
  card.append(el("p", "choice-question", choice.question));
  const options = el("div", "choice-options");
  for (const option of choice.options) {
    const button = el("button", "btn", option);
    button.type = "button";
    button.disabled = !active;
    button.addEventListener("click", () => sendMessage(option));
    options.append(button);
  }
  card.append(options);
  if (!active) card.classList.add("answered");
  addMessageNode(card);
}

function closeChoices() {
  for (const card of els.messages.querySelectorAll(".choice:not(.answered)")) {
    card.classList.add("answered");
    card.querySelectorAll("button").forEach((b) => (b.disabled = true));
  }
  state.pendingChoice = null;
}

function renderTranscript(transcript, pendingChoice) {
  showEmptyState();
  state.activity = null;
  endAssistantText();
  for (const item of transcript) {
    if (item.type === "user") addUserMessage(item.text, item.files);
    else if (item.type === "assistant") {
      addAssistantText(item.text);
      endAssistantText();
    } else if (item.type === "tool") setStep(item.id, item.name, item.input, item.ok ? "ok" : "failed", item.summary);
    else if (item.type === "choice") addChoice(item, Boolean(pendingChoice && pendingChoice.id === item.id));
    else if (item.type === "notice") addNotice(item.text);
  }
  state.activity = null;
  state.pendingChoice = pendingChoice;
  updatePlaceholder();
}

function handleEvent(event) {
  switch (event.type) {
    case "text":
      addAssistantText(event.text);
      break;
    case "text_end":
      endAssistantText();
      break;
    case "tool_start":
      setStep(event.id, event.name, event.input, "running");
      break;
    case "tool_end":
      setStep(event.id, event.name, event.input, event.ok ? "ok" : "failed", event.summary);
      if (event.ok && OUTPUT_TOOLS.includes(event.name)) scheduleOutputsRefresh();
      break;
    case "choice":
      state.pendingChoice = event;
      addChoice(event, true);
      break;
    case "notice":
      addNotice(event.text);
      break;
    case "error":
      addNotice(event.message, true);
      break;
    case "done":
      if (event.title) els.title.textContent = event.title;
      break;
  }
}

// ---------- sessions ----------

async function loadSessions() {
  const sessions = await (await api("/api/sessions")).json();
  els.sessions.replaceChildren();
  for (const session of sessions) {
    const row = el("div", "session" + (session.id === state.sessionId ? " active" : ""));
    row.tabIndex = 0;
    row.append(el("span", "session-title", session.title || "New chat"));
    const remove = el("button", "icon-btn", "×");
    remove.type = "button";
    remove.setAttribute("aria-label", "Delete chat");
    remove.addEventListener("click", (e) => {
      e.stopPropagation();
      deleteSession(session.id, session.title);
    });
    row.append(remove);
    row.addEventListener("click", () => selectSession(session.id));
    row.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && e.target === row) selectSession(session.id);
    });
    els.sessions.append(row);
  }
}

async function selectSession(id) {
  if (state.busy) return;
  const session = await (await api(`/api/sessions/${id}`)).json();
  state.sessionId = id;
  els.title.textContent = session.title || "New chat";
  renderTranscript(session.transcript, session.pending_choice);
  els.app.classList.remove("sidebar-open");
  await Promise.all([loadSessions(), refreshOutputs()]);
}

function newChat() {
  if (state.busy) return;
  state.sessionId = null;
  els.title.textContent = "New chat";
  renderTranscript([], null);
  clearStaged();
  els.app.classList.remove("sidebar-open");
  loadSessions();
  refreshOutputs();
  els.input.focus();
}

async function deleteSession(id, title) {
  if (!confirm(`Delete "${title || "New chat"}" and its images and captions?`)) return;
  try {
    await api(`/api/sessions/${id}`, { method: "DELETE" });
  } catch (error) {
    alert(error.message);
    return;
  }
  if (id === state.sessionId) newChat();
  else loadSessions();
}

// ---------- attachments ----------

function stageFiles(files) {
  for (const file of files) {
    if (isImageName(file.name)) state.staged.push({ file, url: URL.createObjectURL(file) });
    else state.skipped += 1;
  }
  renderTray();
}

function clearStaged() {
  state.staged.forEach((item) => URL.revokeObjectURL(item.url));
  state.staged = [];
  state.skipped = 0;
  renderTray();
}

function renderTray(note) {
  els.tray.replaceChildren();
  state.staged.slice(0, TRAY_PREVIEWS).forEach((item) => {
    const thumb = el("div", "thumb");
    const img = el("img");
    img.src = item.url;
    img.alt = item.file.name;
    img.title = item.file.name;
    const remove = el("button", "", "×");
    remove.type = "button";
    remove.setAttribute("aria-label", `Remove ${item.file.name}`);
    remove.addEventListener("click", () => {
      URL.revokeObjectURL(item.url);
      state.staged = state.staged.filter((other) => other !== item);
      renderTray();
    });
    thumb.append(img, remove);
    els.tray.append(thumb);
  });
  const notes = [];
  if (state.staged.length > TRAY_PREVIEWS) notes.push(`+${state.staged.length - TRAY_PREVIEWS} more`);
  if (state.skipped) notes.push(`${state.skipped} non-image file${state.skipped === 1 ? "" : "s"} skipped`);
  if (note) notes.push(note);
  if (notes.length) els.tray.append(el("span", "tray-note", notes.join(" · ")));
  els.tray.hidden = !els.tray.childElementCount;
}

function pastedName(index) {
  const stamp = new Date().toISOString().replace(/[-:T]/g, "").slice(0, 14);
  return `pasted-${stamp}-${index + 1}.png`;
}

/** Collect files from a drop, descending into dropped folders. */
async function filesFromDrop(dataTransfer) {
  const entries = [...dataTransfer.items].map((item) => item.webkitGetAsEntry?.()).filter(Boolean);
  if (!entries.length) return [...dataTransfer.files];
  const files = [];
  const walk = async (entry) => {
    if (entry.isFile) {
      files.push(await new Promise((resolve, reject) => entry.file(resolve, reject)));
      return;
    }
    const reader = entry.createReader();
    for (;;) {
      const batch = await new Promise((resolve, reject) => reader.readEntries(resolve, reject));
      if (!batch.length) break;
      for (const child of batch) await walk(child);
    }
  };
  for (const entry of entries) await walk(entry);
  return files;
}

async function uploadStaged() {
  const saved = [];
  const rejected = [];
  const files = state.staged.map((item) => item.file);
  for (let start = 0; start < files.length; start += UPLOAD_BATCH) {
    renderTray(`Uploading ${Math.min(start + UPLOAD_BATCH, files.length)} of ${files.length}…`);
    const form = new FormData();
    files.slice(start, start + UPLOAD_BATCH).forEach((file) => form.append("files", file, file.name));
    const result = await (await api(`/api/sessions/${state.sessionId}/files`, { method: "POST", body: form })).json();
    saved.push(...result.saved);
    rejected.push(...result.rejected);
  }
  return { saved, rejected };
}

// ---------- sending ----------

function setBusy(busy) {
  state.busy = busy;
  els.send.disabled = busy;
  els.send.textContent = busy ? "Working…" : "Send";
}

function updatePlaceholder() {
  els.input.placeholder = state.pendingChoice
    ? "Pick an option above, or type your answer"
    : "Drop, paste or attach images, then tell me what you need";
}

async function readEventStream(response) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let boundary;
    while ((boundary = buffer.indexOf("\n\n")) >= 0) {
      const chunk = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      if (chunk.startsWith("data: ")) handleEvent(JSON.parse(chunk.slice(6)));
    }
  }
}

async function sendMessage(overrideText) {
  if (state.busy) return;
  const text = (overrideText ?? els.input.value).trim();
  if (!text && !state.staged.length) return;
  setBusy(true);
  try {
    if (!state.sessionId) {
      state.sessionId = (await (await api("/api/sessions", { method: "POST" })).json()).id;
    }
    let files = [];
    if (state.staged.length) {
      const { saved, rejected } = await uploadStaged();
      files = saved;
      clearStaged();
      if (rejected.length) {
        addNotice(`Not uploaded: ${rejected.map((r) => `${r.name} (${r.reason})`).join("; ")}`, true);
      }
      if (!files.length && !text) return;
    }
    if (overrideText === undefined) {
      els.input.value = "";
      autoGrow();
    }
    closeChoices();
    updatePlaceholder();
    addUserMessage(text, files);
    refreshOutputs();
    const response = await api(`/api/sessions/${state.sessionId}/messages`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, files }),
    });
    await readEventStream(response);
  } catch (error) {
    addNotice(error.message || "Request failed", true);
  } finally {
    endAssistantText();
    state.activity = null;
    setBusy(false);
    updatePlaceholder();
    loadSessions();
    refreshOutputs();
  }
}

// ---------- captions panel ----------

let refreshTimer = null;
function scheduleOutputsRefresh() {
  clearTimeout(refreshTimer);
  refreshTimer = setTimeout(refreshOutputs, 300);
}

async function refreshOutputs() {
  if (!state.sessionId) {
    renderOutputs([]);
    return;
  }
  const id = state.sessionId;
  try {
    const items = await (await api(`/api/sessions/${id}/outputs`)).json();
    if (id === state.sessionId) renderOutputs(items);
  } catch {
    /* the panel keeps its last state */
  }
}

function renderOutputs(items) {
  // Keep unsaved edits across refreshes.
  const drafts = new Map();
  for (const area of els.outputs.querySelectorAll("textarea[data-dirty='1']")) drafts.set(area.dataset.name, area.value);

  const captioned = items.filter((item) => item.caption !== null).length;
  els.outputCount.textContent = items.length ? `${captioned}/${items.length}` : "";
  els.download.hidden = !captioned;
  els.download.href = state.sessionId ? `/api/sessions/${state.sessionId}/download` : "#";
  els.outputs.replaceChildren();
  if (!items.length) {
    els.outputs.append(el("p", "outputs-empty", "Captions appear here as they are written. You can edit them before downloading."));
    return;
  }
  for (const item of items) els.outputs.append(outputCard(item, drafts.get(item.name)));
}

function outputCard(item, draft) {
  const card = el("div", "output");
  const head = el("div", "output-head");
  const base = `/api/sessions/${state.sessionId}/files/${item.kind}/${encodeURIComponent(item.name)}`;
  const link = el("a");
  link.href = base;
  link.target = "_blank";
  link.rel = "noopener";
  const img = el("img");
  img.src = `${base}?thumb=1`;
  img.alt = item.name;
  img.loading = "lazy";
  link.append(img);
  const name = el("div", "output-name", item.name);
  if (item.flipped) name.append(" ", el("span", "badge", "flipped"));
  name.append(el("span", "output-file", item.caption_file));
  head.append(link, name);
  card.append(head);

  if (item.caption === null && draft === undefined) {
    card.append(el("div", "pending", "No caption yet"));
    return card;
  }
  const area = el("textarea");
  area.value = draft ?? item.caption;
  area.dataset.name = item.name;
  area.setAttribute("aria-label", `Caption for ${item.name}`);
  const foot = el("div", "output-foot");
  const status = el("span", "output-status");
  const save = el("button", "btn", "Save");
  save.type = "button";
  save.hidden = draft === undefined;
  if (draft !== undefined) area.dataset.dirty = "1";
  area.addEventListener("input", () => {
    area.dataset.dirty = "1";
    save.hidden = false;
    status.textContent = "";
  });
  save.addEventListener("click", async () => {
    save.disabled = true;
    try {
      await api(`/api/sessions/${state.sessionId}/captions/${encodeURIComponent(item.name)}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: area.value }),
      });
      delete area.dataset.dirty;
      save.hidden = true;
      status.textContent = "Saved";
    } catch (error) {
      status.textContent = error.message;
    } finally {
      save.disabled = false;
    }
  });
  foot.append(status, save);
  card.append(area, foot);
  return card;
}

// ---------- wiring ----------

function autoGrow() {
  els.input.style.height = "auto";
  els.input.style.height = `${Math.min(els.input.scrollHeight, 200)}px`;
}

els.composer.addEventListener("submit", (e) => {
  e.preventDefault();
  sendMessage();
});
els.input.addEventListener("input", autoGrow);
els.input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    sendMessage();
  }
});

$("new-chat").addEventListener("click", newChat);
$("attach-files").addEventListener("click", () => els.fileInput.click());
$("attach-folder").addEventListener("click", () => els.folderInput.click());
$("toggle-panel").addEventListener("click", () => els.app.classList.toggle("panel-closed"));
$("toggle-sidebar").addEventListener("click", () => els.app.classList.toggle("sidebar-open"));

for (const input of [els.fileInput, els.folderInput]) {
  input.addEventListener("change", () => {
    stageFiles([...input.files]);
    input.value = "";
  });
}

document.addEventListener("paste", (e) => {
  const images = [...(e.clipboardData?.files || [])].filter((file) => file.type.startsWith("image/"));
  if (!images.length) return;
  e.preventDefault();
  // Screenshots arrive as a generic "image.png"; give each a distinct name.
  stageFiles(images.map((file, i) => (file.name && file.name !== "image.png" ? file : new File([file], pastedName(i), { type: "image/png" }))));
});

let dragDepth = 0;
els.chat.addEventListener("dragenter", (e) => {
  if (!e.dataTransfer?.types.includes("Files")) return;
  e.preventDefault();
  dragDepth += 1;
  els.dropOverlay.hidden = false;
});
els.chat.addEventListener("dragover", (e) => {
  if (e.dataTransfer?.types.includes("Files")) e.preventDefault();
});
els.chat.addEventListener("dragleave", () => {
  dragDepth = Math.max(0, dragDepth - 1);
  if (!dragDepth) els.dropOverlay.hidden = true;
});
els.chat.addEventListener("drop", async (e) => {
  e.preventDefault();
  dragDepth = 0;
  els.dropOverlay.hidden = true;
  stageFiles(await filesFromDrop(e.dataTransfer));
});

(async function init() {
  if (window.matchMedia("(max-width: 1000px)").matches) els.app.classList.add("panel-closed");
  showEmptyState();
  renderOutputs([]);
  try {
    const config = await (await api("/api/config")).json();
    els.modelLabel.textContent = config.model;
    await loadSessions();
  } catch (error) {
    addNotice(`Could not reach the server: ${error.message}`, true);
  }
})();
