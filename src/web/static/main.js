/**
 * AI VTT Agents Team – Web UI Frontend Logic
 *
 * Handles:
 *  - Config loading / saving via REST API
 *  - Pipeline execution via WebSocket with real-time log streaming
 *  - Markdown result rendering
 */

// ===================== DOM refs =====================
const $ = (sel) => document.querySelector(sel);

const cfgApiKey       = $("#cfg-api-key");
const cfgModelName    = $("#cfg-model-name");
const cfgBaseUrl      = $("#cfg-base-url");
const cfgWhisperModel = $("#cfg-whisper-model");
const cfgSceneThresh  = $("#cfg-scene-threshold");
const cfgMinInterval  = $("#cfg-min-interval");
const cfgMaxDuration  = $("#cfg-max-duration");
const cfgSegDuration  = $("#cfg-segment-duration");
const btnSaveConfig   = $("#btn-save-config");
const configStatus    = $("#config-status");

const inputPrompt     = $("#input-prompt");
const btnStart        = $("#btn-start");
const chatMessages    = $("#chat-messages");
const resultArea      = $("#result-area");
const resultContent   = $("#result-content");
const btnCopyResult   = $("#btn-copy-result");
const btnDownload     = $("#btn-download-result");

// Progress bar refs
const progressBar     = $("#progress-bar");
const progressLabel   = $("#progress-label");
const progressCount   = $("#progress-count");
const progressFill    = $("#progress-fill");

// Batch mode refs (safe – may be null if HTML is cached)
const btnModeSingle   = $("#btn-mode-single");
const btnModeBatch    = $("#btn-mode-batch");
const inputSingle     = $("#input-single");
const inputBatch      = $("#input-batch");
const inputBatchDir   = $("#input-batch-dir");
const batchLanguage   = $("#batch-language");
const batchConcurrency= $("#batch-concurrency");
const btnBatchStart   = $("#btn-batch-start");

// Batch history refs
const batchHistoryList = $("#batch-history-list");
const btnRefreshBatches = $("#btn-refresh-batches");

let ws = null;
let rawArticle = "";
let resultOutputPath = "";
let currentMode = "single";  // "single" | "batch"

// ===================== Config =====================

async function loadConfig() {
  try {
    const resp = await fetch("/api/config");
    const cfg = await resp.json();

    const mc = (cfg.model_configs || [])[0] || {};
    cfgApiKey.value      = mc.api_key && mc.api_key !== "${DASHSCOPE_API_KEY}" ? mc.api_key : "";
    cfgModelName.value    = mc.model_name || "qwen3.6-plus";
    cfgBaseUrl.value      = mc.base_url || "https://dashscope.aliyuncs.com/compatible-mode/v1";

    const ac = cfg.agent_configs || {};
    cfgWhisperModel.value = (ac.transcriber || {}).whisper_model_size || "small";
    cfgSceneThresh.value  = (ac.keyframe_extractor || {}).scene_threshold || 0.08;
    cfgMinInterval.value  = (ac.keyframe_extractor || {}).min_interval_sec || 5;

    // Video split settings are not in config file yet – use defaults
    cfgMaxDuration.value  = 30;
    cfgSegDuration.value  = 10;
  } catch (e) {
    console.error("Failed to load config:", e);
  }
}

function buildConfigPayload() {
  const apiKeyVal = cfgApiKey.value.trim();
  return {
    model_configs: [
      {
        config_name: "dashscope_qwen",
        model_type: "openai_chat",
        model_name: cfgModelName.value.trim(),
        api_key: apiKeyVal || "${DASHSCOPE_API_KEY}",
        base_url: cfgBaseUrl.value.trim(),
      },
    ],
    agent_configs: {
      transcriber: {
        model_config_name: "dashscope_qwen",
        whisper_model_size: cfgWhisperModel.value,
      },
      keyframe_extractor: {
        model_config_name: "dashscope_qwen",
        scene_threshold: parseFloat(cfgSceneThresh.value),
        min_interval_sec: parseInt(cfgMinInterval.value, 10),
      },
      summarizer: { model_config_name: "dashscope_qwen" },
      translator:  { model_config_name: "dashscope_qwen" },
      proofreader: { model_config_name: "dashscope_qwen" },
    },
  };
}

async function saveConfig() {
  btnSaveConfig.disabled = true;
  configStatus.textContent = "";
  try {
    const resp = await fetch("/api/config", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(buildConfigPayload()),
    });
    if (resp.ok) {
      configStatus.textContent = "已保存";
      configStatus.className = "config-status ok";
    } else {
      configStatus.textContent = "保存失败";
      configStatus.className = "config-status err";
    }
  } catch (e) {
    configStatus.textContent = "网络错误";
    configStatus.className = "config-status err";
  } finally {
    btnSaveConfig.disabled = false;
    setTimeout(() => { configStatus.textContent = ""; }, 3000);
  }
}

btnSaveConfig.addEventListener("click", saveConfig);

// ===================== Chat Messages =====================

function addMessage(text, type = "log") {
  const div = document.createElement("div");
  div.className = `msg msg-${type}`;
  div.textContent = text;
  chatMessages.appendChild(div);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function clearMessages() {
  chatMessages.innerHTML = "";
  resultArea.style.display = "none";
  resultContent.innerHTML = "";
  rawArticle = "";
  resultOutputPath = "";
  hideProgress();
}

// ===================== WebSocket Pipeline =====================

function startPipeline() {
  const prompt = inputPrompt.value.trim();
  if (!prompt) {
    addMessage("请输入指令", "error");
    return;
  }

  clearMessages();
  addMessage(prompt, "info");
  btnStart.disabled = true;
  btnStart.textContent = "处理中...";

  const protocol = location.protocol === "https:" ? "wss:" : "ws:";
  ws = new WebSocket(`${protocol}//${location.host}/ws/pipeline`);

  ws.onopen = () => {
    addMessage("已连接服务器，正在解析指令...", "stage");
    ws.send(JSON.stringify({ prompt }));
  };

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      handleMessage(data);
    } catch (e) {
      addMessage(event.data, "log");
    }
  };

  ws.onerror = () => {
    addMessage("WebSocket 连接错误", "error");
    resetStartButton();
  };

  ws.onclose = () => {
    resetStartButton();
  };
}

function startBatchPipeline() {
  const dirEl = document.getElementById("input-batch-dir");
  const langEl = document.getElementById("batch-language");
  const concEl = document.getElementById("batch-concurrency");

  if (!dirEl) {
    addMessage("页面元素未加载，请刷新页面后重试 (Cmd+Shift+R)", "error");
    return;
  }

  const dir = dirEl.value.trim();

  if (!dir) {
    addMessage("请输入视频目录路径", "error");
    return;
  }

  const language = (langEl && langEl.value.trim()) || "中文";
  const concurrency = (concEl && parseInt(concEl.value, 10)) || 3;

  clearMessages();
  addMessage(`批量模式: 扫描目录 ${dir}, 并发度 ${concurrency}, 目标语言: ${language}`, "info");
  btnBatchStart.disabled = true;
  btnBatchStart.textContent = "处理中...";

  const protocol = location.protocol === "https:" ? "wss:" : "ws:";
  ws = new WebSocket(`${protocol}//${location.host}/ws/pipeline`);

  ws.onopen = () => {
    addMessage("已连接服务器，开始批量处理...", "stage");
    ws.send(JSON.stringify({
      video_dir: dir,
      target_language: language,
      max_concurrency: concurrency,
    }));
  };

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      handleMessage(data);
    } catch (e) {
      addMessage(event.data, "log");
    }
  };

  ws.onerror = () => {
    addMessage("WebSocket 连接错误", "error");
    resetBatchButton();
  };

  ws.onclose = () => {
    resetBatchButton();
    loadBatchHistory();
  };
}

function handleMessage(data) {
  switch (data.type) {
    case "log":
      addMessage(data.message, "log");
      break;
    case "stage":
      addMessage(data.message || `${data.stage}: ${data.status}`, "stage");
      break;
    case "progress":
      updateProgress(data);
      break;
    case "error":
      addMessage(data.message, "error");
      hideProgress();
      break;
    case "result":
      addMessage("处理完成!", "result");
      showResult(data.article, data.output_path);
      hideProgress();
      break;
    case "batch_result":
      addMessage("批量处理完成!", "result");
      showBatchResult(data.results || []);
      hideProgress();
      break;
    default:
      addMessage(JSON.stringify(data), "info");
  }
}

// ===================== Progress Bar =====================

function updateProgress(data) {
  if (!progressBar) return;
  const cur = data.current || 0;
  const tot = data.total || 1;
  const pct = Math.round((cur / tot) * 100);
  const videoName = data.video_name || "";
  const stage = data.stage || "processing";

  progressBar.style.display = "";

  if (stage === "processing") {
    progressLabel.textContent = videoName
      ? `正在处理: ${videoName}`
      : "正在处理...";
    progressFill.style.width = `${Math.max(((cur - 1) / tot) * 100, 0)}%`;
    progressFill.className = "progress-fill";
  } else if (stage === "done") {
    progressFill.style.width = `${pct}%`;
    progressFill.className = "progress-fill done";
  }

  progressCount.textContent = `片段 ${cur}/${tot}`;
}

function hideProgress() {
  if (progressBar) progressBar.style.display = "none";
}

// ===================== Result Display =====================

function showResult(article, outputPath) {
  rawArticle = article || "";
  resultOutputPath = outputPath || "";
  resultArea.style.display = "block";

  // Determine the project directory from output_path
  // e.g. /Users/me/videos/topic-name/file.md -> /Users/me/videos/topic-name/
  let projectDir = "";
  if (resultOutputPath) {
    const lastSlash = resultOutputPath.lastIndexOf("/");
    if (lastSlash > 0) projectDir = resultOutputPath.substring(0, lastSlash + 1);
  }

  // Rewrite relative image paths to use /api/file endpoint
  let displayArticle = rawArticle;
  if (projectDir) {
    displayArticle = displayArticle.replace(
      /!\[([^\]]*)\]\((?!\/|https?:\/\/)([^)]+)\)/g,
      (match, alt, src) => {
        const absPath = projectDir + src;
        return `![${alt}](/api/file?path=${encodeURIComponent(absPath)})`;
      }
    );
  }

  if (typeof marked !== "undefined") {
    resultContent.innerHTML = marked.parse(displayArticle);
  } else {
    const pre = document.createElement("pre");
    pre.textContent = rawArticle;
    resultContent.innerHTML = "";
    resultContent.appendChild(pre);
  }
  // Scroll to result
  resultArea.scrollIntoView({ behavior: "smooth" });
}

function resetStartButton() {
  btnStart.disabled = false;
  btnStart.textContent = "发送";
  ws = null;
}

function resetBatchButton() {
  if (btnBatchStart) {
    btnBatchStart.disabled = false;
    btnBatchStart.textContent = "开始批量处理";
  }
  ws = null;
}

function showBatchResult(results) {
  resultArea.style.display = "block";
  const successCount = results.filter(r => r.status === "success").length;
  const failCount = results.filter(r => r.status === "failed").length;

  let html = `<div class="batch-summary">`;
  html += `<h4>批量处理结果: 成功 ${successCount} / 失败 ${failCount} / 共 ${results.length}</h4>`;
  html += `<table class="batch-table"><thead><tr><th>状态</th><th>视频</th><th>操作</th></tr></thead><tbody>`;

  for (const r of results) {
    const icon = r.status === "success" ? "✓" : "✗";
    const cls = r.status === "success" ? "batch-ok" : "batch-fail";
    const name = r.video_path ? r.video_path.split("/").pop() : "未知";
    let action = "";
    if (r.status === "success" && r.output_path) {
      const dlUrl = `/api/download?path=${encodeURIComponent(r.output_path)}`;
      action = `<a href="${dlUrl}" class="btn btn-sm" download>下载 .md</a>`;
    } else if (r.error) {
      action = `<span class="batch-error-text">${r.error}</span>`;
    }
    html += `<tr class="${cls}"><td>${icon}</td><td title="${r.video_path}">${name}</td><td>${action}</td></tr>`;
  }

  html += `</tbody></table></div>`;
  resultContent.innerHTML = html;
  resultArea.scrollIntoView({ behavior: "smooth" });
}

btnStart.addEventListener("click", startPipeline);
if (btnBatchStart) btnBatchStart.addEventListener("click", startBatchPipeline);

// ===================== Mode Toggle =====================

function setMode(mode) {
  currentMode = mode;
  if (mode === "single") {
    if (inputSingle) inputSingle.style.display = "";
    if (inputBatch) inputBatch.style.display = "none";
    if (btnModeSingle) btnModeSingle.classList.add("active");
    if (btnModeBatch) btnModeBatch.classList.remove("active");
  } else {
    if (inputSingle) inputSingle.style.display = "none";
    if (inputBatch) inputBatch.style.display = "";
    if (btnModeSingle) btnModeSingle.classList.remove("active");
    if (btnModeBatch) btnModeBatch.classList.add("active");
    loadBatchHistory();
  }
}

if (btnModeSingle) btnModeSingle.addEventListener("click", () => setMode("single"));
if (btnModeBatch) btnModeBatch.addEventListener("click", () => setMode("batch"));

// Copy result
btnCopyResult.addEventListener("click", () => {
  if (!rawArticle) return;
  navigator.clipboard.writeText(rawArticle).then(() => {
    btnCopyResult.textContent = "已复制";
    setTimeout(() => { btnCopyResult.textContent = "复制"; }, 2000);
  });
});

// Download result
btnDownload.addEventListener("click", () => {
  if (resultOutputPath) {
    // Download via server API
    const url = `/api/download?path=${encodeURIComponent(resultOutputPath)}`;
    const a = document.createElement("a");
    a.href = url;
    a.download = "";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  } else if (rawArticle) {
    // Fallback: download from memory
    const blob = new Blob([rawArticle], { type: "text/markdown;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "article.md";
    document.body.appendChild(a);
    a.click();
    URL.revokeObjectURL(url);
    document.body.removeChild(a);
  }
});

// Enter key starts pipeline
inputPrompt.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !btnStart.disabled) startPipeline();
});

// ===================== Batch History =====================

const STATUS_LABELS = {
  pending: "等待中",
  running: "运行中",
  completed: "已完成",
  partial: "部分完成",
  failed: "失败",
};

const STATUS_CLS = {
  pending: "st-pending",
  running: "st-running",
  completed: "st-completed",
  partial: "st-partial",
  failed: "st-failed",
};

async function loadBatchHistory() {
  try {
    const resp = await fetch("/api/batches?limit=10");
    const batches = await resp.json();
    renderBatchHistory(batches);
  } catch (e) {
    console.error("Failed to load batch history:", e);
    batchHistoryList.innerHTML = '<div class="batch-empty">加载失败</div>';
  }
}

function renderBatchHistory(batches) {
  if (!batches || batches.length === 0) {
    batchHistoryList.innerHTML = '<div class="batch-empty">暂无历史批次</div>';
    return;
  }

  let html = '<table class="batch-table"><thead><tr>'
    + '<th>批次</th><th>目录</th><th>状态</th><th>总数</th><th>创建时间</th><th>操作</th>'
    + '</tr></thead><tbody>';

  for (const b of batches) {
    const stLabel = STATUS_LABELS[b.status] || b.status;
    const stCls = STATUS_CLS[b.status] || "";
    const dir = b.video_dir || "-";
    const shortDir = dir.length > 30 ? "..." + dir.slice(-27) : dir;
    const time = b.created_at ? b.created_at.replace("T", " ").slice(0, 19) : "-";

    let actions = `<button class="btn btn-sm" onclick="viewBatchDetail('${b.batch_id}')">详情</button>`;
    if (b.status === "running" || b.status === "partial") {
      actions += ` <button class="btn btn-sm btn-accent" onclick="resumeBatch('${b.batch_id}')">续传</button>`;
    }

    html += `<tr>
      <td title="${b.batch_id}">${b.batch_id.slice(0, 8)}</td>
      <td title="${dir}">${shortDir}</td>
      <td><span class="st-badge ${stCls}">${stLabel}</span></td>
      <td>${b.total}</td>
      <td>${time}</td>
      <td>${actions}</td>
    </tr>`;
  }

  html += '</tbody></table>';
  batchHistoryList.innerHTML = html;
}

async function viewBatchDetail(batchId) {
  try {
    const resp = await fetch(`/api/batches/${batchId}`);
    const detail = await resp.json();
    if (detail.error) {
      addMessage(detail.error, "error");
      return;
    }
    showBatchDetailPanel(detail.batch, detail.tasks);
  } catch (e) {
    addMessage("查询批次详情失败: " + e.message, "error");
  }
}

function showBatchDetailPanel(batch, tasks) {
  resultArea.style.display = "block";
  const completed = tasks.filter(t => t.status === "completed").length;
  const failed = tasks.filter(t => t.status === "failed").length;
  const processing = tasks.filter(t => t.status === "processing").length;
  const pending = tasks.filter(t => t.status === "pending").length;

  let html = `<div class="batch-summary">`;
  html += `<h4>批次 ${batch.batch_id} 任务详情</h4>`;
  html += `<p class="batch-stats">完成 ${completed} / 失败 ${failed} / 处理中 ${processing} / 等待 ${pending} / 共 ${tasks.length}</p>`;
  html += `<table class="batch-table"><thead><tr><th>状态</th><th>视频</th><th>开始时间</th><th>完成时间</th><th>操作</th></tr></thead><tbody>`;

  for (const t of tasks) {
    const stLabel = STATUS_LABELS[t.status] || t.status;
    const stCls = STATUS_CLS[t.status] || "";
    const name = t.video_name || "未知";
    const started = t.started_at ? t.started_at.replace("T", " ").slice(11, 19) : "-";
    const ended = t.completed_at ? t.completed_at.replace("T", " ").slice(11, 19) : "-";
    let action = "";
    if (t.status === "completed" && t.output_path) {
      const dlUrl = `/api/download?path=${encodeURIComponent(t.output_path)}`;
      action = `<a href="${dlUrl}" class="btn btn-sm" download>下载</a>`;
    } else if (t.status === "failed" && t.error) {
      action = `<span class="batch-error-text" title="${t.error}">${t.error.slice(0, 40)}</span>`;
    }
    html += `<tr><td><span class="st-badge ${stCls}">${stLabel}</span></td><td title="${t.video_path}">${name}</td><td>${started}</td><td>${ended}</td><td>${action}</td></tr>`;
  }

  html += `</tbody></table></div>`;
  resultContent.innerHTML = html;
  resultArea.scrollIntoView({ behavior: "smooth" });
}

async function resumeBatch(batchId) {
  clearMessages();
  addMessage(`恢复批次 ${batchId}，重新处理未完成的视频...`, "info");
  btnBatchStart.disabled = true;
  btnBatchStart.textContent = "处理中...";

  // 先查询待处理的视频路径以获取目录等信息
  let batchInfo;
  try {
    const resp = await fetch(`/api/batches/${batchId}`);
    batchInfo = await resp.json();
  } catch (e) {
    addMessage("查询批次信息失败: " + e.message, "error");
    resetBatchButton();
    return;
  }

  const batch = batchInfo.batch;
  if (!batch) {
    addMessage("批次不存在", "error");
    resetBatchButton();
    return;
  }

  const language = batch.target_language || "中文";
  const concurrency = batch.concurrency || 3;
  const videoDir = batch.video_dir || "";

  const protocol = location.protocol === "https:" ? "wss:" : "ws:";
  ws = new WebSocket(`${protocol}//${location.host}/ws/pipeline`);

  ws.onopen = () => {
    addMessage("已连接服务器，开始恢复批量处理...", "stage");
    // 发送目录 + resume_batch_id
    ws.send(JSON.stringify({
      video_dir: videoDir,
      target_language: language,
      max_concurrency: concurrency,
      resume_batch_id: batchId,
    }));
  };

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      handleMessage(data);
    } catch (e) {
      addMessage(event.data, "log");
    }
  };

  ws.onerror = () => {
    addMessage("WebSocket 连接错误", "error");
    resetBatchButton();
  };

  ws.onclose = () => {
    resetBatchButton();
    loadBatchHistory();
  };
}

if (btnRefreshBatches) btnRefreshBatches.addEventListener("click", loadBatchHistory);

// ===================== Init =====================
loadConfig();
