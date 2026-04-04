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

let ws = null;
let rawArticle = "";
let resultOutputPath = "";

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

function handleMessage(data) {
  switch (data.type) {
    case "log":
      addMessage(data.message, "log");
      break;
    case "stage":
      addMessage(data.message || `${data.stage}: ${data.status}`, "stage");
      break;
    case "error":
      addMessage(data.message, "error");
      break;
    case "result":
      addMessage("处理完成!", "result");
      showResult(data.article, data.output_path);
      break;
    default:
      addMessage(JSON.stringify(data), "info");
  }
}

function showResult(article, outputPath) {
  rawArticle = article || "";
  resultOutputPath = outputPath || "";
  resultArea.style.display = "block";

  // Determine the articles base path from output_path
  // e.g. output/articles/topic-name/file.md -> /articles/topic-name/
  let articlesBase = "";
  if (resultOutputPath) {
    const m = resultOutputPath.match(/articles\/([^/]+)\//); 
    if (m) articlesBase = `/articles/${m[1]}/`;
  }

  // Rewrite relative image paths so browser can find them
  let displayArticle = rawArticle;
  if (articlesBase) {
    displayArticle = displayArticle.replace(
      /!\[([^\]]*)\]\((?!\/|https?:\/\/)([^)]+)\)/g,
      (match, alt, src) => `![${alt}](${articlesBase}${src})`
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

btnStart.addEventListener("click", startPipeline);

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

// ===================== Init =====================
loadConfig();
