const apiBase = `${window.location.protocol}//${window.location.hostname}:8000`;

const promptInput = document.getElementById("prompt-input");
const generationOutput = document.getElementById("generation-output");
const captionInput = document.getElementById("caption-input");
const predictionOutput = document.getElementById("prediction-output");
const summaryOutput = document.getElementById("summary-output");
const historyOutput = document.getElementById("history-output");
const generationsOutput = document.getElementById("generations-output");
const healthBadge = document.getElementById("health-badge");

function renderTable(columns, rows) {
  if (!rows.length) {
    return "<p>Нет данных.</p>";
  }

  const header = columns.map((col) => `<th>${col.label}</th>`).join("");
  const body = rows
    .map((row) => {
      const cells = columns.map((col) => `<td>${row[col.key] ?? ""}</td>`).join("");
      return `<tr>${cells}</tr>`;
    })
    .join("");

  return `<table><thead><tr>${header}</tr></thead><tbody>${body}</tbody></table>`;
}

async function fetchJson(path, options = {}) {
  const response = await fetch(`${apiBase}${path}`, options);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(`${response.status} ${response.statusText}: ${text}`);
  }
  return response.json();
}

async function loadHealth() {
  try {
    const data = await fetchJson("/health");
    const diffusionState = data.diffusion_model_loaded ? "LoRA loaded" : "LoRA lazy";
    healthBadge.textContent = data.status === "ok" ? `Сервис доступен, ${diffusionState}` : "Сервис в degraded режиме";
    healthBadge.className = `badge ${data.status === "ok" ? "ok" : "fail"}`;
  } catch (error) {
    healthBadge.textContent = `Health error: ${error.message}`;
    healthBadge.className = "badge fail";
  }
}

async function generateImage() {
  generationOutput.innerHTML = "<p>Генерация запущена. Первый запрос может быть долгим: модель загружается лениво.</p>";
  try {
    const data = await fetchJson("/generate", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        prompt: promptInput.value,
        width: 256,
        height: 256,
        num_inference_steps: 12,
        guidance_scale: 5.0,
      }),
    });

    generationOutput.innerHTML = `
      <img class="generated-image" src="data:${data.mime_type};base64,${data.image_base64}" alt="Generated diffusion output">
      <pre class="output small">${JSON.stringify({
        seed: data.seed,
        latency_ms: data.latency_ms,
        base_model_checkpoint: data.base_model_checkpoint,
        lora_adapter_path: data.lora_adapter_path,
        output_path: data.output_path,
      }, null, 2)}</pre>
    `;
    await loadHealth();
    await loadGenerations();
  } catch (error) {
    generationOutput.textContent = `Ошибка генерации: ${error.message}`;
  }
}

async function predict() {
  predictionOutput.textContent = "Запрос отправлен...";
  try {
    const data = await fetchJson("/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ caption: captionInput.value }),
    });
    predictionOutput.textContent = JSON.stringify(data, null, 2);
    await loadHistory();
  } catch (error) {
    predictionOutput.textContent = `Ошибка: ${error.message}`;
  }
}

async function loadSummary() {
  try {
    const data = await fetchJson("/experiments/summary");
    summaryOutput.textContent = JSON.stringify(data, null, 2);
  } catch (error) {
    summaryOutput.textContent = `Ошибка загрузки summary: ${error.message}`;
  }
}

async function loadHistory() {
  try {
    const data = await fetchJson("/history?limit=15");
    const rows = data.items.map((item) => ({
      created_at: item.created_at,
      predicted_label: item.predicted_label,
      confidence: item.confidence.toFixed(4),
      latency_ms: item.latency_ms.toFixed(3),
      caption: item.caption,
    }));
    historyOutput.innerHTML = renderTable(
      [
        { key: "created_at", label: "Время" },
        { key: "predicted_label", label: "Label" },
        { key: "confidence", label: "Confidence" },
        { key: "latency_ms", label: "Latency ms" },
        { key: "caption", label: "Caption" },
      ],
      rows,
    );
  } catch (error) {
    historyOutput.textContent = `Ошибка загрузки history: ${error.message}`;
  }
}

async function loadGenerations() {
  try {
    const data = await fetchJson("/generations/history?limit=15");
    const rows = data.items.map((item) => ({
      created_at: item.created_at,
      seed: item.seed,
      latency_ms: item.latency_ms.toFixed(1),
      device: item.device,
      prompt: item.prompt,
      output_path: item.output_path,
    }));
    generationsOutput.innerHTML = renderTable(
      [
        { key: "created_at", label: "Время" },
        { key: "seed", label: "Seed" },
        { key: "latency_ms", label: "Latency ms" },
        { key: "device", label: "Device" },
        { key: "prompt", label: "Prompt" },
        { key: "output_path", label: "Output" },
      ],
      rows,
    );
  } catch (error) {
    generationsOutput.textContent = `Ошибка загрузки generation history: ${error.message}`;
  }
}

document.getElementById("generate-btn").addEventListener("click", generateImage);
document.getElementById("predict-btn").addEventListener("click", predict);
document.getElementById("refresh-history-btn").addEventListener("click", loadHistory);
document.getElementById("refresh-generations-btn").addEventListener("click", loadGenerations);
document.getElementById("refresh-experiments-btn").addEventListener("click", loadSummary);

loadHealth();
loadSummary();
loadHistory();
loadGenerations();
