const apiBase = `${window.location.protocol}//${window.location.hostname}:8000`;

const promptInput = document.getElementById("prompt-input");
const generationOutput = document.getElementById("generation-output");
const topicInput = document.getElementById("topic-input");
const batchCountInput = document.getElementById("batch-count-input");
const batchGenerationOutput = document.getElementById("batch-generation-output");
const captionInput = document.getElementById("caption-input");
const predictionOutput = document.getElementById("prediction-output");
const summaryOutput = document.getElementById("summary-output");
const historyOutput = document.getElementById("history-output");
const generationsOutput = document.getElementById("generations-output");
const healthBadge = document.getElementById("health-badge");
let activeJobPoll = null;

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

function renderGenerationJob(job) {
  const previewCards = job.preview_items
    .map((item, index) => `
      <article class="batch-item">
        <img class="generated-image" src="${apiBase}${item.preview_url}" alt="Generated diffusion output ${index + 1}">
        <p><strong>#${index + 1}</strong> ID: ${item.id}, Seed: ${item.seed}</p>
        <p>${item.prompt}</p>
      </article>
    `)
    .join("");
  const metadataRows = job.items.map((item, index) => ({
    n: index + 1,
    generation_id: item.id,
    batch_id: item.batch_id,
    seed: item.seed,
    prompt: item.prompt,
    output_path: item.output_path,
  }));
  const metadataTable = renderTable(
    [
      { key: "n", label: "#" },
      { key: "generation_id", label: "Generation ID" },
      { key: "batch_id", label: "Batch ID" },
      { key: "seed", label: "Seed" },
      { key: "prompt", label: "Prompt" },
      { key: "output_path", label: "Output path" },
    ],
    metadataRows,
  );
  const manifestLink = job.manifest_url
    ? `<p><a class="download-link" href="${apiBase}${job.manifest_url}" target="_blank" rel="noreferrer">Скачать manifest CSV</a></p>`
    : "<p>Manifest появится после завершения job.</p>";

  batchGenerationOutput.innerHTML = `
    <p>Status: <strong>${job.status}</strong>. Progress: ${job.progress.generated}/${job.progress.requested} (${job.progress.percent}%).</p>
    <p>Job/Batch ID для дальнейшего дообучения: <code>${job.job_id}</code>. Тема: <strong>${job.topic}</strong>.</p>
    ${manifestLink}
    <div class="batch-grid">${previewCards || "<p>Preview появятся после первых сгенерированных изображений.</p>"}</div>
    <div class="table-wrap">${metadataTable}</div>
    <pre class="output small">${JSON.stringify({
      job_id: job.job_id,
      status: job.status,
      base_seed: job.base_seed,
      manifest_path: job.manifest_path,
      error: job.error,
    }, null, 2)}</pre>
  `;
}

async function pollGenerationJob(jobId) {
  const job = await fetchJson(`/generation-jobs/${jobId}`);
  renderGenerationJob(job);
  await loadGenerations();
  if (["done", "failed"].includes(job.status) && activeJobPoll) {
    clearInterval(activeJobPoll);
    activeJobPoll = null;
  }
}

async function generateBatch() {
  const count = Number.parseInt(batchCountInput.value, 10);
  if (!Number.isInteger(count) || count < 1 || count > 1000) {
    batchGenerationOutput.textContent = "Укажи количество от 1 до 1000.";
    return;
  }

  batchGenerationOutput.innerHTML = `<p>Создаю batch job на ${count} изображений. После создания worker продолжит генерацию в фоне, даже если не держать запрос открытым.</p>`;
  try {
    const job = await fetchJson("/generation-jobs", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        topic: topicInput.value,
        count,
        width: 256,
        height: 256,
        num_inference_steps: 12,
        guidance_scale: 5.0,
        preview_limit: 5,
      }),
    });

    renderGenerationJob(job);
    if (activeJobPoll) {
      clearInterval(activeJobPoll);
    }
    activeJobPoll = setInterval(() => {
      pollGenerationJob(job.job_id).catch((error) => {
        batchGenerationOutput.textContent = `Ошибка обновления job status: ${error.message}`;
      });
    }, 5000);
    await pollGenerationJob(job.job_id);
    await loadHealth();
  } catch (error) {
    batchGenerationOutput.textContent = `Ошибка создания batch job: ${error.message}`;
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
      generation_id: item.id,
      batch_id: item.batch_id ?? "",
      source: item.source ?? "",
      seed: item.seed,
      latency_ms: item.latency_ms.toFixed(1),
      device: item.device,
      prompt: item.prompt,
      output_path: item.output_path,
    }));
    generationsOutput.innerHTML = renderTable(
      [
        { key: "created_at", label: "Время" },
        { key: "generation_id", label: "ID" },
        { key: "batch_id", label: "Batch" },
        { key: "source", label: "Source" },
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

document.getElementById("generate-btn")?.addEventListener("click", generateImage);
document.getElementById("batch-generate-btn")?.addEventListener("click", generateBatch);
document.getElementById("predict-btn")?.addEventListener("click", predict);
document.getElementById("refresh-history-btn")?.addEventListener("click", loadHistory);
document.getElementById("refresh-generations-btn")?.addEventListener("click", loadGenerations);
document.getElementById("refresh-experiments-btn")?.addEventListener("click", loadSummary);

loadHealth();
loadSummary();
loadHistory();
loadGenerations();
