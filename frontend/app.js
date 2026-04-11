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
const jobsOutput = document.getElementById("jobs-output");
const monitoringOutput = document.getElementById("monitoring-output");
const healthBadge = document.getElementById("health-badge");

let activeJobPoll = null;

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function formatNumber(value, digits = 0) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "—";
  }
  return Number(value).toFixed(digits);
}

function renderTable(columns, rows) {
  if (!rows.length) {
    return "<p>Нет данных.</p>";
  }

  const header = columns.map((col) => `<th>${escapeHtml(col.label)}</th>`).join("");
  const body = rows
    .map((row) => {
      const cells = columns
        .map((col) => `<td>${row[col.key] ?? ""}</td>`)
        .join("");
      return `<tr>${cells}</tr>`;
    })
    .join("");

  return `<table><thead><tr>${header}</tr></thead><tbody>${body}</tbody></table>`;
}

function metricCard(label, value, tone = "") {
  return `
    <article class="metric-card ${tone}">
      <span class="metric-label">${escapeHtml(label)}</span>
      <strong class="metric-value">${escapeHtml(value)}</strong>
    </article>
  `;
}

function parsePrometheusMetrics(text) {
  const metrics = {};
  const lines = text.split("\n");
  const linePattern = /^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{([^}]*)\})?\s+([0-9eE+\-.]+)$/;

  for (const rawLine of lines) {
    const line = rawLine.trim();
    if (!line || line.startsWith("#")) {
      continue;
    }

    const match = line.match(linePattern);
    if (!match) {
      continue;
    }

    const [, name, , labelsText, valueText] = match;
    const labels = {};
    if (labelsText) {
      for (const part of labelsText.split(",")) {
        const [key, rawValue] = part.split("=");
        labels[key] = rawValue.replace(/^"|"$/g, "");
      }
    }

    if (!metrics[name]) {
      metrics[name] = [];
    }

    metrics[name].push({
      labels,
      value: Number.parseFloat(valueText),
    });
  }

  return metrics;
}

function sumMetric(metricRows, predicate = () => true) {
  return (metricRows || [])
    .filter(predicate)
    .reduce((acc, row) => acc + row.value, 0);
}

function findMetric(metricRows, predicate = () => true) {
  return (metricRows || []).find(predicate)?.value ?? null;
}

async function fetchJson(path, options = {}) {
  const response = await fetch(`${apiBase}${path}`, options);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(`${response.status} ${response.statusText}: ${text}`);
  }
  return response.json();
}

async function fetchText(path) {
  const response = await fetch(`${apiBase}${path}`);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(`${response.status} ${response.statusText}: ${text}`);
  }
  return response.text();
}

async function loadHealth() {
  try {
    const data = await fetchJson("/health");
    const diffusionState = data.diffusion_model_loaded ? "diffusion loaded" : "diffusion lazy";
    healthBadge.textContent =
      data.status === "ok"
        ? `Сервис доступен, ${diffusionState}, DB ok`
        : "Сервис в degraded-режиме";
    healthBadge.className = `badge ${data.status === "ok" ? "ok" : "fail"}`;
  } catch (error) {
    healthBadge.textContent = `Health error: ${error.message}`;
    healthBadge.className = "badge fail";
  }
}

async function loadMonitoring() {
  try {
    const rawMetrics = await fetchText("/metrics");
    const metrics = parsePrometheusMetrics(rawMetrics);

    const totalHttpRequests = sumMetric(metrics.http_requests_total);
    const generateRequests = sumMetric(
      metrics.http_requests_total,
      (row) => row.labels.method === "POST" && row.labels.path === "/generate",
    );
    const generationJobs = sumMetric(
      metrics.http_requests_total,
      (row) => row.labels.method === "POST" && row.labels.path === "/generation-jobs",
    );
    const successGenerations = sumMetric(
      metrics.ml_generations_total,
      (row) => row.labels.status === "success",
    );
    const generationLatencySum = findMetric(metrics.ml_generation_latency_seconds_sum) ?? 0;
    const generationLatencyCount = findMetric(metrics.ml_generation_latency_seconds_count) ?? 0;
    const avgGenerationLatency =
      generationLatencyCount > 0 ? generationLatencySum / generationLatencyCount : null;
    const dbHealth = findMetric(metrics.db_health_status);
    const diffusionLoaded = findMetric(metrics.diffusion_model_loaded_status);

    monitoringOutput.innerHTML = [
      metricCard("HTTP requests", formatNumber(totalHttpRequests), "neutral"),
      metricCard("POST /generate", formatNumber(generateRequests), "neutral"),
      metricCard("POST /generation-jobs", formatNumber(generationJobs), "neutral"),
      metricCard("Успешные одиночные генерации", formatNumber(successGenerations), "good"),
      metricCard(
        "Avg /generate latency, s",
        avgGenerationLatency === null ? "—" : formatNumber(avgGenerationLatency, 2),
        avgGenerationLatency !== null && avgGenerationLatency > 7 ? "warn" : "neutral",
      ),
      metricCard("DB health", dbHealth === 1 ? "ok" : "fail", dbHealth === 1 ? "good" : "bad"),
      metricCard(
        "Diffusion model loaded",
        diffusionLoaded === 1 ? "yes" : "no",
        diffusionLoaded === 1 ? "good" : "warn",
      ),
    ].join("");
  } catch (error) {
    monitoringOutput.innerHTML = `<p>Ошибка загрузки мониторинга: ${escapeHtml(error.message)}</p>`;
  }
}

async function generateImage() {
  generationOutput.innerHTML =
    "<p>Генерация запущена. Первый запрос может быть дольше, потому что модель загружается лениво.</p>";
  try {
    const data = await fetchJson("/generate", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        prompt: promptInput.value,
        width: 320,
        height: 320,
        num_inference_steps: 22,
        guidance_scale: 7.5,
      }),
    });

    generationOutput.innerHTML = `
      <img class="generated-image" src="data:${data.mime_type};base64,${data.image_base64}" alt="Generated diffusion output">
      <pre class="output small">${escapeHtml(JSON.stringify({
        generation_id: data.generation_id,
        seed: data.seed,
        latency_ms: data.latency_ms,
        base_model_checkpoint: data.base_model_checkpoint,
        output_path: data.output_path,
        output_object_uri: data.output_object_uri,
      }, null, 2))}</pre>
    `;

    await Promise.all([loadHealth(), loadMonitoring(), loadGenerations(), loadJobs()]);
  } catch (error) {
    generationOutput.textContent = `Ошибка генерации: ${error.message}`;
  }
}

function renderGenerationJob(job) {
  const previewCards = job.preview_items
    .map(
      (item, index) => `
        <article class="batch-item">
          <img class="generated-image" src="${apiBase}${item.preview_url}" alt="Generated diffusion output ${index + 1}">
          <p><strong>Preview #${index + 1}</strong></p>
          <p>ID: <code>${item.id}</code>, seed: <code>${item.seed}</code></p>
          <p>${escapeHtml(item.prompt)}</p>
        </article>
      `,
    )
    .join("");

  const itemIds = job.items.map((item) => item.id).filter(Boolean);
  const averageLatencyMs =
    job.items.length > 0
      ? job.items.reduce((sum, item) => sum + Number(item.latency_ms || 0), 0) / job.items.length
      : null;
  const sampleRows = job.items.slice(0, 8).map((item, index) => ({
    n: index + 1,
    generation_id: `<code>${item.id}</code>`,
    seed: `<code>${item.seed}</code>`,
    prompt: escapeHtml(item.prompt),
    output_object_uri: item.output_object_uri
      ? `<code>${escapeHtml(item.output_object_uri)}</code>`
      : "—",
  }));
  const moreCount = Math.max(job.items.length - sampleRows.length, 0);

  const manifestLink = job.manifest_url
    ? `<a class="download-link" href="${apiBase}${job.manifest_url}" target="_blank" rel="noreferrer">Скачать manifest CSV</a>`
    : "<span>Manifest появится после завершения job.</span>";

  const cards = `
    <div class="job-summary-grid">
      ${metricCard("Статус", job.status, ["done", "running"].includes(job.status) ? "good" : "warn")}
      ${metricCard("Job ID", job.job_id, "neutral")}
      ${metricCard("Тема", job.topic, "neutral")}
      ${metricCard("Запрошено", formatNumber(job.count_requested), "neutral")}
      ${metricCard("Сгенерировано", formatNumber(job.count_generated), "good")}
      ${metricCard("Preview в UI", formatNumber(job.preview_limit), "neutral")}
      ${metricCard(
        "Диапазон generation_id",
        itemIds.length ? `${Math.min(...itemIds)}-${Math.max(...itemIds)}` : "—",
        "neutral",
      )}
      ${metricCard(
        "Средняя latency, ms",
        averageLatencyMs === null ? "—" : formatNumber(averageLatencyMs, 1),
        "neutral",
      )}
    </div>
  `;

  const datasetMetrics = job.dataset_metrics || {};
  const datasetMetricCards = `
    <div class="job-summary-grid">
      ${metricCard("Dataset sample size", formatNumber(datasetMetrics.sample_count), "neutral")}
      ${metricCard(
        "Avg latency, ms",
        datasetMetrics.avg_latency_ms === null ? "—" : formatNumber(datasetMetrics.avg_latency_ms, 2),
        "neutral",
      )}
      ${metricCard(
        "Min / Max latency, ms",
        datasetMetrics.min_latency_ms === null || datasetMetrics.max_latency_ms === null
          ? "—"
          : `${formatNumber(datasetMetrics.min_latency_ms, 1)} / ${formatNumber(datasetMetrics.max_latency_ms, 1)}`,
        "neutral",
      )}
      ${metricCard(
        "Images per minute",
        datasetMetrics.images_per_minute === null ? "—" : formatNumber(datasetMetrics.images_per_minute, 2),
        "neutral",
      )}
      ${metricCard(
        "Avg PNG size, KB",
        datasetMetrics.avg_file_size_kb === null ? "—" : formatNumber(datasetMetrics.avg_file_size_kb, 2),
        "neutral",
      )}
      ${metricCard(
        "Avg quality score",
        datasetMetrics.avg_quality_score === null ? "—" : formatNumber(datasetMetrics.avg_quality_score, 4),
        "good",
      )}
      ${metricCard(
        "High-quality share",
        datasetMetrics.high_quality_share === null ? "—" : formatNumber(datasetMetrics.high_quality_share, 4),
        "good",
      )}
      ${metricCard(
        "Prompt uniqueness ratio",
        datasetMetrics.prompt_uniqueness_ratio === null ? "—" : formatNumber(datasetMetrics.prompt_uniqueness_ratio, 4),
        "neutral",
      )}
      ${metricCard(
        "Near-duplicate rate",
        datasetMetrics.near_duplicate_rate === null ? "—" : formatNumber(datasetMetrics.near_duplicate_rate, 4),
        datasetMetrics.near_duplicate_rate !== null && datasetMetrics.near_duplicate_rate > 0.1 ? "warn" : "good",
      )}
      ${metricCard("Metrics scope", datasetMetrics.metrics_scope ?? "—", "neutral")}
    </div>
  `;

  const sampleTable = renderTable(
    [
      { key: "n", label: "#" },
      { key: "generation_id", label: "Generation ID" },
      { key: "seed", label: "Seed" },
      { key: "prompt", label: "Prompt" },
      { key: "output_object_uri", label: "Object storage URI" },
    ],
    sampleRows,
  );

  batchGenerationOutput.innerHTML = `
    <p><strong>Фоновая задача создана для генерации датасета.</strong> UI показывает только первые preview и короткую выборку metadata. Полный список лежит в manifest CSV и в PostgreSQL/MinIO.</p>
    ${cards}
    <h3>Dataset-level metrics</h3>
    <p class="hint">Усреднённые метрики считаются по всей batch-партии и помогают показать качество и свойства синтетического датасета на защите.</p>
    ${datasetMetricCards}
    <div class="inline-links">
      ${manifestLink}
      <span>Manifest object: <code>${escapeHtml(job.manifest_object_uri ?? "—")}</code></span>
    </div>
    <div class="batch-grid">${previewCards || "<p>Preview появятся после первых сгенерированных изображений.</p>"}</div>
    <div class="table-wrap">${sampleTable}</div>
    ${
      moreCount > 0
        ? `<p class="hint">В UI скрыто ещё ${moreCount} строк metadata. Для полной партии используй manifest CSV.</p>`
        : ""
    }
    <pre class="output small">${escapeHtml(JSON.stringify({
      job_id: job.job_id,
      status: job.status,
      progress: job.progress,
      base_seed: job.base_seed,
      manifest_path: job.manifest_path,
      error: job.error,
    }, null, 2))}</pre>
  `;
}

async function pollGenerationJob(jobId) {
  const job = await fetchJson(`/generation-jobs/${jobId}`);
  renderGenerationJob(job);
  await Promise.all([loadGenerations(), loadJobs(), loadMonitoring()]);

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

  batchGenerationOutput.innerHTML =
    `<p>Создаю batch job на ${count} изображений. Worker продолжит генерацию в фоне, а в интерфейсе будет только краткая сводка и первые 5 preview.</p>`;

  try {
    const job = await fetchJson("/generation-jobs", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        topic: topicInput.value,
        count,
        width: 320,
        height: 320,
        num_inference_steps: 22,
        guidance_scale: 7.5,
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

    await Promise.all([pollGenerationJob(job.job_id), loadHealth(), loadJobs(), loadMonitoring()]);
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
      created_at: escapeHtml(item.created_at),
      predicted_label: escapeHtml(item.predicted_label),
      confidence: formatNumber(item.confidence, 4),
      latency_ms: formatNumber(item.latency_ms, 3),
      caption: escapeHtml(item.caption),
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
      created_at: escapeHtml(item.created_at),
      generation_id: `<code>${item.id}</code>`,
      batch_id: item.batch_id ? `<code>${escapeHtml(item.batch_id)}</code>` : "—",
      source: escapeHtml(item.source ?? ""),
      topic: escapeHtml(item.topic ?? ""),
      seed: `<code>${item.seed}</code>`,
      latency_ms: formatNumber(item.latency_ms, 1),
      device: escapeHtml(item.device),
      output_object_uri: item.output_object_uri
        ? `<code>${escapeHtml(item.output_object_uri)}</code>`
        : "—",
    }));

    generationsOutput.innerHTML = renderTable(
      [
        { key: "created_at", label: "Время" },
        { key: "generation_id", label: "ID" },
        { key: "batch_id", label: "Batch" },
        { key: "source", label: "Source" },
        { key: "topic", label: "Topic" },
        { key: "seed", label: "Seed" },
        { key: "latency_ms", label: "Latency ms" },
        { key: "device", label: "Device" },
        { key: "output_object_uri", label: "Object storage URI" },
      ],
      rows,
    );
  } catch (error) {
    generationsOutput.textContent = `Ошибка загрузки generation history: ${error.message}`;
  }
}

async function loadJobs() {
  try {
    const data = await fetchJson("/generation-jobs?limit=10");
    const rows = data.items.map((job) => ({
      created_at: escapeHtml(job.created_at),
      job_id: `<code>${escapeHtml(job.job_id)}</code>`,
      topic: escapeHtml(job.topic),
      status: escapeHtml(job.status),
      requested: formatNumber(job.count_requested),
      generated: formatNumber(job.count_generated),
      manifest: job.manifest_url
        ? `<a class="download-link" href="${apiBase}${job.manifest_url}" target="_blank" rel="noreferrer">CSV</a>`
        : "—",
    }));

    jobsOutput.innerHTML = renderTable(
      [
        { key: "created_at", label: "Время" },
        { key: "job_id", label: "Job ID" },
        { key: "topic", label: "Тема" },
        { key: "status", label: "Статус" },
        { key: "requested", label: "Запрошено" },
        { key: "generated", label: "Сгенерировано" },
        { key: "manifest", label: "Manifest" },
      ],
      rows,
    );
  } catch (error) {
    jobsOutput.textContent = `Ошибка загрузки generation jobs: ${error.message}`;
  }
}

document.getElementById("generate-btn")?.addEventListener("click", generateImage);
document.getElementById("batch-generate-btn")?.addEventListener("click", generateBatch);
document.getElementById("predict-btn")?.addEventListener("click", predict);
document.getElementById("refresh-history-btn")?.addEventListener("click", loadHistory);
document.getElementById("refresh-generations-btn")?.addEventListener("click", loadGenerations);
document.getElementById("refresh-experiments-btn")?.addEventListener("click", loadSummary);
document.getElementById("refresh-jobs-btn")?.addEventListener("click", loadJobs);
document.getElementById("refresh-monitoring-btn")?.addEventListener("click", async () => {
  await Promise.all([loadHealth(), loadMonitoring(), loadJobs()]);
});

loadHealth();
loadMonitoring();
loadSummary();
loadHistory();
loadGenerations();
loadJobs();
