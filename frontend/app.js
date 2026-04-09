const apiBase = `${window.location.protocol}//${window.location.hostname}:8000`;

const captionInput = document.getElementById("caption-input");
const predictionOutput = document.getElementById("prediction-output");
const summaryOutput = document.getElementById("summary-output");
const historyOutput = document.getElementById("history-output");
const experimentsOutput = document.getElementById("experiments-output");
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
    throw new Error(`${response.status} ${response.statusText}`);
  }
  return response.json();
}

async function loadHealth() {
  try {
    const data = await fetchJson("/health");
    healthBadge.textContent = data.status === "ok" ? "Сервис доступен" : "Сервис в degraded режиме";
    healthBadge.className = `badge ${data.status === "ok" ? "ok" : "fail"}`;
  } catch (error) {
    healthBadge.textContent = `Health error: ${error.message}`;
    healthBadge.className = "badge fail";
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

async function loadExperiments() {
  try {
    const data = await fetchJson("/experiments/logs?limit=10");
    const rows = data.items.map((item) => ({
      name: item.name,
      val_macro_f1: item.val_metrics.macro_f1.toFixed(4),
      test_macro_f1: item.test_metrics.macro_f1.toFixed(4),
      test_accuracy: item.test_metrics.accuracy.toFixed(4),
      latency_ms: item.latency_ms.toFixed(4),
    }));
    experimentsOutput.innerHTML = renderTable(
      [
        { key: "name", label: "Эксперимент" },
        { key: "val_macro_f1", label: "Val macro F1" },
        { key: "test_macro_f1", label: "Test macro F1" },
        { key: "test_accuracy", label: "Test accuracy" },
        { key: "latency_ms", label: "Latency ms" },
      ],
      rows,
    );
  } catch (error) {
    experimentsOutput.textContent = `Ошибка загрузки experiments: ${error.message}`;
  }
}

document.getElementById("predict-btn").addEventListener("click", predict);
document.getElementById("refresh-history-btn").addEventListener("click", loadHistory);
document.getElementById("refresh-experiments-btn").addEventListener("click", loadExperiments);

loadHealth();
loadSummary();
loadHistory();
loadExperiments();
