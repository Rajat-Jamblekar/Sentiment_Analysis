const form = document.getElementById("analyze-form");
const labelDiv = document.getElementById("label");
const detailPre = document.getElementById("detail");
const ctx = document.getElementById("barChart");

let chart = null;

function colorForLabel(label) {
  if (label === "positive") return "#1f9d55";     // greenish
  if (label === "negative") return "#d64545";     // redish
  return "#3b82f6";                                // blue (neutral)
}

function updateChart(scores) {
  const data = {
    labels: ["Positive", "Neutral", "Negative"],
    datasets: [{
      label: "Sentiment probabilities",
      data: [scores.positive, scores.neutral, scores.negative]
    }]
  };

  if (chart) {
    chart.destroy();
  }
  chart = new Chart(ctx, {
    type: "bar",
    data: data,
    options: {
      responsive: true,
      scales: {
        y: { beginAtZero: true, suggestedMax: 1.0 }
      }
    }
  });
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();

  labelDiv.textContent = "Analyzing...";
  labelDiv.className = "label label-wait";
  detailPre.textContent = "";

  const fd = new FormData(form);

  try {
    const res = await fetch("/api/analyze", {
      method: "POST",
      body: fd
    });

    const payload = await res.json();
    if (!res.ok) throw new Error(payload.error || "Unknown error");

    const label = payload.label || "unknown";
    labelDiv.textContent = label.toUpperCase();
    labelDiv.className = `label label-${label}`;
    updateChart(payload.scores || {positive: 0, neutral: 0, negative: 0});

    const detail = payload.detail ? JSON.stringify(payload.detail, null, 2) : "{}";
    detailPre.textContent = detail;

  } catch (err) {
    labelDiv.textContent = `Error: ${err.message}`;
    labelDiv.className = "label label-error";
    if (chart) chart.destroy();
    detailPre.textContent = "";
  }
});