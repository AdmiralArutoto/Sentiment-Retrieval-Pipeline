const form = document.getElementById("query-form");
const queryInput = document.getElementById("query");
const topKInput = document.getElementById("top-k");
const minScoreInput = document.getElementById("min-score");
const maxTokensInput = document.getElementById("max-output-tokens");
const resultsContainer = document.getElementById("results");
const answerEl = document.getElementById("answer");
const statusEl = document.getElementById("status");

const setStatus = (message, isError = false) => {
  statusEl.textContent = message;
  statusEl.style.color = isError ? "#b91c1c" : "#475569";
};

const applyConfigDefaults = (config) => {
  if (config?.default_top_k !== undefined && topKInput) topKInput.value = config.default_top_k;
  if (config?.default_min_score !== undefined && minScoreInput) minScoreInput.value = config.default_min_score;
  if (config?.default_max_output_tokens !== undefined && maxTokensInput)
    maxTokensInput.value = config.default_max_output_tokens;
};

const loadConfigDefaults = async () => {
  try {
    const response = await fetch("/config");
    if (!response.ok) return;
    const config = await response.json();
    applyConfigDefaults(config);
  } catch (error) {
    console.warn("Could not load server defaults", error);
  }
};

const renderAnswer = (answer) => {
  if (!answerEl) return;
  answerEl.textContent = answer || "No answer generated from the available context.";
};

const renderResults = (results) => {
  resultsContainer.innerHTML = "";
  if (!results.length) {
    resultsContainer.innerHTML = "<p>No supporting chunks met the similarity threshold.</p>";
    return;
  }

  results.forEach((result, index) => {
    const wrapper = document.createElement("article");
    wrapper.className = "result-card";
    wrapper.innerHTML = `
      <div class="result-meta">
        <span class="meta-pill">#${index + 1}</span>
        <span class="meta-pill">${result.chunk_id}</span>
        <span class="meta-pill">Chunk ${result.metadata.chunk_index}</span>
        <span class="meta-pill">Score: ${result.score}</span>
        <span class="meta-pill">Sentiment: ${result.metadata.sentiment}</span>
        <span class="meta-pill">Source: ${result.metadata.source}</span>
      </div>
      <pre>${result.text}</pre>
      <p class="result-meta">
        User ${result.metadata.user_id} • ${result.metadata.location} • ${result.metadata.date}
      </p>
    `;
    resultsContainer.appendChild(wrapper);
  });
};

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const query = queryInput.value.trim();
  if (!query) {
    setStatus("Please enter a query.", true);
    return;
  }

  const submitButton = form.querySelector("button");
  submitButton.disabled = true;
  setStatus("Retrieving context and generating answer...");

  try {
    const topK = Number(topKInput?.value || 4);
    const minScore = Number(minScoreInput?.value || 0.62);
    const maxTokens = Number(maxTokensInput?.value || 256);

    const response = await fetch("/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query,
        top_k: topK,
        min_score: minScore,
        max_output_tokens: maxTokens,
      }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail ?? "Unknown error");
    }

    const payload = await response.json();
    renderAnswer(payload.answer);
    renderResults(payload.citations || []);
    setStatus(`Generated answer with ${payload.citations?.length ?? 0} supporting chunk(s).`);
  } catch (error) {
    console.error(error);
    setStatus(error.message || "Something went wrong.", true);
  } finally {
    submitButton.disabled = false;
  }
});

loadConfigDefaults();
