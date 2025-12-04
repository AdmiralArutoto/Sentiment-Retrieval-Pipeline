const form = document.getElementById("query-form");
const queryInput = document.getElementById("query");
const resultsContainer = document.getElementById("results");
const statusEl = document.getElementById("status");

const setStatus = (message, isError = false) => {
  statusEl.textContent = message;
  statusEl.style.color = isError ? "#b91c1c" : "#475569";
};

const renderResults = (results) => {
  resultsContainer.innerHTML = "";
  if (!results.length) {
    resultsContainer.innerHTML = "<p>No chunks met the similarity threshold.</p>";
    return;
  }

  results.forEach((result, index) => {
    const wrapper = document.createElement("article");
    wrapper.className = "result-card";
    wrapper.innerHTML = `
      <div class="result-meta">
        <span class="meta-pill">#${index + 1}</span>
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
  setStatus("Fetching matching chunks...");

  try {
    const response = await fetch("/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail ?? "Unknown error");
    }

    const payload = await response.json();
    renderResults(payload.results);
    setStatus(`Found ${payload.results.length} chunk(s).`);
  } catch (error) {
    console.error(error);
    setStatus(error.message || "Something went wrong.", true);
  } finally {
    submitButton.disabled = false;
  }
});
