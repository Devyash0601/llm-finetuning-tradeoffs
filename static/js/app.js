function decodeHtmlEntities(text) {
    const textarea = document.createElement("textarea");
    textarea.innerHTML = text;
    return textarea.value;
  }
  
  async function runInference() {
    const prompt = document.getElementById("prompt").value;
    if (!prompt) return alert("Please enter a prompt.");
  
    const btn = document.getElementById("generate-btn");
    btn.innerText = "Generating...";
    btn.disabled = true;
  
    document.getElementById("full-output").innerText = "Running...";
    document.getElementById("lora-output").innerText = "Running...";
    document.getElementById("full-latency").innerText = "";
    document.getElementById("lora-latency").innerText = "";
  
    const payload = { text: prompt, max_new_tokens: 80 };
  
    const full = await fetch("/predict/full", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    }).then(r => r.json());
  
    document.getElementById("full-output").innerText =
      decodeHtmlEntities(full.output);
    document.getElementById("full-latency").innerText =
      `Latency: ${full.latency_seconds}s`;
  
    const lora = await fetch("/predict/lora", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    }).then(r => r.json());
  
    document.getElementById("lora-output").innerText =
      decodeHtmlEntities(lora.output);
    document.getElementById("lora-latency").innerText =
      `Latency: ${lora.latency_seconds}s`;
  
    btn.innerText = "Generate";
    btn.disabled = false;
  }