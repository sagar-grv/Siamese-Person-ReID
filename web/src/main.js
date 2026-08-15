import * as ort from "onnxruntime-web";
import "./style.css";

const state = { session: null, gallery: [], queryBlob: null, queryName: "", queryEmbedding: null };
const $ = (id) => document.getElementById(id);
const status = $("model-status");
const dropzone = $("dropzone");
const liveRegion = $("live-region");

function setStatus(text, ready = false) {
  status.textContent = text;
  status.dataset.state = ready ? "ready" : "loading";
  document.body.classList.toggle("model-ready", ready);
  document.querySelector(".status-dot").classList.toggle("is-ready", ready);
}

function announce(message) {
  if (liveRegion) liveRegion.textContent = message;
}

function validateImageFile(file) {
  const supported = ["image/jpeg", "image/png", "image/webp"];
  if (!file || !supported.includes(file.type)) {
    announce("Please choose a JPG, PNG, or WEBP image.");
    return false;
  }
  if (file.size > 10 * 1024 * 1024) {
    announce("That image is larger than 10 megabytes. Please choose a smaller file.");
    return false;
  }
  return true;
}

async function handleFile(file) {
  if (!validateImageFile(file)) return;
  await setQuery(file, file.name);
  announce(`${file.name} is ready for search.`);
}

function normalizeEmbedding(vector) {
  const norm = Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0)) || 1;
  return vector.map((value) => value / norm);
}

function cosine(left, right) {
  let score = 0;
  for (let i = 0; i < left.length; i += 1) score += left[i] * right[i];
  return score;
}

function renderBars(vector = []) {
  const container = $("embedding-bars");
  container.innerHTML = "";
  const sample = vector.length ? vector.slice(0, 32) : Array.from({ length: 32 }, () => 0.15);
  sample.forEach((value) => {
    const bar = document.createElement("span");
    bar.style.height = `${Math.max(8, Math.min(100, 38 + Math.abs(value) * 110))}%`;
    bar.style.opacity = vector.length ? `${Math.max(0.35, Math.min(1, 0.45 + Math.abs(value)))}` : "0.22";
    container.appendChild(bar);
  });
}

function readImageSource(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}

function loadImage(source) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = reject;
    image.src = source;
  });
}

async function imageTensor(blob) {
  const image = await loadImage(await readImageSource(blob));
  const canvas = document.createElement("canvas");
  canvas.width = 64;
  canvas.height = 128;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  context.drawImage(image, 0, 0, 64, 128);
  const pixels = context.getImageData(0, 0, 64, 128).data;
  const mean = [0.485, 0.456, 0.406];
  const std = [0.229, 0.224, 0.225];
  const data = new Float32Array(3 * 128 * 64);
  for (let y = 0; y < 128; y += 1) {
    for (let x = 0; x < 64; x += 1) {
      const pixel = (y * 64 + x) * 4;
      const offset = y * 64 + x;
      data[offset] = (pixels[pixel] / 255 - mean[0]) / std[0];
      data[128 * 64 + offset] = (pixels[pixel + 1] / 255 - mean[1]) / std[1];
      data[2 * 128 * 64 + offset] = (pixels[pixel + 2] / 255 - mean[2]) / std[2];
    }
  }
  return new ort.Tensor("float32", data, [1, 3, 128, 64]);
}

async function setQuery(blob, name = "query image") {
  state.queryBlob = blob;
  state.queryName = name;
  const source = await readImageSource(blob);
  $("query-image").src = source;
  $("signal-thumb").src = source;
  $("query-name").textContent = name;
  $("dropzone").classList.add("hidden");
  $("query-preview").classList.remove("hidden");
  $("run-button").disabled = !state.session;
  $("embedding-status").textContent = state.session ? "READY" : "LOADING";
}

function renderResults(results) {
  const grid = $("results-grid");
  grid.innerHTML = "";
  $("result-count").textContent = `${results.length} MATCHES`;
  $("top-match-score").textContent = results.length ? results[0].score.toFixed(3) : "—";
  $("results-section").classList.toggle("has-results", results.length > 0);
  results.forEach((item, index) => {
    const card = document.createElement("article");
    card.className = "result-card";
    card.style.animationDelay = `${index * 45}ms`;
    const scorePercent = Math.max(0, Math.min(100, ((item.score + 1) / 2) * 100));
    card.innerHTML = `
      <img class="result-image" src="${item.image}" alt="Retrieved gallery view ${index + 1}" loading="lazy" />
      <div class="result-top"><span class="rank">0${index + 1}</span><span class="score">${item.score.toFixed(3)}</span></div>
      <div class="result-meta">TRACK ${String(item.pid).padStart(4, "0")} · CAM ${item.camid}</div>
      <div class="result-bar"><span style="width:${scorePercent}%"></span></div>`;
    grid.appendChild(card);
  });
}

async function runSearch() {
  if (!state.session || !state.queryBlob) return;
  const button = $("run-button");
  button.disabled = true;
  button.classList.add("is-running");
  button.querySelector("span").textContent = "Encoding query...";
  $("signal-image").classList.add("active");
  $("embedding-status").textContent = "RUNNING";
  const started = performance.now();
  try {
    const tensor = await imageTensor(state.queryBlob);
    const output = await state.session.run({ images: tensor });
    const embedding = normalizeEmbedding(Array.from(output.embeddings.data));
    state.queryEmbedding = embedding;
    const results = state.gallery
      .map((item) => ({ ...item, score: cosine(embedding, item.embedding) }))
      .sort((a, b) => b.score - a.score)
      .slice(0, 10);
    renderBars(embedding);
    $("embedding-status").textContent = "ENCODED";
    $("latency").textContent = `${Math.round(performance.now() - started)} ms`;
    renderResults(results);
    announce(`Search complete. ${results.length} nearest matches are ready.`);
    $("results-section").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    console.error(error);
    $("embedding-status").textContent = "ERROR";
    $("result-count").textContent = "SEARCH FAILED";
    announce("The search failed. Please try another image.");
  } finally {
    button.disabled = false;
    button.classList.remove("is-running");
    $("signal-image").classList.remove("active");
    button.querySelector("span").textContent = "Run nearest-neighbor search";
  }
}

function bindEvents() {
  $("file-input").addEventListener("change", (event) => {
    const file = event.target.files?.[0];
    if (file) handleFile(file);
  });
  ["dragenter", "dragover"].forEach((eventName) => dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.add("dragover");
  }));
  ["dragleave", "drop"].forEach((eventName) => dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.remove("dragover");
  }));
  dropzone.addEventListener("drop", (event) => {
    const file = event.dataTransfer.files?.[0];
    if (file) handleFile(file);
  });
  $("sample-button").addEventListener("click", async () => {
    const response = await fetch("/data/demo-query.jpg");
    await setQuery(await response.blob(), "demo-query.jpg");
    announce("Sample query loaded and ready for search.");
  });
  $("clear-query").addEventListener("click", () => {
    state.queryBlob = null;
    state.queryEmbedding = null;
    $("query-preview").classList.add("hidden");
    $("dropzone").classList.remove("hidden");
    $("signal-thumb").removeAttribute("src");
    $("run-button").disabled = true;
    $("embedding-status").textContent = "WAITING";
    $("result-count").textContent = "AWAITING QUERY";
    $("results-section").classList.remove("has-results");
    renderBars();
    announce("Query cleared.");
  });
  $("run-button").addEventListener("click", runSearch);
}

async function boot() {
  bindEvents();
  renderBars();
  try {
    const [session, galleryResponse] = await Promise.all([
      ort.InferenceSession.create("/model/siamese_encoder.onnx", { executionProviders: ["wasm"] }),
      fetch("/data/gallery.json").then((response) => response.json()),
    ]);
    state.session = session;
    state.gallery = galleryResponse;
    setStatus("Model ready", true);
    $("embedding-status").textContent = "READY";
    announce("Model ready. Add a person crop or use the sample query.");
    const response = await fetch("/data/demo-query.jpg");
    await setQuery(await response.blob(), "demo-query.jpg");
  } catch (error) {
    console.error(error);
    setStatus("Model unavailable");
    $("embedding-status").textContent = "OFFLINE";
    $("result-count").textContent = "MODEL UNAVAILABLE";
    announce("The model could not load. Refresh the page and try again.");
  }
}

boot();
