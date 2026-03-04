import { useState } from "react";

const modules = [
  {
    id: "gpu",
    title: "NVIDIA A100 80GB — Single GPU",
    color: "#1a1a2e",
    textColor: "#76b900",
    items: [
      { name: "ColPali (TomoroAI/tomoro-ai-colqwen3-embed-8b-awq)", vram: "~8 GB", port: "8001", role: "Retrieval", color: "#2d6a4f" },
      { name: "Qwen3-VL-2B-Instruct", vram: "~6 GB", port: "8002", role: "Visual Extraction", color: "#1d3557" },
      { name: "Qwen2.5-7B-Instruct-AWQ", vram: "~14 GB", port: "8003", role: "Text Generation", color: "#6a040f" },
    ],
    details: "Total model weights: ~28 GB | KV-Cache available: ~52 GB | LMCache offloads cold KV entries to system RAM for efficient memory utilization."
  },
  {
    id: "proxy",
    title: "FastAPI Reverse Proxy (:8000)",
    color: "#0d9488",
    items: [
      { path: "/v1/retrieve/*", target: "ColPali :8001", desc: "Visual embedding search" },
      { path: "/v1/extract/*", target: "Qwen3-VL :8002", desc: "Structured data extraction" },
      { path: "/v1/generate/*", target: "Qwen2.5-7B :8003", desc: "Answer generation" },
      { path: "/v1/pipeline/*", target: "Orchestrator", desc: "Full pipeline (all 3 models)" },
    ],
    details: "Routes incoming requests to the appropriate vLLM backend. Handles health checks, rate limiting (100 req/min), circuit breakers, and request queuing."
  },
  {
    id: "pipeline",
    title: "Document Pipeline Flow",
    color: "#7c3aed",
    stages: [
      { step: 1, name: "Ingest", desc: "PDF → page rasterization at 300 DPI → MinIO/S3 storage", icon: "📄" },
      { step: 2, name: "Retrieve", desc: "ColPali encodes pages as multi-vector embeddings → late-interaction scoring", icon: "🔍" },
      { step: 3, name: "Extract", desc: "Qwen3-VL extracts tables, key-values, text blocks as structured JSON", icon: "🧠" },
      { step: 4, name: "Generate", desc: "Qwen2.5-7B synthesizes natural-language answer with page citations", icon: "✍️" },
    ],
    details: "End-to-end latency target: <3s per page. ColPali replaces traditional OCR→chunk→embed with direct visual retrieval."
  },
  {
    id: "infra",
    title: "Supporting Infrastructure",
    color: "#b45309",
    components: [
      { name: "Qdrant", role: "ColPali embedding index (multi-vector, MaxSim scoring)" },
      { name: "PostgreSQL", role: "Document metadata, job tracking, user management" },
      { name: "MinIO / S3", role: "PDF storage, rasterized page images" },
      { name: "Redis + Celery", role: "Async task queue for batch ingestion" },
      { name: "Prometheus + Grafana", role: "GPU metrics, latency dashboards, alerting" },
    ],
    details: "All services orchestrated via Docker Compose. Prometheus scrapes vLLM /metrics endpoints for VRAM usage, tokens/sec, and queue depth."
  },
];

export default function ArchitectureDiagram() {
  const [selected, setSelected] = useState("gpu");
  const [hoveredModel, setHoveredModel] = useState(null);

  const current = modules.find(m => m.id === selected);

  return (
    <div style={{ fontFamily: "'Inter', 'Segoe UI', system-ui, sans-serif", background: "#0f172a", color: "#e2e8f0", minHeight: "100vh", padding: "24px" }}>
      <div style={{ maxWidth: 900, margin: "0 auto" }}>
        {/* Header */}
        <div style={{ textAlign: "center", marginBottom: 32 }}>
          <h1 style={{ fontSize: 22, fontWeight: 700, color: "#f8fafc", margin: 0, letterSpacing: "-0.5px" }}>
            Multi-Model vLLM Deployment Architecture
          </h1>
          <p style={{ color: "#94a3b8", fontSize: 13, marginTop: 6 }}>
            Document Understanding at Scale — ColPali + Qwen3-VL + Qwen2.5-7B on Single A100
          </p>
        </div>

        {/* Navigation Tabs */}
        <div style={{ display: "flex", gap: 8, marginBottom: 24, flexWrap: "wrap", justifyContent: "center" }}>
          {modules.map(m => (
            <button
              key={m.id}
              onClick={() => setSelected(m.id)}
              style={{
                padding: "10px 18px",
                borderRadius: 8,
                border: selected === m.id ? `2px solid ${m.color}` : "2px solid #334155",
                background: selected === m.id ? `${m.color}22` : "#1e293b",
                color: selected === m.id ? "#f8fafc" : "#94a3b8",
                cursor: "pointer",
                fontSize: 13,
                fontWeight: 600,
                transition: "all 0.2s",
              }}
            >
              {m.title.split("(")[0].trim()}
            </button>
          ))}
        </div>

        {/* GPU Module */}
        {selected === "gpu" && (
          <div>
            <div style={{ background: "#1e293b", borderRadius: 12, padding: 24, border: "1px solid #334155", marginBottom: 16 }}>
              <h2 style={{ fontSize: 16, color: "#76b900", marginTop: 0, marginBottom: 20 }}>
                ⚡ {current.title}
              </h2>
              
              {/* VRAM Bar */}
              <div style={{ background: "#0f172a", borderRadius: 8, padding: 16, marginBottom: 20 }}>
                <div style={{ fontSize: 12, color: "#94a3b8", marginBottom: 8 }}>VRAM Allocation (80 GB Total)</div>
                <div style={{ display: "flex", height: 40, borderRadius: 6, overflow: "hidden", gap: 2 }}>
                  {current.items.map((item, i) => {
                    const pct = parseInt(item.vram) / 80 * 100;
                    return (
                      <div
                        key={i}
                        onMouseEnter={() => setHoveredModel(i)}
                        onMouseLeave={() => setHoveredModel(null)}
                        style={{
                          width: `${pct}%`,
                          background: item.color,
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "center",
                          fontSize: 11,
                          color: "#fff",
                          fontWeight: 600,
                          cursor: "pointer",
                          opacity: hoveredModel === null || hoveredModel === i ? 1 : 0.4,
                          transition: "opacity 0.2s",
                        }}
                      >
                        {item.vram}
                      </div>
                    );
                  })}
                  <div style={{ flex: 1, background: "#374151", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 11, color: "#9ca3af", fontWeight: 600 }}>
                    KV-Cache ~52 GB
                  </div>
                </div>
              </div>

              {/* Model Cards */}
              <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 12 }}>
                {current.items.map((item, i) => (
                  <div
                    key={i}
                    onMouseEnter={() => setHoveredModel(i)}
                    onMouseLeave={() => setHoveredModel(null)}
                    style={{
                      background: hoveredModel === i ? `${item.color}33` : "#0f172a",
                      border: `1px solid ${hoveredModel === i ? item.color : "#334155"}`,
                      borderRadius: 8,
                      padding: 16,
                      transition: "all 0.2s",
                      cursor: "pointer",
                    }}
                  >
                    <div style={{ fontSize: 11, color: "#94a3b8", textTransform: "uppercase", letterSpacing: 1, marginBottom: 6 }}>
                      Model {i + 1}
                    </div>
                    <div style={{ fontSize: 14, fontWeight: 700, color: "#f8fafc", marginBottom: 4 }}>
                      {item.name.split("/").pop().split("(")[0]}
                    </div>
                    <div style={{ fontSize: 12, color: "#94a3b8", marginBottom: 12 }}>{item.role}</div>
                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12 }}>
                      <span style={{ color: "#76b900" }}>VRAM: {item.vram}</span>
                      <span style={{ color: "#60a5fa" }}>:{item.port}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
            <div style={{ background: "#1e293b", borderRadius: 8, padding: 14, border: "1px solid #334155", fontSize: 12, color: "#94a3b8", lineHeight: 1.6 }}>
              💡 {current.details}
            </div>
          </div>
        )}

        {/* Proxy Module */}
        {selected === "proxy" && (
          <div>
            <div style={{ background: "#1e293b", borderRadius: 12, padding: 24, border: "1px solid #334155", marginBottom: 16 }}>
              <h2 style={{ fontSize: 16, color: "#0d9488", marginTop: 0, marginBottom: 20 }}>
                🔀 {current.title}
              </h2>
              <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                {current.items.map((item, i) => (
                  <div key={i} style={{ display: "flex", alignItems: "center", gap: 12, background: "#0f172a", borderRadius: 8, padding: 14, border: "1px solid #334155" }}>
                    <code style={{ background: "#0d948822", color: "#5eead4", padding: "4px 10px", borderRadius: 4, fontSize: 13, fontWeight: 600, minWidth: 160, fontFamily: "'JetBrains Mono', monospace" }}>
                      {item.path}
                    </code>
                    <span style={{ color: "#475569", fontSize: 18 }}>→</span>
                    <span style={{ color: "#f8fafc", fontWeight: 600, fontSize: 13, minWidth: 130 }}>{item.target}</span>
                    <span style={{ color: "#94a3b8", fontSize: 12, flex: 1 }}>{item.desc}</span>
                  </div>
                ))}
              </div>
            </div>
            <div style={{ background: "#1e293b", borderRadius: 8, padding: 14, border: "1px solid #334155", fontSize: 12, color: "#94a3b8", lineHeight: 1.6 }}>
              💡 {current.details}
            </div>
          </div>
        )}

        {/* Pipeline Module */}
        {selected === "pipeline" && (
          <div>
            <div style={{ background: "#1e293b", borderRadius: 12, padding: 24, border: "1px solid #334155", marginBottom: 16 }}>
              <h2 style={{ fontSize: 16, color: "#7c3aed", marginTop: 0, marginBottom: 20 }}>
                🔄 {current.title}
              </h2>
              <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                {current.stages.map((stage, i) => (
                  <div key={i}>
                    <div style={{
                      display: "flex",
                      alignItems: "flex-start",
                      gap: 16,
                      background: "#0f172a",
                      borderRadius: 8,
                      padding: 16,
                      border: "1px solid #334155",
                    }}>
                      <div style={{
                        width: 44,
                        height: 44,
                        borderRadius: "50%",
                        background: `${["#2d6a4f", "#1d3557", "#7c3aed", "#6a040f"][i]}44`,
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        fontSize: 20,
                        flexShrink: 0,
                      }}>
                        {stage.icon}
                      </div>
                      <div>
                        <div style={{ fontSize: 14, fontWeight: 700, color: "#f8fafc", marginBottom: 4 }}>
                          Step {stage.step}: {stage.name}
                        </div>
                        <div style={{ fontSize: 12, color: "#94a3b8", lineHeight: 1.5 }}>{stage.desc}</div>
                      </div>
                    </div>
                    {i < current.stages.length - 1 && (
                      <div style={{ textAlign: "center", color: "#475569", fontSize: 18, padding: "2px 0" }}>↓</div>
                    )}
                  </div>
                ))}
              </div>
            </div>
            <div style={{ background: "#1e293b", borderRadius: 8, padding: 14, border: "1px solid #334155", fontSize: 12, color: "#94a3b8", lineHeight: 1.6 }}>
              💡 {current.details}
            </div>
          </div>
        )}

        {/* Infrastructure Module */}
        {selected === "infra" && (
          <div>
            <div style={{ background: "#1e293b", borderRadius: 12, padding: 24, border: "1px solid #334155", marginBottom: 16 }}>
              <h2 style={{ fontSize: 16, color: "#b45309", marginTop: 0, marginBottom: 20 }}>
                🏗️ {current.title}
              </h2>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(2, 1fr)", gap: 10 }}>
                {current.components.map((comp, i) => (
                  <div key={i} style={{
                    background: "#0f172a",
                    borderRadius: 8,
                    padding: 14,
                    border: "1px solid #334155",
                    gridColumn: i === current.components.length - 1 && current.components.length % 2 !== 0 ? "span 2" : "auto",
                  }}>
                    <div style={{ fontSize: 14, fontWeight: 700, color: "#fbbf24", marginBottom: 4 }}>{comp.name}</div>
                    <div style={{ fontSize: 12, color: "#94a3b8", lineHeight: 1.5 }}>{comp.role}</div>
                  </div>
                ))}
              </div>
            </div>
            <div style={{ background: "#1e293b", borderRadius: 8, padding: 14, border: "1px solid #334155", fontSize: 12, color: "#94a3b8", lineHeight: 1.6 }}>
              💡 {current.details}
            </div>
          </div>
        )}

        {/* Summary Table */}
        <div style={{ marginTop: 24, background: "#1e293b", borderRadius: 12, padding: 20, border: "1px solid #334155" }}>
          <h3 style={{ fontSize: 14, color: "#f8fafc", marginTop: 0, marginBottom: 12 }}>📊 Model Summary</h3>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
            <thead>
              <tr style={{ borderBottom: "1px solid #334155" }}>
                {["Role", "Model", "VRAM", "Port", "Quantization"].map(h => (
                  <th key={h} style={{ textAlign: "left", padding: "8px 12px", color: "#94a3b8", fontWeight: 600 }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {[
                ["Retrieval", "TomoroAI/tomoro-ai-colqwen3-embed-8b-awq", "~8 GB", "8001", "AWQ W4A16"],
                ["Visual Extraction", "Qwen3-VL-2B", "~6 GB", "8002", "FP16"],
                ["Text Generation", "Qwen2.5-7B-AWQ", "~14 GB", "8003", "AWQ 4-bit"],
              ].map((row, i) => (
                <tr key={i} style={{ borderBottom: "1px solid #1e293b" }}>
                  {row.map((cell, j) => (
                    <td key={j} style={{ padding: "8px 12px", color: j === 0 ? "#60a5fa" : "#e2e8f0" }}>{cell}</td>
                  ))}
                </tr>
              ))}
              <tr style={{ borderTop: "1px solid #334155" }}>
                <td colSpan={2} style={{ padding: "8px 12px", color: "#76b900", fontWeight: 700 }}>Total</td>
                <td style={{ padding: "8px 12px", color: "#76b900", fontWeight: 700 }}>~28 GB</td>
                <td colSpan={2} style={{ padding: "8px 12px", color: "#94a3b8" }}>52 GB free for KV-cache</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
