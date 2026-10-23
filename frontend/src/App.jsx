
import { useState } from "react";
import "./App.css";

const fields = [
  { name: "Age", label: "Patient age", unit: "years", required: true },
  { name: "Gender", label: "Gender code", unit: "dataset code" },
  { name: "Dioptre_1", label: "Dioptre 1", unit: "D" },
  { name: "Dioptre_2", label: "Dioptre 2", unit: "D" },
  { name: "Astigmatism", label: "Astigmatism", unit: "D" },
  {
    name: "Phakic_Pseudophakic",
    label: "Lens status code",
    unit: "dataset code",
  },
  { name: "IOP_Pneumatic", label: "Pneumatic IOP", unit: "mmHg" },
  { name: "IOP_Perkins", label: "Perkins IOP", unit: "mmHg" },
  { name: "Pachymetry", label: "Corneal thickness", unit: "µm" },
  { name: "Axial_Length", label: "Axial length", unit: "mm" },
  { name: "VF_MD", label: "Visual field MD", unit: "dB" },
];

const diagnosisInfo = {
  Healthy: {
    symbol: "✳",
    tone: "healthy",
    description: "The model classified this input as Healthy.",
  },
  Glaucoma: {
    symbol: "◉",
    tone: "glaucoma",
    description: "The model classified this input as Glaucoma.",
  },
  "Glaucoma suspect": {
    symbol: "◌",
    tone: "suspect",
    description: "The model classified this input as Glaucoma suspect.",
  },
};

function App() {
  const [values, setValues] = useState({});
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const updateField = (name, value) => {
    setValues((previous) => ({ ...previous, [name]: value }));
  };

  const resetForm = () => {
    setValues({});
    setResult(null);
    setError("");
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    setError("");
    setResult(null);
    setLoading(true);

    const payload = {};
    fields.forEach(({ name }) => {
      const value = values[name];
      payload[name] = value === undefined || value.trim() === ""
        ? null
        : Number(value);
    });

    try {
      const response = await fetch("http://127.0.0.1:8000/api/predict/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || "Prediction request failed.");
      }

      setResult(data);
    } catch (err) {
      setError(
        err.message === "Failed to fetch"
          ? "Could not connect to Django. Check that the backend is running."
          : err.message || "Something went wrong. Please try again."
      );
    } finally {
      setLoading(false);
    }
  };

  const diagnosis = result ? diagnosisInfo[result.class_name] : null;
  const probabilities = result?.probabilities || {};
  const probabilityLabels = [
    { key: "0", label: "Healthy", color: "mint" },
    { key: "1", label: "Glaucoma", color: "pink" },
    { key: "2", label: "Glaucoma suspect", color: "lilac" },
  ];

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <a className="brand" href="#home" aria-label="EyeGuard AI home">
          <span className="brand-mark"><span /></span>
          <span className="brand-name">eye<span>guard</span><small>INTELLIGENCE LAB</small></span>
        </a>

        <div className="side-caption">WORKSPACE</div>
        <a className="nav-item active" href="#home">
          <span className="nav-icon">⌘</span>
          <span>Overview</span>
          <span className="nav-dot" />
        </a>
        <a className="nav-item" href="#assessment">
          <span className="nav-icon">◎</span>
          <span>New assessment</span>
        </a>
        <a className="nav-item" href="#results">
          <span className="nav-icon">↗</span>
          <span>Model insights</span>
        </a>

        <div className="sidebar-bottom">
          <div className="orbital-art" aria-hidden="true">
            <div className="orbit orbit-one" />
            <div className="orbit orbit-two" />
            <div className="orbit-core">E</div>
            <span className="orbit-star">✳</span>
          </div>
          <p className="sidebar-note">Clarity begins<br />with a closer look.</p>
          <div className="version-label"><span /> LOCAL RESEARCH BUILD · 01.0</div>
        </div>
      </aside>

      <main className="main-content" id="home">
        <header className="topbar">
          <div className="breadcrumb">WORKSPACE <span>/</span> OVERVIEW</div>
          <div className="topbar-right">
            <span className="system-status"><i /> MODEL ONLINE</span>
            <div className="avatar">EG</div>
          </div>
        </header>

        <section className="hero">
          <div className="hero-copy">
            <div className="eyebrow"><span className="eyebrow-line" /> EXPLORING OCULAR HEALTH</div>
            <h1>See beyond<br />the <em>surface.</em></h1>
            <p className="hero-description">
              A machine-learning research space for exploring eye-health
              patterns, one measurement at a time.
            </p>
            <a href="#assessment" className="hero-link">
              Start an assessment <span>↘</span>
            </a>
          </div>

          <div className="hero-art" aria-label="Abstract eye-inspired orbital illustration">
            <div className="art-ring ring-a" />
            <div className="art-ring ring-b" />
            <div className="art-ring ring-c" />
            <div className="art-eye">
              <div className="eye-iris">
                <div className="eye-pupil" />
                <span className="iris-glint" />
              </div>
            </div>
            <span className="art-coordinate coord-one">17° 23' N</span>
            <span className="art-coordinate coord-two">SCAN / 001</span>
            <span className="art-spark">✳</span>
          </div>

          <div className="hero-index"><span>01</span> / RESEARCH OVERVIEW</div>
        </section>

        <section className="metrics-row" aria-label="System overview">
          <article className="metric-card">
            <div className="metric-top"><span>01 / ENGINE</span><span className="metric-icon">✳</span></div>
            <div className="metric-value">RF<span className="metric-suffix">.ML</span></div>
            <p>Random Forest model</p>
            <div className="metric-line"><span style={{ width: "78%" }} /></div>
          </article>
          <article className="metric-card">
            <div className="metric-top"><span>02 / INPUTS</span><span className="metric-icon">⌁</span></div>
            <div className="metric-value">11<span className="metric-suffix"> / features</span></div>
            <p>Clinical measurements</p>
            <div className="metric-line lavender"><span style={{ width: "88%" }} /></div>
          </article>
          <article className="metric-card">
            <div className="metric-top"><span>03 / OUTPUTS</span><span className="metric-icon">◎</span></div>
            <div className="metric-value">03<span className="metric-suffix"> / classes</span></div>
            <p>Model prediction categories</p>
            <div className="class-dots"><i /><i /><i /></div>
          </article>
        </section>

        <section className="assessment-layout" id="assessment">
          <div className="form-panel">
            <div className="section-heading">
              <div>
                <div className="eyebrow"><span className="eyebrow-line" /> INPUT STUDIO</div>
                <h2>Clinical <em>profile.</em></h2>
              </div>
              <span className="section-number">01—11</span>
            </div>
            <p className="section-description">
              Enter the available measurements. Leave unknown values blank.
            </p>

            <form onSubmit={handleSubmit}>
              <div className="field-grid">
                {fields.map((field, index) => (
                  <label className="field" key={field.name}>
                    <span className="field-label">
                      <span className="field-index">{String(index + 1).padStart(2, "0")}</span>
                      {field.label}
                      {field.required && <span className="required">*</span>}
                    </span>
                    <span className="input-wrap">
                      <input
                        type="number"
                        step="any"
                        name={field.name}
                        value={values[field.name] ?? ""}
                        onChange={(event) => updateField(field.name, event.target.value)}
                        placeholder="—"
                        required={field.required}
                      />
                      <span className="input-unit">{field.unit}</span>
                    </span>
                  </label>
                ))}
              </div>

              <div className="form-footnote">
                <span className="footnote-icon">i</span>
                <span>Use the same coding and units as the PAPILA dataset.</span>
              </div>

              <div className="form-actions">
                <button type="button" className="reset-button" onClick={resetForm}>
                  Reset fields <span>↺</span>
                </button>
                <button type="submit" className="submit-button" disabled={loading}>
                  {loading ? "Analysing…" : "Run assessment"}
                  <span>{loading ? "◌" : "↗"}</span>
                </button>
              </div>
            </form>

            {error && (
              <div className="error-message" role="alert">
                <strong>Connection issue</strong>
                <p>{error}</p>
              </div>
            )}
          </div>

          <aside className="result-panel" id="results">
            <div className="result-panel-top">
              <span className="result-kicker">OUTPUT / 01</span>
              <span className={`live-pill ${result ? "complete" : ""}`}>
                <i /> {result ? "UPDATED" : "AWAITING INPUT"}
              </span>
            </div>

            {result ? (
              <>
                <div className={`result-symbol ${diagnosis?.tone || "suspect"}`}>
                  {diagnosis?.symbol || "◎"}
                </div>
                <div className="result-overline">MODEL CLASSIFICATION</div>
                <h3 className="result-title">{result.class_name}</h3>
                <p className="result-description">
                  {diagnosis?.description || "The model returned a classification for this input."}
                </p>

                <div className="probability-heading">
                  <span>CLASS PROBABILITIES</span><span>MODEL OUTPUT</span>
                </div>
                <div className="probability-list">
                  {probabilityLabels.map((item) => {
                    const probability = Number(probabilities[item.key] ?? 0);
                    return (
                      <div className="probability-item" key={item.key}>
                        <div className="probability-label">
                          <span><i className={`prob-dot ${item.color}`} />{item.label}</span>
                          <strong>{(probability * 100).toFixed(1)}%</strong>
                        </div>
                        <div className="probability-track">
                          <span
                            className={`prob-fill ${item.color}`}
                            style={{ width: `${Math.min(100, Math.max(0, probability * 100))}%` }}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>
                <div className="result-disclaimer">
                  <span>!</span>
                  <p>{result.notice || "Research prototype only; not a medical diagnosis."}</p>
                </div>
              </>
            ) : (
              <div className="empty-result">
                <div className="empty-orbit">
                  <div className="empty-orbit-ring" />
                  <span>◎</span>
                  <i>✳</i>
                </div>
                <div className="empty-overline">YOUR RESULTS WILL APPEAR HERE</div>
                <h3>Waiting for<br /><em>your first scan.</em></h3>
                <p>Complete the profile and run an assessment to reveal the model's output.</p>
                <div className="empty-bottom"><span>01</span> READY WHEN YOU ARE <span>↗</span></div>
              </div>
            )}
          </aside>
        </section>

        <footer className="page-footer">
          <span>EYEGUARD AI <i>·</i> EXPLORATORY RESEARCH</span>
          <span>NOT FOR CLINICAL DECISION-MAKING</span>
        </footer>
      </main>
    </div>
  );
}

export default App;
