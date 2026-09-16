import { useState } from 'react'

const stages = ['Imagery intake', 'Preprocessing', 'Change detection', 'Priority assessment']

function App() {
  const [status, setStatus] = useState('Ready for paired imagery')

  function handleAnalyze(event) {
    event.preventDefault()
    setStatus('Analysis pipeline is not connected yet')
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand-lockup">
          <span className="brand-mark">TG</span>
          <div>
            <p className="eyebrow">Environmental intelligence</p>
            <h1>TerraGuard <span>AI</span></h1>
          </div>
        </div>
        <div className="system-state"><i /> API foundation online</div>
      </header>

      <section className="hero-grid">
        <div className="intro-copy">
          <p className="kicker">Paired observation analysis</p>
          <h2>See what changed.<br /><em>Know what matters.</em></h2>
          <p className="lede">Compare two satellite observations to surface environmental change, explain its evidence, and focus review where it matters most.</p>
          <div className="stage-list" aria-label="Analysis stages">
            {stages.map((stage, index) => (
              <div className="stage" key={stage}>
                <span>0{index + 1}</span>
                <strong>{stage}</strong>
              </div>
            ))}
          </div>
        </div>

        <form className="analysis-panel" onSubmit={handleAnalyze}>
          <div className="panel-heading">
            <div>
              <p className="eyebrow">New analysis</p>
              <h3>Define your observation pair</h3>
            </div>
            <span className="panel-index">01 / 04</span>
          </div>
          <label>
            <span>Area of interest</span>
            <input type="text" placeholder="Search a place or paste coordinates" />
          </label>
          <div className="date-row">
            <label>
              <span>Before date</span>
              <input type="date" />
            </label>
            <label>
              <span>After date</span>
              <input type="date" />
            </label>
          </div>
          <button type="submit">Analyze area <b>↗</b></button>
          <p className="form-status" role="status">{status}</p>
        </form>
      </section>

      <section className="map-preview" aria-label="Map preview">
        <div className="map-grid" />
        <div className="map-label label-north">N</div>
        <div className="map-label label-coast">SENTINEL-2 / DEMO VIEW</div>
        <div className="map-coordinate">18°31' N &nbsp; 73°51' E</div>
        <div className="map-pin"><span /></div>
        <div className="map-caption"><span className="live-dot" /> Awaiting imagery pair <small>MAP VIEW WILL APPEAR HERE</small></div>
      </section>

      <section className="metrics-strip">
        <div><span>Changed area</span><strong>--</strong><small>percentage of valid pixels</small></div>
        <div><span>NDVI difference</span><strong>--</strong><small>vegetation signal</small></div>
        <div><span>Priority</span><strong className="muted-value">Pending</strong><small>explainable assessment</small></div>
        <div className="insight-cell"><span>Action insight</span><p>Run an analysis to generate evidence-backed environmental guidance.</p></div>
      </section>
    </main>
  )
}

export default App
