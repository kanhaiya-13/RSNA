# Frontend Build Spec — Knee MRI AI Assist Tool (Orthopaedic Doctor Viewer)

## 0. Context for the build agent

This is a clinical-facing frontend for orthopaedic doctors reviewing AI-flagged knee MRI
findings. The model is a 5-fold ResNet-34 (2.5D CNN: 24-slice stack, 2D backbone per
slice, depth-aggregated) producing 12 independent multi-label sigmoid probabilities per
case. **There is no spatial (pixel-level) localization** — the model has no Grad-CAM or
bounding-box output. What it does have: per-slice classifier scores, i.e. a probability
curve across all 24 slices per finding, which is what this spec uses in place of a
heatmap. Build this with mocked data conforming exactly to the schema in Section 6.
Desktop-first, neutral low-saturation clinical UI, color reserved for confidence/trust
signaling only (7.2).

### 0.1 Open assumption — flag if wrong
The model runs per series/plane independently and planes are not concatenated before
classification. How per-series predictions combine into one case-level 12-finding list
is not yet confirmed. **This spec assumes a finding may be backed by more than one
series** (the data contract carries a `contributingSeries` list per finding rather than
a single `seriesId`). If it turns out each finding is always sourced from exactly one
fixed plane, simplify `contributingSeries` back to a single `seriesId` — everything else
in this spec is unaffected either way.

---

## 1. Pages

### 1.1 Worklist page (`/`)
Landing page. A queue of pending cases for the logged-in doctor.

### 1.2 Case Review page (`/case/:caseId`)
The core screen. Three-panel layout (Section 2). This is where 90% of time is spent.

### 1.3 Report Summary page (`/case/:caseId/report`)
Read/edit view of the auto-drafted structured report for one case, reachable from the
Case Review page via a "Generate Report" action.

*(No login/auth screens needed yet — assume a single logged-in doctor for now, but leave
a `currentUser` field in app state so auth can be added later without restructuring.)*

---

## 2. Case Review page — panel layout

Three-column layout, resizable dividers, left and right panels collapsible.

```
┌───────────────┬─────────────────────────────────┬──────────────────┐
│  LEFT PANEL   │         CENTER PANEL             │   RIGHT PANEL    │
│  Worklist     │  Image Viewer + Slice Prob Curve │   Findings List  │
│  (collapsible)│                                   │                  │
└───────────────┴─────────────────────────────────┴──────────────────┘
```

### 2.1 Left panel — Worklist (mini, collapsible)
- Same data as the full Worklist page (1.1) but condensed to a scrollable list
- Each row: patient ID (anonymized/masked), study date, AI urgency badge, read status
  (Unread / In Progress / Signed)
- Clicking a row navigates to that case, replacing the center + right panels
- Collapse toggle (chevron) shrinks this panel to an icon rail to free space for the viewer

### 2.2 Center panel — Image Viewer
Largest panel by default (~55% width). Two stacked sub-regions: the image viewer on top,
a slice-probability strip pinned below it.

**Image viewer components:**
- **Series switcher** (tabs/dropdown across the top): whichever series exist for this
  case (sagittal/coronal/axial)
- **Slice scrubber**: horizontal slider + prev/next + readout ("Slice 14 / 24"); scroll
  wheel over the image scrubs slices too
- **AI signal toggle** (top-right, switch labeled "AI Findings: Shown / Hidden"): default
  = Shown; see 7.1 for full interaction spec
- **Window/level control** and **zoom/pan** — standard MRI viewer affordances,
  independent of the AI toggle

**Slice-probability strip (only rendered when a finding is selected in the right panel,
and only visible when the AI toggle is Shown):**
- A small line chart, x-axis = slice index 0–23, y-axis = that finding's sigmoid score
  per slice
- The slice currently shown in the viewer is marked with a vertical indicator
- The peak-scoring slice is marked distinctly (e.g. filled dot + label)
- This is the direct replacement for a heatmap: it tells the doctor how peaked vs. spread
  out the signal is across the volume, which is diagnostically meaningful on its own
  (a sharp single-slice peak reads very differently from a broad plateau)

### 2.3 Right panel — Findings List (~25% width, collapsible)
- List of all 12 findings for this case, each row showing:
  - Finding name
  - **Case confidence badge** — this case's sigmoid probability, tiered (7.2)
  - **Model reliability indicator** — a separate, static badge based on that finding
    type's OOF AUC (7.2) — always shown, never hidden by the AI toggle, since it's a
    property of the model, not of this specific prediction
  - Contributing series (one or more plane labels)
  - Review state icon: unreviewed / reviewed / doctor-flagged-disagree
- Clicking a finding row: switches the viewer to its first contributing series, jumps to
  its peak slice, and renders the slice-probability strip for it
- Row-level actions: **"Mark reviewed"**, **"Flag disagreement"**
- Sticky footer button: **"Generate Report"** → navigates to 1.3

---

## 3. Worklist page (`/`) — full view

- Table/list of all pending cases assigned to the doctor
- Columns: Patient ID (masked), Study Date, # AI-flagged findings, highest-confidence
  finding + tier, read status
- **Default sort: by AI-flagged urgency, not date** — core value of triage
- Filter bar: by finding type, read status, confidence tier
- Row click → Case Review page (1.2)

---

## 4. Report Summary page (`/case/:caseId/report`)

- Auto-populated structured summary: one line per finding with case confidence, model
  reliability tier, and contributing series — pulled from Findings List state (including
  any doctor disagreement notes)
- Each line is **editable free text**; editing marks it "doctor-edited" vs "AI-original"
  visually and distinctly — don't silently merge the two
- Optional comparison block against a prior study, if present, side by side
- Actions: **Save Draft**, **Sign & Finalize** (finalizing locks further edits)
- Print/export affordance (button only for the mock)

---

## 5. Global states to handle

- Loading state per panel
- **No findings flagged** (all 12 low/negative) — right panel should say so explicitly
- **Series missing** — series switcher disables/greys the missing tab rather than
  breaking
- Worklist row status reflects read progress: zero findings reviewed vs. all reviewed

---

## 6. Data contract (mock your data to exactly this shape)

```json
{
  "case": {
    "caseId": "string",
    "patientIdMasked": "string",
    "studyDate": "ISO date string",
    "readStatus": "unread | in_progress | signed",
    "series": [
      { "seriesId": "string", "plane": "sagittal | coronal | axial", "sliceCount": 24 }
    ],
    "findings": [
      {
        "findingId": "string",
        "findingName": "ACL | MCL | Medial Meniscus | Lateral Meniscus | Medial OA | Lateral OA | PF OA | Effusion | Synovitis | Baker's | Contusion | Fracture",
        "probability": "number (0-1, raw sigmoid, 5-fold averaged)",
        "confidenceTier": "high | medium | low",   // see 7.2 — thresholds are empirical, not 0.5/0.9
        "modelReliabilityTier": "high | moderate | verify",  // static, from OOF AUC table (7.2)
        "contributingSeries": ["seriesId", "..."],  // see 0.1 — may be >1
        "sliceScores": [ "24 numbers, per-slice sigmoid probability" ],
        "peakSliceIndex": "number, derived: argmax(sliceScores)",
        "reviewState": "unreviewed | reviewed | flagged_disagree",
        "doctorNote": "optional string"
      }
    ],
    "priorStudy": "optional: same shape as case, for comparison view"
  }
}
```

**Static app-level config (not per-case) — the model reliability table:**
```json
{
  "findingReliability": {
    "Medial Meniscus": { "oofAuc": 0.8662, "tier": "high" },
    "Medial OA":        { "oofAuc": 0.8636, "tier": "high" },
    "Effusion":         { "oofAuc": 0.8622, "tier": "high" },
    "Baker's":          { "oofAuc": 0.8608, "tier": "high" },
    "Fracture":         { "oofAuc": 0.8496, "tier": "high" },
    "ACL":              { "oofAuc": 0.8474, "tier": "high" },
    "Synovitis":        { "oofAuc": 0.8353, "tier": "moderate" },
    "PF OA":            { "oofAuc": 0.8162, "tier": "moderate" },
    "Lateral OA":       { "oofAuc": 0.8121, "tier": "moderate" },
    "Contusion":        { "oofAuc": 0.7984, "tier": "verify" },
    "Lateral Meniscus": { "oofAuc": 0.7901, "tier": "verify" },
    "MCL":              { "oofAuc": 0.7623, "tier": "verify" }
  }
}
```

Build the mock data generator to produce a handful of cases covering: all-negative case,
mixed-confidence case, case with a prior study, case missing a series, and at least one
case with a low-`modelReliabilityTier` finding scored high on `confidenceTier` (this
combination — model unsure of itself in general, but confident on this case — is exactly
the state the UI most needs to represent clearly).

---

## 7. Interaction specs

### 7.1 AI signal toggle
- Default: **Shown**
- Toggling to **Hidden** hides: the slice-probability strip, case confidence badges, and
  contributing-series labels in the right panel. It does **not** hide the model
  reliability badge or the finding names/list itself — those aren't "AI output for this
  case," they're static reference info the doctor may want regardless.
- On toggling from Hidden back to Shown, set `independentReadPerformed: true` in that
  case's local state (no UI needed yet — just needs to exist for later use in the report
  or backend).

### 7.2 Two confidence signals — do not merge them into one badge
- **Case confidence tier** (per-prediction, this case only): High / Medium / Low, with
  thresholds chosen empirically per finding from validation PR/ROC curves — do not hardcode
  0.5/0.9 globally; leave the threshold values as a config object per finding name so they
  can be tuned once real operating points are chosen.
- **Model reliability tier** (per finding type, static, from the `findingReliability`
  config in Section 6): High / Moderate / "Verify manually" — MCL and Lateral Meniscus
  should always carry the "verify manually" treatment regardless of how confident the
  model is on a specific case, because the finding type itself tops out lower (OOF AUC
  0.76 and 0.79).
- Never use red/green alone for either tier (accessibility) — pair color with an icon or
  outline-style distinction.

### 7.3 Finding-click → viewer response
- Switches to the first entry in `contributingSeries`, scrubs to `peakSliceIndex`,
  renders the slice-probability strip for that finding, and briefly pulses/highlights the
  peak marker on the strip so the doctor's eye goes there immediately.

---

## 8. Explicitly out of scope for this frontend pass
- Authentication/login flows
- Real report PDF generation
- Multi-doctor assignment/handoff
- Any actual model inference — all findings data is mocked per Section 6
- Pixel-level spatial overlays (model doesn't currently support this — see 0)
