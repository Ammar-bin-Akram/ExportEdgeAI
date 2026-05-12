"""
Generate a self-contained HTML test report for the ExportEdge AI test suite.

Usage (from Code/):
    python tests/generate_report.py                # all tests
    python tests/generate_report.py unit           # unit tests only
    python tests/generate_report.py integration    # integration tests only
    python tests/generate_report.py performance    # performance tests only
    python tests/generate_report.py validation     # data validation tests only
    python tests/generate_report.py reliability    # reliability tests only
    python tests/generate_report.py streamlit      # Streamlit UI tests only

Output: tests/report_<category>.html
No extra pip packages required.
"""
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

CODE_DIR  = Path(__file__).resolve().parent.parent
TESTS_DIR = CODE_DIR / "tests"
XML_PATH  = TESTS_DIR / "_results.xml"

# ── Category → test file mapping ─────────────────────────────────────────────
CATEGORIES: dict[str, dict] = {
    "all": {
        "label": "Full Test Suite",
        "files": [],          # empty = run whole tests/ directory
        "accent": "#6366f1",
    },
    "unit": {
        "label": "Unit Tests",
        "files": [
            "test_config.py",
            "test_preprocessing.py",
            "test_postprocessing.py",
            "test_pipeline_factory.py",
            "test_regex_and_chunker.py",
            "test_vision.py",
        ],
        "accent": "#22c55e",
    },
    "integration": {
        "label": "Integration Tests",
        "files": [
            "test_integration.py",
            "test_integration_pipeline.py",
        ],
        "accent": "#3b82f6",
    },
    "performance": {
        "label": "Performance Tests",
        "files": ["test_performance.py"],
        "accent": "#f59e0b",
    },
    "validation": {
        "label": "Data Validation Tests",
        "files": ["test_data_validation.py"],
        "accent": "#ec4899",
    },
    "reliability": {
        "label": "Reliability Tests",
        "files": ["test_reliability.py"],
        "accent": "#14b8a6",
    },
    "streamlit": {
        "label": "Streamlit UI Tests",
        "files": ["test_streamlit_app.py"],
        "accent": "#f97316",
    },
}


def _resolve_targets(category: str) -> list[str]:
    files = CATEGORIES[category]["files"]
    if not files:
        return [str(TESTS_DIR)]
    return [str(TESTS_DIR / f) for f in files]


def run_pytest(targets: list[str]) -> bool:
    print(f"Running pytest on: {', '.join(Path(t).name for t in targets)}")
    result = subprocess.run(
        [
            sys.executable, "-m", "pytest",
            *targets,
            f"--junit-xml={XML_PATH}",
            "--tb=short",
            "-v",
            "-q",
        ],
        capture_output=True,
        text=True,
        cwd=str(CODE_DIR),
    )
    tail = result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout
    print(tail)
    if result.stderr:
        print(result.stderr[-800:])
    return XML_PATH.exists()


def parse_xml(xml_path: Path) -> dict:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    suite = root if root.tag == "testsuite" else root.find("testsuite")
    if suite is None:
        suite = root

    total    = int(suite.get("tests",    0))
    errors   = int(suite.get("errors",   0))
    failures = int(suite.get("failures", 0))
    skipped  = int(suite.get("skipped",  0))
    passed   = total - errors - failures - skipped
    duration = float(suite.get("time",   0))

    cases: list[dict] = []
    for tc in root.iter("testcase"):
        name      = tc.get("name", "")
        classname = tc.get("classname", "")
        time_s    = float(tc.get("time", 0))

        failure = tc.find("failure")
        error   = tc.find("error")
        skip    = tc.find("skipped")

        if failure is not None:
            status  = "FAILED"
            message = (failure.get("message") or "") + "\n" + (failure.text or "")
        elif error is not None:
            status  = "ERROR"
            message = (error.get("message") or "") + "\n" + (error.text or "")
        elif skip is not None:
            status  = "SKIPPED"
            message = skip.get("message", "")
        else:
            status  = "PASSED"
            message = ""

        module = classname.split(".")[-1] if classname else "unknown"
        cases.append({
            "name":     name,
            "module":   module,
            "status":   status,
            "duration": time_s,
            "message":  message.strip(),
        })

    modules: dict[str, list] = {}
    for c in cases:
        modules.setdefault(c["module"], []).append(c)

    return {
        "total":    total,
        "passed":   passed,
        "failed":   failures,
        "errors":   errors,
        "skipped":  skipped,
        "duration": duration,
        "modules":  modules,
        "cases":    cases,
    }


def _badge(status: str) -> str:
    colors = {
        "PASSED":  ("#22c55e", "#dcfce7"),
        "FAILED":  ("#ef4444", "#fee2e2"),
        "ERROR":   ("#f97316", "#ffedd5"),
        "SKIPPED": ("#a855f7", "#f3e8ff"),
    }
    fg, bg = colors.get(status, ("#6b7280", "#f3f4f6"))
    return (
        f'<span style="background:{bg};color:{fg};'
        f'font-size:0.72rem;font-weight:700;padding:2px 9px;'
        f'border-radius:999px;letter-spacing:0.05em;">{status}</span>'
    )


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
    )


def build_html(data: dict, label: str, accent: str) -> str:
    total    = data["total"]
    passed   = data["passed"]
    failed   = data["failed"]
    errors   = data["errors"]
    skipped  = data["skipped"]
    duration = data["duration"]
    modules  = data["modules"]

    pass_pct = round(passed / total * 100, 1) if total else 0
    fail_pct = round((failed + errors) / total * 100, 1) if total else 0
    skip_pct = round(skipped / total * 100, 1) if total else 0

    all_ok        = failed == 0 and errors == 0
    overall_color = "#22c55e" if all_ok else "#ef4444"
    overall_label = "ALL PASSED" if all_ok else "SOME FAILED"
    now           = datetime.now().strftime("%d %b %Y, %H:%M")

    # ── per-module accordion blocks ───────────────────────────────────────────
    module_rows = ""
    for mod, cases in sorted(modules.items()):
        m_total   = len(cases)
        m_passed  = sum(1 for c in cases if c["status"] == "PASSED")
        m_failed  = sum(1 for c in cases if c["status"] in ("FAILED", "ERROR"))
        m_skipped = sum(1 for c in cases if c["status"] == "SKIPPED")
        m_dur     = sum(c["duration"] for c in cases)
        m_pct     = round(m_passed / m_total * 100) if m_total else 0
        bar_color = "#22c55e" if m_failed == 0 else "#ef4444"
        mod_id    = mod.replace(".", "_").replace(" ", "_")

        test_rows = ""
        for c in cases:
            msg_html = ""
            if c["message"]:
                esc = _escape(c["message"])
                msg_html = (
                    f'<div style="margin-top:6px;background:#1e1e2e;color:#cdd6f4;'
                    f'font-family:monospace;font-size:0.75rem;padding:10px 14px;'
                    f'border-radius:6px;white-space:pre-wrap;max-height:200px;'
                    f'overflow-y:auto;">{esc}</div>'
                )
            test_rows += f"""
            <tr class="test-row" data-status="{c['status']}">
              <td style="padding:10px 12px;vertical-align:top;">
                <span style="font-size:0.82rem;color:#e2e8f0;">{_escape(c['name'])}</span>
                {msg_html}
              </td>
              <td style="padding:10px 12px;text-align:center;vertical-align:top;">{_badge(c['status'])}</td>
              <td style="padding:10px 12px;text-align:right;vertical-align:top;color:#94a3b8;font-size:0.8rem;white-space:nowrap;">{c['duration']:.3f}s</td>
            </tr>"""

        skipped_html = (
            f'<span style="font-size:0.8rem;color:#a855f7;">skip {m_skipped}</span>'
            if m_skipped else ""
        )
        failed_html = (
            f'<span style="font-size:0.8rem;color:#ef4444;">fail {m_failed}</span>'
            if m_failed else ""
        )

        module_rows += f"""
        <div class="module-block" style="margin-bottom:16px;background:#1e293b;border-radius:12px;overflow:hidden;border:1px solid #334155;">
          <div onclick="toggle('{mod_id}')"
               style="display:flex;align-items:center;gap:14px;padding:14px 18px;
                      cursor:pointer;user-select:none;background:#1e293b;transition:background .15s;"
               onmouseover="this.style.background='#263348'"
               onmouseout="this.style.background='#1e293b'">
            <span style="font-size:1rem;color:#e2e8f0;flex:1;font-weight:600;">{_escape(mod)}</span>
            <span style="font-size:0.8rem;color:#94a3b8;">{m_total} tests</span>
            <span style="font-size:0.8rem;color:#22c55e;">pass {m_passed}</span>
            {failed_html}
            {skipped_html}
            <div style="width:72px;background:#334155;border-radius:999px;height:6px;overflow:hidden;">
              <div style="width:{m_pct}%;height:100%;background:{bar_color};border-radius:999px;"></div>
            </div>
            <span style="font-size:0.8rem;color:{bar_color};width:36px;text-align:right;">{m_pct}%</span>
            <span style="color:#64748b;font-size:0.8rem;">{m_dur:.2f}s</span>
            <span id="arrow-{mod_id}" style="color:#64748b;font-size:0.85rem;transition:transform .2s;">&#9660;</span>
          </div>
          <div id="{mod_id}" style="display:none;border-top:1px solid #334155;">
            <table style="width:100%;border-collapse:collapse;">
              <thead>
                <tr style="background:#0f172a;">
                  <th style="padding:8px 12px;text-align:left;font-size:0.72rem;color:#64748b;font-weight:600;text-transform:uppercase;letter-spacing:.06em;">Test</th>
                  <th style="padding:8px 12px;text-align:center;font-size:0.72rem;color:#64748b;font-weight:600;text-transform:uppercase;letter-spacing:.06em;">Status</th>
                  <th style="padding:8px 12px;text-align:right;font-size:0.72rem;color:#64748b;font-weight:600;text-transform:uppercase;letter-spacing:.06em;">Time</th>
                </tr>
              </thead>
              <tbody>{test_rows}
              </tbody>
            </table>
          </div>
        </div>"""

    module_id_list = list(modules.keys())

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>ExportEdge AI -- {label}</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0f172a; color: #e2e8f0; min-height: 100vh; padding: 32px 20px; }}
    .container {{ max-width: 1100px; margin: 0 auto; }}
    .filter-btn {{ padding: 6px 16px; border-radius: 999px; border: 1px solid #334155;
                   background: transparent; color: #94a3b8; cursor: pointer; font-size: 0.82rem;
                   transition: all .15s; }}
    .filter-btn:hover, .filter-btn.active {{ background: #334155; color: #e2e8f0; }}
    tr.test-row {{ border-bottom: 1px solid #1e293b; }}
    tr.test-row:last-child {{ border-bottom: none; }}
    tr.test-row:hover td {{ background: rgba(255,255,255,.03); }}
  </style>
</head>
<body>
<div class="container">

  <!-- Header -->
  <div style="display:flex;align-items:flex-start;justify-content:space-between;flex-wrap:wrap;gap:12px;margin-bottom:32px;">
    <div>
      <div style="font-size:0.75rem;font-weight:700;letter-spacing:.1em;color:{accent};text-transform:uppercase;margin-bottom:4px;">ExportEdge AI</div>
      <h1 style="font-size:1.9rem;font-weight:800;color:#f1f5f9;letter-spacing:-.02em;">{label}</h1>
      <p style="color:#64748b;margin-top:4px;">Generated {now}</p>
    </div>
    <div style="background:{overall_color}22;border:1.5px solid {overall_color};border-radius:10px;padding:10px 22px;text-align:center;">
      <div style="font-size:1.4rem;font-weight:900;color:{overall_color};">{overall_label}</div>
      <div style="font-size:0.78rem;color:{overall_color}aa;margin-top:2px;">{duration:.2f}s runtime</div>
    </div>
  </div>

  <!-- Summary cards -->
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:14px;margin-bottom:28px;">
    <div style="background:#1e293b;border-radius:12px;padding:20px 22px;border:1px solid #334155;">
      <div style="font-size:0.7rem;font-weight:700;letter-spacing:.08em;color:#64748b;text-transform:uppercase;">Total</div>
      <div style="font-size:2.4rem;font-weight:900;color:#f1f5f9;line-height:1.1;margin-top:6px;">{total}</div>
      <div style="font-size:0.78rem;color:#475569;margin-top:4px;">test cases</div>
    </div>
    <div style="background:#1e293b;border-radius:12px;padding:20px 22px;border:1px solid #166534;">
      <div style="font-size:0.7rem;font-weight:700;letter-spacing:.08em;color:#4ade80;text-transform:uppercase;">Passed</div>
      <div style="font-size:2.4rem;font-weight:900;color:#22c55e;line-height:1.1;margin-top:6px;">{passed}</div>
      <div style="font-size:0.78rem;color:#4ade8088;margin-top:4px;">{pass_pct}%</div>
    </div>
    <div style="background:#1e293b;border-radius:12px;padding:20px 22px;border:1px solid {'#7f1d1d' if (failed+errors) > 0 else '#334155'};">
      <div style="font-size:0.7rem;font-weight:700;letter-spacing:.08em;color:{'#f87171' if (failed+errors) > 0 else '#475569'};text-transform:uppercase;">Failed</div>
      <div style="font-size:2.4rem;font-weight:900;color:{'#ef4444' if (failed+errors) > 0 else '#475569'};line-height:1.1;margin-top:6px;">{failed + errors}</div>
      <div style="font-size:0.78rem;color:{'#f8717188' if (failed+errors) > 0 else '#334155'};margin-top:4px;">{fail_pct}%</div>
    </div>
    <div style="background:#1e293b;border-radius:12px;padding:20px 22px;border:1px solid {'#4c1d9544' if skipped > 0 else '#334155'};">
      <div style="font-size:0.7rem;font-weight:700;letter-spacing:.08em;color:{'#c084fc' if skipped > 0 else '#475569'};text-transform:uppercase;">Skipped</div>
      <div style="font-size:2.4rem;font-weight:900;color:{'#a855f7' if skipped > 0 else '#475569'};line-height:1.1;margin-top:6px;">{skipped}</div>
      <div style="font-size:0.78rem;color:{'#c084fc88' if skipped > 0 else '#334155'};margin-top:4px;">{skip_pct}%</div>
    </div>
    <div style="background:#1e293b;border-radius:12px;padding:20px 22px;border:1px solid #334155;">
      <div style="font-size:0.7rem;font-weight:700;letter-spacing:.08em;color:#64748b;text-transform:uppercase;">Modules</div>
      <div style="font-size:2.4rem;font-weight:900;color:#f1f5f9;line-height:1.1;margin-top:6px;">{len(modules)}</div>
      <div style="font-size:0.78rem;color:#475569;margin-top:4px;">test files</div>
    </div>
  </div>

  <!-- Pass rate bar -->
  <div style="background:#1e293b;border-radius:12px;padding:20px 24px;border:1px solid #334155;margin-bottom:28px;">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
      <span style="font-size:0.82rem;font-weight:600;color:#94a3b8;">Pass Rate</span>
      <span style="font-size:1.1rem;font-weight:800;color:{overall_color};">{pass_pct}%</span>
    </div>
    <div style="background:#334155;border-radius:999px;height:10px;overflow:hidden;">
      <div style="display:flex;height:100%;border-radius:999px;overflow:hidden;">
        <div style="width:{pass_pct}%;background:#22c55e;"></div>
        <div style="width:{fail_pct}%;background:#ef4444;"></div>
        <div style="width:{skip_pct}%;background:#a855f7;"></div>
      </div>
    </div>
    <div style="display:flex;gap:18px;margin-top:10px;flex-wrap:wrap;">
      <span style="font-size:0.75rem;color:#4ade80;">&#9632; Passed {pass_pct}%</span>
      <span style="font-size:0.75rem;color:#f87171;">&#9632; Failed/Error {fail_pct}%</span>
      <span style="font-size:0.75rem;color:#c084fc;">&#9632; Skipped {skip_pct}%</span>
    </div>
  </div>

  <!-- Controls -->
  <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px;margin-bottom:16px;">
    <div style="display:flex;gap:8px;flex-wrap:wrap;" id="filter-bar">
      <button class="filter-btn active" onclick="filterModules('all',this)">All</button>
      <button class="filter-btn" onclick="filterModules('failed',this)">Failed only</button>
      <button class="filter-btn" onclick="filterModules('passed',this)">Passed only</button>
    </div>
    <div style="display:flex;gap:8px;">
      <button class="filter-btn" onclick="expandAll()">Expand all</button>
      <button class="filter-btn" onclick="collapseAll()">Collapse all</button>
    </div>
  </div>

  <!-- Modules -->
  <div id="modules-container">
    {module_rows}
  </div>

  <p style="text-align:center;color:#334155;font-size:0.78rem;margin-top:36px;">
    ExportEdge AI FYP &mdash; generate_report.py
  </p>

</div>
<script>
  var moduleIds = {module_id_list!r};

  function toggle(id) {{
    var el    = document.getElementById(id);
    var arrow = document.getElementById('arrow-' + id);
    var open  = el.style.display !== 'none';
    el.style.display       = open ? 'none' : 'block';
    arrow.style.transform  = open ? ''     : 'rotate(180deg)';
  }}

  function expandAll() {{
    moduleIds.forEach(function(id) {{
      var safe  = id.replace(/\\./g, '_').replace(/ /g, '_');
      var el    = document.getElementById(safe);
      var arrow = document.getElementById('arrow-' + safe);
      if (el) {{ el.style.display = 'block'; arrow.style.transform = 'rotate(180deg)'; }}
    }});
  }}

  function collapseAll() {{
    moduleIds.forEach(function(id) {{
      var safe  = id.replace(/\\./g, '_').replace(/ /g, '_');
      var el    = document.getElementById(safe);
      var arrow = document.getElementById('arrow-' + safe);
      if (el) {{ el.style.display = 'none'; arrow.style.transform = ''; }}
    }});
  }}

  function filterModules(type, btn) {{
    document.querySelectorAll('#filter-bar .filter-btn').forEach(function(b) {{
      b.classList.remove('active');
    }});
    btn.classList.add('active');
    document.querySelectorAll('.module-block').forEach(function(block) {{
      if (type === 'all') {{ block.style.display = ''; return; }}
      var rows     = block.querySelectorAll('tr.test-row');
      var hasFail  = false;
      var hasPass  = false;
      rows.forEach(function(r) {{
        var s = r.getAttribute('data-status');
        if (s === 'FAILED' || s === 'ERROR') hasFail = true;
        if (s === 'PASSED') hasPass = true;
      }});
      if (type === 'failed') block.style.display = hasFail ? '' : 'none';
      if (type === 'passed') block.style.display = (hasPass && !hasFail) ? '' : 'none';
    }});
  }}
</script>
</body>
</html>"""


def main():
    valid = list(CATEGORIES.keys())

    # resolve category from argv
    category = "all"
    if len(sys.argv) > 1:
        category = sys.argv[1].lower()
        if category not in valid:
            print(f"Unknown category '{category}'. Choose from: {', '.join(valid)}")
            sys.exit(1)

    meta    = CATEGORIES[category]
    targets = _resolve_targets(category)

    html_path = TESTS_DIR / f"report_{category}.html"

    ok = run_pytest(targets)
    if not ok:
        print("ERROR: pytest did not produce an XML file. Aborting.")
        sys.exit(1)

    data = parse_xml(XML_PATH)
    html = build_html(data, label=meta["label"], accent=meta["accent"])
    html_path.write_text(html, encoding="utf-8")

    XML_PATH.unlink(missing_ok=True)

    print(f"\n{'='*56}")
    print(f"  [{meta['label']}]")
    print(f"  Report : {html_path}")
    print(f"  Total  : {data['total']}  Passed: {data['passed']}  "
          f"Failed: {data['failed']}  Skipped: {data['skipped']}")
    print(f"{'='*56}\n")

    import webbrowser
    webbrowser.open(html_path.as_uri())


if __name__ == "__main__":
    main()
