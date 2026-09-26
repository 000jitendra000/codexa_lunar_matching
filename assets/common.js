// Codexa — Lunar Match & Register — shared helpers used across every page.
window.CODEXA = (function () {
    const API_BASE = window.location.origin.startsWith("http") ? window.location.origin : "http://127.0.0.1:8000";
    const RESULT_KEY = "lunar_last_result";
    const STATS_KEY = "lunar_session_stats";

    const VIZ_PAGES = [
        {
            key: "correspondence", file: "results-correspondence.html", label: "Correspondences Plot",
            short: "Line-by-line vectors connecting every matched keypoint between the two frames."
        },
        {
            key: "overlay", file: "results-overlay.html", label: "Registration Overlay",
            short: "Image B warped and blended over Image A in false color to check alignment."
        },
        {
            key: "checkerboard", file: "results-checkerboard.html", label: "Checkerboard Comparison",
            short: "Alternating tiles from A and warped B — seams reveal any residual misalignment."
        },
        {
            key: "cmap_a", file: "results-cmap-a.html", label: "Image A Evidence Map",
            short: "Heatmap of match confidence across Image A's surface."
        },
        {
            key: "cmap_b", file: "results-cmap-b.html", label: "Image B Evidence Map",
            short: "Heatmap of match confidence across Image B's surface."
        }
    ];

    function saveResult(jobId, result, viz) {
        const payload = { jobId, result, viz, timestamp: new Date().toISOString() };
        try { localStorage.setItem(RESULT_KEY, JSON.stringify(payload)); } catch (e) { }
        return payload;
    }

    function loadResult() {
        try {
            const raw = localStorage.getItem(RESULT_KEY);
            return raw ? JSON.parse(raw) : null;
        } catch (e) { return null; }
    }

    function clearResult() {
        try { localStorage.removeItem(RESULT_KEY); } catch (e) { }
    }

    function getStats() {
        try {
            const raw = localStorage.getItem(STATS_KEY);
            return raw ? JSON.parse(raw) : { tested: 0, accepted: 0, rejected: 0 };
        } catch (e) { return { tested: 0, accepted: 0, rejected: 0 }; }
    }

    function bumpStats(matched) {
        const s = getStats();
        s.tested++;
        if (matched) s.accepted++; else s.rejected++;
        try { localStorage.setItem(STATS_KEY, JSON.stringify(s)); } catch (e) { }
        return s;
    }

    function renderSessionBadge(elId) {
        const el = document.getElementById(elId);
        if (!el) return;
        const s = getStats();
        el.textContent = `Pairs Tested: ${s.tested} | Accepted: ${s.accepted} | Rejected: ${s.rejected}`;
    }

    async function checkHealth(badgeId, textId) {
        const badge = document.getElementById(badgeId);
        const text = document.getElementById(textId);
        if (!badge || !text) return;
        try {
            const res = await fetch(`${API_BASE}/health`);
            if (!res.ok) throw new Error("HTTP error");
            const data = await res.json();
            text.textContent = data.model_available ? "API Ready & Model Loaded" : "API Online (Model Loading)";
            badge.classList.remove("offline");
        } catch (err) {
            text.textContent = `API Offline (${API_BASE})`;
            badge.classList.add("offline");
        }
    }

    function vizUrl(payload, key) {
        if (!payload || !payload.viz) return null;
        const map = {
            correspondence: payload.viz.correspondence_image,
            overlay: payload.viz.registration_overlay,
            checkerboard: payload.viz.checkerboard,
            cmap_a: payload.viz.confidence_map_a,
            cmap_b: payload.viz.confidence_map_b
        };
        const path = map[key];
        return path ? `${API_BASE}${path}` : null;
    }

    return { API_BASE, VIZ_PAGES, saveResult, loadResult, clearResult, getStats, bumpStats, renderSessionBadge, checkHealth, vizUrl };
})();