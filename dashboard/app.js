/**
 * COVID-19 Wave Analytics: Interactive Dashboard Application Logic
 * Renders dynamic regression charts, interactive PCA biplot, clustering archetypes,
 * and data panel explorer.
 */

document.addEventListener('DOMContentLoaded', () => {
    // 1. Load Data (from window.COVID_DATA or fetch fallback)
    if (window.COVID_DATA) {
        initDashboard(window.COVID_DATA);
    } else {
        fetch('data.json')
            .then(res => res.json())
            .then(data => initDashboard(data))
            .catch(err => {
                console.error("Failed to load data.json:", err);
                document.querySelector('.dashboard-container').innerHTML = `
                    <div style="padding: 2rem; color: #ef4444; background: #1e1e1e; border-radius: 8px;">
                        <h3>Error Loading Dashboard Data</h3>
                        <p>${err.message}</p>
                    </div>
                `;
            });
    }

    // Theme Toggle
    const themeBtn = document.getElementById('themeToggle');
    const themeIcon = document.getElementById('themeIcon');
    themeBtn.addEventListener('click', () => {
        document.body.classList.toggle('light-theme');
        themeIcon.textContent = document.body.classList.contains('light-theme') ? '🌙' : '☀️';
        // Re-render canvases with theme colors
        if (window.CURRENT_DATA) {
            renderRegressionChart();
            renderPcaBiplot();
        }
    });
});

let DATA = null;
const CLUSTER_COLORS = ['#e7298a', '#d95f02', '#2b5c8f', '#10b981', '#a855f7'];

function initDashboard(data) {
    DATA = data;
    window.CURRENT_DATA = data;

    // Populate State Selector
    const stateSel = document.getElementById('stateSelector');
    stateSel.innerHTML = '';

    // Sort states alphabetically by State Name
    const sortedStates = [...data.state_results].sort((a, b) => a.State_Name.localeCompare(b.State_Name));

    // Add Maharashtra, Delhi, Kerala first as key highlights, then all
    sortedStates.forEach(st => {
        const opt = document.createElement('option');
        opt.value = st.State_Code;
        opt.textContent = `${st.State_Name} (${st.State_Code})`;
        stateSel.appendChild(opt);
    });

    // Default select Maharashtra ('MH') if present
    if ([...stateSel.options].some(o => o.value === 'MH')) {
        stateSel.value = 'MH';
    }

    // Event Listeners for Selectors
    stateSel.addEventListener('change', () => {
        updateScorecard();
        renderRegressionChart();
    });

    document.getElementById('waveSelector').addEventListener('change', () => {
        updateScorecard();
        renderRegressionChart();
    });

    // Initialize UI Components
    updateScorecard();
    renderRegressionChart();
    renderPcaBiplot();
    renderPcaLoadingsTable();
    renderArchetypes();
    renderPanelTable();

    // Resize handler for responsive canvas redraws
    window.addEventListener('resize', () => {
        renderRegressionChart();
        renderPcaBiplot();
    });
}

/**
 * Update Scorecard Information
 */
function updateScorecard() {
    const stateCode = document.getElementById('stateSelector').value;
    const wave = document.getElementById('waveSelector').value;

    const stateObj = DATA.state_results.find(s => s.State_Code === stateCode);
    const waveObj = DATA.regression_metrics.find(m => m.State_Code === stateCode && m.Wave === wave);

    if (stateObj) {
        document.getElementById('scorecardStateCode').textContent = stateObj.State_Code;
        document.getElementById('scorecardStateName').textContent = stateObj.State_Name;
    }

    if (waveObj) {
        document.getElementById('scorePeakDate').textContent = waveObj.Peak_Date;
        document.getElementById('scorePeakCases').textContent = Number(waveObj.Peak_Daily_Cases).toLocaleString() + " /day";
        document.getElementById('scorePeakIncidence').textContent = waveObj.Peak_Per_100k + " /100k";
        document.getElementById('scoreDaysToPeak').textContent = `${waveObj.Days_To_Peak} days`;

        document.getElementById('scorePolyDeg').textContent = `deg ${waveObj.Poly_Degree}`;
        document.getElementById('scorePolyR2').textContent = waveObj.Poly_R2.toFixed(3);
        document.getElementById('scorePolyRMSE').textContent = Number(waveObj.Poly_RMSE).toLocaleString();

        document.getElementById('scoreRidgeAlpha').textContent = `α = ${waveObj.Ridge_Alpha}`;
        document.getElementById('scoreRidgeR2').textContent = waveObj.Ridge_R2.toFixed(3);
        document.getElementById('scoreRidgeRMSE').textContent = Number(waveObj.Ridge_RMSE).toLocaleString();
    } else {
        document.getElementById('scorePeakDate').textContent = "N/A";
        document.getElementById('scorePeakCases').textContent = "N/A";
        document.getElementById('scorePeakIncidence').textContent = "N/A";
        document.getElementById('scoreDaysToPeak').textContent = "N/A";
        document.getElementById('scorePolyR2').textContent = "N/A";
        document.getElementById('scoreRidgeR2').textContent = "N/A";
    }
}

/**
 * Render Epidemic Curve and Regression Overlays on HTML5 Canvas
 */
function renderRegressionChart() {
    const canvas = document.getElementById('regressionChart');
    const wrapper = canvas.parentElement;
    canvas.width = wrapper.clientWidth * window.devicePixelRatio;
    canvas.height = wrapper.clientHeight * window.devicePixelRatio;

    const ctx = canvas.getContext('2d');
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
    const width = wrapper.clientWidth;
    const height = wrapper.clientHeight;

    const isLight = document.body.classList.contains('light-theme');
    const textColor = isLight ? '#475569' : '#94a3b8';
    const gridColor = isLight ? 'rgba(0,0,0,0.06)' : 'rgba(255,255,255,0.06)';
    const actualColor = isLight ? '#64748b' : '#94a3b8';
    const polyColor = '#f97316';
    const ridgeColor = '#38bdf8';

    ctx.clearRect(0, 0, width, height);

    const stateCode = document.getElementById('stateSelector').value;
    const wave = document.getElementById('waveSelector').value;

    const curveData = DATA.fitted_curves[wave] ? DATA.fitted_curves[wave][stateCode] : null;

    if (!curveData || !curveData.actual || curveData.actual.length === 0) {
        ctx.fillStyle = textColor;
        ctx.font = '14px Outfit';
        ctx.textAlign = 'center';
        ctx.fillText(`No regression data available for ${stateCode} in ${wave}`, width / 2, height / 2);
        return;
    }

    const actual = curveData.actual;
    const polyFit = curveData.poly_fit;
    const ridgeFit = curveData.ridge_fit;
    const dates = curveData.dates;
    const n = actual.length;

    const maxVal = Math.max(...actual, ...polyFit, ...ridgeFit) * 1.1;
    const padding = { top: 25, right: 30, bottom: 40, left: 65 };
    const chartW = width - padding.left - padding.right;
    const chartH = height - padding.top - padding.bottom;

    // Helper functions
    const getX = i => padding.left + (i / (n - 1)) * chartW;
    const getY = val => padding.top + chartH - (val / maxVal) * chartH;

    // Draw Grid & Y Labels
    ctx.lineWidth = 1;
    ctx.strokeStyle = gridColor;
    ctx.fillStyle = textColor;
    ctx.font = '11px JetBrains Mono';
    ctx.textAlign = 'right';

    const yTicks = 5;
    for (let i = 0; i <= yTicks; i++) {
        const val = (maxVal / yTicks) * i;
        const y = getY(val);
        ctx.beginPath();
        ctx.moveTo(padding.left, y);
        ctx.lineTo(width - padding.right, y);
        ctx.stroke();
        ctx.fillText(Math.round(val).toLocaleString(), padding.left - 10, y + 4);
    }

    // Draw X Dates
    ctx.textAlign = 'center';
    const xStep = Math.floor(n / 5);
    for (let i = 0; i < n; i += xStep) {
        const x = getX(i);
        const dateStr = dates[i] ? dates[i].substring(5) : `${i}d`;
        ctx.fillText(dateStr, x, height - 12);
    }

    // 1. Draw Actual Observed (Filled Area & Line)
    ctx.beginPath();
    ctx.moveTo(getX(0), getY(actual[0]));
    for (let i = 1; i < n; i++) {
        ctx.lineTo(getX(i), getY(actual[i]));
    }
    ctx.strokeStyle = actualColor;
    ctx.lineWidth = 1.8;
    ctx.setLineDash([]);
    ctx.stroke();

    // Area fill
    ctx.lineTo(getX(n - 1), padding.top + chartH);
    ctx.lineTo(getX(0), padding.top + chartH);
    ctx.closePath();
    ctx.fillStyle = isLight ? 'rgba(100, 116, 139, 0.08)' : 'rgba(148, 163, 184, 0.08)';
    ctx.fill();

    // 2. Draw Polynomial Fit Line
    ctx.beginPath();
    ctx.moveTo(getX(0), getY(polyFit[0]));
    for (let i = 1; i < n; i++) {
        ctx.lineTo(getX(i), getY(polyFit[i]));
    }
    ctx.strokeStyle = polyColor;
    ctx.lineWidth = 2.4;
    ctx.setLineDash([5, 4]);
    ctx.stroke();

    // 3. Draw Ridge Fit Line
    ctx.beginPath();
    ctx.moveTo(getX(0), getY(ridgeFit[0]));
    for (let i = 1; i < n; i++) {
        ctx.lineTo(getX(i), getY(ridgeFit[i]));
    }
    ctx.strokeStyle = ridgeColor;
    ctx.lineWidth = 2.2;
    ctx.setLineDash([2, 3]);
    ctx.stroke();
    ctx.setLineDash([]); // Reset dash

    // Mark Peak Point
    const peakIdx = actual.indexOf(Math.max(...actual));
    if (peakIdx >= 0) {
        const peakX = getX(peakIdx);
        const peakY = getY(actual[peakIdx]);
        ctx.fillStyle = '#ef4444';
        ctx.beginPath();
        ctx.arc(peakX, peakY, 5, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 1.5;
        ctx.stroke();

        ctx.fillStyle = isLight ? '#0f172a' : '#ffffff';
        ctx.font = 'bold 11px Outfit';
        ctx.textAlign = 'center';
        ctx.fillText(`Peak: ${Math.round(actual[peakIdx]).toLocaleString()}`, peakX, peakY - 10);
    }
}

/**
 * Interactive 2D PCA Biplot on HTML5 Canvas
 */
let pcaPoints = [];

function renderPcaBiplot() {
    const canvas = document.getElementById('pcaCanvas');
    const wrapper = canvas.parentElement;
    canvas.width = wrapper.clientWidth * window.devicePixelRatio;
    canvas.height = wrapper.clientHeight * window.devicePixelRatio;

    const ctx = canvas.getContext('2d');
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
    const width = wrapper.clientWidth;
    const height = wrapper.clientHeight;

    const isLight = document.body.classList.contains('light-theme');
    const textColor = isLight ? '#334155' : '#94a3b8';
    const axisColor = isLight ? '#cbd5e1' : '#334155';

    ctx.clearRect(0, 0, width, height);

    const states = DATA.state_results;
    if (!states || states.length === 0) return;

    // PC1 and PC2 ranges
    const pc1Vals = states.map(s => s.PC1);
    const pc2Vals = states.map(s => s.PC2);

    const padFactor = 1.3;
    const maxPC = Math.max(
        Math.abs(Math.min(...pc1Vals)), Math.abs(Math.max(...pc1Vals)),
        Math.abs(Math.min(...pc2Vals)), Math.abs(Math.max(...pc2Vals))
    ) * padFactor;

    const cx = width / 2;
    const cy = height / 2;
    const scale = (Math.min(width, height) / 2 - 40) / maxPC;

    // Draw Axes
    ctx.strokeStyle = axisColor;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(30, cy);
    ctx.lineTo(width - 30, cy);
    ctx.moveTo(cx, 30);
    ctx.lineTo(cx, height - 30);
    ctx.stroke();

    // Axis Labels
    ctx.fillStyle = textColor;
    ctx.font = '11px Outfit';
    ctx.textAlign = 'right';
    ctx.fillText(`PC1 (${(DATA.meta.pca_explained_var[0] * 100).toFixed(1)}%) →`, width - 35, cy - 8);
    ctx.textAlign = 'center';
    ctx.fillText(`↑ PC2 (${(DATA.meta.pca_explained_var[1] * 100).toFixed(1)}%)`, cx, 20);

    // Draw Feature Loading Vectors
    const loadings = DATA.pca_loadings;
    const vecScale = scale * 1.8;
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = 'rgba(239, 68, 68, 0.6)';

    if (loadings && loadings.PC1_Loading) {
        Object.keys(loadings.PC1_Loading).forEach(feat => {
            const vx = cx + loadings.PC1_Loading[feat] * vecScale;
            const vy = cy - loadings.PC2_Loading[feat] * vecScale;

            ctx.beginPath();
            ctx.moveTo(cx, cy);
            ctx.lineTo(vx, vy);
            ctx.stroke();

            // Label
            ctx.fillStyle = '#ef4444';
            ctx.font = '9px Outfit';
            const shortName = feat.replace('_Per_100k', '').replace('_Pct', '').replace('Wave', 'W');
            ctx.fillText(shortName, vx + (vx > cx ? 6 : -6), vy + (vy > cy ? 6 : -6));
        });
    }

    // Draw State Points
    pcaPoints = [];
    states.forEach(st => {
        const px = cx + st.PC1 * scale;
        const py = cy - st.PC2 * scale;
        const color = CLUSTER_COLORS[st.KMeans_Cluster % CLUSTER_COLORS.length];

        // Store screen coordinates for hover collision
        pcaPoints.push({ x: px, y: py, r: 8, state: st, color: color });

        // Point outer circle
        ctx.beginPath();
        ctx.arc(px, py, 6, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 1;
        ctx.stroke();

        // Label state code
        ctx.fillStyle = isLight ? '#0f172a' : '#f8fafc';
        ctx.font = 'bold 9px JetBrains Mono';
        ctx.textAlign = 'left';
        ctx.fillText(st.State_Code, px + 7, py + 3);
    });

    // PCA Cluster Legend
    const legendContainer = document.getElementById('pcaClusterLegend');
    if (legendContainer && legendContainer.children.length === 0) {
        Object.keys(DATA.cluster_profiles).forEach(cid => {
            const color = CLUSTER_COLORS[cid % CLUSTER_COLORS.length];
            const item = document.createElement('span');
            item.className = 'legend-item';
            item.innerHTML = `<span class="dot" style="background-color: ${color};"></span> Cluster ${cid}`;
            legendContainer.appendChild(item);
        });
    }

    // Attach Mousemove listener for Hover Tooltip
    canvas.onmousemove = e => {
        const rect = canvas.getBoundingClientRect();
        const mx = e.clientX - rect.left;
        const my = e.clientY - rect.top;

        const hovered = pcaPoints.find(p => {
            const dx = p.x - mx;
            const dy = p.y - my;
            return Math.sqrt(dx * dx + dy * dy) <= p.r + 4;
        });

        const hoverCard = document.getElementById('stateHoverCard');
        if (hovered) {
            const s = hovered.state;
            hoverCard.innerHTML = `
                <div style="width: 100%;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.35rem;">
                        <strong style="color: ${hovered.color}; font-size: 1rem;">${s.State_Name} (${s.State_Code})</strong>
                        <span style="font-family: var(--font-mono); font-size: 0.75rem; background: rgba(255,255,255,0.1); padding: 0.15rem 0.4rem; border-radius: 4px;">Cluster ${s.KMeans_Cluster}</span>
                    </div>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.35rem; font-size: 0.78rem;">
                        <div>PC1: <strong>${s.PC1}</strong> | PC2: <strong>${s.PC2}</strong></div>
                        <div>Delta Surge: <strong>${s.Delta_Surge_Ratio}x</strong></div>
                        <div>Dose 1: <strong>${s.Dose1_Coverage_Pct}%</strong></div>
                        <div>W2 Peak/100k: <strong>${s.Wave2_Peak_Per_100k}</strong></div>
                    </div>
                </div>
            `;
        } else {
            hoverCard.innerHTML = `<span class="hover-tip">Hover over any point in the PCA biplot to view state details.</span>`;
        }
    };
}

/**
 * Render PCA Loadings Table
 */
function renderPcaLoadingsTable() {
    const tbody = document.getElementById('pcaLoadingsTableBody');
    tbody.innerHTML = '';

    const loadings = DATA.pca_loadings;
    if (!loadings || !loadings.PC1_Loading) return;

    Object.keys(loadings.PC1_Loading).forEach(feat => {
        const pc1 = loadings.PC1_Loading[feat];
        const pc2 = loadings.PC2_Loading[feat];
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td><strong>${feat.replace(/_/g, ' ')}</strong></td>
            <td style="color: ${pc1 >= 0 ? '#10b981' : '#ef4444'};">${pc1 > 0 ? '+' : ''}${pc1.toFixed(3)}</td>
            <td style="color: ${pc2 >= 0 ? '#10b981' : '#ef4444'};">${pc2 > 0 ? '+' : ''}${pc2.toFixed(3)}</td>
        `;
        tbody.appendChild(tr);
    });
}

/**
 * Render Archetype Cards
 */
function renderArchetypes() {
    const container = document.getElementById('archetypesContainer');
    container.innerHTML = '';

    const profiles = DATA.cluster_profiles;
    Object.keys(profiles).forEach(cid => {
        const prof = profiles[cid];
        const card = document.createElement('div');
        card.className = 'archetype-card';

        const stateChips = prof.state_codes.map(c => `<span class="state-chip">${c}</span>`).join('');

        card.innerHTML = `
            <div class="archetype-top">
                <span class="archetype-cluster-badge cluster-${cid}-badge">Cluster ${cid}</span>
                <span class="archetype-count">${prof.count} States / UTs</span>
            </div>
            <div class="archetype-name">${prof.archetype || `Cluster Archetype ${cid}`}</div>
            <div class="archetype-stats">
                <div class="arch-stat">Delta Surge Multiplier: <strong>${prof.mean_delta_surge}x</strong></div>
                <div class="arch-stat">Wave 2 Peak / 100k: <strong>${prof.mean_w2_peak}</strong></div>
                <div class="arch-stat">Dose 1 Coverage: <strong>${prof.mean_dose1_cov}%</strong></div>
                <div class="arch-stat">Dose 2 Coverage: <strong>${prof.mean_dose2_cov}%</strong></div>
            </div>
            <div class="archetype-states-list">
                ${stateChips}
            </div>
        `;
        container.appendChild(card);
    });
}

/**
 * Render Data in Brief Preview Table
 */
function renderPanelTable() {
    const tbody = document.getElementById('panelTableBody');
    tbody.innerHTML = '';

    const states = [...DATA.state_results].sort((a, b) => b.Wave2_Peak_Per_100k - a.Wave2_Peak_Per_100k);

    states.forEach(st => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td><strong>${st.State_Code}</strong></td>
            <td>${st.State_Name}</td>
            <td><span class="badge" style="background: rgba(255,255,255,0.06); font-size: 0.72rem;">Cluster ${st.KMeans_Cluster}</span></td>
            <td>${st.Wave1_Peak_Per_100k}</td>
            <td><strong>${st.Wave2_Peak_Per_100k}</strong></td>
            <td>${st.Delta_Surge_Ratio}x</td>
            <td>${st.Dose1_Coverage_Pct}%</td>
            <td>${st.Dose2_Coverage_Pct}%</td>
            <td>${st.Case_Fatality_Rate_Pct}%</td>
            <td>${st.Silhouette_Value}</td>
        `;
        tbody.appendChild(tr);
    });
}

/**
 * Modal Image Viewer
 */
function openFigureModal(src, caption) {
    const modal = document.getElementById('figModal');
    const modalImg = document.getElementById('modalImg');
    const modalCaption = document.getElementById('modalCaption');
    modal.style.display = 'block';
    modalImg.src = src;
    modalCaption.textContent = caption;
}

function closeFigureModal() {
    document.getElementById('figModal').style.display = 'none';
}
