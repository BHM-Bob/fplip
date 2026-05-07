/**
 * All-Atom Visualization Web Application
 *
 * Main application logic for interactive visualization.
 */

// Global state
const appState = {
    stage: null,                    // NGL Stage instance
    interactions: [],               // Current interactions
    selectedInteractions: new Set(), // Selected interaction IDs
    groups: {},                     // Interaction groups
    filters: {
        types: [],
        residues: [],
        distanceMin: null,
        distanceMax: null
    },
    representation: 'cartoon',      // Current molecular representation
    interactionShapes: [],          // NGL shape components for interactions
    interactionData: {},            // Map of shape name to interaction data
    selectedInteraction: null,      // Currently selected interaction
    focusMode: {                    // Focus mode settings
        enabled: false,
        centerChains: [],
        radius: 5.0
    },
    structureComponent: null        // NGL structure component
};

// Color scheme matching main PLIP (CSS uses 0-255 range)
const INTERACTION_COLORS_CSS = {
    'hbond': [0, 102, 204],           // Blue
    'hbond_possible': [153, 204, 255], // Light blue
    'saltbridge': [255, 204, 0],      // Yellow
    'hydrophobic': [128, 128, 128],   // Grey
    'pistacking': [0, 153, 0],        // Green
    'pication': [255, 102, 0],        // Orange
    'halogen': [0, 204, 204],         // Cyan
    'water_bridge': [153, 204, 255],  // Light blue
    'water_bridge_possible': [153, 204, 255],  // Light blue
    'metal': [153, 0, 204]            // Purple
};

// Color scheme for NGL (uses 0-1 range)
const INTERACTION_COLORS = {};
for (const [type, rgb] of Object.entries(INTERACTION_COLORS_CSS)) {
    INTERACTION_COLORS[type] = rgb.map(v => v / 255);
}

// User custom color preferences (persisted in memory during session)
const userColorPreferences = {};

/**
 * Convert RGB array to hex color string
 */
function rgbToHex(rgb) {
    return '#' + rgb.map(v => {
        const hex = v.toString(16);
        return hex.length === 1 ? '0' + hex : hex;
    }).join('');
}

/**
 * Convert hex color string to RGB array
 */
function hexToRgb(hex) {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result ? [
        parseInt(result[1], 16),
        parseInt(result[2], 16),
        parseInt(result[3], 16)
    ] : [200, 200, 200];
}

/**
 * Get color for interaction type (respects user preferences)
 */
function getInteractionColor(type) {
    if (userColorPreferences[type]) {
        return userColorPreferences[type];
    }
    return INTERACTION_COLORS_CSS[type] || [200, 200, 200];
}

/**
 * Get color for interaction type in NGL format (0-1 range)
 */
function getInteractionColorNGL(type) {
    const color = getInteractionColor(type);
    return color.map(v => v / 255);
}



/**
 * Set custom color for interaction type
 */
function setInteractionColor(type, color) {
    // Store user preference
    userColorPreferences[type] = color;

    // Update CSS color map
    INTERACTION_COLORS_CSS[type] = color;

    // Update NGL color map
    INTERACTION_COLORS[type] = color.map(v => v / 255);

    // Update color indicator in UI
    const colorIndicators = document.querySelectorAll('.type-color-indicator');
    const typeItems = document.querySelectorAll('#typeFilters .type-filter-item');

    typeItems.forEach((item, index) => {
        const checkbox = item.querySelector('input[type="checkbox"]');
        if (checkbox && checkbox.value === type) {
            const indicator = item.querySelector('.type-color-indicator');
            if (indicator) {
                indicator.style.backgroundColor = `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
            }
        }
    });

    // Refresh interaction display with new colors
    updateInteractionDisplay();

    showToast(`Color updated for ${formatTypeName(type)}`, 'success');
}

// Initialize application
document.addEventListener('DOMContentLoaded', function() {
    initNGL();
    loadInitialData();
});

/**
 * Initialize NGL Viewer
 */
function initNGL() {
    appState.stage = new NGL.Stage('viewport', {
        backgroundColor: 'black',
        cameraType: 'perspective'
    });

    // Handle window resize
    window.addEventListener('resize', function() {
        appState.stage.handleResize();
    });

    // Setup click handler for interaction selection
    appState.stage.signals.clicked.add(function(pickingProxy) {
        if (pickingProxy && pickingProxy.cylinder) {
            const cylinder = pickingProxy.cylinder;
            const name = cylinder.name;

            if (name && appState.interactionData[name]) {
                const interaction = appState.interactionData[name];
                selectInteraction(interaction);
            }
        }
    });
}

/**
 * Load initial data from API
 */
async function loadInitialData() {
    try {
        // Load interaction types
        const typesResponse = await fetch('/api/interactions/types');
        const typesData = await typesResponse.json();
        populateTypeFilters(typesData.types);

        // Load residues
        const residuesResponse = await fetch('/api/residues');
        const residuesData = await residuesResponse.json();
        populateResidueSelect(residuesData.residues);

        // Load statistics
        const statsResponse = await fetch('/api/interactions/summary');
        const statsData = await statsResponse.json();
        updateStatistics(statsData);

        // Initialize with empty interactions (user must apply filters to see them)
        appState.interactions = [];
        updateInteractionTable();
        updateInteractionDisplay();
        updateInteractionCount(0);

        // Load PDB structure
        await loadPDBStructure();

        showToast('Data loaded. Use filters to display interactions.', 'info');

    } catch (error) {
        console.error('Error loading initial data:', error);
        showToast('Error loading data', 'danger');
    }
}

/**
 * Load PDB structure into NGL Viewer
 */
async function loadPDBStructure() {
    try {
        const response = await fetch('/api/pdb');
        const data = await response.json();

        if (data.has_pdb_content) {
            // Load PDB content from our API
            const pdbResponse = await fetch('/api/pdb/content');
            const pdbContent = await pdbResponse.text();

            // Create a Blob from the PDB content
            const blob = new Blob([pdbContent], { type: 'text/plain' });
            const url = URL.createObjectURL(blob);

            // Load into NGL
            appState.stage.loadFile(url, { ext: 'pdb' }).then(function(component) {
                appState.structureComponent = component;
                component.addRepresentation(appState.representation, {
                    color: 'chainname'
                });
                component.autoView();
                showToast('PDB structure loaded', 'success');
            }).catch(function(error) {
                console.error('Error loading PDB into NGL:', error);
                showToast('Error loading PDB structure', 'danger');
            });
        } else if (data.pdb_file) {
            // Fallback: try to load from RCSB if it's a PDB ID
            const pdbId = data.pdb_file.split('/').pop().replace('.pdb', '');
            if (pdbId.length === 4) {
                appState.stage.loadFile('rcsb://' + pdbId).then(function(component) {
                    appState.structureComponent = component;
                    component.addRepresentation(appState.representation, {
                        color: 'chainname'
                    });
                    component.autoView();
                    showToast('PDB structure loaded from RCSB', 'success');
                }).catch(function(error) {
                    console.error('Error loading from RCSB:', error);
                    showToast('Could not load PDB structure', 'warning');
                });
            } else {
                showToast('No PDB structure available', 'warning');
            }
        } else {
            showToast('No PDB structure available', 'warning');
        }
    } catch (error) {
        console.error('Error loading PDB:', error);
        showToast('Error loading PDB structure', 'danger');
    }
}

/**
 * Populate interaction type filter checkboxes
 */
function populateTypeFilters(types) {
    const container = document.getElementById('typeFilters');
    container.innerHTML = '';

    types.forEach(type => {
        const color = getInteractionColor(type);
        const colorStr = `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
        const colorHex = rgbToHex(color);

        const div = document.createElement('div');
        div.className = 'type-filter-item';
        div.innerHTML = `
            <input type="checkbox" id="type_${type}" value="${type}" checked>
            <label for="color_${type}" class="type-color-indicator" style="background-color: ${colorStr}; cursor: pointer; margin-bottom: 0;" title="Click to change color"></label>
            <input type="color" id="color_${type}" value="${colorHex}" style="position: absolute; opacity: 0; pointer-events: none; width: 0; height: 0;" data-type="${type}">
            <label for="type_${type}">${formatTypeName(type)}</label>
        `;
        container.appendChild(div);

        // Add change listener to color input
        const colorInput = div.querySelector(`#color_${type}`);
        colorInput.addEventListener('change', function() {
            const newColor = hexToRgb(this.value);
            setInteractionColor(type, newColor);
            // Update the visual indicator
            const indicator = div.querySelector('.type-color-indicator');
            if (indicator) {
                indicator.style.backgroundColor = `rgb(${newColor[0]}, ${newColor[1]}, ${newColor[2]})`;
            }
        });
    });
}

/**
 * Populate residue select dropdown
 */
function populateResidueSelect(residues) {
    const select = document.getElementById('residueSelect');
    select.innerHTML = '';

    residues.forEach(res => {
        const option = document.createElement('option');
        option.value = `${res.name}:${res.chain}:${res.num}`;
        option.textContent = `${res.name}:${res.chain}:${res.num}`;

        // Add double-click handler to focus residue in NGL
        option.addEventListener('dblclick', function() {
            focusResidue(res.chain, res.num);
        });

        select.appendChild(option);
    });

    // Also add change handler for single click (when not multi-selecting)
    select.addEventListener('change', function() {
        const selectedOption = select.options[select.selectedIndex];
        if (selectedOption) {
            const res = residues.find(r =>
                `${r.name}:${r.chain}:${r.num}` === selectedOption.value
            );
            if (res) {
                // Small delay to allow multi-select
                setTimeout(() => {
                    if (select.selectedOptions.length === 1) {
                        focusResidue(res.chain, res.num);
                    }
                }, 100);
            }
        }
    });

    // Populate chains for Focus Mode
    populateChainsSelect(residues);
}

/**
 * Populate chains select dropdown for Focus Mode
 */
function populateChainsSelect(residues) {
    const select = document.getElementById('focusChains');
    if (!select) return;

    // Get unique chains
    const chains = [...new Set(residues.map(r => r.chain))].sort();

    select.innerHTML = '';
    chains.forEach(chain => {
        const option = document.createElement('option');
        option.value = chain;
        option.textContent = `Chain ${chain}`;
        select.appendChild(option);
    });
}

/**
 * Toggle Focus Mode on/off
 */
function toggleFocusMode() {
    const chainsSelect = document.getElementById('focusChains');
    const radiusInput = document.getElementById('focusRadius');
    const toggleBtn = document.getElementById('focusToggleBtn');
    const resetBtn = document.getElementById('focusResetBtn');

    // Get selected chains
    const selectedChains = Array.from(chainsSelect.selectedOptions).map(opt => opt.value);

    if (selectedChains.length === 0) {
        showToast('Please select at least one chain as center', 'warning');
        return;
    }

    if (appState.focusMode.enabled) {
        // Disable focus mode
        disableFocusMode();
        toggleBtn.textContent = 'Enable Focus';
        toggleBtn.classList.remove('btn-danger');
        toggleBtn.classList.add('btn-warning');
        resetBtn.disabled = true;
        chainsSelect.disabled = false;
        radiusInput.disabled = false;
    } else {
        // Enable focus mode
        const radius = parseFloat(radiusInput.value) || 5.0;
        enableFocusMode(selectedChains, radius);
        toggleBtn.textContent = 'Disable Focus';
        toggleBtn.classList.remove('btn-warning');
        toggleBtn.classList.add('btn-danger');
        resetBtn.disabled = false;
        chainsSelect.disabled = true;
        radiusInput.disabled = true;
    }
}

/**
 * Enable Focus Mode
 */
function enableFocusMode(chains, radius) {
    appState.focusMode.enabled = true;
    appState.focusMode.centerChains = chains;
    appState.focusMode.radius = radius;

    // Update structure display with focus
    updateStructureDisplay();

    // Re-apply filters to update interactions
    applyFilters();

    showToast(`Focus Mode enabled: ${chains.length} chain(s), ${radius} Å radius`, 'success');
}

/**
 * Disable Focus Mode
 */
function disableFocusMode() {
    appState.focusMode.enabled = false;

    // Reset structure display
    updateStructureDisplay();

    // Re-apply filters
    applyFilters();

    showToast('Focus Mode disabled', 'info');
}

/**
 * Reset Focus Mode
 */
function resetFocusMode() {
    const chainsSelect = document.getElementById('focusChains');
    const radiusInput = document.getElementById('focusRadius');
    const toggleBtn = document.getElementById('focusToggleBtn');
    const resetBtn = document.getElementById('focusResetBtn');

    // Clear selections
    chainsSelect.selectedIndex = -1;
    radiusInput.value = '5.0';

    // Disable focus mode if enabled
    if (appState.focusMode.enabled) {
        disableFocusMode();
    }

    // Reset UI
    toggleBtn.textContent = 'Enable Focus';
    toggleBtn.classList.remove('btn-danger');
    toggleBtn.classList.add('btn-warning');
    resetBtn.disabled = true;
    chainsSelect.disabled = false;
    radiusInput.disabled = false;

    showToast('Focus Mode reset', 'info');
}

/**
 * Update structure display based on Focus Mode
 */
function updateStructureDisplay() {
    if (!appState.structureComponent) {
        console.warn('No structure component available');
        return;
    }

    console.log('Updating structure display. Focus mode:', appState.focusMode.enabled,
                'Chains:', appState.focusMode.centerChains,
                'Radius:', appState.focusMode.radius);

    // Remove existing representations
    appState.structureComponent.removeAllRepresentations();

    if (appState.focusMode.enabled && appState.focusMode.centerChains.length > 0) {
        // Build NGL selection for focus mode
        const chains = appState.focusMode.centerChains;
        const radius = appState.focusMode.radius;

        // Build selection for center chains
        const chainSelections = chains.map(c => `:${c}`).join(' or ');

        // Use NGL API to get atoms within radius
        const structure = appState.structureComponent.structure;
        const selection = new NGL.Selection(chainSelections);
        const atomSet = structure.getAtomSetWithinSelection(selection, radius);

        // Expand selection to include complete residues
        const atomSet2 = structure.getAtomSetWithinGroup(atomSet);

        // Convert to selection string
        const seleString = atomSet2.toSeleString();

        console.log('Center chains:', chainSelections);
        console.log('Atoms within', radius, 'Å:', seleString.substring(0, 200) + '...');

        // Add representation with the selection
        appState.structureComponent.addRepresentation(appState.representation, {
            sele: seleString,
            color: 'chainname'
        });

        showToast(`Showing pocket around chain(s): ${chains.join(', ')} (${radius} Å)`, 'info');
    } else {
        // Show all
        console.log('Showing all structure');
        appState.structureComponent.addRepresentation(appState.representation, {
            color: 'chainname'
        });
    }
}

/**
 * Calculate minimum distance from interaction to any center chain
 */
function getMinDistanceToCenterChains(interaction) {
    if (!appState.focusMode.enabled || appState.focusMode.centerChains.length === 0) {
        return 0;
    }

    const chains = appState.focusMode.centerChains;

    // Check if either residue is in a center chain
    const resAInCenter = chains.includes(interaction.res_a_chain);
    const resBInCenter = chains.includes(interaction.res_b_chain);

    // If either residue is in center chain, distance is 0
    if (resAInCenter || resBInCenter) {
        return 0;
    }

    // Otherwise, we need to calculate distance to center chains
    // Since we don't have all atom coordinates, we use a heuristic:
    // Return a large value to filter out, but in practice we should
    // rely on the structure display to show only nearby atoms
    return Infinity;
}

/**
 * Focus NGL view on a specific residue
 */
function focusResidue(chain, resNum) {
    if (!appState.stage) {
        return;
    }

    // Find the component with the structure
    let targetComponent = null;
    appState.stage.eachComponent(component => {
        if (component.structure) {
            targetComponent = component;
        }
    });

    if (!targetComponent) {
        return;
    }

    // Create selection string
    const selection = `${resNum}:${chain}`;

    // Add temporary highlight representation
    const highlightRepr = targetComponent.addRepresentation('ball+stick', {
        sele: selection,
        color: 'red',
        radius: 0.5
    });

    // Center view on selection
    targetComponent.autoView(selection, 1000);  // 1000ms animation duration

    // Remove highlight after 3 seconds
    setTimeout(() => {
        targetComponent.removeRepresentation(highlightRepr);
    }, 3000);

    showToast(`Focused on residue ${resNum} (Chain ${chain})`, 'info');
}

/**
 * Update statistics panel
 */
function updateStatistics(data) {
    const container = document.getElementById('statistics');

    let html = `
        <div class="stat-item">
            <span class="stat-label">Total Interactions</span>
            <span class="stat-value">${data.total_interactions}</span>
        </div>
    `;

    // Add by-type statistics
    if (data.by_type) {
        Object.entries(data.by_type)
            .sort((a, b) => b[1] - a[1])
            .forEach(([type, count]) => {
                html += `
                    <div class="stat-item">
                        <span class="stat-label">${formatTypeName(type)}</span>
                        <span class="stat-value">${count}</span>
                    </div>
                `;
            });
    }

    container.innerHTML = html;
}

/**
 * Apply filters and update display
 */
async function applyFilters() {
    // Collect filter values
    const typeCheckboxes = document.querySelectorAll('#typeFilters input[type="checkbox"]:checked');
    appState.filters.types = Array.from(typeCheckboxes).map(cb => cb.value);

    const residueSelect = document.getElementById('residueSelect');
    appState.filters.residues = Array.from(residueSelect.selectedOptions).map(opt => opt.value);

    appState.filters.distanceMin = document.getElementById('distMin').value || null;
    appState.filters.distanceMax = document.getElementById('distMax').value || null;

    // Build query parameters
    const params = new URLSearchParams();
    appState.filters.types.forEach(type => params.append('types', type));
    appState.filters.residues.forEach(res => params.append('residues', res));
    if (appState.filters.distanceMin) params.append('distance_min', appState.filters.distanceMin);
    if (appState.filters.distanceMax) params.append('distance_max', appState.filters.distanceMax);

    try {
        const response = await fetch('/api/interactions?' + params.toString());
        const data = await response.json();

        let interactions = data.interactions;

        // Apply Focus Mode filter if enabled
        if (appState.focusMode.enabled && appState.focusMode.centerChains.length > 0) {
            const chains = appState.focusMode.centerChains;
            interactions = interactions.filter(i => {
                // Keep interactions where either residue is in center chains
                return chains.includes(i.res_a_chain) || chains.includes(i.res_b_chain);
            });
        }

        appState.interactions = interactions;
        updateInteractionTable();
        updateInteractionDisplay();
        updateInteractionCount(interactions.length);

    } catch (error) {
        showToast('Error applying filters', 'danger');
    }
}

/**
 * Clear all filters
 */
function clearFilters() {
    // Uncheck all type filters (no types selected = no interactions displayed)
    document.querySelectorAll('#typeFilters input[type="checkbox"]').forEach(cb => {
        cb.checked = false;
    });

    // Clear residue selection
    document.getElementById('residueSelect').selectedIndex = -1;

    // Clear distance inputs
    document.getElementById('distMin').value = '';
    document.getElementById('distMax').value = '';

    // Apply cleared filters (will result in empty interactions)
    applyFilters();
}

/**
 * Update interaction table
 */
function updateInteractionTable() {
    const tbody = document.getElementById('interactionTableBody');
    tbody.innerHTML = '';

    appState.interactions.forEach(interaction => {
        const row = document.createElement('tr');
        row.dataset.id = interaction.id;

        if (appState.selectedInteractions.has(interaction.id)) {
            row.classList.add('selected');
        }

        row.innerHTML = `
            <td>${formatTypeName(interaction.type)}</td>
            <td>${interaction.res_a_name}:${interaction.res_a_chain}:${interaction.res_a_num}</td>
            <td>${interaction.res_b_name}:${interaction.res_b_chain}:${interaction.res_b_num}</td>
            <td>${interaction.distance.toFixed(2)}</td>
            <td>
                <button class="btn btn-sm btn-outline-primary" onclick="toggleInteractionSelection(${interaction.id})">
                    ${appState.selectedInteractions.has(interaction.id) ? 'Deselect' : 'Select'}
                </button>
            </td>
        `;

        row.addEventListener('click', function(e) {
            if (!e.target.closest('button')) {
                toggleInteractionSelection(interaction.id);
            }
        });

        tbody.appendChild(row);
    });
}

/**
 * Update interaction count display
 */
function updateInteractionCount(count) {
    document.getElementById('interactionCount').textContent = `${count} interactions`;
}

/**
 * Toggle interaction selection
 */
function toggleInteractionSelection(id) {
    if (appState.selectedInteractions.has(id)) {
        appState.selectedInteractions.delete(id);
    } else {
        appState.selectedInteractions.add(id);
    }

    updateInteractionTable();
    updateSelectedCount();
}

/**
 * Update selected count in create group modal
 */
function updateSelectedCount() {
    const count = appState.selectedInteractions.size;
    document.getElementById('selectedInteractionsCount').textContent =
        `${count} interaction${count !== 1 ? 's' : ''} selected`;
}

/**
 * Update 3D display of interactions
 */
function updateInteractionDisplay() {
    // Remove existing interaction shapes
    appState.interactionShapes.forEach(shape => {
        appState.stage.removeComponent(shape);
    });
    appState.interactionShapes = [];

    // Filter interactions with coordinates
    const interactionsWithCoords = appState.interactions.filter(i =>
        i.coords_a && i.coords_b
    );

    if (interactionsWithCoords.length === 0) {
        return;
    }

    // Clear previous interaction data
    appState.interactionData = {};

    // Add new interaction lines
    const shape = new NGL.Shape('interactions');

    interactionsWithCoords.forEach((interaction) => {
        const color = getInteractionColorNGL(interaction.type);

        // Check if this is a water bridge
        const isWaterBridge = interaction.type === 'water_bridge' || interaction.type === 'water_bridge_possible';

        if (isWaterBridge && interaction.coords_water && Array.isArray(interaction.coords_water) && interaction.coords_water.length === 3) {
            // Water bridge: draw two segments (A-Water and Water-B)
            const [x1, y1, z1] = interaction.coords_a;
            const [xw, yw, zw] = interaction.coords_water;
            const [x2, y2, z2] = interaction.coords_b;

            // Create unique names for each segment
            const seg1Name = `${interaction.type}_${interaction.id}_aw`;
            const seg2Name = `${interaction.type}_${interaction.id}_wb`;

            // Store interaction data for both segments
            appState.interactionData[seg1Name] = interaction;
            appState.interactionData[seg2Name] = interaction;

            // First segment: A to Water
            shape.addCylinder(
                [x1, y1, z1],  // position1
                [xw, yw, zw],  // position2
                color,          // color
                0.08,           // radius (slightly thinner)
                seg1Name        // name for picking
            );

            // Second segment: Water to B
            shape.addCylinder(
                [xw, yw, zw],  // position1
                [x2, y2, z2],  // position2
                color,          // color
                0.08,           // radius
                seg2Name        // name for picking
            );
        } else {
            // Regular interaction: single line
            const [x1, y1, z1] = interaction.coords_a;
            const [x2, y2, z2] = interaction.coords_b;

            // Create unique name for this cylinder
            const shapeName = `${interaction.type}_${interaction.id}`;

            // Store interaction data
            appState.interactionData[shapeName] = interaction;

            shape.addCylinder(
                [x1, y1, z1],  // position1
                [x2, y2, z2],  // position2
                color,          // color
                0.1,            // radius
                shapeName       // name for picking
            );
        }
    });

    const shapeComponent = appState.stage.addComponentFromObject(shape);
    shapeComponent.addRepresentation('buffer');
    appState.interactionShapes.push(shapeComponent);
}

/**
 * Select an interaction and display its details in the sidebar
 */
function selectInteraction(interaction) {
    appState.selectedInteraction = interaction;

    const detailsDiv = document.getElementById('selectedInteractionDetails');
    if (!detailsDiv) return;

    // Build detailed info HTML
    let html = `
        <div class="interaction-details">
            <h6 class="border-bottom pb-2 mb-3">${interaction.type.toUpperCase()} #${interaction.id}</h6>

            <div class="mb-3">
                <strong>Residue A:</strong><br>
                <span class="ms-2">${interaction.res_a_name} ${interaction.res_a_chain}:${interaction.res_a_num}</span><br>
                <span class="ms-2 text-muted">Atom: ${interaction.atom_a_name} (idx: ${interaction.atom_a_idx})</span>
            </div>

            <div class="mb-3">
                <strong>Residue B:</strong><br>
                <span class="ms-2">${interaction.res_b_name} ${interaction.res_b_chain}:${interaction.res_b_num}</span><br>
                <span class="ms-2 text-muted">Atom: ${interaction.atom_b_name} (idx: ${interaction.atom_b_idx})</span>
            </div>

            <div class="mb-3">
                <strong>Geometry:</strong><br>
                <span class="ms-2">Distance: ${interaction.distance.toFixed(2)} Å</span>
    `;

    if (interaction.angle !== undefined && interaction.angle !== null) {
        html += `<br><span class="ms-2">Angle: ${interaction.angle.toFixed(1)}°</span>`;
    }

    html += `</div>`;

    // Add water bridge specific info
    if (interaction.type === 'water_bridge' || interaction.type === 'water_bridge_possible') {
        html += `
            <div class="mb-3">
                <strong>Water Bridge Details:</strong>
        `;

        if (interaction.water_residue) {
            html += `<br><span class="ms-2">Water: ${interaction.water_residue}</span>`;
        }

        if (interaction.distance_aw !== undefined) {
            html += `<br><span class="ms-2">Acceptor-Water: ${interaction.distance_aw.toFixed(2)} Å</span>`;
        }

        if (interaction.distance_bw !== undefined) {
            html += `<br><span class="ms-2">Water-Donor: ${interaction.distance_bw.toFixed(2)} Å</span>`;
        }

        if (interaction.distance_dw !== undefined) {
            html += `<br><span class="ms-2">Donor-Water: ${interaction.distance_dw.toFixed(2)} Å</span>`;
        }

        if (interaction.d_angle !== undefined) {
            html += `<br><span class="ms-2">Donor Angle: ${interaction.d_angle.toFixed(1)}°</span>`;
        }

        if (interaction.w_angle !== undefined) {
            html += `<br><span class="ms-2">Water Angle: ${interaction.w_angle.toFixed(1)}°</span>`;
        }

        html += `</div>`;
    }

    // Add coordinates info (collapsible)
    html += `
        <div class="mt-3 pt-2 border-top">
            <small class="text-muted">
                <strong>Coordinates:</strong><br>
                Atom A: [${interaction.coords_a.map(c => c.toFixed(2)).join(', ')}]<br>
                Atom B: [${interaction.coords_b.map(c => c.toFixed(2)).join(', ')}]
    `;

    if (interaction.coords_water) {
        html += `<br>Water O: [${interaction.coords_water.map(c => c.toFixed(2)).join(', ')}]`;
    }

    html += `
            </small>
        </div>
    </div>
    `;

    detailsDiv.innerHTML = html;

    // Highlight the interaction in the table
    highlightInteractionInTable(interaction);
}

/**
 * Highlight the selected interaction in the table
 */
function highlightInteractionInTable(interaction) {
    // Remove previous highlights
    document.querySelectorAll('#interactionTableBody tr').forEach(row => {
        row.classList.remove('table-primary');
    });

    // Find and highlight the row
    const rows = document.querySelectorAll('#interactionTableBody tr');
    rows.forEach(row => {
        const rowId = parseInt(row.dataset.id);
        if (rowId === interaction.id) {
            row.classList.add('table-primary');
            row.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
    });
}

/**
 * Set molecular representation
 */
function setRepresentation(representation) {
    appState.representation = representation;

    // Use updateStructureDisplay which handles Focus Mode
    updateStructureDisplay();
}

/**
 * Create new group (show modal)
 */
function createGroup() {
    updateSelectedCount();
    const modal = new bootstrap.Modal(document.getElementById('createGroupModal'));
    modal.show();
}

/**
 * Confirm group creation
 */
async function confirmCreateGroup() {
    const name = document.getElementById('newGroupName').value.trim();

    if (!name) {
        showToast('Please enter a group name', 'warning');
        return;
    }

    if (appState.selectedInteractions.size === 0) {
        showToast('Please select at least one interaction', 'warning');
        return;
    }

    try {
        const response = await fetch('/api/groups', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                name: name,
                interaction_ids: Array.from(appState.selectedInteractions)
            })
        });

        const data = await response.json();

        if (response.ok) {
            showToast(`Group "${name}" created`, 'success');
            bootstrap.Modal.getInstance(document.getElementById('createGroupModal')).hide();
            document.getElementById('newGroupName').value = '';

            // Clear selection after creating group
            appState.selectedInteractions.clear();
            updateInteractionTable();
            updateSelectedCount();

            await loadGroups();
        } else {
            showToast(data.error || 'Error creating group', 'danger');
        }
    } catch (error) {
        showToast('Error creating group', 'danger');
    }
}

/**
 * Load groups from server
 */
async function loadGroups() {
    try {
        const response = await fetch('/api/groups');
        const data = await response.json();

        appState.groups = data.groups;
        updateGroupsList();
    } catch (error) {
        showToast('Error loading groups', 'warning');
    }
}

/**
 * Update groups list display
 */
function updateGroupsList() {
    const container = document.getElementById('groupsList');
    container.innerHTML = '';

    if (Object.keys(appState.groups).length === 0) {
        container.innerHTML = '<p class="text-muted text-center">No groups created</p>';
        return;
    }

    Object.entries(appState.groups).forEach(([name, group]) => {
        const div = document.createElement('div');
        div.className = 'group-item';
        div.innerHTML = `
            <div class="group-header">
                <span class="group-name" onclick="toggleGroupVisibility('${name}')">
                    ${group.visible ? '▼' : '▶'} ${name}
                </span>
                <span class="group-count">${group.interaction_ids.length}</span>
            </div>
            <div class="group-actions">
                <button class="btn btn-sm btn-outline-primary" onclick="showGroup('${name}')">Show</button>
                <button class="btn btn-sm btn-outline-danger" onclick="deleteGroup('${name}')">Delete</button>
            </div>
        `;
        container.appendChild(div);
    });
}

/**
 * Toggle group visibility
 */
async function toggleGroupVisibility(name) {
    const group = appState.groups[name];
    if (!group) return;

    const newVisibility = !group.visible;

    try {
        await fetch(`/api/groups/${name}/visibility`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ visible: newVisibility })
        });

        group.visible = newVisibility;
        updateGroupsList();
    } catch (error) {
        showToast('Error toggling group visibility', 'danger');
    }
}

/**
 * Show group interactions
 */
async function showGroup(name) {
    const group = appState.groups[name];
    if (!group) {
        showToast(`Group "${name}" not found`, 'warning');
        return;
    }

    // Clear current filters and show only group interactions
    // First, get all interactions
    const response = await fetch('/api/interactions');
    const data = await response.json();

    // Filter to show only group interactions
    appState.interactions = data.interactions.filter(i =>
        group.interaction_ids.includes(i.id)
    );

    // Update display
    updateInteractionTable();
    updateInteractionDisplay();
    updateInteractionCount(appState.interactions.length);

    // Also select them
    appState.selectedInteractions = new Set(group.interaction_ids);
    updateInteractionTable();
    updateSelectedCount();

    showToast(`Showing ${appState.interactions.length} interactions from "${name}"`, 'info');
}

/**
 * Delete group
 */
async function deleteGroup(name) {
    if (!confirm(`Delete group "${name}"?`)) return;

    try {
        const response = await fetch(`/api/groups/${name}`, {
            method: 'DELETE'
        });

        if (response.ok) {
            showToast(`Group "${name}" deleted`, 'success');
            await loadGroups();
        }
    } catch (error) {
        showToast('Error deleting group', 'danger');
    }
}

/**
 * Export to PyMOL
 */
async function exportPyMOL() {
    const interactionIds = appState.interactions.map(i => i.id);

    if (interactionIds.length === 0) {
        showToast('No interactions to export', 'warning');
        return;
    }

    try {
        const response = await fetch('/api/export/pymol', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ interaction_ids: interactionIds })
        });

        if (response.ok) {
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'interactions.py';
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);

            showToast('PyMOL script exported', 'success');
        }
    } catch (error) {
        showToast('Error exporting PyMOL script', 'danger');
    }
}

/**
 * Show help modal
 */
function showHelp() {
    const modal = new bootstrap.Modal(document.getElementById('helpModal'));
    modal.show();
}

/**
 * Show toast notification
 */
function showToast(message, type = 'info') {
    const toastContainer = document.querySelector('.toast-container') || createToastContainer();

    const toast = document.createElement('div');
    toast.className = `toast align-items-center text-white bg-${type} border-0`;
    toast.setAttribute('role', 'alert');
    toast.innerHTML = `
        <div class="d-flex">
            <div class="toast-body">${message}</div>
            <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
        </div>
    `;

    toastContainer.appendChild(toast);

    const bsToast = new bootstrap.Toast(toast, { delay: 3000 });
    bsToast.show();

    toast.addEventListener('hidden.bs.toast', () => {
        toast.remove();
    });
}

/**
 * Create toast container
 */
function createToastContainer() {
    const container = document.createElement('div');
    container.className = 'toast-container';
    document.body.appendChild(container);
    return container;
}

/**
 * Format interaction type name for display
 */
function formatTypeName(type) {
    return type
        .split('_')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ');
}

// ============================================
// Conformation Management (for Docking Data)
// ============================================

/**
 * Load conformation information from API
 */
async function loadConformationInfo() {
    try {
        const response = await fetch('/api/conformations');
        const data = await response.json();

        if (data.is_docking) {
            // Show conformation selector
            const selectorDiv = document.getElementById('conformationSelector');
            const infoDiv = document.getElementById('conformationInfo');

            if (selectorDiv) selectorDiv.style.display = 'block';
            if (infoDiv) infoDiv.style.display = 'block';

            // Populate conformation select
            populateConformationSelect(data.conformations, data.current);

            // Update conformation info display
            updateConformationInfo(data.current);

            showToast(`Loaded docking data with ${data.conformations.length} conformations`, 'info');
        } else {
            // Hide conformation selector for standard data
            const selectorDiv = document.getElementById('conformationSelector');
            const infoDiv = document.getElementById('conformationInfo');

            if (selectorDiv) selectorDiv.style.display = 'none';
            if (infoDiv) infoDiv.style.display = 'none';
        }
    } catch (error) {
        console.error('Error loading conformation info:', error);
    }
}

/**
 * Populate conformation select dropdown
 */
function populateConformationSelect(conformations, current) {
    const select = document.getElementById('conformationSelect');
    if (!select) return;

    select.innerHTML = '';

    conformations.forEach(conf => {
        const option = document.createElement('option');
        option.value = conf.index;

        const vina = conf.vina_result;
        if (vina) {
            option.text = `Model ${conf.model_num} (Affinity: ${vina.affinity.toFixed(2)})`;
        } else {
            option.text = `Model ${conf.model_num}`;
        }

        if (conf.index === current.index) {
            option.selected = true;
        }

        select.appendChild(option);
    });
}

/**
 * Update conformation info display
 */
function updateConformationInfo(current) {
    const infoDiv = document.getElementById('conformationInfo');
    if (!infoDiv || !current) return;

    const vina = current.vina_result;
    if (vina) {
        infoDiv.innerHTML = `
            <span class="badge bg-info">Affinity: ${vina.affinity.toFixed(3)} kcal/mol</span>
            ${vina.rmsd_lb > 0 ? `<span class="badge bg-secondary ms-1">RMSD: ${vina.rmsd_lb.toFixed(2)}</span>` : ''}
        `;
    } else {
        infoDiv.innerHTML = `<span class="badge bg-secondary">Model ${current.model_num}</span>`;
    }
}

/**
 * Switch to a different conformation
 */
async function switchConformation(index) {
    const idx = parseInt(index);
    if (isNaN(idx)) return;

    showToast(`Switching to conformation ${idx + 1}...`, 'info');

    try {
        const response = await fetch(`/api/conformations/${idx}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });

        const data = await response.json();

        if (response.ok) {
            // Update conformation info display
            updateConformationInfo(data.current);

            // Reload all data for the new conformation
            await reloadConformationData();

            showToast(`Switched to conformation ${data.current.model_num}`, 'success');
        } else {
            showToast(data.error || 'Error switching conformation', 'danger');
        }
    } catch (error) {
        console.error('Error switching conformation:', error);
        showToast('Error switching conformation', 'danger');
    }
}

/**
 * Reload data after conformation switch
 */
async function reloadConformationData() {
    try {
        // Clear current interactions
        appState.interactions = [];
        appState.selectedInteractions.clear();
        appState.groups = {};

        // Reload interaction types
        const typesResponse = await fetch('/api/interactions/types');
        const typesData = await typesResponse.json();
        populateTypeFilters(typesData.types);

        // Reload residues
        const residuesResponse = await fetch('/api/residues');
        const residuesData = await residuesResponse.json();
        populateResidueSelect(residuesData.residues);

        // Reload statistics
        const statsResponse = await fetch('/api/interactions/summary');
        const statsData = await statsResponse.json();
        updateStatistics(statsData);

        // Clear interaction display
        updateInteractionTable();
        updateInteractionDisplay();
        updateInteractionCount(0);
        updateGroupsList();

        // Reload PDB structure (if coordinates changed)
        await reloadPDBStructure();

    } catch (error) {
        console.error('Error reloading conformation data:', error);
        showToast('Error reloading data', 'danger');
    }
}

/**
 * Reload PDB structure
 */
async function reloadPDBStructure() {
    try {
        // Remove existing structure component
        if (appState.structureComponent) {
            appState.stage.removeComponent(appState.structureComponent);
            appState.structureComponent = null;
        }

        // Load new structure
        await loadPDBStructure();
    } catch (error) {
        console.error('Error reloading PDB structure:', error);
    }
}

/**
 * Export current conformation as PDB file
 */
async function exportConformationPDB() {
    try {
        const response = await fetch('/api/export/conformation', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });

        if (response.ok) {
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;

            // Get current conformation info for filename
            const infoResponse = await fetch('/api/conformations');
            const infoData = await infoResponse.json();
            const currentModel = infoData.current ? infoData.current.model_num : 'unknown';
            a.download = `conformation_${currentModel}.pdb`;

            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);

            showToast('Conformation exported as PDB', 'success');
        } else {
            showToast('Error exporting conformation', 'danger');
        }
    } catch (error) {
        console.error('Error exporting conformation:', error);
        showToast('Error exporting conformation', 'danger');
    }
}

// Modify loadInitialData to also load conformation info
const originalLoadInitialData = loadInitialData;
loadInitialData = async function() {
    await originalLoadInitialData();
    await loadConformationInfo();
};
