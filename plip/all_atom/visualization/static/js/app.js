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
    interactionShapes: []           // NGL shape components for interactions
};

// Color scheme matching main PLIP
const INTERACTION_COLORS = {
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
        console.log('Loading PDB structure...');
        const response = await fetch('/api/pdb');
        const data = await response.json();
        console.log('PDB info:', data);

        if (data.has_pdb_content) {
            // Load PDB content from our API
            console.log('Loading PDB content from API...');
            const pdbResponse = await fetch('/api/pdb/content');
            const pdbContent = await pdbResponse.text();
            console.log(`Loaded PDB content: ${pdbContent.length} characters`);

            // Create a Blob from the PDB content
            const blob = new Blob([pdbContent], { type: 'text/plain' });
            const url = URL.createObjectURL(blob);

            // Load into NGL
            appState.stage.loadFile(url, { ext: 'pdb' }).then(function(component) {
                console.log('PDB loaded successfully');
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
            console.log(`Trying to load from RCSB: ${pdbId}`);
            if (pdbId.length === 4) {
                appState.stage.loadFile('rcsb://' + pdbId).then(function(component) {
                    console.log('PDB loaded from RCSB');
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
                console.warn('No valid PDB ID found');
                showToast('No PDB structure available', 'warning');
            }
        } else {
            console.warn('No PDB file available');
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
        const color = INTERACTION_COLORS[type] || [200, 200, 200];
        const colorStr = `rgb(${color[0]}, ${color[1]}, ${color[2]})`;

        const div = document.createElement('div');
        div.className = 'type-filter-item';
        div.innerHTML = `
            <input type="checkbox" id="type_${type}" value="${type}" checked>
            <span class="type-color-indicator" style="background-color: ${colorStr};"></span>
            <label for="type_${type}">${formatTypeName(type)}</label>
        `;
        container.appendChild(div);
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
}

/**
 * Focus NGL view on a specific residue
 */
function focusResidue(chain, resNum) {
    console.log(`Focusing on residue: chain ${chain}, number ${resNum}`);

    if (!appState.stage) {
        console.warn('NGL stage not initialized');
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
        console.warn('No structure component found');
        return;
    }

    // Create selection string
    const selection = `${resNum}:${chain}`;
    console.log(`NGL selection: ${selection}`);

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

        appState.interactions = data.interactions;
        updateInteractionTable();
        updateInteractionDisplay();
        updateInteractionCount(data.count);

    } catch (error) {
        console.error('Error applying filters:', error);
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
    console.log(`Updating interaction display: ${appState.interactions.length} interactions`);

    // Remove existing interaction shapes
    appState.interactionShapes.forEach(shape => {
        appState.stage.removeComponent(shape);
    });
    appState.interactionShapes = [];

    // Filter interactions with coordinates
    const interactionsWithCoords = appState.interactions.filter(i =>
        i.coords_a && i.coords_b
    );

    console.log(`Interactions with coordinates: ${interactionsWithCoords.length}`);

    if (interactionsWithCoords.length === 0) {
        console.warn('No interactions with coordinates to display');
        return;
    }

    // Add new interaction lines
    const shape = new NGL.Shape('interactions');

    interactionsWithCoords.forEach((interaction) => {
        const color = INTERACTION_COLORS[interaction.type] || [255, 255, 255];

        // Check if this is a water bridge
        const isWaterBridge = interaction.type === 'water_bridge' || interaction.type === 'water_bridge_possible';

        if (isWaterBridge && interaction.coords_water && Array.isArray(interaction.coords_water) && interaction.coords_water.length === 3) {
            // Water bridge: draw two segments (A-Water and Water-B)
            const [x1, y1, z1] = interaction.coords_a;
            const [xw, yw, zw] = interaction.coords_water;
            const [x2, y2, z2] = interaction.coords_b;

            // First segment: A to Water
            shape.addCylinder(
                [x1, y1, z1],  // position1
                [xw, yw, zw],  // position2
                color,          // color
                0.08,           // radius (slightly thinner)
            );

            // Second segment: Water to B
            shape.addCylinder(
                [xw, yw, zw],  // position1
                [x2, y2, z2],  // position2
                color,          // color
                0.08,           // radius
            );
        } else {
            // Regular interaction: single line
            const [x1, y1, z1] = interaction.coords_a;
            const [x2, y2, z2] = interaction.coords_b;

            shape.addCylinder(
                [x1, y1, z1],  // position1
                [x2, y2, z2],  // position2
                color,          // color
                0.1,            // radius
            );
        }
    });

    const shapeComponent = appState.stage.addComponentFromObject(shape);
    shapeComponent.addRepresentation('buffer');
    appState.interactionShapes.push(shapeComponent);

    console.log(`Created shape component with ${interactionsWithCoords.length} interactions`);
}

/**
 * Set molecular representation
 */
function setRepresentation(representation) {
    appState.representation = representation;

    // Update all components
    appState.stage.eachComponent(component => {
        if (component.structure) {
            component.removeAllRepresentations();
            component.addRepresentation(representation, {
                color: 'chainname'
            });
        }
    });
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
        console.error('Error creating group:', error);
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
        console.error('Error loading groups:', error);
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
        console.error('Error toggling group visibility:', error);
    }
}

/**
 * Show group interactions
 */
async function showGroup(name) {
    const group = appState.groups[name];
    if (!group) {
        console.error(`Group "${name}" not found`);
        return;
    }

    console.log(`Showing group "${name}" with ${group.interaction_ids.length} interactions`);

    // Clear current filters and show only group interactions
    // First, get all interactions
    const response = await fetch('/api/interactions');
    const data = await response.json();

    // Filter to show only group interactions
    appState.interactions = data.interactions.filter(i =>
        group.interaction_ids.includes(i.id)
    );

    console.log(`Filtered to ${appState.interactions.length} interactions from group`);

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
        console.error('Error deleting group:', error);
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
        console.error('Error exporting PyMOL:', error);
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
