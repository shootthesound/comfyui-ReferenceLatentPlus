// Reference Latent+ — progressive UI:
// - image_N + mask_N slots reveal as the previous slot gets connected
// - per-image collapsible mask-config sections appear only when their image is connected
// - disconnecting an earlier slot never removes a slot that has an active link
//
// We register against both the new class name and the legacy "Klein" name so
// saved workflows from before the rename keep working.

import { app } from "/scripts/app.js";

const NODE_CLASSES = ["ReferenceLatentPlus", "KleinReferenceLatentPlus"];

const GROUPS = [
    { idx: 1, prefix: "image1_", label: "Image 1 mask + timestep" },
    { idx: 2, prefix: "image2_", label: "Image 2 mask + timestep" },
    { idx: 3, prefix: "image3_", label: "Image 3 mask + timestep" },
    { idx: 4, prefix: "image4_", label: "Image 4 mask + timestep" },
];

const HIDDEN_SIZE = [0, -4];

// --- widget visibility ----------------------------------------------------

function setWidgetHidden(widget, hidden) {
    widget.hidden = hidden;
    if (hidden) {
        widget.computeSize = () => HIDDEN_SIZE;
    } else {
        delete widget.computeSize;
    }
}

function updateGroupVisibility(node, group) {
    // Hide the entire group (header + body widgets) if its image isn't connected.
    const imgConnected = isSlotConnected(node, `image_${group.idx}`);
    const collapsed = node._reflatentplus_collapsed[group.prefix];

    if (group.headerWidget) {
        setWidgetHidden(group.headerWidget, !imgConnected);
        group.headerWidget.name = (collapsed ? "▶ " : "▼ ") + group.label;
    }
    for (const w of group.widgets) {
        // Body widgets hidden if image disconnected OR group collapsed.
        setWidgetHidden(w, !imgConnected || collapsed);
    }
}

// --- slot management (progressive disclosure) -----------------------------

function isSlotConnected(node, slotName) {
    const slot = node.inputs?.find((s) => s.name === slotName);
    return !!(slot && slot.link != null);
}

function highestConnectedImage(node) {
    let highest = 0;
    for (let i = 1; i <= 4; i++) {
        if (isSlotConnected(node, `image_${i}`)) {
            highest = i;
        }
    }
    return highest;
}

function ensureSlotPresent(node, slotName, slotType, beforeName) {
    if (node.inputs?.some((s) => s.name === slotName)) {
        return;  // already present
    }
    // addInput appends to end; we then splice to the desired position.
    node.addInput(slotName, slotType);
    if (!beforeName) return;
    const newIdx = node.inputs.length - 1;
    const beforeIdx = node.inputs.findIndex((s) => s.name === beforeName);
    if (beforeIdx >= 0 && beforeIdx < newIdx) {
        const moved = node.inputs.splice(newIdx, 1)[0];
        node.inputs.splice(beforeIdx, 0, moved);
    }
}

function ensureSlotAbsentIfUnconnected(node, slotName) {
    const idx = node.inputs?.findIndex((s) => s.name === slotName);
    if (idx == null || idx < 0) return;
    const slot = node.inputs[idx];
    if (slot.link != null) return;  // never remove a slot with an active wire
    node.removeInput(idx);
}

function updateSlots(node) {
    // Target = (highest connected image) + 1, clamped to [1, 4].
    // Ensures one "next available" slot is always shown.
    const highest = highestConnectedImage(node);
    const target = Math.max(1, Math.min(4, highest + 1));

    // Add slots up to the target, in order.
    for (let i = 2; i <= target; i++) {
        ensureSlotPresent(node, `image_${i}`, "IMAGE", null);
    }
    // mask_N slots track image_N visibility (mask_1 always visible since image_1
    // is required; the optional masks come and go with their images).
    for (let i = 1; i <= 4; i++) {
        const need = i === 1 || i <= target;
        if (need) {
            ensureSlotPresent(node, `mask_${i}`, "MASK", null);
        }
    }

    // Remove slots beyond the target IF they have no active connection.
    // (Disconnecting an earlier slot won't yank a later connected one.)
    for (let i = target + 1; i <= 4; i++) {
        ensureSlotAbsentIfUnconnected(node, `image_${i}`);
        ensureSlotAbsentIfUnconnected(node, `mask_${i}`);
    }
}

// --- main extension -------------------------------------------------------

function refreshUI(node) {
    updateSlots(node);
    for (const g of node._reflatentplus_groups || []) {
        updateGroupVisibility(node, g);
    }
    node.size = node.computeSize();
    node.setDirtyCanvas(true, true);
}

app.registerExtension({
    name: "RefLatentPlus.ProgressiveUI",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!NODE_CLASSES.includes(nodeData.name)) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);

            // Find existing widgets that belong to each group.
            const groups = GROUPS.map((g) => ({
                ...g,
                widgets: this.widgets.filter((w) => w.name.startsWith(g.prefix)),
                headerWidget: null,
            }));
            this._reflatentplus_groups = groups;

            // Per-node collapsed state (default collapsed).
            this._reflatentplus_collapsed = this._reflatentplus_collapsed || {};
            for (const g of groups) {
                if (!(g.prefix in this._reflatentplus_collapsed)) {
                    this._reflatentplus_collapsed[g.prefix] = true;
                }
            }

            // Add a header button per group, then splice it into position right
            // before the group's first widget.
            for (const g of groups) {
                const btn = this.addWidget("button", "▶ " + g.label, null, () => {
                    this._reflatentplus_collapsed[g.prefix] =
                        !this._reflatentplus_collapsed[g.prefix];
                    refreshUI(this);
                });
                btn.serialize = false;
                g.headerWidget = btn;

                if (g.widgets.length > 0) {
                    const firstIdx = this.widgets.indexOf(g.widgets[0]);
                    const btnIdx = this.widgets.indexOf(btn);
                    if (firstIdx >= 0 && btnIdx >= 0 && btnIdx !== firstIdx) {
                        this.widgets.splice(btnIdx, 1);
                        this.widgets.splice(firstIdx, 0, btn);
                    }
                }
            }

            refreshUI(this);
            return result;
        };

        // Hook connection changes to refresh slots + group visibility.
        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function (type, slotIndex, isConnected, link, slotInfo) {
            const result = onConnectionsChange?.apply(this, arguments);
            // Defer to next frame so node.inputs has the updated link state.
            setTimeout(() => refreshUI(this), 0);
            return result;
        };
    },
});
