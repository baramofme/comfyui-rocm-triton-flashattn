import { app } from "../../../scripts/app.js";

app.registerExtension({
  name: "LTX25Resolution.UI",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "LTX25Controls" && nodeData.name !== "LTX25Resolution") return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated?.apply(this, arguments);
      const arW = this.widgets?.find((w) => w.name === "aspect_ratio");
      const cwW = this.widgets?.find((w) => w.name === "custom_width");
      const chW = this.widgets?.find((w) => w.name === "custom_height");

      const applyCustomState = () => {
        if (!arW || (!cwW && !chW)) return;
        const enabled = arW.value === "Custom (manual)";
        for (const w of [cwW, chW]) {
          if (!w) continue;
          w.disabled = !enabled;
          if (w.computeSize) w.computeSize();
        }
        if (this.setDirtyCanvas) this.setDirtyCanvas(true, true);
      };

      // LTX25Controls only: "Duration drives frames" -> manual_frames useless,
      // "Manual exact frames" -> duration_seconds ignored (server: "duration field ignored")
      const tmW = this.widgets?.find((w) => w.name === "timing_mode");
      const mfW = this.widgets?.find((w) => w.name === "manual_frames");
      const durW = this.widgets?.find((w) => w.name === "duration_seconds");

      const applyTimingState = () => {
        if (!tmW) return;
        const isDuration = tmW.value === "Duration drives frames";
        if (mfW) {
          mfW.disabled = isDuration;
          if (mfW.computeSize) mfW.computeSize();
        }
        if (durW) {
          durW.disabled = !isDuration;
          if (durW.computeSize) durW.computeSize();
        }
        if (this.setDirtyCanvas) this.setDirtyCanvas(true, true);
      };

      if (arW) arW.callback = applyCustomState;
      if (tmW) tmW.callback = applyTimingState;
      const onConfigure = this.onConfigure;
      this.onConfigure = function () {
        const r2 = onConfigure?.apply(this, arguments);
        applyCustomState();
        applyTimingState();
        return r2;
      };
      applyCustomState();
      applyTimingState();
      return r;
    };
  },
});