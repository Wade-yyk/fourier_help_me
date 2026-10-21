document.querySelectorAll("[data-hybrid-viewer]").forEach((viewer) => {
  const slider = viewer.querySelector('input[type="range"]');
  const image = viewer.querySelector("img");
  const output = viewer.querySelector("output");
  const stage = viewer.querySelector(".hybrid-stage");

  const update = () => {
    const size = Number(slider.value);
    image.style.width = `${size}px`;
    output.value = `${size} px`;
    slider.setAttribute("aria-valuetext", `${size} pixels wide`);
  };

  const fit = () => {
    const maximum = Math.max(Number(slider.min), Math.min(384, Math.floor(stage.clientWidth)));
    slider.max = String(maximum);
    slider.value = String(Math.min(Number(slider.value), maximum));
    update();
  };

  slider.addEventListener("input", update);
  new ResizeObserver(fit).observe(stage);
  fit();
});
