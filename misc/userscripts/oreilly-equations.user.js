// ==UserScript==
// @name         O'Reilly readable equations
// @namespace    leet-gpu
// @version      2.0.3
// @description  Render hidden MathML as crisp SVG equations.
// @match        https://learning.oreilly.com/library/view/*
// @run-at       document-idle
// @inject-into  content
// @require      https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/mml-svg.js
// @grant        GM_getValue
// @grant        GM_setValue
// @grant        GM_registerMenuCommand
// ==/UserScript==

(async () => {
  "use strict";

  const namespace = "http://www.w3.org/1998/Math/MathML";
  const defaults = {display : 1.35, inline : 1};
  const scales = {};
  const rendered = new WeakMap();
  const style = document.createElement("style");
  style.textContent = `
    #sbo-rt-content .oreilly-svg-equation {
      display: inline-block;
      max-width: 100%;
      overflow: visible;
      font-size: calc(1em * var(--oreilly-inline-scale));
      vertical-align: baseline;
    }
    #sbo-rt-content .oreilly-svg-equation mjx-container {
      margin: 0 !important;
    }
    #sbo-rt-content .oreilly-svg-equation svg {
      max-width: none !important;
    }
    #sbo-rt-content .oreilly-svg-equation-display {
      display: block;
      font-size: calc(1em * var(--oreilly-display-scale));
      text-align: center;
      overflow-x: auto;
      padding: 0.5em 0;
    }
    #sbo-rt-content img.oreilly-svg-original {
      display: none !important;
    }
  `;
  document.head.append(style);

  function setScale(kind, value) {
    scales[kind] = value;
    document.documentElement.style.setProperty(
      `--oreilly-${kind}-scale`,
      value,
    );
  }

  for (const [kind, fallback] of Object.entries(defaults)) {
    const stored = Number(GM_getValue(`svg${kind}Scale`, fallback));
    setScale(
      kind,
      Number.isFinite(stored) && stored >= 0.5 && stored <= 5 ? stored : fallback,
    );
    GM_registerMenuCommand(`Set ${kind} equation scale…`, () => {
      const answer = window.prompt(
        `${kind} equation scale (0.5–5):`,
        scales[kind],
      );
      if (answer === null)
        return;
      const value = Number(answer);
      if (!Number.isFinite(value) || value < 0.5 || value > 5) {
        window.alert("Enter a number from 0.5 to 5.");
        return;
      }
      GM_setValue(`svg${kind}Scale`, value);
      setScale(kind, value);
    });
  }

  async function loadRenderer() {
    // @require shares the userscript's isolated context.
    const mathjax = window.MathJax;
    await mathjax?.startup?.promise;
    if (typeof mathjax?.mathml2svgPromise !== "function") {
      throw new Error("MathJax SVG renderer unavailable.");
    }
    return mathjax;
  }

  function findMath(source) {
    return [...source.children ].find(
      (node) => node.localName.split(":").pop() === "math",
    );
  }

  function normalizeMath(node, repairFences) {
    if (node.nodeType === Node.TEXT_NODE)
      return document.createTextNode(node.textContent);
    if (node.nodeType !== Node.ELEMENT_NODE)
      return null;
    const name = node.localName.split(":").pop();
    const copy = document.createElementNS(namespace, name);
    for (const attribute of node.attributes) {
      if (!attribute.name.startsWith("xmlns"))
        copy.setAttribute(attribute.name, attribute.value);
    }
    // Verified Chapter 8 assets omit opening fences.
    if (repairFences && name === "mfenced" && copy.getAttribute("open") === "") {
      const opening = {")" : "(", "]" : "[", "}" : "{"}[copy.getAttribute("close")];
      if (opening)
        copy.setAttribute("open", opening);
    }
    for (const child of node.childNodes) {
      const normalized = normalizeMath(child, repairFences);
      if (normalized)
        copy.append(normalized);
    }
    return copy;
  }

  let renderer;
  try {
    renderer = await loadRenderer();
  } catch (error) {
    console.warn("O'Reilly SVG equations:", error);
    return;
  }

  async function renderEquations() {
    const reader = document.getElementById("sbo-rt-content");
    if (!reader)
      return;
    for (const wrapper of reader.querySelectorAll(".oreilly-svg-equation")) {
      const image = wrapper.previousElementSibling;
      const state = image && rendered.get(image);
      if (state?.wrapper === wrapper && state.source.matches(".hiddenClass") &&
          findMath(state.source) && state.source.nextElementSibling === image.parentElement)
        continue;
      wrapper.remove();
      image?.classList.remove("oreilly-svg-original");
    }
    for (const source of reader.querySelectorAll(".hiddenClass")) {
      const target = source.nextElementSibling;
      const image = target?.querySelector("img");
      const math = findMath(source);
      if (!math || !image)
        continue;
      const display = target.classList.contains("showClass");
      const original = math.outerHTML;
      const imageSource = image.src;
      const key = JSON.stringify([ original, imageSource, display ]);
      const previous = rendered.get(image);
      if (previous?.key === key && (!previous.wrapper || previous.wrapper.parentElement === target))
        continue;
      previous?.wrapper?.remove();
      image.classList.remove("oreilly-svg-original");
      const repairFences =
        /\/urn:orm:book:9780323984638\/files\/images\/F000173si\d+\.png(?:[?#]|$)/.test(
          imageSource,
        );
      const normalized = normalizeMath(math, repairFences);
      normalized.setAttribute("display", display ? "block" : "inline");
      const isCurrent = () =>
        image.isConnected && reader.contains(image) && source.matches(".hiddenClass") &&
        target.classList.contains("showClass") === display && image.parentElement === target &&
        source.nextElementSibling === target && source.contains(math) &&
        math.outerHTML === original && image.src === imageSource;
      try {
        const output = await renderer.mathml2svgPromise(
          new XMLSerializer().serializeToString(normalized),
          {display},
        );
        if (!output.querySelector("svg") || output.querySelector('[data-mml-node="merror"]')) {
          throw new Error("Equation conversion failed.");
        }
        if (!isCurrent())
          continue;
        const wrapper = document.createElement("span");
        wrapper.className = "oreilly-svg-equation";
        if (display)
          wrapper.classList.add("oreilly-svg-equation-display");
        wrapper.append(output);
        image.after(wrapper);
        image.classList.add("oreilly-svg-original");
        rendered.set(image, {key, wrapper, source});
      } catch (error) {
        if (!isCurrent())
          continue;
        rendered.set(image, {key, source});
        console.warn("O'Reilly SVG equation:", image.src, error);
      }
    }
  }

  let pending = false;
  let running = false;
  async function scheduleRender() {
    pending = true;
    if (running)
      return;
    running = true;
    try {
      while (pending) {
        pending = false;
        await renderEquations();
      }
    } finally {
      running = false;
    }
  }
  function needsRender(record) {
    const reader = document.getElementById("sbo-rt-content");
    const target =
      record.target.nodeType === Node.ELEMENT_NODE ? record.target : record.target.parentElement;
    if (target?.closest(".oreilly-svg-equation"))
      return false;
    if (record.type !== "childList") {
      return (
        reader?.contains(target) &&
        (target.closest(".hiddenClass") ||
         (record.attributeName === "class" &&
          (target.previousElementSibling?.matches(".hiddenClass") || findMath(target))) ||
         (record.attributeName === "src" && target.tagName === "IMG"))
      );
    }
    return [...record.addedNodes, ...record.removedNodes ].some((node) => {
      if (node.nodeType !== Node.ELEMENT_NODE)
        return reader?.contains(target) && target?.closest(".hiddenClass");
      if (node.classList.contains("oreilly-svg-equation"))
        return !node.isConnected && reader?.contains(target);
      return (
        reader?.contains(target) || node.id === "sbo-rt-content" ||
        !!node.querySelector("#sbo-rt-content")
      );
    });
  }
  new MutationObserver((records) => {
    if (records.some(needsRender))
      void scheduleRender();
  }).observe(document.body, {
    childList : true,
    subtree : true,
    attributes : true,
    characterData : true,
  });
  await scheduleRender();
})();
