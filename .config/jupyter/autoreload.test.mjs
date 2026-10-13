import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";

const source = fs.readFileSync(".config/jupyter/autoreload.js", "utf8")
                 .replace(/^import .*;$/gm, "")
                 .replace("export default plugin;", "globalThis.plugin = plugin;");
let tick;
let onSave;
let resolveMetadata;
let reads = 0;
let reverts = 0;
const nb = {
  cells : [ {cell_type : "markdown", source : "original"} ]
};
const context = {
  path : "notebook.md",
  ready : Promise.resolve(),
  contentsModel : {last_modified : "original", writable : true},
  model : {toJSON : () => nb, fromJSON : () => {}, readOnly : false},
  sessionContext : {kernelDisplayStatus : "idle"},
  saveState : {
    connect : (callback) => {
      onSave = callback;
    },
  },
  revert : async () => {
    reverts += 1;
    context.contentsModel.last_modified = "external";
  },
};
const panel = {
  context,
  isDisposed : false,
  content : {activeCellIndex : 0, node : {scrollTop : 0}, widgets : [ 0 ]},
};
const app = {
  restored : Promise.resolve(),
  serviceManager : {
    contents : {
      get : () => {
        reads += 1;
        return new Promise((resolve) => {
          resolveMetadata = resolve;
        });
      },
    },
  },
};
const sandbox = {
  INotebookTracker : {},
  Notification : {info : () => {}},
  setInterval : (callback) => {
    tick = callback;
    return 1;
  },
  clearInterval : () => {},
  requestAnimationFrame : (callback) => callback(),
  console,
};
vm.createContext(sandbox);
vm.runInContext(source, sandbox);
sandbox.plugin.activate(app, {forEach : (callback) => callback(panel)});
const flush = () => new Promise((resolve) => setImmediate(resolve));
await flush();

tick();
await flush();
onSave(context, "started");
context.contentsModel.last_modified = "saved";
onSave(context, "completed");
resolveMetadata({last_modified : "original"});
await flush();
assert.equal(reverts, 0, "An in-flight poll reloaded a browser save");

tick();
await flush();
resolveMetadata({last_modified : "saved"});
await flush();
assert.equal(reverts, 0, "A completed browser save triggered reload");

onSave(context, "started");
const before = reads;
tick();
await flush();
assert.equal(reads, before, "Polling continued during browser save");
onSave(context, "failed");

tick();
await flush();
resolveMetadata({last_modified : "external"});
await flush();
assert.equal(reverts, 1, "External changes stopped reloading");
console.log("Browser saves, overlapping polls, and external reload passed");
