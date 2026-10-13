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
let onNotebook;
function codeCell() {
  return {
    model : {type : "code", metadata : {}},
    syncCollapse : true,
    _inputHidden : false,
    _outputHidden : true,
    get inputHidden() {
      return this._inputHidden;
    },
    set inputHidden(value) {
      this._inputHidden = value;
      if (this.syncCollapse)
        this.model.metadata.source_hidden = value;
    },
    get outputHidden() {
      return this._outputHidden;
    },
    set outputHidden(value) {
      this._outputHidden = value;
      if (this.syncCollapse)
        this.model.metadata.outputs_hidden = value;
    },
  };
}
const code = codeCell();
const markdown = {
  model : {type : "markdown"},
  inputHidden : false
};
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
  content : {activeCellIndex : 0, node : {scrollTop : 0}, widgets : [ code, markdown ]},
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
sandbox.plugin.activate(app, {
  forEach : (callback) => callback(panel),
  widgetAdded : {
    connect : (callback) => {
      onNotebook = callback;
    },
    disconnect : (callback) => {
      assert.equal(callback, onNotebook, "Wrong notebook handler removed");
      onNotebook = undefined;
    },
  },
});
const flush = () => new Promise((resolve) => setImmediate(resolve));
await flush();
assert.equal(code.inputHidden, true, "Code remained visible on startup");
assert.equal(code.outputHidden, false, "Outputs remained hidden on startup");
assert.deepEqual(code.model.metadata, {}, "View defaults wrote cell metadata");
assert.equal(markdown.inputHidden, false, "Markdown was hidden");

const opened = codeCell();
onNotebook(null, {...panel, content : {...panel.content, widgets : [ opened ]}});
await flush();
assert.equal(opened.inputHidden, true, "New notebooks kept code visible");
assert.deepEqual(opened.model.metadata, {}, "Opening notebooks wrote metadata");

const ipynb = codeCell();
onNotebook(null, {
  ...panel,
  context : {...context, path : "notebook.ipynb"},
  content : {...panel.content, widgets : [ ipynb ]},
});
await flush();
assert.equal(ipynb.inputHidden, false, "Unrelated notebooks changed");

code.inputHidden = false;
const added = codeCell();
panel.content.widgets.push(added);

tick();
await flush();
assert.equal(code.inputHidden, false, "Polling reset manual visibility");
assert.equal(added.inputHidden, true, "New code cells stayed visible");
assert.deepEqual(added.model.metadata, {}, "New cells gained view metadata");
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
sandbox.plugin.deactivate();
assert.equal(onNotebook, undefined, "Notebook handler survived deactivation");
console.log("View defaults, metadata isolation, saves, and reload passed");
