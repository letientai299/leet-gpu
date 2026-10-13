import {Notification} from "@jupyterlab/apputils";
import {INotebookTracker} from "@jupyterlab/notebook";

let timer;
let stop;

function cellKey(cell) {
  const source = Array.isArray(cell.source) ? cell.source.join("") : cell.source;
  return JSON.stringify([ cell.cell_type, source ]);
}

function sources(notebook) {
  return JSON.stringify(notebook.cells.map(cellKey));
}

function mergeCells(incoming, previous) {
  const cells = new Map();
  for (const cell of previous) {
    const key = cellKey(cell);
    if (!cells.has(key))
      cells.set(key, []);
    cells.get(key).push(cell);
  }
  return incoming.map((cell) => {
    const old = cells.get(cellKey(cell))?.shift();
    if (!old)
      return cell;
    if (cell.cell_type !== "code")
      return {...cell, id : old.id};
    return {
      ...cell,
      id : old.id,
      outputs : old.outputs,
      execution_count : old.execution_count,
    };
  });
}

const plugin = {
  id : "leet-gpu:markdown-autoreload",
  autoStart : true,
  requires : [ INotebookTracker ],
  activate : (app, tracker) => {
    const states = new WeakMap();
    const viewed = new WeakSet();
    let active = true;

    function showOutputs(panel) {
      if (!active || panel.isDisposed || !panel.context.path.endsWith(".md"))
        return;
      for (const cell of panel.content.widgets) {
        if (cell.model.type !== "code" || viewed.has(cell))
          continue;
        cell.syncCollapse = false;
        cell.inputHidden = true;
        cell.outputHidden = false;
        viewed.add(cell);
      }
    }

    function initialize(panel) {
      panel.context.ready.then(() => showOutputs(panel))
        .catch(
          (error) => console.warn("Notebook view failed", error),
        );
    }

    const notebookAdded = (_, panel) => initialize(panel);
    tracker.widgetAdded.connect(notebookAdded);
    stop = () => {
      active = false;
      clearInterval(timer);
      tracker.widgetAdded.disconnect(notebookAdded);
    };

    async function reload(panel) {
      const context = panel.context;
      if (panel.isDisposed || !context.path.endsWith(".md"))
        return;
      await context.ready;
      showOutputs(panel);
      let state = states.get(panel);
      if (!state) {
        state = {
          modified : context.contentsModel.last_modified,
          sources : sources(context.model.toJSON()),
          busy : false,
          saving : false,
          generation : 0,
          warned : false,
        };
        states.set(panel, state);
        context.saveState.connect((_, status) => {
          if (status === "started") {
            state.saving = true;
            state.generation += 1;
            return;
          }
          state.saving = false;
          if (status !== "completed")
            return;
          state.modified = context.contentsModel.last_modified;
          state.sources = sources(context.model.toJSON());
          state.warned = false;
        });
      }
      if (state.busy || state.saving || context.sessionContext.kernelDisplayStatus === "busy")
        return;
      state.busy = true;
      try {
        const contents = app.serviceManager.contents;
        const generation = state.generation;
        const file = await contents.get(context.path, {content : false});
        if (state.saving || state.generation !== generation)
          return;
        if (file.last_modified === state.modified || panel.isDisposed)
          return;
        if (panel.isDisposed || context.sessionContext.kernelDisplayStatus === "busy")
          return;
        const previous = context.model.toJSON();
        if (sources(previous) !== state.sources) {
          if (!state.warned) {
            Notification.info("Save browser edits to resume notebook reload.", {
              autoClose : 5000,
            });
            state.warned = true;
          }
          return;
        }
        const index = panel.content.activeCellIndex;
        const scroll = panel.content.node.scrollTop;
        const readOnly = context.model.readOnly;
        context.model.readOnly = true;
        try {
          await context.revert();
          if (panel.isDisposed)
            return;
          const incoming = context.model.toJSON();
          const cells = mergeCells(incoming.cells, previous.cells);
          context.model.fromJSON({...incoming, cells});
        } finally {
          if (!panel.isDisposed) {
            context.model.readOnly = readOnly || context.contentsModel.writable === false;
          }
        }
        context.model.dirty = false;
        panel.content.activeCellIndex = Math.min(
          index,
          panel.content.widgets.length - 1,
        );
        requestAnimationFrame(() => {
          if (!panel.isDisposed) {
            showOutputs(panel);
            panel.content.node.scrollTop = scroll;
          }
        });
        state.sources = sources(context.model.toJSON());
        state.modified = context.contentsModel.last_modified;
        state.warned = false;
      } finally {
        state.busy = false;
      }
    }

    app.restored.then(() => {
      if (!active)
        return;
      tracker.forEach(initialize);
      timer = setInterval(() => {
        tracker.forEach((panel) => {
          reload(panel).catch(
            (error) => console.warn("Notebook reload failed", error),
          );
        });
      }, 1000);
    });
  },
  deactivate : () => stop?.(),
};

export default plugin;
