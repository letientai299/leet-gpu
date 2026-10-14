-- Project config, auto-loaded by nvim's 'exrc' (trusted via lib.exrc
-- trust_roots under ~/dev/mine). clangd runs inside the leet-gpu-cuda container
-- (see .nvim/clangd), so go-to-definition into CUDA / libstdc++ / gcc headers
-- yields container paths that don't exist on the host. Fetch them read-only
-- from the container so those jumps open.

local this = debug.getinfo(1, "S").source:sub(2) -- strip leading '@'
local root = vim.fn.fnamemodify(this, ":h")

-- Reuse CONTAINER_NAME from .env (the same source .nvim/clangd reads).
local container = "leet-gpu-cuda"
local env = root .. "/.env"
if vim.uv.fs_stat(env) then
  for line in io.lines(env) do
    local v = line:match("^%s*CONTAINER_NAME%s*=%s*(.-)%s*$")
    if v then
      container = (v:gsub('^"(.*)"$', "%1"):gsub("^'(.*)'$", "%1"))
    end
  end
end

require("lib.container_files").setup({
  container = container,
  -- Prefixes clangd's cc1 args reference for system/toolchain headers.
  prefixes = {
    "/usr/local/cuda",
    "/usr/local/include",
    "/usr/include",
    "/usr/lib/gcc",
    "/opt/rh",
    "/usr/lib/llvm-21",
  },
})

-- Clang-tidy checks clangd fixes on save.
local fix_on_save = { ["misc-const-correctness"] = true }

-- Fixes target the text clangd last diagnosed.
local diagnosed_tick = {} ---@type table<integer, integer>
local group = vim.api.nvim_create_augroup("leet_gpu_tidy_fix", { clear = true })

---@param buf integer
---@return vim.lsp.Client?
local function clangd_client(buf)
  return vim.lsp.get_clients({ bufnr = buf, name = "clangd" })[1]
end

vim.api.nvim_create_autocmd("DiagnosticChanged", {
  group = group,
  pattern = root .. "/*",
  callback = function(ev)
    local client = clangd_client(ev.buf)
    local first = ev.data.diagnostics[1]
    if
      client
      and first
      and first.namespace
        == vim.lsp.diagnostic.get_namespace(client.id, false)
    then
      diagnosed_tick[ev.buf] = vim.b[ev.buf].changedtick
    end
  end,
})

vim.api.nvim_create_autocmd("BufWritePre", {
  group = group,
  pattern = root .. "/*",
  callback = function(ev)
    local client = clangd_client(ev.buf)
    if not client or diagnosed_tick[ev.buf] ~= vim.b[ev.buf].changedtick then
      return
    end
    local ns = vim.lsp.diagnostic.get_namespace(client.id, false)
    local diags = {}
    for _, d in ipairs(vim.diagnostic.get(ev.buf, { namespace = ns })) do
      local lsp = d.user_data and d.user_data.lsp
      if lsp and fix_on_save[lsp.code] then
        table.insert(diags, lsp)
      end
    end
    if #diags == 0 then
      return
    end

    local last = vim.api.nvim_buf_line_count(ev.buf) - 1
    local resp = client:request_sync("textDocument/codeAction", {
      textDocument = vim.lsp.util.make_text_document_params(ev.buf),
      range = {
        start = { line = 0, character = 0 },
        ["end"] = { line = last, character = 0 },
      },
      context = { diagnostics = diags, only = { "quickfix" } },
    }, 1000, ev.buf)
    if not resp or resp.err then
      return
    end

    -- Merge edits so same-line fixes apply together.
    local uri = vim.uri_from_bufnr(ev.buf)
    local edits, seen = {}, {}
    for _, action in ipairs(resp.result or {}) do
      local we = action.edit or {}
      local lists = { (we.changes or {})[uri] }
      for _, dc in ipairs(we.documentChanges or {}) do
        if dc.textDocument and dc.textDocument.uri == uri then
          table.insert(lists, dc.edits)
        end
      end
      for _, list in ipairs(lists) do
        for _, e in ipairs(list) do
          local key = vim.json.encode({ e.range, e.newText })
          if not seen[key] then
            seen[key] = true
            table.insert(edits, e)
          end
        end
      end
    end
    if #edits == 0 then
      return
    end
    vim.lsp.util.apply_text_edits(edits, ev.buf, client.offset_encoding)
    -- Conform may run first; reformat to west const.
    require("conform").format({ bufnr = ev.buf })
  end,
})
