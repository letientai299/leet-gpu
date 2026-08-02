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
    "/usr/lib/llvm-21",
  },
})
