# Writes OUTPUT with the git commit, plus "-dirty" when built sources differ
# from it. Rewrites only on change, so unchanged builds skip the relink.
# cmake -DSOURCE_DIR=... -DOUTPUT=... -P build_revision.cmake

# The container builds as root in a tree owned by another user.
set(
  git
  git
  -c
  "safe.directory=${SOURCE_DIR}"
  -C
  "${SOURCE_DIR}"
)
execute_process(
  COMMAND ${git} rev-parse --short=12 HEAD
  OUTPUT_VARIABLE revision
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE failed
  ERROR_QUIET
)
if(failed)
  set(revision unknown)
else()
  execute_process(
    COMMAND ${git} status --porcelain -- src CMakeLists.txt
    OUTPUT_VARIABLE changes
    ERROR_QUIET
  )
  if(changes)
    string(APPEND revision -dirty)
  endif()
endif()

file(
  WRITE "${OUTPUT}.tmp"
  "namespace lg {\nconst char* build_revision() { return \"${revision}\"; }\n}\n"
)
file(COPY_FILE "${OUTPUT}.tmp" "${OUTPUT}" ONLY_IF_DIFFERENT)
file(REMOVE "${OUTPUT}.tmp")
