#pragma once

namespace lg {

/// Git commit the binary was built from, "-dirty" when sources differed.
const char* build_revision();

} // namespace lg
