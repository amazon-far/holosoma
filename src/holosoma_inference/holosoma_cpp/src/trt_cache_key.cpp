#include "holosoma_cpp/trt_cache_key.hpp"

#include <functional>
#include <sstream>

namespace holosoma {

std::string trt_engine_file_name(const std::string& model_stem, const std::string& model_key,
                                 const char* tf32_override) {
  // Only "0" disables TF32; the raw value still goes into the hash, so an engine is never shared
  // between two settings even where they would build the same plan.
  const bool fp32 = tf32_override != nullptr && std::string(tf32_override) == "0";
  std::ostringstream key;
  key << model_key << ":tf32_override=" << (tf32_override != nullptr ? tf32_override : "<unset>");
  std::ostringstream name;
  name << model_stem << "-" << (fp32 ? "fp32" : "tf32") << "-" << std::hex << std::hash<std::string>{}(key.str())
       << ".engine";
  return name.str();
}

}  // namespace holosoma
