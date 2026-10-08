// Observation assembly (BasePolicy._prepare_group_observations).
//
// Terms of a group are concatenated in sorted (alphabetical) order. Each term
// is scaled, kept in a per-term history of `history_length` steps (oldest
// first, zero padded) and flattened time-major, exactly like the Python runtime.
#pragma once

#include <deque>
#include <map>
#include <string>
#include <vector>

#include "holosoma_cpp/config.hpp"

namespace holosoma {

// Unscaled term values for one control step.
using TermValues = std::map<std::string, std::vector<double>>;

class ObservationGroup {
 public:
  ObservationGroup(std::string name, std::vector<std::string> terms, const ObservationConfig& config);

  const std::string& name() const { return name_; }
  const std::vector<std::string>& sorted_terms() const { return terms_; }
  int history_length() const { return history_length_; }
  int term_dim(const std::string& term) const;
  int dim() const { return dim_; }

  // Appends the current values to the history and writes the flattened group into `out` (size dim()).
  void update(const TermValues& values, float* out);

  // Clears the history (zero padding again).
  void reset();

  // Python's zero-initialized buffer, used before the first update (e.g. the WBT probe inference).
  std::vector<float> zeros() const { return std::vector<float>(static_cast<size_t>(dim_), 0.0f); }

 private:
  struct Term {
    std::string name;
    int dim = 0;
    double scale = 1.0;
    std::deque<std::vector<float>> history;
  };

  std::string name_;
  std::vector<std::string> terms_;
  std::vector<Term> term_state_;
  int history_length_ = 1;
  int dim_ = 0;
};

class ObservationBuilder {
 public:
  explicit ObservationBuilder(const ObservationConfig& config);

  bool has_group(const std::string& group) const;
  ObservationGroup& group(const std::string& group);
  const ObservationGroup& group(const std::string& group) const;

  // Every configured term name (all groups).
  std::vector<std::string> all_terms() const;

 private:
  std::vector<ObservationGroup> groups_;
};

}  // namespace holosoma
