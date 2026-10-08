#include "holosoma_cpp/observation.hpp"

#include <algorithm>
#include <set>
#include <stdexcept>

namespace holosoma {

ObservationGroup::ObservationGroup(std::string name, std::vector<std::string> terms, const ObservationConfig& config)
    : name_(std::move(name)), terms_(std::move(terms)) {
  std::sort(terms_.begin(), terms_.end());
  const auto history = config.history_length_dict.find(name_);
  history_length_ = history == config.history_length_dict.end() ? 1 : history->second;
  for (const auto& term : terms_) {
    Term state;
    state.name = term;
    state.dim = config.obs_dims.at(term);
    state.scale = config.obs_scales.at(term);
    dim_ += state.dim * history_length_;
    term_state_.push_back(std::move(state));
  }
}

int ObservationGroup::term_dim(const std::string& term) const {
  for (const auto& state : term_state_) {
    if (state.name == term) {
      return state.dim;
    }
  }
  throw std::out_of_range("Observation group '" + name_ + "' has no term '" + term + "'");
}

void ObservationGroup::update(const TermValues& values, float* out) {
  size_t offset = 0;
  for (auto& term : term_state_) {
    const auto found = values.find(term.name);
    if (found == values.end()) {
      throw std::runtime_error("Observation term '" + term.name + "' missing from current observation buffer.");
    }
    const std::vector<double>& raw = found->second;
    if (static_cast<int>(raw.size()) != term.dim) {
      throw std::runtime_error("Observation term '" + term.name + "' has " + std::to_string(raw.size()) +
                               " values, configured dimension is " + std::to_string(term.dim));
    }
    // (term_obs * scale).astype(float32): scale in double, then round once to float.
    std::vector<float> scaled(raw.size());
    for (size_t i = 0; i < raw.size(); ++i) {
      scaled[i] = static_cast<float>(raw[i] * term.scale);
    }
    term.history.push_back(std::move(scaled));
    while (static_cast<int>(term.history.size()) > history_length_) {
      term.history.pop_front();
    }

    const int missing = history_length_ - static_cast<int>(term.history.size());
    const size_t term_dim = static_cast<size_t>(term.dim);
    std::fill(out + offset, out + offset + term_dim * static_cast<size_t>(missing), 0.0f);
    offset += term_dim * static_cast<size_t>(missing);
    for (const auto& step : term.history) {
      std::copy(step.begin(), step.end(), out + offset);
      offset += term_dim;
    }
  }
}

void ObservationGroup::reset() {
  for (auto& term : term_state_) {
    term.history.clear();
  }
}

ObservationBuilder::ObservationBuilder(const ObservationConfig& config) {
  for (const auto& [name, terms] : config.obs_dict) {
    groups_.emplace_back(name, terms, config);
  }
}

bool ObservationBuilder::has_group(const std::string& group) const {
  return std::any_of(groups_.begin(), groups_.end(), [&](const ObservationGroup& g) { return g.name() == group; });
}

ObservationGroup& ObservationBuilder::group(const std::string& group) {
  for (auto& g : groups_) {
    if (g.name() == group) {
      return g;
    }
  }
  throw std::out_of_range("Observation group '" + group + "' is not configured for this policy.");
}

const ObservationGroup& ObservationBuilder::group(const std::string& group) const {
  return const_cast<ObservationBuilder*>(this)->group(group);
}

std::vector<std::string> ObservationBuilder::all_terms() const {
  std::set<std::string> terms;
  for (const auto& g : groups_) {
    terms.insert(g.sorted_terms().begin(), g.sorted_terms().end());
  }
  return {terms.begin(), terms.end()};
}

}  // namespace holosoma
