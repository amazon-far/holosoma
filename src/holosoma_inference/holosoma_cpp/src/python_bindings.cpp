#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "holosoma_cpp/api.hpp"
#include "holosoma_cpp/contextual_policy.hpp"
#include "holosoma_cpp/holosoma_inference_core.hpp"
#include "holosoma_cpp/wbt_policy.hpp"

namespace py = pybind11;
using namespace holosoma;

namespace {
std::vector<float> observation(py::array_t<float, py::array::c_style | py::array::forcecast> a) {
  if (a.ndim() != 1 && !(a.ndim() == 2 && a.shape(0) == 1))
    throw py::value_error("Expected one observation with shape (N,) or (1,N)");
  return {a.data(), a.data() + a.size()};
}
py::array_t<double> array(const std::vector<double>& values) {
  py::array_t<double> out({py::ssize_t(1), static_cast<py::ssize_t>(values.size())});
  std::copy(values.begin(), values.end(), out.mutable_data());
  return out;
}
}  // namespace

PYBIND11_MODULE(_native, m) {
  m.doc() = "Transport-independent native inference; construction never opens a robot interface.";
  py::class_<ContextualInput>(m, "ContextualInput")
      .def(py::init<>())
      .def_readwrite("depth", &ContextualInput::depth)
      .def_readwrite("monotonic_time", &ContextualInput::monotonic_time)
      .def_readwrite("velocity", &ContextualInput::velocity)
      .def_readwrite("command_index", &ContextualInput::command_index)
      .def_readwrite("speed", &ContextualInput::speed)
      .def_readwrite("fixed_arm", &ContextualInput::fixed_arm)
      .def_readwrite("dz", &ContextualInput::dz)
      .def_readwrite("allow_climb_at_stand", &ContextualInput::allow_climb_at_stand)
      .def_readwrite("allow_climb_continuation", &ContextualInput::allow_climb_continuation)
      .def_readwrite("key", &ContextualInput::key);
  py::class_<RobotState>(m, "RobotState")
      .def(py::init<>())
      .def_readwrite("base_pos", &RobotState::base_pos)
      .def_readwrite("base_quat_wxyz", &RobotState::base_quat_wxyz)
      .def_readwrite("dof_pos", &RobotState::dof_pos)
      .def_readwrite("base_lin_vel", &RobotState::base_lin_vel)
      .def_readwrite("base_ang_vel", &RobotState::base_ang_vel)
      .def_readwrite("dof_vel", &RobotState::dof_vel)
      .def_readwrite("projected_gravity", &RobotState::projected_gravity)
      .def_readwrite("tick_ms", &RobotState::tick_ms);
  py::class_<JointCommand>(m, "JointCommand")
      .def_readonly("q", &JointCommand::q)
      .def_readonly("dq", &JointCommand::dq)
      .def_readonly("tau", &JointCommand::tau)
      .def_readonly("kp", &JointCommand::kp)
      .def_readonly("kd", &JointCommand::kd);
  py::class_<ReferenceFrame>(m, "PolicyReferenceFrame")
      .def(py::init<>())
      .def_readwrite("q", &ReferenceFrame::q)
      .def_readwrite("dq", &ReferenceFrame::dq)
      .def_readwrite("quat", &ReferenceFrame::quat);
  py::class_<ReferenceWindow>(m, "ReferenceWindow")
      .def(py::init<>())
      .def_readwrite("q", &ReferenceWindow::q)
      .def_readwrite("dq", &ReferenceWindow::dq)
      .def_readwrite("quat", &ReferenceWindow::quat)
      .def_readwrite("current", &ReferenceWindow::current);
  // Keep the former engine name source-compatible while exposing the public front door.
  using InferenceEngine = Policy;
  py::class_<Policy>(m, "Policy", py::dynamic_attr())
      .def(py::init<const std::string&, const std::string&>(), py::arg("preset"), py::arg("checkpoint") = "")
      .def_static(
          "from_config",
          [](const std::string& config, bool autostart) {
            return std::make_unique<Policy>(config_from_yaml(config), autostart);
          },
          py::arg("config"), py::arg("autostart") = true)
      .def_static(
          "from_live_model",
          [](const std::string& checkpoint, bool use_gpu, int threads, const std::string& history_initialization) {
            auto config = live_policy_config(checkpoint, use_gpu, threads);
            config.task.wbt_history_initialization = history_initialization;
            return std::make_unique<Policy>(config);
          },
          py::arg("checkpoint"), py::arg("use_gpu") = false, py::arg("intra_op_threads") = 2,
          py::arg("history_initialization") = "repeat_first_valid_obs")
      .def(
          "set_reference",
          [](py::object self, py::object window) {
            self.cast<Policy&>().set_reference(window.is_none() ? nullptr : window.cast<const ReferenceWindow*>());
            self.attr("_reference") = window;  // replace ownership rather than retaining every past window
          },
          py::arg("window"))
      .def(
          "step", [](Policy& policy, const RobotState& state) { return JointCommand(policy.step(state)); },
          py::arg("state"), py::call_guard<py::gil_scoped_release>())
      .def(
          "step",
          [](Policy& policy, const RobotState& state, ContextualInput input) {
            return JointCommand(policy.step(state, std::move(input)));
          },
          py::arg("state"), py::arg("inputs"), py::call_guard<py::gil_scoped_release>())
      // The depth contract is read off the loaded graph, so a consumer cannot be configured to
      // preprocess a frame the model preprocesses itself. Empty for a non-contextual policy.
      .def_property_readonly("depth_input",
                             [](const Policy& policy) {
                               const auto* ctx = dynamic_cast<const ContextualPolicy*>(&policy.policy());
                               return ctx ? ctx->core().depth_input() : std::string{};
                             })
      .def("num_dofs", &Policy::num_dofs)
      .def_property_readonly("dof_names", &Policy::dof_names)
      .def_property_readonly("lookahead_offsets",
                             [](const Policy& policy) {
                               const auto* tracker = dynamic_cast<const GeneralWBTPolicy*>(&policy.policy());
                               return tracker ? tracker->core().TrainedLookaheadOffsets() : std::vector<int>{};
                             })
      .def(
          "set_elapsed_seconds",
          [](Policy& policy, std::optional<double> elapsed) {
            auto* tracker = dynamic_cast<GeneralWBTPolicy*>(&policy.policy());
            if (!tracker) throw py::value_error("Elapsed reference time requires live WBT");
            if (elapsed)
              tracker->core().SetElapsedSecondsFn([value = *elapsed] { return value; });
            else
              tracker->core().SetElapsedSecondsFn({});
          },
          py::arg("elapsed_seconds"))
      .def("set_control_interval",
           [](Policy& policy, double interval) {
             auto* tracker = dynamic_cast<GeneralWBTPolicy*>(&policy.policy());
             if (!tracker) throw py::value_error("Reference control interval requires live WBT");
             tracker->core().SetControlInterval(interval);
           })
      .def("reset_heading",
           [](Policy& policy) {
             auto* tracker = dynamic_cast<GeneralWBTPolicy*>(&policy.policy());
             if (!tracker) throw py::value_error("Reference heading reset requires live WBT");
             tracker->core().ResetHeading();
           })
      .def("step",
           [](InferenceEngine& engine, py::array_t<float, py::array::c_style | py::array::forcecast> obs) {
             auto values = observation(obs);
             std::vector<double> result;
             {
               py::gil_scoped_release release;
               result = engine.step(values);
             }
             return array(result);
           })
      .def("step_state", &InferenceEngine::step_state, py::arg("state"), py::arg("advance_phase") = true,
           py::call_guard<py::gil_scoped_release>())
      .def("set_state", &InferenceEngine::set_state)
      .def("infer_offsets",
           [](InferenceEngine& e, const RobotState& state) {
             std::vector<double> result;
             {
               py::gil_scoped_release release;
               result = e.infer_offsets(state);
             }
             return array(result);
           })
      .def("command",
           [](InferenceEngine& e, const std::string& name) {
             auto command = state_command_from_string(name);
             if (!command) throw py::value_error("Unknown state command: " + name);
             e.command(*command);
           })
      .def("velocity", [](InferenceEngine& e, double x, double y, double yaw) { e.velocity({x, y, yaw}); })
      .def("update_phase", &InferenceEngine::update_phase)
      .def("reset", &InferenceEngine::reset)
      .def_property_readonly("active", [](const InferenceEngine& e) { return e.policy().use_policy_action(); })
      .def_property_readonly("ready", [](const InferenceEngine& e) { return e.policy().get_ready_state(); })
      .def_property_readonly("active_index", [](const InferenceEngine& e) { return e.policy().active_index(); })
      .def_property_readonly("motion_timestep",
                             [](const InferenceEngine& e) -> std::optional<int> {
                               const auto* wbt = dynamic_cast<const WholeBodyTrackingPolicy*>(&e.policy());
                               return wbt ? std::optional<int>(wbt->motion_timestep()) : std::nullopt;
                             })
      .def_property_readonly("initial_motion",
                             [](const InferenceEngine& e) {
                               const auto* wbt = dynamic_cast<const WholeBodyTrackingPolicy*>(&e.policy());
                               if (!wbt)
                                 throw py::value_error("Only embedded-motion WBT has an initial motion reference");
                               return array(wbt->initial_motion());
                             })
      .def_property_readonly("last_observation", [](const InferenceEngine& e) { return e.policy().last_observation(); })
      .def_property_readonly("joint_kp", [](const InferenceEngine& e) { return e.policy().joint_kp(); })
      .def_property_readonly("joint_kd", [](const InferenceEngine& e) { return e.policy().joint_kd(); })
      .def_property_readonly("velocity_zero_requests", &InferenceEngine::velocity_zero_requests);

  // The serialized-config preview constructor remains available for existing callers.
  m.def(
      "InferenceEngine",
      [](const std::string& config, bool autostart) {
        return std::make_unique<Policy>(config_from_yaml(config), autostart);
      },
      py::arg("config"), py::arg("autostart") = true);

  py::class_<live::RobotObservation>(m, "LiveRobotObservation")
      .def(py::init<>())
      .def_readwrite("root_quat_wxyz", &live::RobotObservation::root_quat_wxyz)
      .def_readwrite("dof_pos", &live::RobotObservation::dof_pos_mj)
      .def_readwrite("dof_vel", &live::RobotObservation::dof_vel_mj)
      .def_readwrite("base_ang_vel", &live::RobotObservation::base_ang_vel);
  py::class_<live::WbtReferenceFrame>(m, "ReferenceFrame")
      .def(py::init<>())
      .def_readwrite("dof_pos", &live::WbtReferenceFrame::dof_pos)
      .def_readwrite("dof_vel", &live::WbtReferenceFrame::dof_vel)
      .def_readwrite("root_quat_wxyz", &live::WbtReferenceFrame::root_quat_wxyz);
  py::class_<live::WbtReferenceSequence>(m, "ReferenceSequence")
      .def(py::init<>())
      .def_readwrite("current", &live::WbtReferenceSequence::current)
      .def_readwrite("lookahead", &live::WbtReferenceSequence::lookahead);
  py::class_<live::WbtPolicy>(m, "LiveWbtPolicy")
      .def(py::init([](const std::string& path, bool use_gpu, const std::string& history_initialization) {
             auto policy = std::make_unique<live::WbtPolicy>();
             live::WbtHistoryInitialization history;
             if (!live::ParseWbtHistoryInitialization(history_initialization, history))
               throw py::value_error("history_initialization must be repeat_first_valid_obs or zero_pad, not '" +
                                     history_initialization + "'");
             policy->SetHistoryInitialization(history);
             if (!policy->Initialize(path, use_gpu))
               throw py::value_error("Cannot initialize live WBT checkpoint: " + path);
             return policy;
           }),
           py::arg("model_path"), py::arg("use_gpu") = false,
           py::arg("history_initialization") = "repeat_first_valid_obs")
      .def_property_readonly(
          "history_initialization",
          [](const live::WbtPolicy& p) { return std::string(live::ToString(p.HistoryInitialization())); })
      .def_property_readonly("action_clip", &live::WbtPolicy::ActionClip)
      .def(
          "step",
          [](live::WbtPolicy& p, const live::RobotObservation& obs, const live::WbtReferenceSequence& ref,
             std::optional<double> elapsed_seconds) {
            if (elapsed_seconds)
              p.SetElapsedSecondsFn([value = *elapsed_seconds] { return value; });
            else
              p.SetElapsedSecondsFn({});
            std::array<float, live::kActionDim> offsets{};
            {
              py::gil_scoped_release release;
              if (!p.ComputeAction(obs, ref, offsets.data())) throw std::runtime_error("Live WBT inference failed");
            }
            std::vector<double> targets(live::kActionDim);
            for (int i = 0; i < live::kActionDim; ++i) targets[i] = offsets[i] + p.DefaultDofPos()[i];
            return array(targets);
          },
          py::arg("observation"), py::arg("reference"), py::arg("elapsed_seconds") = py::none())
      .def("set_control_interval", &live::WbtPolicy::SetControlInterval)
      .def("reset", &live::WbtPolicy::Reset)
      .def("reset_heading", &live::WbtPolicy::ResetHeading)
      .def("lookahead_offsets", &live::WbtPolicy::LookaheadSampleOffsets)
      .def_property_readonly("last_observation", &live::WbtPolicy::LastObservation)
      .def_property_readonly("last_raw_action", &live::WbtPolicy::LastRawAction)
      .def_property_readonly("last_clipped_action", &live::WbtPolicy::LastClippedAction)
      .def_property_readonly("kp", &live::WbtPolicy::Kp)
      .def_property_readonly("kd", &live::WbtPolicy::Kd);
}
