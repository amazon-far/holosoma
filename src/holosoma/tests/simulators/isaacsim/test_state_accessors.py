from __future__ import annotations

from unittest.mock import Mock

import pytest
import torch

from holosoma.simulator.isaacsim.proxy_utils import RootStatesProxy
from holosoma.simulator.isaacsim.state_adapter import IsaacSimStateAdapter
from holosoma.simulator.isaacsim.state_utils import fullstate_wxyz_to_xyzw
from holosoma.simulator.shared.object_registry import ObjectRegistry, ObjectType

# Pure torch/mock accessor tests (no isaaclab); runs in the no_sim CPU job, not the isaacsim job.
pytestmark = pytest.mark.no_sim


class TestStateAccessors:
    """Test state access patterns for IsaacSim objects.

    Covers the IsaacSimStateAdapter index path (``get_states_by_index`` /
    ``write_states_by_index``), RootStatesProxy, and ObjectRegistry. End-to-end behavior of the
    unified view on a live sim is covered by ``all_root_states_unified_assert.py``.
    """

    # fmt: off

    # Test data constants
    env_ids = torch.tensor([0])

    # Robot test data - wxyz format (IsaacSim internal)
    robot_wxyz = torch.tensor([[
        1.0, 0.0, 1.0,
        1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0
    ]], dtype=torch.float32)

    robot_new_wxyz = torch.tensor([[
        2.0, 1.0, 1.5,
        0.707, 0.707, 0.0, 0.0,
        0.5, 0.2, 0.1,
        0.1, 0.05, 0.02
    ]], dtype=torch.float32)

    # Box test data - wxyz format (IsaacSim internal)
    box_wxyz = torch.tensor([[
        2.0, 1.0, 0.5,
        1.0, 0.2, 0.4, 0.6,
        0.1, 0.2, 0.0,
        0.01, 0.02, 0.03
    ]], dtype=torch.float32)

    box_new_wxyz = torch.tensor([[
        3.0, 2.0, 1.0,
        0.707, 0.1, 0.2, 0.3,
        0.5, 0.3, 0.1,
        0.05, 0.03, 0.08
    ]], dtype=torch.float32)

    # Multi-environment test data (2 envs)
    multi_robot_wxyz = torch.tensor([
        [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.5, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
    ], dtype=torch.float32)
    multi_box_wxyz = torch.tensor([
        [2.0, 1.0, 0.5, 0.707, 0.707, 0.0, 0.0, 0.1, 0.1, 0.0, 0.01, 0.01, 0.01],
        [2.5, 1.0, 0.5, 0.707, 0.707, 0.0, 0.0, 0.2, 0.1, 0.0, 0.02, 0.01, 0.01],
    ], dtype=torch.float32)

    # Scene object test data - wxyz format (IsaacSim internal)
    scene_wxyz = torch.tensor([[
        0.0, 0.0, 0.5,
        1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0
    ]], dtype=torch.float32)

    # Convert manually, used to assert tests after read/writes
    robot_xyzw          = fullstate_wxyz_to_xyzw(robot_wxyz)
    robot_new_xyzw      = fullstate_wxyz_to_xyzw(robot_new_wxyz)
    box_xyzw            = fullstate_wxyz_to_xyzw(box_wxyz)
    box_new_xyzw        = fullstate_wxyz_to_xyzw(box_new_wxyz)
    multi_robot_xyzw    = fullstate_wxyz_to_xyzw(multi_robot_wxyz)
    multi_box_xyzw      = fullstate_wxyz_to_xyzw(multi_box_wxyz)
    scene_xyzw          = fullstate_wxyz_to_xyzw(scene_wxyz)

    # fmt: on

    def _create_mock_object(self, wxyz_state: torch.Tensor, object_name: str = "object"):
        """Test helper to create a mock object (robot or rigid_object) with functional write methods."""

        # Create a mutable tensor that can be updated by write operations
        # Ensure it's 2D: [num_envs, 13]
        if wxyz_state.dim() == 1:
            wxyz_state = wxyz_state.unsqueeze(0)

        # Create the mock object structure
        mock_object = Mock()
        mock_data = Mock()

        # Store the state tensor (mutable, in wxyz format like IsaacSim)
        state_tensor = wxyz_state.clone()
        mock_data.root_state_w = state_tensor  # For rigid objects

        # Functional write methods that update the internal tensor
        def write_root_pose_to_sim(pose_tensor, env_ids):
            state_tensor[env_ids, 0:7] = pose_tensor

        def write_root_velocity_to_sim(velocity_tensor, env_ids):
            state_tensor[env_ids, 7:13] = velocity_tensor

        # Attach the working write methods
        mock_object.write_root_pose_to_sim = write_root_pose_to_sim
        mock_object.write_root_velocity_to_sim = write_root_velocity_to_sim
        mock_object.data = mock_data  # For rigid objects
        mock_object._internal_state = state_tensor  # For direct access

        return mock_object

    def _create_mock_robot_states(self, xyzw_state: torch.Tensor):
        """Test helper to create a mock robot states object with the given xyzw state.

        NOTE: this builds a *standalone* read-only mock (returns the given constant) for the
        cases that only read once. For a genuine write->read round-trip use
        ``_bind_robot_states_to_write_tensor`` so the read observes what the write actually did.
        """
        mock_robot_states = Mock()
        mock_robot_states.__getitem__ = Mock(return_value=xyzw_state)
        return mock_robot_states

    def _bind_robot_states_to_write_tensor(self, adapter):
        """Back the robot-states *read* path with the SAME tensor the *write* path mutates.

        The production robot read path (``IsaacSimStateAdapter.get_object_states``) returns
        ``self._robot_states[env_ids]`` (already xyzw), while the write path mutates a separate
        ``self._robot`` handle's wxyz tensor via ``write_root_pose_to_sim`` /
        ``write_root_velocity_to_sim``. If the robot-states mock just returns a hand-injected
        constant, the read can never observe the write — so a regression in pose/velocity
        writing OR the xyzw<->wxyz conversion is invisible.

        Here we point the robot-states mock's ``__getitem__`` at the robot handle's backing wxyz
        tensor (the one ``write_object_states`` actually writes into), converting wxyz->xyzw on
        the way out — exactly mirroring production (write does xyzw->wxyz in, this read does
        wxyz->xyzw out). A round-trip ``write(state) -> read()`` therefore returns ``state``
        ONLY if both the write landed and the conversion is correct; drop either and it fails.
        """
        write_tensor_wxyz = adapter._robot._internal_state  # mutated by write_root_*_to_sim

        def read_xyzw(env_ids):
            return fullstate_wxyz_to_xyzw(write_tensor_wxyz[env_ids])

        adapter._robot_states.__getitem__ = Mock(side_effect=read_xyzw)

    def _create_state_adapter(self, object_configs: list[tuple], default_robot_wxyz_state=None, num_envs=1):
        """Test helper to create a IsaacSimStateAdapter with mocks

           Simplifies each test case, avoids repetition.

        Example:
            object_configs = [
                ("robot", {"robot": mock_robot_states}),
                ("individual", {"box1": wxyz_tensor, "box2": wxyz_tensor})
            ]
        """

        def create_object_registry():
            # Create real ObjectRegistry with minimal setup
            object_registry = ObjectRegistry(device="cpu")

            # Count objects for range setup
            robot_count = 1 if any(obj_type == "robot" for obj_type, _ in object_configs) else 0
            individual_count = sum(
                len(objects_dict) for obj_type, objects_dict in object_configs if obj_type == "individual"
            )
            scene_count = sum(len(objects_dict) for obj_type, objects_dict in object_configs if obj_type == "scene")
            object_registry.setup_ranges(
                num_envs=num_envs, robot_count=robot_count, scene_count=scene_count, individual_count=individual_count
            )

            # Register objects with dummy poses (we don't actually use the poses in our tests)
            dummy_pose = torch.zeros(num_envs, 7, dtype=torch.float32)
            dummy_pose[:, 6] = 1.0  # Set identity quaternions
            position_counters = {"robot": 0, "scene": 0, "individual": 0}

            for object_type, objects_dict in object_configs:
                for obj_name in objects_dict:
                    obj_type_enum = (
                        ObjectType.ROBOT
                        if object_type == "robot"
                        else (ObjectType.SCENE if object_type == "scene" else ObjectType.INDIVIDUAL)
                    )
                    object_registry.register_object(
                        name=obj_name,
                        object_type=obj_type_enum,
                        position_in_type=position_counters[object_type],
                        initial_poses=dummy_pose,
                    )
                    position_counters[object_type] += 1

            object_registry.finalize_registration()
            return object_registry

        # Setup references for StateAdapter, collect all object for the real registry, robot state mocks
        object_type_mapping = {}
        all_rigid_objects = {}
        robot_states_mock = Mock()
        robot_wxyz_state = (
            default_robot_wxyz_state
            if default_robot_wxyz_state is not None
            else torch.tensor([[0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
        )
        robot_mock = self._create_mock_object(robot_wxyz_state, "robot")

        # Process each object type configuration
        for object_type, objects_dict in object_configs:
            for obj_name, obj_data in objects_dict.items():
                object_type_mapping[obj_name] = object_type
                if object_type == "individual":
                    all_rigid_objects[obj_name] = self._create_mock_object(obj_data, obj_name)
                elif object_type == "robot":
                    robot_states_mock = obj_data
                elif object_type == "scene":
                    all_rigid_objects[obj_name] = self._create_mock_object(obj_data, obj_name)
                else:
                    raise NotImplementedError("Unknown type")

        # Setup scene with all rigid objects
        mock_scene = Mock()
        mock_scene.env_origins = torch.zeros(num_envs, 3)
        mock_scene.rigid_objects = all_rigid_objects

        # Add _get_permuted_clone method to robot_states_mock for clone() compatibility
        def get_permuted_clone():
            # Return empty tensor to avoid double-counting robot in clone()
            # The clone() method will get robot states from the registry iteration
            return torch.empty(0, 13, dtype=torch.float32)

        robot_states_mock._get_permuted_clone = get_permuted_clone

        return IsaacSimStateAdapter(
            device=torch.device("cpu"),
            object_registry=create_object_registry(),
            scene=mock_scene,
            robot=robot_mock,
            robot_states=robot_states_mock,
        )

    def _assert_full_state_equal(self, actual: torch.Tensor, expected: torch.Tensor, msg: str = ""):
        """Helper to validate all 13 state elements with clear error messages."""
        assert actual.shape == expected.shape
        assert actual.shape[-1] == 13, f"Expected 13-element state, got {actual.shape[-1]} elements"
        assert expected.shape[-1] == 13, f"Expected 13-element state, got {expected.shape[-1]} elements"

        # Position [0:3]
        assert torch.allclose(actual[..., 0:3], expected[..., 0:3], atol=1e-6), (
            f"{msg} Position mismatch. Expected: {expected[..., 0:3]}, Got: {actual[..., 0:3]}"
        )

        # Quaternion [3:7] - xyzw format
        assert torch.allclose(actual[..., 3:7], expected[..., 3:7], atol=1e-6), (
            f"{msg} Quaternion mismatch. Expected: {expected[..., 3:7]}, Got: {actual[..., 3:7]}"
        )

        # Linear velocity [7:10]
        assert torch.allclose(actual[..., 7:10], expected[..., 7:10], atol=1e-6), (
            f"{msg} Linear velocity mismatch. Expected: {expected[..., 7:10]}, Got: {actual[..., 7:10]}"
        )

        # Angular velocity [10:13]
        assert torch.allclose(actual[..., 10:13], expected[..., 10:13], atol=1e-6), (
            f"{msg} Angular velocity mismatch. Expected: {expected[..., 10:13]}, Got: {actual[..., 10:13]}"
        )

    def test_robot_state_roundtrip_via_root_states_proxy(self):
        """Test robot state read/write using RootStatesProxy."""

        proxy = RootStatesProxy(self.robot_wxyz)

        # Test read
        actual_state = proxy[self.env_ids, :]
        self._assert_full_state_equal(actual_state, self.robot_xyzw, "Read")

        # Test write (via RootStatesProxy __setitem__ and __getitem__)
        proxy[self.env_ids, :] = self.robot_new_xyzw[0]
        actual_state = proxy[self.env_ids, :]
        self._assert_full_state_equal(actual_state, self.robot_new_xyzw, "Write")

    def test_robot_state_roundtrip_via_index_path(self):
        """Robot state read/write through the adapter index path (get/write_states_by_index)."""

        # Create StateAdapter with internal IsaacSim robot states
        mock_robot_states = self._create_mock_robot_states(self.robot_xyzw)
        object_configs = [("robot", {"robot": mock_robot_states})]
        adapter = self._create_state_adapter(object_configs, self.robot_wxyz)
        # Back the READ path with the SAME wxyz tensor the WRITE path mutates, so the
        # round-trip below genuinely observes the write + the xyzw<->wxyz conversion (not a
        # separately-injected constant).
        self._bind_robot_states_to_write_tensor(adapter)

        # robot is the only actor -> index 0 in env 0
        indices = torch.tensor([0])

        # Test read
        actual_result = adapter.get_states_by_index(indices)
        self._assert_full_state_equal(actual_result, self.robot_xyzw, "Read")

        # Test write via the index path, then read back. The read is now backed by the write
        # tensor, so this fails if the pose/velocity write is a no-op or the xyzw->wxyz->xyzw
        # conversion is dropped.
        adapter.write_states_by_index(indices, self.robot_new_xyzw)
        actual_result_after_write = adapter.get_states_by_index(indices)
        self._assert_full_state_equal(actual_result_after_write, self.robot_new_xyzw, "Write roundtrip")

    def test_robot_states_roundtrip_via_state_adapter(self):
        """Test robot state read/write via StateAdapter get/write methods"""
        # Create StateAdapter with internal IsaacSim robot states
        mock_robot_states = self._create_mock_robot_states(self.robot_xyzw)
        object_configs = [("robot", {"robot": mock_robot_states})]
        adapter = self._create_state_adapter(object_configs, self.robot_wxyz)
        # Back the READ path with the wxyz tensor the WRITE path mutates (see helper docstring).
        self._bind_robot_states_to_write_tensor(adapter)

        # Test read
        actual_state = adapter.get_object_states("robot", self.env_ids)
        self._assert_full_state_equal(actual_state, self.robot_xyzw, "Robot read")

        # Test write, then read back the SAME tensor it mutated. This exercises the real
        # write_root_pose_to_sim / write_root_velocity_to_sim split AND the xyzw->wxyz
        # conversion in write_object_states (state_adapter ~line 114): a dropped conversion or a
        # no-op write would make the read-back disagree with robot_new_xyzw and fail.
        adapter.write_object_states("robot", self.robot_new_xyzw, self.env_ids)

        actual_result_after_write = adapter.get_object_states("robot", self.env_ids)
        self._assert_full_state_equal(actual_result_after_write, self.robot_new_xyzw, "Write roundtrip")

    def test_object_states_roundtrip_via_state_adapter(self):
        """Test object (non-robot) state read/write via StateAdapter get/write methods"""
        object_configs = [("individual", {"box": self.box_wxyz})]
        adapter = self._create_state_adapter(object_configs)

        # Test read
        actual_state = adapter.get_object_states("box", self.env_ids)
        self._assert_full_state_equal(actual_state, self.box_xyzw, "Read")

        # Test write
        adapter.write_object_states("box", self.box_new_xyzw, self.env_ids)

        actual_result_after_write = adapter.get_object_states("box", self.env_ids)
        self._assert_full_state_equal(actual_result_after_write, self.box_new_xyzw, "Write")

    def test_object_states_roundtrip_via_index_path(self):
        """Individual object read/write through the adapter index path."""
        # Create adapter with individual object
        object_configs = [("individual", {"box": self.box_wxyz})]
        adapter = self._create_state_adapter(object_configs)
        indices = torch.tensor([0])  # box is the only actor

        # Test read (should convert wxyz → xyzw)
        actual_state = adapter.get_states_by_index(indices)
        self._assert_full_state_equal(actual_state, self.box_xyzw, "Read")

        # Test write (should convert xyzw → wxyz internally)
        adapter.write_states_by_index(indices, self.box_new_xyzw)

        # Test roundtrip - read back to verify write worked
        actual_after_write = adapter.get_states_by_index(indices)
        self._assert_full_state_equal(actual_after_write, self.box_new_xyzw, "Roundtrip")

    def test_multiple_envs_and_objects_roundtrip_via_index_path(self):
        """Multi-object, multi-env read/write through the adapter index path."""
        # Create real adapter with multiple object types and 2 environments
        mock_robot_states = self._create_mock_robot_states(self.multi_robot_xyzw)
        object_configs = [("robot", {"robot": mock_robot_states}), ("individual", {"box": self.multi_box_wxyz})]
        adapter = self._create_state_adapter(object_configs, self.multi_robot_wxyz, num_envs=2)
        # Back the robot READ path with the wxyz tensor the WRITE path mutates, so the robot
        # rows of the round-trip below observe the real write (the box rows already round-trip
        # through the box mock's own backing tensor).
        self._bind_robot_states_to_write_tensor(adapter)

        # Test read - should concatenate robot + box states (2 objects x 2 envs = 4 states)
        nobjects = torch.arange(4)
        actual_states = adapter.get_states_by_index(nobjects)  # 2 objects x 2 envs = 4 states

        # Expected states in xyzw format (what the index path should return)
        expected_states = torch.cat([self.multi_robot_xyzw, self.multi_box_xyzw], dim=0)
        self._assert_full_state_equal(actual_states, expected_states, "Multi-object read")

        # Test write
        new_states = torch.zeros(4, 13, dtype=torch.float32)
        new_states[:, 6] = 1.0  # Set identity quaternions
        adapter.write_states_by_index(nobjects, new_states)

        # Read back. Both the robot rows (backed by the write tensor) and the box rows (own
        # backing tensor) now reflect what the write actually wrote, so a dropped write or a bad
        # conversion on either branch fails this assertion.
        actual_states_after_write = adapter.get_states_by_index(nobjects)
        self._assert_full_state_equal(actual_states_after_write, new_states, "Write verification")

    def test_reset_environments_via_index_path(self):
        """Multi-env, multi-object partial/batch reset through the adapter index path.

        Covers strided per-env subset writes, read-modify-write of pose-only columns, and the
        dirty-flag bookkeeping.
        """

        # Setup comprehensive multi-object scenario with scene objects (2 envs)
        num_boxes = 3
        box_configs = {f"box_{i}": self.multi_box_wxyz for i in range(num_boxes)}
        box_names = list(box_configs.keys())
        mock_robot_states = self._create_mock_robot_states(self.multi_robot_xyzw)
        multi_scene_wxyz = self.scene_wxyz.repeat(2, 1)  # Duplicate for 2 envs

        object_configs = [
            ("robot", {"robot": mock_robot_states}),
            ("scene", {"table": multi_scene_wxyz}),
            ("individual", box_configs),
        ]
        adapter = self._create_state_adapter(object_configs, self.multi_robot_wxyz, num_envs=2)
        registry = adapter._object_registry

        expected_total_objects = 10  # 1 robot + 1 table + 3 boxes, each in 2 envs
        assert len(registry.objects) * registry.num_envs == expected_total_objects, (
            f"Expected {expected_total_objects} total objects, got {len(registry.objects) * registry.num_envs}"
        )

        # Get initial poses from registry (covers get_initial_poses_batch)
        all_objects = ["table", "robot"] + box_names
        env_ids = torch.tensor([0, 1])
        initial_poses = registry.get_initial_poses_batch(all_objects, env_ids)

        # Verify initial poses shape: (5 objects * 2 envs, 7)
        assert initial_poses.shape == (len(all_objects) * 2, 7), (
            f"Expected ({len(all_objects) * 2}, 7) initial poses, got {initial_poses.shape}"
        )

        # Check get_object_indices
        box_indices = registry.get_object_indices(box_names, env_ids=None)
        assert box_indices.shape == (num_boxes * 2,), f"Expected ({num_boxes * 2},) indices, got {box_indices.shape}"

        # Capture original states for env_ids=0 before reset operations
        env_0_ids = torch.tensor([0])
        env_0_indices = registry.get_object_indices(["robot", "table"] + box_names, env_0_ids)
        original_env_0_states = adapter.get_states_by_index(env_0_indices).clone()

        # Reset specific environments (covers partial updates)
        reset_env_ids = torch.tensor([1])

        # Reset robot
        robot_reset_indices = registry.get_object_indices("robot", reset_env_ids)
        reset_state = torch.zeros(1, 13, dtype=torch.float32)
        reset_state[:, 6] = 1.0  # Identity quaternion
        adapter.write_states_by_index(robot_reset_indices, reset_state)

        # Reset scene and individual objects (read-modify-write pose-only columns)
        object_reset_indices = registry.get_object_indices(["table"] + box_names, reset_env_ids)
        current_object_states = adapter.get_states_by_index(object_reset_indices)
        current_object_states[:, 0:7] = initial_poses[object_reset_indices, 0:7]  # Update poses only
        adapter.write_states_by_index(object_reset_indices, current_object_states)

        # Verify robot reset worked
        reset_robot_state = adapter.get_states_by_index(robot_reset_indices)
        assert torch.allclose(reset_robot_state[:, 3:7], torch.tensor([[0.0, 0.0, 0.0, 1.0]])), (
            "Robot quaternion not reset to identity"
        )

        # Verify objects were reset to initial poses
        reset_object_states = adapter.get_states_by_index(object_reset_indices)
        assert torch.allclose(reset_object_states[:, 0:7], initial_poses[object_reset_indices, 0:7]), (
            "Objects not reset to initial poses"
        )

        # Verify env_ids=0 states remain unchanged during env_ids=1 reset
        current_env_0_states = adapter.get_states_by_index(env_0_indices)
        assert torch.allclose(current_env_0_states, original_env_0_states), (
            "Env 0 states should remain unchanged during env 1 reset"
        )

        # Verify that env_ids=0 states are different from env_ids=1 reset states
        # (confirms reset actually changed something)
        # Compare only the corresponding objects (table + boxes, excluding robot)
        env_0_object_indices = registry.get_object_indices(["table"] + box_names, env_0_ids)
        current_env_0_object_states = adapter.get_states_by_index(env_0_object_indices)
        assert not torch.allclose(current_env_0_object_states, reset_object_states), (
            "Env 0 states should be different from reset env 1 states"
        )

        # Verify dirty flag after reset operations
        assert adapter.is_dirty(), "Adapter should be dirty after reset writes"
        adapter.clear_dirty()
        assert not adapter.is_dirty(), "Adapter should be clean after clear_dirty"

    def test_get_initial_pose_methods(self):
        """Test ObjectRegistry get_initial_pose singular and batch."""

        # Create registry with *distinct* initial poses for validation
        test_registry = ObjectRegistry(device="cpu")
        test_registry.setup_ranges(num_envs=2, robot_count=1, scene_count=1, individual_count=1)

        # Register objects with DISTINCT initial poses
        robot_pose = torch.tensor(
            [[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0], [1.1, 2.1, 3.1, 0.0, 0.0, 0.0, 1.0]], dtype=torch.float32
        )
        table_pose = torch.tensor(
            [[4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 1.0], [4.1, 5.1, 6.1, 0.0, 0.0, 0.0, 1.0]], dtype=torch.float32
        )
        box_pose = torch.tensor(
            [[7.0, 8.0, 9.0, 0.0, 0.0, 0.0, 1.0], [7.1, 8.1, 9.1, 0.0, 0.0, 0.0, 1.0]], dtype=torch.float32
        )

        test_registry.register_object("robot", ObjectType.ROBOT, 0, robot_pose)
        test_registry.register_object("table", ObjectType.SCENE, 0, table_pose)
        test_registry.register_object("box", ObjectType.INDIVIDUAL, 0, box_pose)
        test_registry.finalize_registration()

        # Test get_initial_pose returns the CORRECT object's pose
        robot_single_pose = test_registry.get_initial_pose("robot", 0)
        expected_robot_pose = torch.tensor([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0], dtype=torch.float32)

        assert torch.allclose(robot_single_pose, expected_robot_pose), (
            f"get_initial_pose('robot', 0) returned wrong pose. "
            f"Expected: {expected_robot_pose}, Got: {robot_single_pose}"
        )

        # Test different environment
        robot_env1_pose = test_registry.get_initial_pose("robot", 1)
        expected_robot_env1_pose = torch.tensor([1.1, 2.1, 3.1, 0.0, 0.0, 0.0, 1.0], dtype=torch.float32)

        assert torch.allclose(robot_env1_pose, expected_robot_env1_pose), (
            f"get_initial_pose('robot', 1) returned wrong pose. "
            f"Expected: {expected_robot_env1_pose}, Got: {robot_env1_pose}"
        )

        # Validate object differentiation - should return different poses for different objects
        table_single_pose = test_registry.get_initial_pose("table", 0)
        box_single_pose = test_registry.get_initial_pose("box", 0)

        assert not torch.allclose(robot_single_pose, table_single_pose), (
            "get_initial_pose should return different poses for robot vs table"
        )
        assert not torch.allclose(robot_single_pose, box_single_pose), (
            "get_initial_pose should return different poses for robot vs box"
        )

        # Validate consistency with batch method
        batch_poses = test_registry.get_initial_poses_batch(["robot", "table", "box"], torch.tensor([0]))
        assert torch.allclose(robot_single_pose, batch_poses[0]), (
            "get_initial_pose should match get_initial_poses_batch for robot"
        )
        assert torch.allclose(table_single_pose, batch_poses[1]), (
            "get_initial_pose should match get_initial_poses_batch for table"
        )
        assert torch.allclose(box_single_pose, batch_poses[2]), (
            "get_initial_pose should match get_initial_poses_batch for box"
        )

    def test_full_range_gather(self):
        """Full-range gather over mixed object types."""

        # Setup multi-object scenario
        mock_robot_states = self._create_mock_robot_states(self.multi_robot_xyzw)
        object_configs = [
            ("robot", {"robot": mock_robot_states}),
            ("individual", {"box": self.multi_box_wxyz}),
            ("scene", {"table": self.scene_wxyz.repeat(2, 1)}),
        ]
        adapter = self._create_state_adapter(object_configs, self.multi_robot_wxyz, num_envs=2)
        registry = adapter._object_registry
        total = len(registry.objects) * registry.num_envs

        all_states = adapter.get_states_by_index(torch.arange(total))

        # Verify shape/device/dtype
        assert all_states.shape == (total, 13), f"Gather shape {all_states.shape} != ({total}, 13)"
        assert all_states.device == adapter.device, "Device should match adapter device"
        assert all_states.dtype == torch.float32, f"Expected float32 dtype, got {all_states.dtype}"

        # A fresh gather equals it (read is stable across calls)
        again = adapter.get_states_by_index(torch.arange(total))
        assert torch.allclose(all_states, again), "Repeated gather should match"

        # The gather is a fresh tensor (cat output), independent of subsequent reads
        all_states[0, 0] = 999.0  # mutate the returned tensor
        after = adapter.get_states_by_index(torch.arange(total))
        assert not torch.allclose(all_states[0], after[0]), "Returned gather must be a copy, not a live view"

    def test_error_recovery_and_edge_cases(self):
        """Test error handling in realistic scenarios."""

        adapter = self._create_state_adapter([("individual", {"box": self.box_wxyz})])

        # 1. Test graceful handling of partial failures
        # Mix valid and invalid indices -> registry resolve_indices raises out-of-range.
        mixed_indices = torch.tensor([0, 999])  # 0 valid, 999 invalid
        with pytest.raises((ValueError, KeyError), match=r"out of range|No objects found"):
            adapter.get_states_by_index(mixed_indices)

        # 2. Empty indices return an empty [0, 13] tensor.
        empty_indices = torch.tensor([], dtype=torch.long)
        empty_states = adapter.get_states_by_index(empty_indices)
        assert empty_states.shape == (0, 13), f"Expected (0, 13) empty states, got {empty_states.shape}"

        # 3. Test registry operations on empty registry
        empty_registry = ObjectRegistry(device="cpu")
        empty_registry.setup_ranges(num_envs=1, robot_count=0, scene_count=0, individual_count=0)
        empty_registry.finalize_registration()

        # Should return empty results, not crash
        empty_objects = empty_registry.list_all_objects()
        assert len(empty_objects) == 0, f"Expected empty object list, got {empty_objects}"

        empty_poses = empty_registry.get_initial_poses_batch([], torch.tensor([0]))
        assert empty_poses.shape == (0, 7), f"Expected (0, 7) empty poses, got {empty_poses.shape}"

        # 4. Test adapter with unknown object type error handling
        with pytest.raises(KeyError, match="not found"):
            adapter.get_object_states("nonexistent_object", torch.tensor([0]))

    def test_register_object_after_finalization_raises_error(self):
        """Test that registering objects after finalization raises RuntimeError."""
        registry = ObjectRegistry(device="cpu")
        registry.setup_ranges(num_envs=1, robot_count=1, scene_count=0, individual_count=0)

        # Register and finalize
        dummy_pose = torch.zeros(1, 7, dtype=torch.float32)
        dummy_pose[:, 6] = 1.0
        registry.register_object("robot", ObjectType.ROBOT, 0, dummy_pose)
        registry.finalize_registration()

        # Attempt to register after finalization should fail
        with pytest.raises(RuntimeError, match="ObjectRegistry already finalized, cannot register object 'box'"):
            registry.register_object("box", ObjectType.INDIVIDUAL, 0, dummy_pose)


if __name__ == "__main__":
    pytest.main([__file__])
