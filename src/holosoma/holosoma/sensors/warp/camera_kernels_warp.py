import warp as wp

NO_HIT_RAY_VAL = wp.constant(1000.0)


class DepthCameraWarpKernels:
    """Warp kernels for static and articulated-scene depth rendering."""

    @staticmethod
    @wp.kernel
    def draw_optimized_kernel_pointcloud(
        terrain_mesh_id: wp.uint64,
        cam_poss: wp.array(dtype=wp.vec3, ndim=2),  # type: ignore[valid-type]
        cam_quats: wp.array(dtype=wp.quat, ndim=2),  # type: ignore[valid-type]
        k_inv: wp.mat44,
        far_plane: float,
        pixels: wp.array(dtype=wp.vec3, ndim=4),  # type: ignore[valid-type]
        pointcloud_in_world_frame: bool,
    ) -> None:
        # wp.tid() returns a 4-tuple for a 4D launch, but Warp's stub types a 0-arg call as a scalar int,
        # so mypy cannot type the unpacked indices. Comment-only ignores keep this valid in the Warp DSL
        # (which rejects cast()).
        env_id, cam_id, x, y = wp.tid()  # type: ignore[misc]
        cam_pos = cam_poss[env_id, cam_id]  # type: ignore[has-type]
        cam_quat = cam_quats[env_id, cam_id]  # type: ignore[has-type]
        cam_coords = wp.vec3(float(x), float(y), 1.0)  # type: ignore[has-type]
        uv = wp.normalize(wp.transform_vector(k_inv, cam_coords))
        ray_direction = wp.normalize(wp.quat_rotate(cam_quat, uv))

        # These explicit constructors produce mutable Warp scalars required by
        # mesh_query_ray's output parameters. Bare literals are constants.
        distance = float(0.0)
        u = float(0.0)
        v = float(0.0)
        sign = float(0.0)
        normal = wp.vec3()
        face = int(0)
        if not wp.mesh_query_ray(  # type: ignore[call-arg, truthy-bool]
            terrain_mesh_id,
            cam_pos,
            ray_direction,
            far_plane,  # type: ignore[arg-type]
            distance,
            u,
            v,
            sign,
            normal,
            face,
        ):
            distance = NO_HIT_RAY_VAL

        if pointcloud_in_world_frame:
            pixels[env_id, cam_id, y, x] = cam_pos + distance * ray_direction  # type: ignore[has-type]
        else:
            pixels[env_id, cam_id, y, x] = distance * uv  # type: ignore[has-type]

    @staticmethod
    @wp.kernel
    def draw_optimized_kernel_depth_range(
        terrain_mesh_id: wp.uint64,
        cam_poss: wp.array(dtype=wp.vec3, ndim=2),  # type: ignore[valid-type]
        cam_quats: wp.array(dtype=wp.quat, ndim=2),  # type: ignore[valid-type]
        k_inv: wp.mat44,
        far_plane: float,
        pixels: wp.array(dtype=wp.float32, ndim=4),  # type: ignore[valid-type]
        c_x: int,
        c_y: int,
        calculate_depth: bool,
    ) -> None:
        # wp.tid() returns a 4-tuple for a 4D launch, but Warp's stub types a 0-arg call as a scalar int,
        # so mypy cannot type the unpacked indices. Comment-only ignores keep this valid in the Warp DSL
        # (which rejects cast()).
        env_id, cam_id, x, y = wp.tid()  # type: ignore[misc]
        cam_pos = cam_poss[env_id, cam_id]  # type: ignore[has-type]
        cam_quat = cam_quats[env_id, cam_id]  # type: ignore[has-type]
        uv = wp.transform_vector(k_inv, wp.vec3(float(x), float(y), 1.0))  # type: ignore[has-type]
        uv_principal = wp.transform_vector(k_inv, wp.vec3(float(c_x), float(c_y), 1.0))
        ray_direction = wp.normalize(wp.quat_rotate(cam_quat, uv))
        principal_direction = wp.normalize(wp.quat_rotate(cam_quat, uv_principal))

        multiplier = wp.float32(1.0)
        if calculate_depth:
            multiplier = wp.min(
                wp.max(wp.dot(ray_direction, principal_direction), wp.float32(1.0e-6)),
                wp.float32(1.0),
            )

        distance = float(0.0)
        u = float(0.0)
        v = float(0.0)
        sign = float(0.0)
        normal = wp.vec3()
        face = int(0)
        depth = wp.float32(NO_HIT_RAY_VAL)
        if wp.mesh_query_ray(  # type: ignore[call-arg, truthy-bool]
            terrain_mesh_id,
            cam_pos,
            ray_direction,
            far_plane / multiplier,
            distance,
            u,
            v,
            sign,
            normal,
            face,
        ):
            depth = multiplier * distance
        pixels[env_id, cam_id, y, x] = depth  # type: ignore[has-type]

    @staticmethod
    @wp.kernel
    def draw_optimized_kernel_depth_range_dynamic(
        terrain_id: wp.uint64,
        robot_ids: wp.array(dtype=wp.uint64),  # type: ignore[valid-type]
        body_poss: wp.array(dtype=wp.vec3, ndim=2),  # type: ignore[valid-type]
        body_quats: wp.array(dtype=wp.quat, ndim=2),  # type: ignore[valid-type]
        cam_poss: wp.array(dtype=wp.vec3, ndim=2),  # type: ignore[valid-type]
        cam_quats: wp.array(dtype=wp.quat, ndim=2),  # type: ignore[valid-type]
        k_inv: wp.mat44,
        far_plane: float,
        pixels: wp.array(dtype=wp.float32, ndim=4),  # type: ignore[valid-type]
        c_x: int,
        c_y: int,
        calculate_depth: bool,
        num_bodies: int,
    ) -> None:
        # wp.tid() returns a 4-tuple for a 4D launch, but Warp's stub types a 0-arg call as a scalar int,
        # so mypy cannot type the unpacked indices. Comment-only ignores keep this valid in the Warp DSL
        # (which rejects cast()).
        env_id, cam_id, x, y = wp.tid()  # type: ignore[misc]
        cam_pos = cam_poss[env_id, cam_id]  # type: ignore[has-type]
        cam_quat = cam_quats[env_id, cam_id]  # type: ignore[has-type]
        uv = wp.transform_vector(k_inv, wp.vec3(float(x), float(y), 1.0))  # type: ignore[has-type]
        uv_principal = wp.transform_vector(k_inv, wp.vec3(float(c_x), float(c_y), 1.0))
        ray_direction = wp.normalize(wp.quat_rotate(cam_quat, uv))
        principal_direction = wp.normalize(wp.quat_rotate(cam_quat, uv_principal))

        multiplier = wp.float32(1.0)
        if calculate_depth:
            multiplier = wp.min(
                wp.max(wp.dot(ray_direction, principal_direction), wp.float32(1.0e-6)),
                wp.float32(1.0),
            )

        best_depth = wp.float32(NO_HIT_RAY_VAL)
        far_bound = wp.float32(far_plane)

        for body_index in range(num_bodies):
            body_quat = body_quats[env_id, body_index]  # type: ignore[has-type]
            body_pos = body_poss[env_id, body_index]  # type: ignore[has-type]
            local_origin = wp.quat_rotate_inv(body_quat, cam_pos - body_pos)
            local_direction = wp.quat_rotate_inv(body_quat, ray_direction)

            distance = float(0.0)
            u = float(0.0)
            v = float(0.0)
            sign = float(0.0)
            normal = wp.vec3()
            face = int(0)
            if wp.mesh_query_ray(  # type: ignore[call-arg, truthy-bool]
                robot_ids[body_index],
                local_origin,
                local_direction,
                far_bound / multiplier,
                distance,
                u,
                v,
                sign,
                normal,
                face,
            ):
                depth = multiplier * distance
                if best_depth == NO_HIT_RAY_VAL or depth < best_depth:
                    best_depth = depth
                    far_bound = depth

        distance = float(0.0)
        u = float(0.0)
        v = float(0.0)
        sign = float(0.0)
        normal = wp.vec3()
        face = int(0)
        if wp.mesh_query_ray(  # type: ignore[call-arg, truthy-bool]
            terrain_id,
            cam_pos,
            ray_direction,
            far_bound / multiplier,
            distance,
            u,
            v,
            sign,
            normal,
            face,
        ):
            best_depth = multiplier * distance

        pixels[env_id, cam_id, y, x] = best_depth  # type: ignore[has-type]
