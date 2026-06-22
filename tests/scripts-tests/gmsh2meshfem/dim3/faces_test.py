import numpy as np

from gmsh2meshfem.dim3.model.faces import vectorized_bbox_calc


def test_bbox_singletons():
    ref_face = np.array(
        [
            [-1, -1, 0],
            [1, -1, 0],
            [1, 1, 0],
            [-1, 1, 0],
            [0, -1, 0],
            [1, 0, 0],
            [0, 1, 0],
            [-1, 0, 0],
            [0, 0, 0],
        ],
        dtype=float,
    )

    def verify_by_extrema_of_nodes(face, context_string: str):
        bbox = vectorized_bbox_calc(face)
        assert bbox.shape == (6,), (
            f"({context_string}) Single element should give only 1 set of bounds"
        )
        low = np.min(face, axis=-2)
        high = np.max(face, axis=-2)

        bbox_by_nodes = np.concatenate((low, high), axis=-1)

        face_arr_recov_rows = [
            "[" + ",".join(f"{v:.10f}" for v in row) + "]" for row in face
        ]

        np.testing.assert_array_almost_equal(
            bbox,
            bbox_by_nodes,
            err_msg=f"{context_string}.\n"
            f"input array: np.array([{','.join(face_arr_recov_rows)}])",
        )

    def rot2d(theta):
        return np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )

    def euler_rotation_matrix(a, b, c):
        combined = np.eye(3)
        combined[1:, 1:] = rot2d(a)

        mat = np.eye(3)
        mat[::2, ::2] = rot2d(b)
        combined @= mat

        mat[:, :] = np.eye(3)
        mat[:-1, :-1] = rot2d(c)
        combined @= mat

        return combined

    def test_for_rotations(face, element_ref_string):
        verify_by_extrema_of_nodes(face, f"[{element_ref_string}]")
        verify_by_extrema_of_nodes(
            face @ euler_rotation_matrix(0, 0, np.pi / 4),
            f"[{element_ref_string}] z-rot 45°",
        )
        verify_by_extrema_of_nodes(
            face @ euler_rotation_matrix(0, np.pi / 4, 0),
            f"[{element_ref_string}] y-rot 45°",
        )
        verify_by_extrema_of_nodes(
            face @ euler_rotation_matrix(np.pi / 4, 0, 0),
            f"[{element_ref_string}] x-rot 45°",
        )
        verify_by_extrema_of_nodes(
            face @ euler_rotation_matrix(np.pi / 4, 0, -np.pi / 4),
            f"[{element_ref_string}] x-rot 45° * z-rot -45°",
        )

    test_for_rotations(ref_face, "reference element")
    face = ref_face.copy()
    face[4, 1] = -1
    test_for_rotations(face, "ref - middle-front nudge -y")
    face = ref_face.copy()
    face[4, 2] = 0.1
    test_for_rotations(face, "ref - middle-front nudge +z")
    face = ref_face.copy()
    face[5, 1] = -0.1
    test_for_rotations(face, "ref - middle-right nudge -y")
    face = ref_face.copy()
    face[5, 2] = 0.1
    test_for_rotations(face, "ref - middle-right nudge +z")
