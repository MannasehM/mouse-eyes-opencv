import cv2
import numpy as np

class HeadPose:
    def __init__(self, model_points, landmark_ids, camera_matrix, dist_coeffs):
        self.model_points = model_points      # 3D reference points
        self.landmark_ids = landmark_ids      # Corresponding 2D points in image
        self.camera_matrix = camera_matrix    # Intrinsic camera parameters
        self.dist_coeffs = dist_coeffs        # Lens distortion coefficients

    def get_head_pose(self, landmarks, image):
        """
        Compute rotation and translation vectors using solvePnP.
        Returns rotation vector, translation vector, and 2D projected points.
        """
        points_2D = []

        for idx in self.landmark_ids:
            lm = landmarks[idx]
            x = int(lm.x * image.shape[1])
            y = int(lm.y * image.shape[0])
            points_2D.append((x, y))
            cv2.circle(image, (x, y), 3, (0, 0, 255))  # Draw 2D landmark points

        points_2D = np.array(points_2D, dtype="double")

        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.model_points, points_2D,
            self.camera_matrix, self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return None, None, None

        return rotation_vector, translation_vector, points_2D

    def draw_nose_line(self, rotation_vector, translation_vector, image, points_2D):
        """
        Draws a line from the nose tip forward to indicate head direction.
        Returns 2D endpoints (p1, p2) for mouse control.
        """
        nose_end_point3D = np.array([[0, 0, 1000.0]])
        nose_end_point2D, _ = cv2.projectPoints(
            nose_end_point3D, rotation_vector, translation_vector,
            self.camera_matrix, self.dist_coeffs
        )

        p1 = (int(points_2D[0][0]), int(points_2D[0][1]))
        p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
        cv2.line(image, p1, p2, (255, 0, 0), 2)  # Draw nose direction line
        return p1, p2