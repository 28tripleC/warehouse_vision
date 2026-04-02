import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
import time
import json
import os
from datetime import datetime

class ArucoDetector(Node):
    def __init__(self):
        super().__init__('aruco_detector')
        
        self.sub_img = self.create_subscription(
            Image, '/esp32_img', self.image_callback, 1)
        
        self.pub_img = self.create_publisher(Image, '/aruco_detected_img', 1)
        self.pub_markers = self.create_publisher(MarkerArray, '/inventory_markers', 1)

        self.bridge = CvBridge()

        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters_create()

        calib_file = os.path.expanduser('~/camera_calibration/calib_data.npz')
        if os.path.exists(calib_file):
            data = np.load(calib_file)
            self.camera_matrix = data['camera_matrix']
            self.dist_coeffs = data['dist_coeffs']
            self.get_logger().info("Loaded camera calibration from file")
        else:
            self.get_logger().warning("Calibration file not found, using default parameters")
            self.camera_matrix = np.array([
                [400, 0, 320],
                [0, 400, 240],
                [0, 0, 1]
            ], dtype=np.float64)
            self.dist_coeffs = np.zeros((4,1), dtype=np.float64)

        self.marker_size = 0.08

        self.inventory = {}
        self.log_dir = os.path.expanduser('~/warehouse_log')
        os.makedirs(self.log_dir, exist_ok=True)

        self.item_names = {
            0: "Package_A",
            1: "Package_B",
            2: "Package_C",
            3: "Package_D",
            4: "Package_E",
            5: "Package_F",
            6: "Package_G",
            7: "Package_H",
            8: "Package_I",
            9: "Package_J",
        }

        self.get_logger().info("ArUco Detector started, waiting for images...")
    
    def image_callback(self, msg):
        start_time = time.time()
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, rejected = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            for i in range(len(ids)):
                marker_id = ids[i][0]
                corner = corners[i][0]

                obj_points = np.array([
                    [-self.marker_size/2, self.marker_size/2, 0],
                    [self.marker_size/2, self.marker_size/2, 0],
                    [self.marker_size/2, -self.marker_size/2, 0],
                    [-self.marker_size/2, -self.marker_size/2, 0]
                ], dtype=np.float64)

                success, rvec, tvec = cv2.solvePnP(obj_points, corner, self.camera_matrix, self.dist_coeffs)

                if success:
                    cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs, rvec, tvec, 0.05)

                    distance = np.linalg.norm(tvec)
                    item_name = self.item_names.get(marker_id, f"Unknown_{marker_id}")

                    cx = int(corner[:, 0].mean())
                    cy = int(corner[:, 1].mean())
                    cv2.putText(frame, f"ID:{marker_id} {item_name}",
                                (cx - 60, cy - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.putText(frame, f"Dist:{distance:.2f}m",
                                (cx - 60, cy + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    self.log_detection(marker_id, item_name, tvec, distance)
                    self.get_logger().info(f"Detected {item_name} (ID: {marker_id}) at distance {distance:.2f}m")
                
        img_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        self.pub_img.publish(img_msg)

        self.publish_markers()
        
        cv2.imshow("ArUco Detection", frame)
        cv2.waitKey(1)

        end_time = time.time()
        self.get_logger().info(f"Processing time: {(end_time - start_time)*1000:.2f} ms")

    def log_detection(self, marker_id, item_name, tvec, distance):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.inventory[int(marker_id)] = {
            "item_name": item_name,
            "marker_id": int(marker_id),
            "x": float(tvec[0][0]),
            "y": float(tvec[1][0]),
            "z": float(tvec[2][0]),
            "distance": float(distance),
            "last_seen": now
        }
    
    def publish_markers(self):
        marker_array = MarkerArray()
        for marker_id, data in self.inventory.items():
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "inventory"
            marker.id = int(marker_id)
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position.x = data["x"]
            marker.pose.position.y = data["y"]
            marker.pose.position.z = 0.1
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 1.0
            marker.lifetime.sec = 30
            marker_array.markers.append(marker)
        self.pub_markers.publish(marker_array)
    
    def save_inventory_report(self):
        report_path = os.path.join(
            self.log_dir,
            f"inventory_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(report_path, 'w') as f:
            json.dump(self.inventory, f, indent=2, ensure_ascii=False)
        self.get_logger().info(f"Inventory report saved to {report_path}")

    def destroy_node(self):
        self.save_inventory_report()
        cv2.destroyAllWindows()
        super().destroy_node()
    
def main(args=None):
    rclpy.init(args=args)
    node = ArucoDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
