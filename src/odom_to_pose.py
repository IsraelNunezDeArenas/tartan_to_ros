from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped

class PoseReader(Node):
    def __init__(self):
        super().__init__('pose_reader')
        self.create_subscription(
            Odometry,
            '/odometry/filtered',
            self.cb,
            10
        )
        self.pub = self.create_publisher(
            PoseWithCovarianceStamped,
            '/pose/filtered',
            10
        )

    def cb(self, msg: Odometry):
        out = PoseWithCovarianceStamped()

        # Mismo timestamp y frame que el EKF
        out.header.stamp    = msg.header.stamp
        out.header.frame_id = msg.header.frame_id

        # Pose
        out.pose.pose = msg.pose.pose

        # Covarianza completa (6x6 → 36 elementos)
        out.pose.covariance = msg.pose.covariance

        self.pub.publish(out)