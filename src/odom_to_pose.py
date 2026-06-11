from nav_msgs.msg import Odometry

class PoseReader(Node):
    def __init__(self):
        super().__init__('pose_reader')
        self.create_subscription(
            Odometry,
            '/odometry/filtered',
            self.cb,
            10
        )

    def cb(self, msg: Odometry):
        # Posición
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        z = msg.pose.pose.position.z

        # Orientación (quaternion)
        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w

        # Covarianza de pose (matriz 6x6 aplanada, orden x,y,z,roll,pitch,yaw)
        cov = msg.pose.covariance   # lista de 36 elementos
        var_x   = cov[0]
        var_y   = cov[7]
        var_z   = cov[14]
        var_yaw = cov[35]