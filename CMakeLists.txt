cmake_minimum_required(VERSION 3.8)
project(tartan_to_ros)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ── Dependencias ──────────────────────────────────────────────────────────────
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(cv_bridge REQUIRED)

# ── Scripts ejecutables ───────────────────────────────────────────────────────
install(
  PROGRAMS
    src/map_tartanground.py
    src/tartanground_debug.py
  DESTINATION lib/${PROJECT_NAME}
)

# ── Launch (destino correcto: share/<pkg>/launch/) ────────────────────────────
install(
  DIRECTORY launch/ rviz/
  DESTINATION share/${PROJECT_NAME}
)

install(
  FILES config/ekf_tartanground.yaml 
  DESTINATION share/${PROJECT_NAME}/config
)

install(
  FILES rviz/TartanGroundRVIZ.rviz
  DESTINATION share/${PROJECT_NAME}/rviz
)
# ── Tests ─────────────────────────────────────────────────────────────────────
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
