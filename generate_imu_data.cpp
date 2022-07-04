#include <Eigen/Core>
#include <basalt/spline/se3_spline.h>
#include <fstream>
#include <iostream>
#include <json.hpp>

// Parameters for simulations
int NUM_POINTS = 1000;
double POINT_DIST = 10.0;
constexpr int NUM_FRAMES = 1000;
constexpr int CAM_FREQ = 20;
constexpr int IMU_FREQ = 500;

static const int knot_time = 3;
constexpr int64_t time_interval_ns = knot_time * 1e9;
static const double obs_std_dev = 0.5;

Eigen::Vector3d g(0, 0, -9.81);
basalt::Se3Spline<5> gt_spline(time_interval_ns);

void gen_data() {

  constexpr double seconds = NUM_FRAMES / CAM_FREQ;
  constexpr int num_knots = seconds / knot_time + 5;
  gt_spline.genRandomTrajectory(num_knots, true);

  int64_t t_ns = 0;

  int64_t dt_ns = int64_t(1e9) / IMU_FREQ;
  std::cout << dt_ns << std::endl;
  // return;

  int64_t offset =
      dt_ns / 2; // Offset to make IMU in the center of the interval
  t_ns = offset;
  int count = 0;

  nlohmann::ordered_json oj;

  std::vector<std::string> vd_name = {"qx", "qy", "qz", "qw", "px", "py", "pz"};
  // for (int i = 0; i < cam_poses[cam].size(); ++i) {
  // }

  while (t_ns < int64_t(seconds * 1e9)) {
    Sophus::SE3d pose = gt_spline.pose(t_ns);
    nlohmann::ordered_json temp_j;
    temp_j["time_stamp"] = t_ns;
    std::vector<double> vd(pose.data(), pose.data() + 7);
    for (int p = 0; p < vd_name.size(); ++p) {
      temp_j[vd_name[p]] = vd[p];
    }

    Eigen::Vector3d accel_body =
        pose.so3().inverse() * (gt_spline.transAccelWorld(t_ns) - g);
    temp_j["ax"] = accel_body.x();
    temp_j["ay"] = accel_body.y();
    temp_j["az"] = accel_body.z();
    Eigen::Vector3d rot_vel_body = gt_spline.rotVelBody(t_ns);
    temp_j["rx"] = rot_vel_body.x();
    temp_j["ry"] = rot_vel_body.y();
    temp_j["rz"] = rot_vel_body.z();

    oj[std::to_string(count)] = temp_j;
    std::cout << count++ << std::endl;

    // gt_accel.emplace_back(accel_body);
    // gt_gyro.emplace_back(rot_vel_body);
    // gt_vel.emplace_back(gt_spline.transVelWorld(t_ns));

    // accel_body[0] += accel_noise_dist(gen);
    // accel_body[1] += accel_noise_dist(gen);
    // accel_body[2] += accel_noise_dist(gen);

    // accel_body += gt_accel_bias.back();

    // rot_vel_body[0] += gyro_noise_dist(gen);
    // rot_vel_body[1] += gyro_noise_dist(gen);
    // rot_vel_body[2] += gyro_noise_dist(gen);

    // rot_vel_body += gt_gyro_bias.back();

    // noisy_accel.emplace_back(accel_body);
    // noisy_gyro.emplace_back(rot_vel_body);

    // gt_imu_t_ns.emplace_back(t_ns + offset);

    t_ns += dt_ns;
  }

  std::ofstream o("imu_data.json");
  o << std::setw(4) << oj;
  o.close();
}

int main(int, char **) { gen_data(); }
