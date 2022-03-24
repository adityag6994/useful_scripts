//Aditya Gupta


// Code for image registration given image and depth map
// There are 4 differnt methods which can be tried
// 0 : RegistrationRANSACBasedOnFeatureMatching (default)
// 1 : RegistrationRANSACBasedOnCorrespondence
// 2 : FastGlobalRegistrationBasedOnFeatureMatching
// 3 : RegistrationGeneralizedICP
// output's merged version of target and query point cloud

#include <Eigen/Dense>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include "open3d/Open3D.h"
#include "open3d/core/Tensor.h"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "open3d/Open3D.h"

using namespace open3d;
using namespace cv;


/// Function which reads image path and does following
/// Step1 : downsample based on voxel_size
/// Step2 : extracts normal from downsampled point cloud
/// Step3 : extract FPFHFeature from point cloud from Step 2

std::tuple<std::shared_ptr<geometry::PointCloud>,
           std::shared_ptr<geometry::PointCloud>,
           std::shared_ptr<pipelines::registration::Feature>>
CreatePointCloud(const char *file_name,
                 const float voxel_size,
                 const camera::PinholeCameraIntrinsic intrinsic) {
    std::string img_path(file_name);
    img_path += ".jpeg";
    std::string dep_path(file_name);
    dep_path += ".depth.tiff";

    // error handling for paths
    if(!utility::filesystem::FileExists(img_path)){
        throw std::invalid_argument("Something wrong with Image Path!");
    }
    if(!utility::filesystem::FileExists(dep_path)){
        throw std::invalid_argument("Something wrong with Depth Path!");
    }
    auto color = io::CreateImageFromFile(img_path);
    Mat depth_cv = imread(dep_path, cv::IMREAD_UNCHANGED);

    // Note: rotate depth map for target, as it's inverted 90 ACW
    if(depth_cv.cols == 256){
        cv::rotate(depth_cv, depth_cv, cv::ROTATE_90_CLOCKWISE);
    }

    // Insert data from opencv format to open3d format
    int width  = color->width_;
    int height = color->height_;
    Mat depth_cv_resize;
    // resize to image size
    cv::resize(depth_cv, depth_cv_resize, Size(width, height), CV_INTER_AREA);
    geometry::Image image;
    image.Prepare(width, height, 1, 4);//(width, height, num_of_channels, bytes_per_channel)
    float* data_ptr = reinterpret_cast<float*>(image.data_.data());
    for(int i=0; i<width*height; i++){
        data_ptr[i] = depth_cv_resize.at<float>(i/width,i%width);
    }
	
    // Create point cloud from RGB image, depth map and instrinsic matrix
    auto rgbd = geometry::RGBDImage::CreateFromColorAndDepth(*color, image,1,3);
    auto pcd = geometry::PointCloud::CreateFromRGBDImage(*rgbd, intrinsic);
    auto pcd_down = pcd->VoxelDownSample(voxel_size);
    // Extract Normals
    pcd_down->EstimateNormals(open3d::geometry::KDTreeSearchParamHybrid(2 * voxel_size, 30));
    // Extract Features
    auto pcd_fpfh = pipelines::registration::ComputeFPFHFeature(*pcd_down,
                                                                open3d::geometry::KDTreeSearchParamHybrid(5 * voxel_size, 100));

    return std::make_tuple(pcd, pcd_down, pcd_fpfh);
}

/// Function to visualise results, given 2 point clouds and transformation matrix
void VisualizeRegistration(const open3d::geometry::PointCloud &source,
                           const open3d::geometry::PointCloud &target,
                           const Eigen::Matrix4d &Transformation) {
    std::shared_ptr<geometry::PointCloud> source_transformed_ptr(
            new geometry::PointCloud);
    std::shared_ptr<geometry::PointCloud> target_ptr(new geometry::PointCloud);
    *source_transformed_ptr = source;
    *target_ptr = target;
    source_transformed_ptr->Transform(Transformation);
    visualization::DrawGeometries({source_transformed_ptr, target_ptr},
                                  "Registration result");
}

void PrintHelp() {
    using namespace open3d;
    utility::LogInfo("\n"
                     "|------------------------------------------------------------\n"
                     "|target and query folder must have image and depth with same name : \n"
                     "|eg : image_match --target folder1/good1 --query folder2/target \n"
                     "|folder must have these files => folder1/good1.jpeg, folder1/good1.depth.tiff \n"
                     "|folder must have these files => folder2/target.jpeg, folder2/target.depth.tiff ");

    utility::LogInfo("\n"
                     "|image_match --target <path/to/target/image> --query <path/to/query/image>\n"
                     "|--camera <path/to/query/image>{in open3D format}\n"
                     "|--method=feature_matching \n"
                     "|--voxel_size=0.01 \n"
                     "|--distance_multiplier=1.5 \n"
                     "|--max_iterations 1000000 \n"
                     "|--confidence 0.999 \n"
                     "|--mutual_filter False");
}

int main(int argc, char *argv[]) {
    using namespace open3d;
    using namespace cv;

    utility::SetVerbosityLevel(utility::VerbosityLevel::Debug);

    if (argc < 3 ||
        utility::ProgramOptionExistsAny(argc, argv, {"-h", "--help"})) {
        PrintHelp();
        return 1;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////
    // Set method
    // 0 : RegistrationRANSACBasedOnFeatureMatching
    // 1 : RegistrationRANSACBasedOnCorrespondence
    // 2 : FastGlobalRegistrationBasedOnFeatureMatching
    // 3 : RegistrationGeneralizedICP
    int method = utility::GetProgramOptionAsInt(argc, argv, "--method", 0);;

    bool mutual_filter = false;
    if (utility::ProgramOptionExists(argc, argv, "--mutual_filter")) {
        mutual_filter = true;
    }
    float voxel_size =
            utility::GetProgramOptionAsDouble(argc, argv, "--voxel_size", 0.05);
    float distance_multiplier = utility::GetProgramOptionAsDouble(
            argc, argv, "--distance_multiplier", 1.5);
    float distance_threshold = voxel_size * distance_multiplier;
    int max_iterations = utility::GetProgramOptionAsInt(
            argc, argv, "--max_iterations", 1000000);
    float confidence = utility::GetProgramOptionAsDouble(argc, argv,
                                                         "--confidence", 0.7);

    std::string intrinsic_path = "../camera.json";
    camera::PinholeCameraIntrinsic intrinsic;
    if (intrinsic_path.empty() ||
        !io::ReadIJsonConvertible(intrinsic_path, intrinsic)) {
        utility::LogWarning(
                "Failed to read intrinsic parameters for depth image.");
        utility::LogWarning("Using default value for Primesense camera.");
        intrinsic = camera::PinholeCameraIntrinsic(
                camera::PinholeCameraIntrinsicParameters::PrimeSenseDefault);
    }

    ///////////////////////////////////////////////////////////////////////////////////////////
    // Prepare input
    std::shared_ptr<geometry::PointCloud> source_pc, source_down, target_pc, target_down;
    std::shared_ptr<pipelines::registration::Feature> source_fpfh, target_fpfh;
    try {
        std::tie(source_pc, source_down, source_fpfh) = CreatePointCloud(argv[1], voxel_size, intrinsic);
        std::tie(target_pc, target_down, target_fpfh) = CreatePointCloud(argv[2], voxel_size, intrinsic);
    } catch (std::invalid_argument& e) {
        std::cerr << e.what() << std::endl;
        return -1;
    }

    pipelines::registration::RegistrationResult registration_result;
    std::vector<std::reference_wrapper<
            const pipelines::registration::CorrespondenceChecker>>
            correspondence_checker;
    auto correspondence_checker_edge_length =
            pipelines::registration::CorrespondenceCheckerBasedOnEdgeLength(
                    0.9);
    auto correspondence_checker_distance =
            pipelines::registration::CorrespondenceCheckerBasedOnDistance(
                    distance_threshold);
    correspondence_checker.push_back(correspondence_checker_edge_length);
    correspondence_checker.push_back(correspondence_checker_distance);
    //////////////////////////////////////////////////////////////////////////////////////
    // Apply registration methods on estimated point cloud
    if(method == 0){
    	// Method 0
        std::cout << "|| Method : RegistrationRANSACBasedOnFeatureMatching" << std::endl;
        registration_result = pipelines::registration::
                RegistrationRANSACBasedOnFeatureMatching(
                        *source_down,
                        *target_down,
                        *source_fpfh,
                        *target_fpfh,
                        false,
                        distance_threshold,
                        pipelines::registration::TransformationEstimationPointToPoint(false),
                        3,
                        correspondence_checker,
                        pipelines::registration::RANSACConvergenceCriteria(max_iterations, confidence));
    }else if(method == 1) {
	// Method 1
        std::cout << "|| Method : RegistrationRANSACBasedOnCorrespondence" << std::endl;
        // Manually search correspondences
        int nPti = int(source_down->points_.size());
        int nPtj = int(target_down->points_.size());

        geometry::KDTreeFlann feature_tree_i(*source_fpfh);
        geometry::KDTreeFlann feature_tree_j(*target_fpfh);

        pipelines::registration::CorrespondenceSet corres_ji;
        std::vector<int> i_to_j(nPti, -1);

        // Buffer all correspondences
        for (int j = 0; j < nPtj; j++) {
            std::vector<int> corres_tmp(1);
            std::vector<double> dist_tmp(1);

            feature_tree_i.SearchKNN(Eigen::VectorXd(target_fpfh->data_.col(j)),
                                     1, corres_tmp, dist_tmp);
            int i = corres_tmp[0];
            corres_ji.push_back(Eigen::Vector2i(i, j));
        }

        if (mutual_filter) {
            pipelines::registration::CorrespondenceSet mutual_corres;
            for (auto &corres : corres_ji) {
                int j = corres(1);
                int j2i = corres(0);

                std::vector<int> corres_tmp(1);
                std::vector<double> dist_tmp(1);
                feature_tree_j.SearchKNN(
                        Eigen::VectorXd(source_fpfh->data_.col(j2i)), 1,
                        corres_tmp, dist_tmp);
                int i2j = corres_tmp[0];
                if (i2j == j) {
                    mutual_corres.push_back(corres);
                }
            }

            //utility::LogDebug("{:d} points remain after mutual filter", mutual_corres.size());
            registration_result = pipelines::registration::
                    RegistrationRANSACBasedOnCorrespondence(
                            *source_down, *target_down, mutual_corres,
                            distance_threshold,
                            pipelines::registration::
                                    TransformationEstimationPointToPoint(false),
                            3, correspondence_checker,
                            pipelines::registration::RANSACConvergenceCriteria(
                                    max_iterations, confidence));
        } else {
            //utility::LogDebug("{:d} points remain", corres_ji.size());
            registration_result = pipelines::registration::
                    RegistrationRANSACBasedOnCorrespondence(
                            *source_down, *target_down, corres_ji,
                            distance_threshold,
                            pipelines::registration::
                                    TransformationEstimationPointToPoint(false),
                            3, correspondence_checker,
                            pipelines::registration::RANSACConvergenceCriteria(
                                    max_iterations, confidence));
        }
    }else if(method == 2){
        // Method 2
        std::cout << "|| Method : FastGlobalRegistrationBasedOnFeatureMatching" << std::endl;
        registration_result = pipelines::registration::
                        FastGlobalRegistrationBasedOnFeatureMatching(
                                *source_down,
                                *target_down,
                                *source_fpfh,
                                *target_fpfh,
                                pipelines::registration::FastGlobalRegistrationOption(
                                                /* decrease_mu =  */ 1.4, true,
                                                true, distance_threshold,
                                                64,
                                                /* tuple_scale =  */ 0.95,
                                                1000));
    }else if(method == 3){
    	// Method 3
	std::cout << "|| Method : RegistrationGeneralizedICP" << std::endl;
        Eigen::Matrix4d trans = Eigen::Matrix4d::Identity();
        registration_result = pipelines::registration::
                RegistrationGeneralizedICP(
                    *source_down,
                    *target_down,
                    0.07,
                    trans,
                    pipelines::registration::
                            TransformationEstimationForGeneralizedICP(),
                    pipelines::registration::ICPConvergenceCriteria(1e-6, 1e-6,
                                                                    30));
    }else{
        std::cout << "|| Check method name " << std::endl;
    }

    //////////////////////////////////////////////////////////////////////////////////////
    // print result
    std::cout << "|| correspondence_set_ size : " << registration_result.correspondence_set_.size() << std::endl;
    std::cout << "|| rmse : " << registration_result.inlier_rmse_ << std::endl;
    std::cout << "|| fitness : " << registration_result.fitness_ << std::endl;

    // check if there is sufficient match, 0.6 is emperically calculated
    if(registration_result.fitness_ < 0.6){
        std::cout << "|| ==> Mismatch, not enough points to match target and source" << std::endl;
        return 0;
    }
    std::cout << registration_result.transformation_ << std::endl;
    //////////////////////////////////////////////////////////////////////////////////////
    // visulize result     	
    VisualizeRegistration(*source_pc,
                          *target_pc,
                          registration_result.transformation_);
    //////////////////////////////////////////////////////////////////////////////////////
    return 0;
}
