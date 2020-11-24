#include "igl/opengl/glfw/Viewer.h"
#include "igl/readOBJ.h"
#include "igl/unproject.h"
#include "igl/unproject_onto_mesh.h"
#include "ascent/Ascent.h"

static Eigen::Vector3d pos_start{0, 0, 0};
static Eigen::Vector3d pos_end;
static double brush_strength;

static Eigen::MatrixXd V_input;
static Eigen::MatrixXd V_deformed;
static Eigen::MatrixXi F_input;
static Eigen::MatrixXi F_deformed;
static Eigen::MatrixXd result;

/**
 * NOTE(brendan): Equation 6
 */
struct RegularizedKelvinlet {
        void operator()(const asc::state_t& s, asc::state_t& sd, const double)
        {
                Eigen::Vector3d x{s[0], s[1], s[2]};
                Eigen::Vector3d x0{s[3], s[4], s[5]};
                Eigen::Vector3d f{s[6], s[7], s[8]};
                double brush_radius;

                double dt = 0.1;
                double poisson_ratio = 0.5;
                double shear_modulus = 1.0;
                double a = 1.0/(4.0 * igl::PI * shear_modulus);
                double b = a/(4.0*(1.0 - poisson_ratio));
                double c = 2.0/(3.0*a - 2.0*b);

                Eigen::Vector3d r = x - (x0 + (f / c / brush_radius)*dt);
                double r_norm_sq = r.squaredNorm();

                double r_eps = sqrt(r_norm_sq + brush_radius*brush_radius);
                double r_eps_3 = r_eps * r_eps * r_eps;
                Eigen::Vector3d t1 = ((a - b) / r_eps) * f;
                Eigen::Vector3d t2 = ((b / r_eps_3) * r * r.transpose()) * f;
                Eigen::Vector3d t3 = ((a * brush_radius * brush_radius) / (2 * r_eps_3)) * f;

                Eigen::Vector3d u = t1 + t2 + t3;
                sd[0] = u(0) - x(0);
                sd[1] = u(1) - x(1);
                sd[2] = u(2) - x(2);
        }
};

static bool
callback_mouse_move(igl::opengl::glfw::Viewer& viewer, int button, int modifie)
{
        if (!pos_start.isZero() && !pos_start.hasNaN()) {
                pos_end = igl::unproject(Eigen::Vector3f(viewer.current_mouse_x,
                                                         viewer.core().viewport[3] -
                                                         static_cast<float>(viewer.current_mouse_y),
                                                         viewer.down_mouse_z),
                                         viewer.core().view,
                                         viewer.core().proj,
                                         viewer.core().viewport).template cast<double>();

                Eigen::Vector3d f = (pos_end - pos_start)*brush_strength;

                double t = 0.0;
                double dt = 0.1;
                double t_end = 1.0;
                for (int i = 0;
                     i < V_deformed.rows();
                     ++i) {
                        asc::RK4 integrator;
                        RegularizedKelvinlet system;

                        Eigen::Vector3d x = V_deformed.row(i);
                        Eigen::Vector3d x0 = pos_start;
                        asc::state_t s{x(0), x(1), x(2), x0(0), x0(1), x0(2), f(0), f(1), f(2)};
                        while (t < t_end)
                                integrator(system, s, t, dt);

                        x = {s[0], s[1], s[2]};
                        result.row(i) = x;
                }

                viewer.data().set_mesh(result, F_deformed);
                viewer.core().align_camera_center(result, F_deformed);

                return true;
        }
        return false;
}

static bool
key_down(igl::opengl::glfw::Viewer& viewer, unsigned char key, int modifier)
{
        if (key == '1')
        {
                viewer.data().clear();
                viewer.data().set_mesh(V_input, F_input);
                viewer.core().align_camera_center(V_input, F_input);
        }
        else if (key == '2')
        {
                viewer.data().clear();
                viewer.data().set_mesh(V_deformed, F_deformed);
                viewer.core().align_camera_center(V_deformed, F_deformed);
        }
        return false;
}

static bool
callback_mouse_down(igl::opengl::glfw::Viewer& viewer, int button, int modifier)
{
        Eigen::Vector3f bc;
        int fid;
        double x = viewer.current_mouse_x;
        double y = viewer.core().viewport(3) - static_cast<float>(viewer.current_mouse_y);
        if (
                igl::unproject_onto_mesh(
                        Eigen::Vector2f(x, y),
                        viewer.core().view,
                        viewer.core().proj,
                        viewer.core().viewport,
                        V_deformed,
                        F_deformed,
                        fid,
                        bc
                )
        ) {
                pos_start = igl::unproject(Eigen::Vector3f(x, y, viewer.down_mouse_z),
                                           viewer.core().view,
                                           viewer.core().proj,
                                           viewer.core().viewport).template cast<double>();
                return true;
        }
        return false;
}

int main(int argc, char *argv[])
{
        igl::readOBJ("/home/bduke/work/libigl/tutorial/data/elephant.obj", V_input, F_input);

        V_deformed = V_input;
        F_deformed = F_input;
        result.resize(V_deformed.rows(), V_deformed.cols());

        brush_strength = (
                V_deformed.colwise().maxCoeff() - V_deformed.colwise().minCoeff()
        ).norm();

        igl::opengl::glfw::Viewer viewer;

        viewer.callback_mouse_down = &callback_mouse_down;
        viewer.callback_mouse_move = &callback_mouse_move;
        viewer.callback_key_down = &key_down;
        viewer.callback_mouse_up =
                [&](igl::opengl::glfw::Viewer& viewer, int, int) -> bool {
                        if (!pos_start.isZero()) {
                                V_deformed = result;
                                pos_start.setZero();
                                return true;
                        }
                        return false;
                };

        viewer.data().set_mesh(V_deformed, F_deformed);
        viewer.core().align_camera_center(V_deformed, F_deformed);
        viewer.launch();
}
