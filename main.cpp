#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <string>
#include <vector>
#include <math.h>
#include <iostream>
#include <memory>
#include <assert.h>

///////////////////////////////////////////////////////////////////
//
// Vec3
//
///////////////////////////////////////////////////////////////////
class Vec3
{
    private:
        float _v[3];
    public:
        Vec3() {};

        Vec3(float x, float y, float z, bool normalized = false)
        {
            _v[0] = x; _v[1] = y; _v[2] = z;
            if(normalized) {
                Vec3 v = this->normalized();
                _v[0] = v.x(); _v[1] = v.y(); _v[2] = v.z();
            }
        }

        inline Vec3 clone() const
        {
            return Vec3(_v[0], _v[1], _v[2]);
        }

        inline float x() const { return _v[0]; }
        inline float y() const { return _v[1]; }
        inline float z() const { return _v[2]; }

        inline float r() const { return _v[0]; }
        inline float g() const { return _v[1]; }
        inline float b() const { return _v[2]; }


        inline float sqr() const
        {
            return _v[0]*_v[0] + _v[1]*_v[1] + _v[2]*_v[2];
        }

        inline float norm() const
        {
            return sqrt(this->sqr());
        }

        inline Vec3 negative() const
        {
            return Vec3(-_v[0], -_v[1], -_v[2]);
        }

        inline Vec3 normalized() const
        {
            float k = 1/this->norm();
            return Vec3(_v[0]*k, _v[1]*k, _v[2]*k);
        }

        inline Vec3 operator/(const Vec3& v2)  const
        {
            return Vec3(_v[0]/v2._v[0], _v[1]/v2._v[1], _v[2]/v2._v[2]);
        }

        inline Vec3 operator*(const Vec3& v2)  const
        {
            return Vec3(_v[0]*v2._v[0], _v[1]*v2._v[1], _v[2]*v2._v[2]);
        }

        inline Vec3 operator+(const Vec3& v2)  const
        {
            return Vec3(_v[0]+v2._v[0], _v[1]+v2._v[1], _v[2]+v2._v[2]);
        }

        inline Vec3 operator-(const Vec3& v2)  const
        {
            return Vec3(_v[0]-v2._v[0], _v[1]-v2._v[1], _v[2]-v2._v[2]);
        }

        inline Vec3 operator+=(const Vec3& v2)
        {
            _v[0] += v2._v[0]; _v[1] += v2._v[1]; _v[2] += v2._v[2];
        }

        inline Vec3 operator*(float s)  const
        {
            return Vec3(_v[0]*s, _v[1]*s, _v[2]*s);
        }

        inline float dot(const Vec3& v2) const
        {
            return _v[0]*v2._v[0] + _v[1]*v2._v[1] + _v[2]*v2._v[2];
        }

        inline Vec3 cross(const Vec3& v2) const
        {
            return Vec3(
                _v[1]*v2._v[2] - _v[2]*v2._v[1],
                -_v[0]*v2._v[2] + _v[2]*v2._v[0],
                _v[0]*v2._v[1] - _v[1]*v2._v[0]
            );
        }

        inline float distance_to(const Vec3& v2) const
        {
            return (*this-v2).norm();
        }
};

Vec3 max(const Vec3& v1, const Vec3& v2)
{
    return Vec3(
        std::max(v1.x(), v2.x()),
        std::max(v1.y(), v2.y()),
        std::max(v1.z(), v2.z()));
}

float random_uniform(float min = 0, float max = 1)
{
    return min + (float(random())/RAND_MAX)/(max-min);
}

std::ostream& operator << (std::ostream& out, const Vec3& v)
{
    out << "( " << v.x() << ", " << v.y() << ", " << v.z() <<")";
    return out;
}


///////////////////////////////////////////////////////////////////
//
// Ray
//
///////////////////////////////////////////////////////////////////
class Ray
{
    public:
        Vec3 origin;
        Vec3 direction;
        Vec3 color;
        float intensity;

        Ray(const Vec3& o, const Vec3& v, const Vec3& c = Vec3(0,0,0), float i = 0)
            :origin(o), direction(v), color(c), intensity(i)
            {};
};

///////////////////////////////////////////////////////////////////
//
// Camera
//
///////////////////////////////////////////////////////////////////
class Camera
{
    public:
        Vec3 _origin;
        Vec3 _toward; // local z
        Vec3 _down;   // local y
        Vec3 _right;  // local x (determined by y and z)
        float _focal_length;
        Vec3 _focal_point;
        float _view_angle; // in degrees
        int _width;
        int _height;
        bool _is_perspective;

        float _film_width;
        float _film_height;

    public:
        Camera(
            const Vec3& origin,
            const Vec3& toward,
            const Vec3& down,
            float focal_length,
            float view_angle,
            int width,
            int height,
            bool is_perspective = true
        ) : _origin(origin), _toward(toward), _down(down),
        _right(_down.cross(_toward).normalized())
        {
            _toward = _toward.normalized();
            _down = _toward.cross(_right).normalized();
            _focal_point = _origin - _toward*focal_length;

            _focal_length = focal_length;
            _view_angle = view_angle;
            _width = width;
            _height = height;
            _is_perspective = is_perspective;

            _film_width = 2*_focal_length*tan(view_angle/180*M_PI*0.5);
            _film_height = _film_width*_height/_width;
        };

        const Vec3& focal_ponit() const
        {
            return _focal_point;
        }

        inline Ray cast_ray_from_plane(float x, float y) const
        {
            if(_is_perspective)
            {
                return Ray(this->_focal_point, (_right*x + _down*y + _origin - _focal_point).normalized(), Vec3(0,0,0), 1);
            }
            else
            {
                return Ray(_right*x + _down*y + _origin, _toward, Vec3(0,0,0), 1);
            }
        }

        inline Ray cast_ray_from_pixel(float x, float y) const
        {
            float xx = -_film_width/2 + _film_width/(_width-1)*x;
            float yy = -_film_height/2 + _film_height/(_height-1)*y;
            return cast_ray_from_plane(xx, yy);
        }
};

///////////////////////////////////////////////////////////////////
//
// Lights
//
///////////////////////////////////////////////////////////////////
class Light
{
    protected:
        Vec3 _color;
        float _intensity;

    public:
        Light(const Vec3& color, float intensity) : _color(color), _intensity(intensity) {};
        virtual Ray incidence_at(const Vec3& point) const = 0;
};

class SunLight: public Light
{
    private:
        Vec3 _direction;

    public:
        SunLight(const Vec3& color, float intensity, const Vec3& direction) : Light(color, intensity), _direction(direction.normalized())
        {}

        Ray incidence_at(const Vec3& point) const
        {
            return Ray(Vec3(0,0,0), _direction, _color, _intensity);
        }
};

class SpotLight: public Light
{
    private:
        Vec3 _position;

    public:
        SpotLight(const Vec3& color, float intensity, const Vec3& position): Light(color, intensity), _position(position)
        {}

        Ray incidence_at(const Vec3& point) const
        {
            Vec3 v = point - _position;
            float intensity = _intensity/(4*M_PI*v.sqr());
            v = v.normalized();
            return Ray(_position, v, _color, intensity);
        }
};

///////////////////////////////////////////////////////////////////
//
// Geometry
//
///////////////////////////////////////////////////////////////////
class Geometry
{
    private:
        float _roughness;
        Vec3 _albedo;

    public:
        Geometry(float roughness, const Vec3& albedo)
        : _roughness(roughness), _albedo(albedo)
        {}

        inline float roughness() const { return _roughness; }
        inline Vec3 albedo() const { return _albedo; }

        virtual Vec3 normal_at(const Vec3& point) const = 0;
        virtual bool interception_with_ray(const Ray& ray, float& interception_distance) const = 0;
};

inline float abs(float v)
{
    return v < 0 ? -v: v;
}

class InfinitePlane: public Geometry
{
    private:
        float _distance;
        Vec3 _normal;
    public:
        InfinitePlane(float distance, const Vec3& normal, float roughness, const Vec3& albedo)
        : Geometry(roughness, albedo),
        _distance(distance), _normal(normal.normalized())
        {};

        Vec3 normal_at(const Vec3& point) const
        {
            return _normal;
        }

        bool interception_with_ray(const Ray& ray, float& interception_distance) const
        {
            float discriminant = _normal.dot(ray.direction);
            if(abs(discriminant) > 1e-3)
            {
                float lambda = -(_distance+_normal.dot(ray.origin))/discriminant;
                if(lambda < 0) return false;
                interception_distance = lambda;
                return true;
            }
            else return false;
        }
};

class Sphere: public Geometry
{
    private:
        Vec3 _center;
        float _radius;

    public:
        Sphere(const Vec3& center, float radius, float roughness, const Vec3& albedo)
        : Geometry(roughness, albedo), _center(center), _radius(radius)
        {}

        Vec3 normal_at(const Vec3& point) const
        {
            return (point-_center).normalized();
        }

        bool interception_with_ray(const Ray& ray, float& interception_distance) const
        {
            Vec3 t = ray.origin - _center;
            float a = ray.direction.sqr();
            float b = t.dot(ray.direction)*2;
            float c = t.sqr() - _radius*_radius;
            float discriminant = b*b - 4*a*c;
            if(discriminant >= 0)
            {
                float lambda1 = (-b + sqrt(discriminant))/(2*a);
                float lambda2 = (-b - sqrt(discriminant))/(2*a);
                // lambda1 is supposed to be the smaller one.
                if(lambda1 > lambda2)
                {
                    float tmp = lambda1;
                    lambda1 = lambda2;
                    lambda2 = tmp;
                }
                float lambda = 0;
                if(lambda1 >= 0)
                    lambda = lambda1;
                else if(lambda2 >=0)
                    lambda = lambda2;
                else // interception on the invisible side of the image plane
                    return false;

                interception_distance = lambda;
                return true;
            }
            else // no interceptions
                return false;
        }
};

///////////////////////////////////////////////////////////////////
//
// Ray Tracer
//
///////////////////////////////////////////////////////////////////
class RayTracer
{
    private:
        std::vector<std::shared_ptr<Geometry>> _geometries;
        std::vector<std::shared_ptr<Light>> _lights;
        Vec3 _ambient_color;

    public:
        RayTracer(
            const std::vector<std::shared_ptr<Geometry>>& geometries,
            const std::vector<std::shared_ptr<Light>>& lights,
            const Vec3& ambient_color)
            : _geometries(geometries),
            _lights(lights),
            _ambient_color(ambient_color)
        {}

        int run(Ray& camera_ray, int max_num_bounces)
        {
            int num_bounces = 0;
            while(1)
            {
                if(camera_ray.intensity < 1e-4) break;
                if(num_bounces >= max_num_bounces) break;

                Vec3 interception_point, interception_point_normal;
                std::shared_ptr<Geometry> intercepted_geometry = nullptr;
                bool camera_ray_intercepted = this->find_interception_point_with_ray(camera_ray, interception_point, interception_point_normal, intercepted_geometry);

                if(camera_ray_intercepted)
                {
                    num_bounces ++;
                    Vec3 interception_point_color = this->calc_interception_point_color(interception_point, interception_point_normal,  intercepted_geometry);

                    // update camera rate
                    camera_ray.color += interception_point_color*camera_ray.intensity;
                    camera_ray.origin = interception_point;
                    camera_ray.direction = camera_ray.direction-interception_point_normal*(interception_point_normal.dot(camera_ray.direction)*2);
                    camera_ray.intensity *= (1-intercepted_geometry->roughness());
                }
                else // Ray will not meet any geometries and go to infinity.
                {
                    if(num_bounces == 0)
                        camera_ray.color += _ambient_color*camera_ray.intensity;
                    camera_ray.intensity = 0; // to make the loop break
                }
            }

            return num_bounces;
        }

    private:
        bool find_interception_point_with_ray(const Ray& ray, Vec3& interception_point, Vec3& interception_point_normal, std::shared_ptr<Geometry>& intercepted_geometry) const
        {
            bool has_interception = false;
            float min_interception_distance = std::numeric_limits<float>::max();

            for(auto geometry: _geometries)
            {
                float interception_distance;
                bool intercepted = geometry->interception_with_ray(ray, interception_distance);
                if(intercepted && interception_distance > 1e-4)
                {
                    if(interception_distance < min_interception_distance)
                    {
                        has_interception = true;
                        min_interception_distance = interception_distance;
                        intercepted_geometry = geometry;
                    }
                }
            }

            if(has_interception)
            {
                interception_point = ray.origin + ray.direction*min_interception_distance;

                interception_point_normal = intercepted_geometry->normal_at(interception_point);
            }

            return has_interception;
        }

        bool incident_light_ray_is_blocked(const Ray& incident_light_ray, const Vec3 interception_point, const std::shared_ptr<Geometry>& intercepted_geometry) const
        {
            bool light_ray_blocked = false;

            float dist_to_light = interception_point.distance_to(incident_light_ray.origin);

            Ray reverse_incident_light_ray = Ray(interception_point, incident_light_ray.direction.negative());

            for(auto geometry: _geometries)
            {
                if(geometry != intercepted_geometry)
                {
                    float dist_to_geometry;
                    bool intercepted = geometry->interception_with_ray(reverse_incident_light_ray, dist_to_geometry);

                    if(intercepted && dist_to_geometry > 0 && dist_to_geometry < dist_to_light)
                    {
                        light_ray_blocked = true;
                        break;
                    }
                }
            }

            return light_ray_blocked;
        }

        Vec3 calc_interception_point_color(const Vec3& interception_point, const Vec3& interception_point_normal, const std::shared_ptr<Geometry>& intercepted_geometry) const
        {
            Vec3 interception_point_color = _ambient_color*intercepted_geometry->albedo()*intercepted_geometry->roughness();

            for(auto light: _lights)
            {
                Ray incident_light_ray = light->incidence_at(interception_point);

                // Test if the incident light ray is blocked.
                if(!this->incident_light_ray_is_blocked(incident_light_ray, interception_point, intercepted_geometry))
                {
                    float dot = interception_point_normal.dot(incident_light_ray.direction.negative());

                    if(dot > 0)
                    {
                        interception_point_color += incident_light_ray.color*incident_light_ray.intensity*intercepted_geometry->albedo()*intercepted_geometry->roughness()*dot;
                    }
                }
            }

            return interception_point_color;
        }
};

///////////////////////////////////////////////////////////////////
//
// Tests
//
///////////////////////////////////////////////////////////////////
void test_Vec3()
{
    Vec3 v1(random_uniform(), random_uniform(), random_uniform());
    Vec3 v2 = v1.normalized();
    std::cout << v1 << std::endl;
    std::cout << v1.norm() << std::endl;
    std::cout << v2 << std::endl;
    std::cout << v2.norm() << std::endl;
    std::cout << v1/v2 << std::endl;
    std::cout << Vec3(1,0,0).cross(Vec3(0,1,0)) << std::endl;
}

void test_Camera()
{
    Camera camera(Vec3(-1,1,1),
        Vec3(random_uniform(), random_uniform(), random_uniform()),
        Vec3(random_uniform(), random_uniform(), random_uniform()),
        1, 30, 100, 200);
    std::cout << camera._toward.dot(camera._toward) << std::endl;
    std::cout << camera._toward.dot(camera._down) << std::endl;
    std::cout << camera._toward.dot(camera._right) << std::endl;
    std::cout << camera._down.dot(camera._toward) << std::endl;
    std::cout << camera._down.dot(camera._down) << std::endl;
    std::cout << camera._down.dot(camera._right) << std::endl;
    std::cout << camera._right.dot(camera._toward) << std::endl;
    std::cout << camera._right.dot(camera._down) << std::endl;
    std::cout << camera._right.dot(camera._right) << std::endl;
}


void test_scene()
{
    size_t W = 3000, H = 1200;
    cv::Mat img(H, W, CV_8UC3);

    Camera camera(Vec3(0,0.3,0), Vec3(0,0,1), Vec3(0,1,0), 1, 80, W, H, true);
    std::vector<std::shared_ptr<Light>> lights;
    //lights.push_back(std::shared_ptr<Light>(new SunLight(Vec3(1,1,1), 1, Vec3(1,1,0))));
    //lights.push_back(std::shared_ptr<Light>(new SunLight(Vec3(1,1,1), 1, Vec3(-1,1,0))));
     lights.push_back(std::shared_ptr<Light>(new SpotLight(Vec3(1,1,1), 200, Vec3(0,-2,2))));


    std::vector<std::shared_ptr<Geometry>> geometries;
    geometries.push_back(std::shared_ptr<Geometry>(new InfinitePlane(1, Vec3(0,-1,0), 1, Vec3(0,0,1))));
    geometries.push_back(std::shared_ptr<Geometry>(new Sphere(Vec3(0,0,3), 0.8, 0.5, Vec3(1,0,0))));
    geometries.push_back(std::shared_ptr<Geometry>(new Sphere(Vec3(-1.8,0,3), 0.8, 0.2, Vec3(1,1,0))));
    geometries.push_back(std::shared_ptr<Geometry>(new Sphere(Vec3(1.8,0,3), 0.8, 0.9, Vec3(0,1,0))));

    Vec3 ambient_color(0.2, 0.2, 0.2);
    RayTracer ray_tracer(geometries, lights, ambient_color);
    int max_num_bounces = 100;
    int total_num_bounces = 0;

    for(float y = 0; y < H; y++)
    {
        for(float x = 0; x < W; x++)
        {
            Ray camera_ray = camera.cast_ray_from_pixel(x, y);
            total_num_bounces += ray_tracer.run(camera_ray, max_num_bounces);

            int idx = img.step*int(y) + 3*int(x);
            img.data[idx+0] = std::min(255, int(0.5+255*camera_ray.color.b()));
            img.data[idx+1] = std::min(255, int(0.5+255*camera_ray.color.g()));
            img.data[idx+2] = std::min(255, int(0.5+255*camera_ray.color.r()));
        }
    }

    std::cout << "Average: " << float(total_num_bounces)/(W*H) << " bounces/pixel." << std::endl;

    cv::imwrite("test.jpg", img);
}
///////////////////////////////////////////////////////////////////
//
// main()
//
///////////////////////////////////////////////////////////////////
int main()
{
    // test_Vec3();
    // test_Camera();
    test_scene();
    return 0;
}