#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <string>
#include <vector>
#include <list>
#include <math.h>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <memory>
#include <assert.h>
#include <thread>
#include <mutex>
#include <algorithm>

///////////////////////////////////////////////////////////////////
//
// facilites
//
///////////////////////////////////////////////////////////////////
inline float abs(float v)
{
    return v < 0 ? -v: v;
}

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

        inline const float operator[](int i) const
        {
            // for efficiency, no boundary check.
            return _v[i];
        }


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

        inline void operator+=(const Vec3& v2)
        {
            _v[0] += v2._v[0]; _v[1] += v2._v[1]; _v[2] += v2._v[2];
        }

        inline void operator*=(float s)
        {
            _v[0] *= s; _v[1] *= s; _v[2] *= s;
        }

        inline void operator*=(const Vec3& v2)
        {
            _v[0] *= v2[0]; _v[1] *= v2[1]; _v[2] *= v2[2];
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

Vec3 operator *(float s, const Vec3& v)
{
    return Vec3(s*v.x(), s*v.y(), s*v.z());
}

Vec3 max(const Vec3& v1, const Vec3& v2)
{
    return Vec3(
        std::max(v1.x(), v2.x()),
        std::max(v1.y(), v2.y()),
        std::max(v1.z(), v2.z()));
}

float random_uniform(float min = 0, float max = 1)
{
    return min + (float(random())/RAND_MAX)*(max-min);
}

Vec3 random_in_unit_sphere()
{
    Vec3 v;
    do {
        v = Vec3(random_uniform(-1,1), random_uniform(-1,1), random_uniform(-1,1));
    } while(v.sqr() > 1.0);
    return v;
}

std::ostream& operator << (std::ostream& out, const Vec3& v)
{
    out << "( " << v.x() << ", " << v.y() << ", " << v.z() <<")";
    return out;
}

///////////////////////////////////////////////////////////////////
//
// Mat3x3
//
///////////////////////////////////////////////////////////////////
class Mat3x3
{
    private:
        float _v[3][3];

    public:
        Vec3 column(int idx) const
        {
            return Vec3(_v[0][idx], _v[1][idx], _v[2][idx]);
        }

        Vec3 row(int idx) const
        {
            return Vec3(_v[idx][0], _v[idx][1], _v[idx][2]);
        }

        Mat3x3 operator+(const Mat3x3 &left) const
        {
            Mat3x3 result;
            for(int i = 0; i < 3; i++)
                for(int j = 0; j < 3; j++)
                    result._v[i][j] = _v[i][j] + left._v[i][j];
            return result;
        }

        Vec3 operator*(const Vec3& v) const
        {
            return Vec3(row(0).dot(v), row(1).dot(v), row(2).dot(v));
        }

        Mat3x3 operator*(const Mat3x3& m) const
        {
            Mat3x3 result;
            for(int i = 0; i < 3; i++)
            {
                Vec3 r = row(i);
                for(int j = 0; j < 3; j++)
                    result._v[i][j] = r.dot(m.column(j));
            }
            return result;
        }

        static Mat3x3 diagonal(float s)
        {
            return diagonal(Vec3(s,s,s));
        }

        static Mat3x3 Rx(float angle)
        {
            float angle_ = angle/180*M_PI;
            float cos_angle = cos(angle_);
            float sin_angle = sin(angle_);

            Mat3x3 result;
            result._v[0][0] = 1;
            result._v[0][1] = 0;
            result._v[0][2] = 0;
            result._v[1][0] = 0;
            result._v[1][1] = cos_angle;
            result._v[1][2] = -sin_angle;
            result._v[2][0] = 0;
            result._v[2][1] = sin_angle;
            result._v[2][2] = cos_angle;

            return result;
        }

        static Mat3x3 Ry(float angle)
        {
            float angle_ = angle/180*M_PI;
            float cos_angle = cos(angle_);
            float sin_angle = sin(angle_);

            Mat3x3 result;
            result._v[0][0] = cos_angle;
            result._v[0][1] = 0;
            result._v[0][2] = -sin_angle;
            result._v[1][0] = 0;
            result._v[1][1] = 1;
            result._v[1][2] = 0;
            result._v[2][0] = sin_angle;
            result._v[2][1] = 0;
            result._v[2][2] = cos_angle;

            return result;
        }

        static Mat3x3 Rz(float angle)
        {
            float angle_ = angle/180*M_PI;
            float cos_angle = cos(angle_);
            float sin_angle = sin(angle_);

            Mat3x3 result;
            result._v[0][0] = cos_angle;
            result._v[0][1] = -sin_angle;
            result._v[0][2] = 0;
            result._v[1][0] = sin_angle;
            result._v[1][1] = cos_angle;
            result._v[1][2] = 0;
            result._v[2][0] = 0;
            result._v[2][1] = 0;
            result._v[2][2] = 1;

            return result;
        }

        static Mat3x3 diagonal(const Vec3& v)
        {
            Mat3x3 result;
            result._v[0][0] = v.x();
            result._v[0][1] = 0;
            result._v[0][2] = 0;
            result._v[1][0] = 0;
            result._v[1][1] = v.y();
            result._v[1][2] = 0;
            result._v[2][0] = 0;
            result._v[2][1] = 0;
            result._v[2][2] = v.z();

            return result;
        }

        static Mat3x3 outer_product(const Vec3& a, const Vec3& b)
        {
            Mat3x3 result;
            result._v[0][0] = a.x()*b.x();
            result._v[0][1] = a.x()*b.y();
            result._v[0][2] = a.x()*b.z();
            result._v[1][0] = a.y()*b.x();
            result._v[1][1] = a.y()*b.y();
            result._v[1][2] = a.y()*b.z();
            result._v[2][0] = a.z()*b.x();
            result._v[2][1] = a.z()*b.y();
            result._v[2][2] = a.z()*b.z();

            return result;
        }

        static Mat3x3 cross(const Vec3& v)
        {
            Mat3x3 result;
            result._v[0][0] = 0;
            result._v[0][1] = -v.z();
            result._v[0][2] = v.y();
            result._v[1][0] = v.z();
            result._v[1][1] = 0;
            result._v[1][2] = -v.x();
            result._v[2][0] = -v.y();
            result._v[2][1] = v.x();
            result._v[2][2] = 0;

            return result;
        }
};

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
        Vec3 intensity;
        bool debug;

        Ray(const Vec3& o, const Vec3& v, const Vec3& c = Vec3(0,0,0), const Vec3& i = Vec3(0,0,0))
            :origin(o), direction(v), color(c), intensity(i), debug(false)
            {};
};

///////////////////////////////////////////////////////////////////
//
// Transform
//
///////////////////////////////////////////////////////////////////
class Transform
{
    public:
        Vec3 x, y, z, t;
    public:
        Transform(): x(1,0,0), y(0,1,0), z(0,0,1), t(0,0,0) {};
        Transform(const Vec3& x_, const Vec3& y_, const Vec3& z_, const Vec3& t_): x(x_), y(y_), z(z_), t(t_) {};

        inline Vec3 forward(const Vec3& v) const
        {
            return x*v.x() + y*v.y() + z*v.z() + t;
        }

        inline Vec3 reverse(const Vec3& v) const
        {
            Vec3 vv = v - t;
            return Vec3(x.dot(vv), y.dot(vv), z.dot(vv));
        }

        inline Vec3 forward_vector(const Vec3& v) const
        {
            return x*v.x() + y*v.y() + z*v.z();
        }

        inline Vec3 reverse_vector(const Vec3& v) const
        {
            return Vec3(x.dot(v), y.dot(v), z.dot(v));
        }

        void euler(float x_angle, float y_angle, float z_angle)
        {
            Mat3x3 R = Mat3x3::Rx(x_angle)*Mat3x3::Ry(y_angle)*Mat3x3::Rz(z_angle);
            x = (R*x).normalized();
            y = (R*y).normalized();
            z = (R*z).normalized();
        }


        // void rotate_along_vector(const Vec3& v, float theta)
        // // theta --- in degrees
        // {
        //     Vec3 v_ = v.normalized();
        //     float cos_theta = cos(theta/180*M_PI);

        //     Mat3x3 R = Mat3x3::cross(v_) + Mat3x3::outer_product(v_, v_) + Mat3x3::diagonal(cos_theta);
        //     x = (R*x).normalized();
        //     y = (R*y).normalized();
        //     z = (R*z).normalized();
        // }
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
                Vec3 point_on_the_plane_co = _right*x + _down*y + _origin;
                return Ray(point_on_the_plane_co, (point_on_the_plane_co - _focal_point).normalized(), Vec3(0,0,0), Vec3(1,1,1));
            }
            else
            {
                return Ray(_right*x + _down*y + _origin, _toward, Vec3(0,0,0), Vec3(1,1,1));
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
            return Ray(Vec3(0,0,0), _direction, _color, Vec3(1,1,1)*_intensity);
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
            return Ray(_position, v, _color, Vec3(1,1,1)*intensity);
        }
};


///////////////////////////////////////////////////////////////////
//
// BoundingBox
//
///////////////////////////////////////////////////////////////////
bool box_hit_by_ray(const Vec3& min_corner, const Vec3& max_corner, const Vec3& origin, const Vec3& direction, float& tmin, float& tmax)
{
    for(int i = 0; i < 3; i++)
    {
        float invD = 1.0f/direction[i];
        float t0 = (min_corner[i]-origin[i])*invD;
        float t1 = (max_corner[i]-origin[i])*invD;
        if(invD < 0) std::swap(t0, t1);
        tmin = t0 > tmin ? t0 : tmin;
        tmax = t1 < tmax ? t1 : tmax;
        if (tmax <= tmin) return false;
    }
    return true;
}

class BoundingBox
{
    private:
        Vec3 _min_corner;
        Vec3 _max_corner;
    public:
        BoundingBox(): _min_corner(0,0,0), _max_corner(0,0,0) {};
        BoundingBox(const Vec3& min_corner, const Vec3& max_corner)
        : _min_corner(min_corner), _max_corner(max_corner)
        {}

        void setMinCorner(const Vec3& min)
        {
            _min_corner = min;
        }

        void setMaxCorner(const Vec3& max)
        {
            _max_corner = max;
        }

        bool is_valid() const
        {
            return (_max_corner.x() > _min_corner.x()) && (_max_corner.y() > _min_corner.y()) && (_max_corner.z() > _min_corner.z());
        }

        bool hit_by_ray(const Ray& ray) const
        {
            float tmax = std::numeric_limits<float>::max();
            float tmin = -tmax;
            return box_hit_by_ray(_min_corner, _max_corner, ray.origin, ray.direction, tmin, tmax);
        }
};

///////////////////////////////////////////////////////////////////
//
// PerlinNoise
//
///////////////////////////////////////////////////////////////////

class PerlinNoise {
    private:
        // The permutation vector
        std::vector<int> _p;

    private:
        float fade(float t) const
        {
            return t * t * t * (t * (t * 6 - 15) + 10);
        }

        float lerp(float t, float a, float b) const
        {
            return a + t * (b - a);
        }

        float grad(int hash, float x, float y, float z) const
        {
            int h = hash & 15;
            // Convert lower 4 bits of hash into 12 gradient directions
            float u = h < 8 ? x : y,
                v = h < 4 ? y : h == 12 || h == 14 ? x : z;
            return ((h & 1) == 0 ? u : -u) + ((h & 2) == 0 ? v : -v);
        }

    public:
        PerlinNoise()
        {
            _p.resize(256);
            for(int i = 0; i < 256; i++)
                _p[i] = i;
            std::random_shuffle(_p.begin(), _p.end());
            // Duplicate the permutation vector
            _p.insert(_p.end(), _p.begin(), _p.end());
        }

        float get(const Vec3& vv) const
        {
            float x = vv.x(), y = vv.y(), z = vv.z();

            // Find the unit cube that contains the point
            int X = (int) floor(x) & 255;
            int Y = (int) floor(y) & 255;
            int Z = (int) floor(z) & 255;

            // Find relative x, y,z of point in cube
            x -= floor(x);
            y -= floor(y);
            z -= floor(z);

            // Compute fade curves for each of x, y, z
            float u = fade(x);
            float v = fade(y);
            float w = fade(z);

            // Hash coordinates of the 8 cube corners
            int A = _p[X] + Y;
            int AA = _p[A] + Z;
            int AB = _p[A + 1] + Z;
            int B = _p[X + 1] + Y;
            int BA = _p[B] + Z;
            int BB = _p[B + 1] + Z;

            // Add blended results from 8 corners of cube
            double res = lerp(w, lerp(v, lerp(u, grad(_p[AA], x, y, z), grad(_p[BA], x-1, y, z)), lerp(u, grad(_p[AB], x, y-1, z), grad(_p[BB], x-1, y-1, z))), lerp(v, lerp(u, grad(_p[AA+1], x, y, z-1), grad(_p[BA+1], x-1, y, z-1)), lerp(u, grad(_p[AB+1], x, y-1, z-1), grad(_p[BB+1], x-1, y-1, z-1))));
            return (res + 1.0)/2.0;
        }
};

///////////////////////////////////////////////////////////////////
//
// Texture
//
///////////////////////////////////////////////////////////////////
class Texture
{
    public:
        virtual Vec3 value(const Vec3& co) const = 0;

        // uv or 3d texture.
        virtual bool is_uv_texture() const = 0;
};

class ConstantTexture : public Texture
{
    private:
        Vec3 _albedo;

    public:
        ConstantTexture(const Vec3& albedo) : _albedo(albedo) {};
        Vec3 value(const Vec3& co) const
        {
            return _albedo;
        }

        bool is_uv_texture() const
        {
            return false;
        }
};

class CheckerBoxTexture: public Texture
{
    private:
        Vec3 _albedo1;
        Vec3 _albedo2;
        Vec3 _frequency;

    public:
        CheckerBoxTexture(const Vec3& albedo1, const Vec3& albedo2, const Vec3& frequency) : _albedo1(albedo1), _albedo2(albedo2), _frequency(2*M_PI*frequency) {};

        Vec3 value(const Vec3& co) const
        {
            Vec3 v = _frequency*co;
            bool sign = sin(v.x())*sin(v.y())*sin(v.z()) > 0;
            return sign ? _albedo1 : _albedo2;
        }

        bool is_uv_texture() const
        {
            return false;
        }
};

class PerlinNoiseTexture: public Texture
{
    public:
        typedef Vec3 (*COORDINATE_TRANSFORM_FCN)(const Vec3&);
        typedef Vec3 (*VALUE_TRANSFORM_FCN)(const Vec3&, float);

    private:
        PerlinNoise _noise;
        COORDINATE_TRANSFORM_FCN _coord_transform_fcn;
        VALUE_TRANSFORM_FCN _value_transform_fcn;

    public:
        PerlinNoiseTexture(COORDINATE_TRANSFORM_FCN coord_fcn = nullptr, VALUE_TRANSFORM_FCN value_fcn = nullptr)
        {
            _coord_transform_fcn = coord_fcn;
            _value_transform_fcn = value_fcn;
        }

        Vec3 value(const Vec3& co) const
        {
            float s = _noise.get(_coord_transform_fcn ? _coord_transform_fcn(co) : co);
            return _value_transform_fcn ? _value_transform_fcn(co, s) : Vec3(s,s,s);
        }

        bool is_uv_texture() const
        {
            return false;
        }
};

///////////////////////////////////////////////////////////////////
//
// Geometry
//
///////////////////////////////////////////////////////////////////
class Geometry
{
    protected:
        float _roughness;
        float _specularity;
        std::shared_ptr<Texture> _texture;
        BoundingBox _bounding_box;
        bool _is_emitter;


    private:
        bool bounding_box_hit_by_ray(const Ray& ray)
        {
            if(!_bounding_box.is_valid())
                return true;
            else
                return _bounding_box.hit_by_ray(ray);
        }

        virtual bool geometry_hit_by_ray(const Ray& ray, float& interception_distance, Vec3& interception_point, Vec3& normal) const = 0;

        virtual Vec3 map_to_uv_texture_space(const Vec3& xyz) const
        {
            return Vec3(0,0,0);
        }

    public:
        Geometry(float roughness, float specularity, const std::shared_ptr<Texture>& texture, bool is_emitter)
        : _roughness(roughness), _specularity(specularity), _texture(texture), _is_emitter(is_emitter)
        {}

        inline bool is_emitter() const { return _is_emitter; }
        inline float roughness() const { return _roughness; }
        inline float specularity() const { return _specularity; }
        inline const BoundingBox& bounding_box() const { return _bounding_box; }

        inline Vec3 texture_at(const Vec3& co) const
        {
            if(_texture->is_uv_texture())
                return _texture->value(map_to_uv_texture_space(co));
            else
                return _texture->value(co);
        }

        virtual bool interception_with_ray(const Ray& ray, float& interception_distance, Vec3& interception_point, Vec3& normal)
        {
            bool hit_by_ray = false;
            if(bounding_box_hit_by_ray(ray))
            {
                hit_by_ray = geometry_hit_by_ray(ray, interception_distance, interception_point, normal);
            }
            return hit_by_ray;
        }
};

class InfinitePlane: public Geometry
{
    private:
        float _distance;
        Vec3 _normal;

    public:
        InfinitePlane(float distance, const Vec3& normal, float roughness, float specularity, const std::shared_ptr<Texture>& texture, bool is_emitter = false)
        : Geometry(roughness, specularity, texture, is_emitter),
        _distance(distance), _normal(normal.normalized())
        {};

    private:
        bool geometry_hit_by_ray(const Ray& ray, float& interception_distance, Vec3& interception_point, Vec3& normal) const
        {
            float discriminant = _normal.dot(ray.direction);
            if(abs(discriminant) > 1e-3)
            {
                float lambda = -(_distance+_normal.dot(ray.origin))/discriminant;
                if(lambda < 0) return false;
                interception_distance = lambda;

                normal = _normal;
                interception_point = ray.origin + interception_distance*ray.direction;
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
        Sphere(const Vec3& center, float radius, float roughness, float specularity, const std::shared_ptr<Texture>& texture, bool is_emitter = false)
        : Geometry(roughness, specularity, texture, is_emitter), _center(center), _radius(radius)
        {
            _bounding_box.setMinCorner(_center-Vec3(radius, radius, radius));
            _bounding_box.setMaxCorner(_center+Vec3(radius, radius, radius));
        }

        bool has_interception_with_sphere(const Sphere& another_sphere) const
        {
            return _center.distance_to(another_sphere._center) <= (_radius + another_sphere._radius);
        }

    private:
        bool geometry_hit_by_ray(const Ray& ray, float& interception_distance, Vec3& interception_point, Vec3& normal) const
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
                interception_point = ray.origin + interception_distance*ray.direction;
                normal = (interception_point-_center).normalized();
                return true;
            }
            else // no interceptions
                return false;
        }
};

class TransformGeometry: public Geometry
{
    private:
        Transform _transform;
    public:
        TransformGeometry(float roughness, float specularity, const std::shared_ptr<Texture>& texture, bool is_emitter)
        : Geometry(roughness, specularity, texture, is_emitter)
        {}

        Transform& transform() { return _transform; }
        const Transform& transform() const { return _transform; }
};

class Rectangle: public TransformGeometry
{
    private:
        float _half_width;
        float _half_height;
    public:
        Rectangle(float half_width, float half_height, float roughness, float specularity, const std::shared_ptr<Texture>& texture, bool is_emitter = false) :
        TransformGeometry(roughness, specularity, texture, is_emitter)
        {
            _half_width = half_width;
            _half_height = half_height;
        }

    private:
        bool geometry_hit_by_ray(const Ray& ray, float& interception_distance, Vec3& interception_point, Vec3& normal) const
        {
            Vec3 origin = transform().reverse(ray.origin);
            Vec3 direction = transform().reverse_vector(ray.direction);

            if(direction.z() != 0)
            {
                float lambda = -origin.z()/direction.z();
                if(lambda > 0)
                {
                    float interception_x = origin.x() + lambda*direction.x();
                    float interception_y = origin.y() + lambda*direction.y();

                    if((interception_x > -_half_width) && (interception_x < _half_width) && (interception_y > -_half_height) && (interception_y < _half_height))
                    {
                        interception_distance = lambda;
                        interception_point = ray.origin + interception_distance*ray.direction;
                        normal = transform().z;
                        return true;
                    }
                }
            }

            return false;
        }
};

class Cube: public TransformGeometry
{
    private:
        float _half_x;
        float _half_y;
        float _half_z;
    public:
        Cube(float half_x, float half_y, float half_z, float roughness, float specularity, const std::shared_ptr<Texture>& texture, bool is_emitter = false) :
        TransformGeometry(roughness, specularity, texture, is_emitter)
        {
            _half_x = half_x;
            _half_y = half_y;
            _half_z = half_z;
        }

    private:
        bool geometry_hit_by_ray(const Ray& ray, float& interception_distance, Vec3& interception_point, Vec3& normal) const
        {
            Vec3 origin = transform().reverse(ray.origin);
            Vec3 direction = transform().reverse_vector(ray.direction);

            const float eps = 1e-6;
            float tmax = std::numeric_limits<float>::max();
            float tmin = -tmax;
            bool hit = box_hit_by_ray(Vec3(-_half_x, -_half_y, -_half_z), Vec3(_half_x, _half_y, _half_z), origin, direction, tmin, tmax);

            if(hit)
            {
                if(tmin > 0)
                {
                    interception_distance = tmin;
                    interception_point = ray.origin + interception_distance*ray.direction;

                    Vec3 p = origin + interception_distance*direction;
                    if(abs(p.x()-_half_x) < eps) { normal = Vec3(1,0,0); }
                    else if(abs(p.x()+_half_x) < eps) { normal = Vec3(-1,0,0); }
                    else if(abs(p.y()-_half_y) < eps) { normal = Vec3(0,1,0); }
                    else if(abs(p.y()+_half_y) < eps) { normal = Vec3(0,-1,0); }
                    else if(abs(p.z()-_half_z) < eps) { normal = Vec3(0,0,1); }
                    else if(abs(p.z()+_half_z) < eps) { normal = Vec3(0,0,-1); }
                    normal = transform().forward_vector(normal);

                    return true;
                }
            }
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
                if(camera_ray.intensity.norm() < 1e-4) break;
                if(num_bounces >= max_num_bounces) break;

                Vec3 interception_point, interception_point_normal;
                std::shared_ptr<Geometry> intercepted_geometry = nullptr;
                bool camera_ray_intercepted = this->find_interception_point_with_ray(camera_ray, interception_point, interception_point_normal, intercepted_geometry);

                if(camera_ray_intercepted)
                {
                    num_bounces ++;

                    Vec3 interception_point_albedo = intercepted_geometry->texture_at(interception_point);
                    Vec3 interception_point_color = this->calc_interception_point_color(interception_point, interception_point_normal,  intercepted_geometry, interception_point_albedo);

                    // update camera ray
                    camera_ray.color += interception_point_color*camera_ray.intensity;

                    if(intercepted_geometry->is_emitter())
                    {
                        camera_ray.intensity = Vec3(0,0,0);
                    }
                    else
                    {
                        camera_ray.origin = interception_point;
                        float dot = 0.0f;
                        for(int i = 0; i < 10; i++)
                        {
                            Vec3 reflection_dir = get_reflection_dir(camera_ray.direction, interception_point_normal, intercepted_geometry->specularity());
                            dot = interception_point_normal.dot(reflection_dir);
                            if(dot > 0)
                            {
                                camera_ray.direction = reflection_dir;
                                break;
                            }
                        }
                        camera_ray.intensity *= (1-intercepted_geometry->roughness())*dot*interception_point_albedo;
                    }
                }
                else // Ray will not meet any geometries and go to infinity.
                {
                    if(num_bounces == 0)
                        camera_ray.color += _ambient_color*camera_ray.intensity;
                    camera_ray.intensity = Vec3(0,0,0); // to make the loop break
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
                Vec3 interception_point_;
                Vec3 interception_point_normal_;
                bool intercepted = geometry->interception_with_ray(ray, interception_distance, interception_point_, interception_point_normal_);
                if(intercepted && interception_distance > 1e-4)
                {
                    if(interception_distance < min_interception_distance)
                    {
                        has_interception = true;
                        min_interception_distance = interception_distance;
                        intercepted_geometry = geometry;
                        interception_point = interception_point_;
                        interception_point_normal = interception_point_normal_;
                    }
                }
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
                    Vec3 dummy1, dummy2;
                    bool intercepted = geometry->interception_with_ray(reverse_incident_light_ray, dist_to_geometry, dummy1, dummy2);

                    if(intercepted && dist_to_geometry > 0 && dist_to_geometry < dist_to_light)
                    {
                        light_ray_blocked = true;
                        break;
                    }
                }
            }

            return light_ray_blocked;
        }

        Vec3 calc_interception_point_color(const Vec3& interception_point, const Vec3& interception_point_normal, const std::shared_ptr<Geometry>& intercepted_geometry, const Vec3& interception_point_albedo) const
        {
            Vec3 interception_point_color;
            if(intercepted_geometry->is_emitter())
            {
                interception_point_color = interception_point_albedo;
            }
            else
            {
                interception_point_color = _ambient_color*interception_point_albedo*intercepted_geometry->roughness();

                for(auto light: _lights)
                {
                    Ray incident_light_ray = light->incidence_at(interception_point);

                    // Test if the incident light ray is blocked.
                    if(!this->incident_light_ray_is_blocked(incident_light_ray, interception_point, intercepted_geometry))
                    {
                        float dot = interception_point_normal.dot(incident_light_ray.direction.negative());

                        if(dot > 0)
                        {
                            interception_point_color += incident_light_ray.color*incident_light_ray.intensity*interception_point_albedo*intercepted_geometry->roughness()*dot;
                        }
                    }
                }
            }

            return interception_point_color;
        }

        Vec3 get_reflection_dir(const Vec3& incident_dir, const Vec3& normal, float specularity, bool perturb_normal = true)
        {
            Vec3 reflection_dir;

            if(perturb_normal) // perturb reflection through perturbing normal
            {
                Vec3 perturbed_normal = (normal + random_in_unit_sphere()*(1-specularity)).normalized();
                reflection_dir = incident_dir - perturbed_normal*(perturbed_normal.dot(incident_dir)*2);
            }
            else // pertube reflection directly
            {
                Vec3 idea_reflection_dir = incident_dir - normal*(incident_dir.dot(normal)*2);
                reflection_dir = (idea_reflection_dir + random_in_unit_sphere()*(1-specularity)).normalized();
            }
            return reflection_dir;
        }
};

///////////////////////////////////////////////////////////////////
//
// Tests
//
///////////////////////////////////////////////////////////////////
struct Parameters
{
    int S;
    int num_samples;
    int num_threads;
    int max_num_bounces;
};

void usage()
{
    std::cout << "ray_tracer [S=15] [num_samples=10] [num_threads=8] [max_num_bounces=100]" << std::endl;
}

Parameters parse_params(int argc, char* argv[])
{
    Parameters params = {
        .S = 15,
        .num_samples = 10,
        .num_threads = 8,
        .max_num_bounces = 100,
    };
    if(argc > 1) params.S = atoi(argv[1]);
    if(argc > 2) params.num_samples = atoi(argv[2]);
    if(argc > 3) params.num_threads = atoi(argv[3]);
    if(argc > 4) params.max_num_bounces = atoi(argv[4]);

    return params;
}

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

std::vector<Vec3> get_antialiasing_samples(int num_samples, bool even = false)
{
    std::vector<Vec3> samples;
    const float eps = 1e-4;

    if(even) {
        int N = int(sqrt(num_samples));
        float delta = (1-eps)/N;
        for(float x = 0; x < 1; x+=delta)
            for(float y = 0; y < 1; y+=delta)
                samples.push_back(Vec3(x, y, 0));
    } else {
        for(int i = 0; i < num_samples; i++)
            samples.push_back(Vec3(
                random_uniform(), random_uniform(), 0
            ));
    }

    return samples;
}

void test_scene()
{
    int S = 10;
    size_t W = 30*S, H = 12*S;
    cv::Mat img(H, W, CV_8UC3);

    Camera camera(Vec3(0,0.3,0), Vec3(0,0,1), Vec3(0,1,0), 1, 80, W, H, true);
    std::vector<std::shared_ptr<Light>> lights;
    //lights.push_back(std::shared_ptr<Light>(new SunLight(Vec3(1,1,1), 1, Vec3(1,1,0))));
    //lights.push_back(std::shared_ptr<Light>(new SunLight(Vec3(1,1,1), 1, Vec3(-1,1,0))));
     lights.push_back(std::shared_ptr<Light>(new SpotLight(Vec3(1,1,1), 200, Vec3(0,-2,2))));


    std::vector<std::shared_ptr<Geometry>> geometries;
    geometries.push_back(std::shared_ptr<Geometry>(new InfinitePlane(/*distnace*/ 1, /*normal*/ Vec3(0,-1,0), /*roughness*/ 1, /*specularity*/ 0.5, /*texture*/ std::shared_ptr<Texture>(new ConstantTexture(Vec3(0,0,1))))));
    geometries.push_back(std::shared_ptr<Geometry>(new Sphere(/*center*/ Vec3(0,0,3), /*radius*/ 0.8, /*roughness*/ 0.2, /*specularity*/ 0.9, /*texture*/ std::shared_ptr<Texture>(new ConstantTexture(Vec3(1,0,0))))));
    geometries.push_back(std::shared_ptr<Geometry>(new Sphere(/*center*/Vec3(-1.8,0,3), /*radius*/ 0.8, /*roughness*/ 0.1, /*specularity*/ 0.9, /*texture*/ std::shared_ptr<Texture>(new ConstantTexture(Vec3(1,1,0))))));
    geometries.push_back(std::shared_ptr<Geometry>(new Sphere(/*center*/ Vec3(1.8,0,3), /*radius*/ 0.8, /*roughness*/ 0.9, /*specularity*/ 0.4, /*texture*/ std::shared_ptr<Texture>(new ConstantTexture(Vec3(0,1,0))))));

    Vec3 ambient_color(0.2, 0.2, 0.2);
    RayTracer ray_tracer(geometries, lights, ambient_color);
    const int max_num_bounces = 20;
    int total_num_bounces = 0;
    std::vector<Vec3> antianliasing_samples = get_antialiasing_samples(10);

    for(float y = 0; y < H; y++)
    {
        for(float x = 0; x < W; x++)
        {
            Vec3 color(0,0,0);
            for(auto aa_sample: antianliasing_samples)
            {
                Ray camera_ray = camera.cast_ray_from_pixel(x+aa_sample.x(), y+aa_sample.y());
                total_num_bounces += ray_tracer.run(camera_ray, max_num_bounces);
                color += camera_ray.color;
            }
            color *= 1.0f/(antianliasing_samples.size());

            int idx = img.step*int(y) + 3*int(x);
            img.data[idx+0] = std::min(255, int(0.5+255*color.b()));
            img.data[idx+1] = std::min(255, int(0.5+255*color.g()));
            img.data[idx+2] = std::min(255, int(0.5+255*color.r()));
        }

        std::cout << "\r" << std::round(float(y+1)/H*100) << "%" << " completed.";
        std::cout.flush();
    }

    std::cout << "\nAverage: " << float(total_num_bounces)/(W*H) << " bounces/pixel." << std::endl;

    cv::imwrite("test.png", img);
}

bool spheres_have_interceptions(const std::shared_ptr<Sphere>& test_sphere, const std::vector<std::shared_ptr<Geometry>>& geometries)
{
    for(auto geometry: geometries)
    {
        Sphere* another_sphere = dynamic_cast<Sphere*>(geometry.get());
        if(another_sphere != nullptr)
            if(test_sphere->has_interception_with_sphere(*another_sphere))
                return true;
    }

    return false;
}

cv::Mat generate_image(int H, int W, int num_threads, int max_num_bounces, int num_samples,
    const Camera& camera,
    const std::vector<std::shared_ptr<Geometry>>& geometries,
    const std::vector<std::shared_ptr<Light>>& lights,
    const Vec3& ambient_color)
{
    cv::Mat img(H, W, CV_8UC3);
    std::vector<std::thread> threads;

    std::vector<Vec3> antialiasing_samples = get_antialiasing_samples(num_samples);

    for(int i = 0; i < num_threads; i++)
    {
        threads.push_back(std::thread(
            [i, num_threads, H, W, max_num_bounces, &camera, &geometries, &lights, &ambient_color, &antialiasing_samples, &img]
            {
                    RayTracer ray_tracer(geometries, lights, ambient_color);

                    int start_H  = int(float(H)/num_threads*i);
                    int end_H = int(float(H)/num_threads*(i+1));

                    for(float y = start_H; y < end_H; y++)
                    {
                        for(float x = 0; x < W; x++)
                        {
                            Vec3 color(0,0,0);
                            for(auto aa_sample: antialiasing_samples)
                            {
                                Ray camera_ray = camera.cast_ray_from_pixel(x+aa_sample.x(), y+aa_sample.y());
                                ray_tracer.run(camera_ray, max_num_bounces);
                                color += camera_ray.color;
                            }
                            color *= 1.0f/(antialiasing_samples.size());

                            int idx = img.step*int(y) + 3*int(x);
                            img.data[idx+0] = std::min(255, int(0.5+255*color.b()));
                            img.data[idx+1] = std::min(255, int(0.5+255*color.g()));
                            img.data[idx+2] = std::min(255, int(0.5+255*color.r()));
                        }

                        if(i == 0)
                        {
                            std::cout << "\r" << std::round(float(y+1)/(end_H-start_H)*100) << "%" << " completed.";
                            std::cout.flush();
                        }
                    }
            }
        ));
    }

    for(auto& thread: threads)
        thread.join();

    return img;
}

cv::Mat generate_image2(int H, int W, int num_threads, int max_num_bounces, int num_samples,
    const Camera& camera,
    const std::vector<std::shared_ptr<Geometry>>& geometries,
    const std::vector<std::shared_ptr<Light>>& lights,
    const Vec3& ambient_color)
{
    cv::Mat img(H, W, CV_8UC3, cv::Scalar::all(0));
    std::vector<std::thread> threads;

    std::vector<Vec3> antialiasing_samples = get_antialiasing_samples(num_samples);

    std::list<Vec3> pixel_coords;
    for(int y = 0; y < H; y++)
        for(int x = 0; x < W; x++)
            pixel_coords.push_back(Vec3(x, y, 0));
    const int total_pixels = H*W;

    std::mutex pop_pixel_mutex;
    for(int i = 0; i < num_threads; i++)
    {
        threads.push_back(std::thread(
            [i, num_threads, max_num_bounces, total_pixels, &camera, &geometries, &lights, &ambient_color, &antialiasing_samples, &img, &pixel_coords, &pop_pixel_mutex]()
            {
                RayTracer ray_tracer(geometries, lights, ambient_color);

                while(1)
                {
                    pop_pixel_mutex.lock();
                    int pixels_left = pixel_coords.size();
                    if(pixels_left <= 0) {
                        pop_pixel_mutex.unlock();
                        break;
                    }
                    Vec3 pixel_coord = pixel_coords.front();
                    pixel_coords.pop_front();
                    pop_pixel_mutex.unlock();

                    float x = pixel_coord.x();
                    float y = pixel_coord.y();

                    Vec3 color(0,0,0);
                    for(auto aa_sample: antialiasing_samples)
                    {
                        Ray camera_ray = camera.cast_ray_from_pixel(x+aa_sample.x(), y+aa_sample.y());
                        ray_tracer.run(camera_ray, max_num_bounces);
                        color += camera_ray.color;
                    }
                    color *= 1.0f/(antialiasing_samples.size());

                    int idx = img.step*int(y) + 3*int(x);
                    img.data[idx+0] = std::min(255, int(0.5+255*color.b()));
                    img.data[idx+1] = std::min(255, int(0.5+255*color.g()));
                    img.data[idx+2] = std::min(255, int(0.5+255*color.r()));

                    if(i == 0)
                    {
                        std::cout << "\r" << std::round(float(total_pixels - pixels_left)/total_pixels*100) << "%" << " completed.";
                        std::cout.flush();
                    }
                }
            }
        ));
    }

    for(auto& thread: threads)
        thread.join();

    return img;
}

void test_scene2(const Parameters& params)
{
    // parameters
    int S = params.S;
    int num_samples = params.num_samples;
    int num_threads = params.num_threads;
    int max_num_bounces = params.max_num_bounces;

    Vec3 ambient_color(0.2, 0.2, 0.2);

    // camera
    size_t W = 30*S, H = 20*S;
    Camera camera(Vec3(0,-0.8,0), Vec3(0,0.4,1), Vec3(0,1,0), 10, 32, W, H, true);

    // lights
    std::vector<std::shared_ptr<Light>> lights;
    // lights.push_back(std::shared_ptr<Light>(new SpotLight(Vec3(1,1,1), 100, Vec3(-2,-4,3))));
    // lights.push_back(std::shared_ptr<Light>(new SpotLight(Vec3(1,1,1), 100, Vec3(2,-4,3))));
    // lights.push_back(std::shared_ptr<Light>(new SunLight(Vec3(1,1,1), 0.8, Vec3(0,1,0))));

    // geometries
    std::vector<std::shared_ptr<Geometry>> geometries;
    geometries.push_back(std::shared_ptr<Geometry>(new InfinitePlane(/*distnace*/ 1, /*normal*/ Vec3(0,-1,0), /*roughness*/ 0.1, /*specularity*/ 0.5, /*texture*/ std::shared_ptr<Texture>(new ConstantTexture(Vec3(0.5,0.5,0.8))), /*is_emitter*/ false)));
    geometries.push_back(std::shared_ptr<Geometry>(new Sphere(/*center*/Vec3(1.2,0,2), /*radius*/ 1.0, /*roughness*/ 0.1, /*specularity*/ 0.9, /*texture*/ std::shared_ptr<Texture>(new ConstantTexture(Vec3(0.6,0.6,0))))));
    geometries.push_back(std::shared_ptr<Geometry>(new Sphere(/*center*/ Vec3(0,0,3), /*radius*/ 1.0, /*roughness*/ 0.2, /*specularity*/ 1, /*texture*/ std::shared_ptr<Texture>(new ConstantTexture(Vec3(0.6,0,0))))));
    geometries.push_back(std::shared_ptr<Geometry>(new Sphere(/*center*/ Vec3(-1.2,0,4), /*radius*/ 1.0, /*roughness*/ 0.9, /*specularity*/ 0.4, /*texture*/ std::shared_ptr<Texture>(new ConstantTexture(Vec3(0,0.8,0))), /*is_emitter*/ true)));

    int num_small_balls = 400;
    float small_ball_radius = 0.25;
    float distribute_range = 12;
    for(int i = 0; i < num_small_balls;) {
        std::shared_ptr<Sphere> small_ball(new Sphere(
            /*center*/ Vec3(
                random_uniform(-distribute_range, distribute_range),
                1-small_ball_radius,
                random_uniform(-distribute_range, distribute_range)),
            /*radius*/ small_ball_radius,
            /*roughness*/ random_uniform(0.1, 1),
            /*specularity*/ random_uniform(0.1, 1),
            /*texture*/ std::shared_ptr<Texture>(new ConstantTexture(
                Vec3(
                random_uniform(),
                random_uniform(),
                random_uniform()
                ))),
            /*is_emitter*/ random_uniform() > 0.8
        ));

        if(!spheres_have_interceptions(small_ball, geometries))
        {
            geometries.push_back(small_ball);
            i++;
        }
    }

    // generate image using ray tracing
    cv::Mat img = generate_image2(H, W, num_threads, max_num_bounces, num_samples, camera, geometries, lights, ambient_color);

    cv::imwrite("test.png", img);
}

void test_scene3(const Parameters& params)
{
    // parameters
    int S = params.S;
    int num_samples = params.num_samples;
    int num_threads = params.num_threads;
    int max_num_bounces = params.max_num_bounces;
    Vec3 ambient_color(0, 0, 0);

    // camera
    size_t W = 30*S, H = 20*S;
    Camera camera(Vec3(0,-0.8,0), Vec3(0,0.4,1), Vec3(0,1,0), 10, 32, W, H, true);

    // lights
    std::vector<std::shared_ptr<Light>> lights;

    // geometries
    std::vector<std::shared_ptr<Geometry>> geometries;
    geometries.push_back(std::shared_ptr<Geometry>(new InfinitePlane(/*distnace*/ 1, /*normal*/ Vec3(0,-1,0), /*roughness*/ 0.1, /*specularity*/ 0.2, /*texture*/ std::shared_ptr<Texture>(new ConstantTexture(Vec3(1,1,1))), /*is_emitter*/ false)));
    geometries.push_back(std::shared_ptr<Geometry>(new Sphere(/*center*/ Vec3(0,0,3), /*radius*/ 1.0, /*roughness*/ 0.5, /*specularity*/ 0.5, /*texture*/ std::shared_ptr<Texture>(new ConstantTexture(Vec3(1,0,0))), /*is_emitter*/ false)));

    {
        std::shared_ptr<TransformGeometry> geometry = std::shared_ptr<TransformGeometry>(new Rectangle(/*half_width*/ 2, /*half_height*/ 2, /*roughness*/ 0, /*specularity*/ 1, /*texture*/ std::shared_ptr<Texture>(new ConstantTexture(Vec3(1,1,1)*10)), /*is_emitter*/ true));
        geometry->transform().x = Vec3(0, 0,1);
        geometry->transform().y = Vec3(0, 1,0);
        geometry->transform().z = Vec3(-1,0,0);
        geometry->transform().t = Vec3(2.8,0.2,3.5);
        geometries.push_back(geometry);
    }

    // generate image using ray tracing
    cv::Mat img = generate_image2(H, W, num_threads, max_num_bounces, num_samples, camera, geometries, lights, ambient_color);

    cv::imwrite("test.png", img);
}

void test_scene4(const Parameters& params)
{
    // parameters
    int S = params.S;
    int num_samples = params.num_samples;
    int num_threads = params.num_threads;
    int max_num_bounces = params.max_num_bounces;
    Vec3 ambient_color(0.2,0.2,0.2);

    // camera
    size_t W = 10*S, H = 10*S;
    Camera camera(Vec3(0,-0.2,-1), Vec3(0,0.2,1), Vec3(0,1,0), 4, 28, W, H, true);

    // lights
    std::vector<std::shared_ptr<Light>> lights;

    // geometries
    std::vector<std::shared_ptr<Geometry>> geometries;

    float roughness = 0.4;
    float specularity = 0.1;
    // bottom
    geometries.push_back(std::shared_ptr<Geometry>(new InfinitePlane(/*distnace*/ 1, /*normal*/ Vec3(0,-1,0), /*roughness*/ roughness, /*specularity*/ specularity, /*texture*/ std::shared_ptr<Texture>(new ConstantTexture(Vec3(1,1,1))), /*is_emitter*/ false)));
    // top
    geometries.push_back(std::shared_ptr<Geometry>(new InfinitePlane(/*distnace*/ 1, /*normal*/ Vec3(0,1,0), /*roughness*/ roughness, /*specularity*/ specularity, /*texture*/ std::shared_ptr<Texture>(new ConstantTexture(Vec3(1,1,1))), /*is_emitter*/ false)));
    // left
    geometries.push_back(std::shared_ptr<Geometry>(new InfinitePlane(/*distnace*/ 1, /*normal*/ Vec3(1,0,0), /*roughness*/ roughness, /*specularity*/ specularity, /*texture*/ std::shared_ptr<Texture>(new ConstantTexture(Vec3(0,1,0))), /*is_emitter*/ false)));
    // right
    geometries.push_back(std::shared_ptr<Geometry>(new InfinitePlane(/*distnace*/ 1, /*normal*/ Vec3(-1,0,0), /*roughness*/ roughness, /*specularity*/ specularity, /*texture*/ std::shared_ptr<Texture>(new ConstantTexture(Vec3(1,0,0))), /*is_emitter*/ false)));
    // front
    geometries.push_back(std::shared_ptr<Geometry>(new InfinitePlane(/*distnace*/ 2.5, /*normal*/ Vec3(0,0,-1), /*roughness*/ roughness, /*specularity*/ specularity, /*texture*/ std::shared_ptr<Texture>(new ConstantTexture(Vec3(1,1,1))), /*is_emitter*/ false)));

    {
        std::shared_ptr<TransformGeometry> cube(new Cube(/*half_x*/ 0.3, /*half_y*/ 0.3, /*half_z*/ 0.3, /*roughness*/ 0.8, /*specularity*/ 0.5, /*texture*/ std::shared_ptr<Texture>(new ConstantTexture(Vec3(1,1,1))), /*is_emitter*/ false));
        cube->transform().euler(0, -30, 0);
        cube->transform().t = Vec3(0.3,0.7,0);
        geometries.push_back(cube);
    }
    {
        std::shared_ptr<TransformGeometry> cube(new Cube(/*half_x*/ 0.3, /*half_y*/ 0.6, /*half_z*/ 0.3, /*roughness*/ 0.8, /*specularity*/ 0.5, /*texture*/ std::shared_ptr<Texture>(new ConstantTexture(Vec3(1,1,1))), /*is_emitter*/ false));
        cube->transform().euler(0, -30, 0);
        cube->transform().t = Vec3(-0.2,0.4,1);
        geometries.push_back(cube);
    }
    {
        std::shared_ptr<TransformGeometry> geometry(new Rectangle(/*half_width*/ 0.3, /*half_height*/ 0.3, /*roughness*/ 0, /*specularity*/ 1, /*texture*/ std::shared_ptr<Texture>(new ConstantTexture(Vec3(1,1,1)*50)), /*is_emitter*/ true));
        geometry->transform().x = Vec3(1,0,0);
        geometry->transform().y = Vec3(0,0,1);
        geometry->transform().z = Vec3(0,1,0);
        geometry->transform().t = Vec3(0,-0.999,1);
        geometries.push_back(geometry);
    }

    // generate image using ray tracing
    cv::Mat img = generate_image2(H, W, num_threads, max_num_bounces, num_samples, camera, geometries, lights, ambient_color);

    cv::imwrite("test.png", img);
}

void test_scene5(const Parameters& params)
{
    // parameters
    int S = params.S;
    int num_samples = params.num_samples;
    int num_threads = params.num_threads;
    int max_num_bounces = params.max_num_bounces;
    Vec3 ambient_color(0.6,0.6,0.8);

    // camera
    size_t W = 30*S, H = 20*S;
    Camera camera(Vec3(0,-0.2,-1), Vec3(0,0.18,1), Vec3(0,1,0), 4, 25, W, H, true);

    // lights
    std::vector<std::shared_ptr<Light>> lights;
    lights.push_back(std::shared_ptr<Light>(new SunLight(Vec3(1,1,1), 2, Vec3(0,1,0))));

    // geometries
    std::vector<std::shared_ptr<Geometry>> geometries;
    geometries.push_back(std::shared_ptr<Geometry>(new InfinitePlane(/*distnace*/ 1, /*normal*/ Vec3(0,-1,0), /*roughness*/ 0.5, /*specularity*/ 0.2, /*texture*/ std::shared_ptr<Texture>(new PerlinNoiseTexture()), /*is_emitter*/ false)));
    geometries.push_back(std::shared_ptr<Geometry>(new Sphere(/*center*/ Vec3(0,0,5), /*radius*/ 1.0, /*roughness*/ 0.5, /*specularity*/ 0.5, /*texture*/ std::shared_ptr<Texture>(new PerlinNoiseTexture(
        [](const Vec3& co)
        {
            return 8*co;
        },
        [](const Vec3& co, float noise)
        {
            float v = 0.8*(0.5*(1+sin(8*co.x()+4*noise)));
            return Vec3(v,v,v);
        }
    )), /*is_emitter*/ false)));

    // generate image using ray tracing
    cv::Mat img = generate_image2(H, W, num_threads, max_num_bounces, num_samples, camera, geometries, lights, ambient_color);

    cv::imwrite("test.png", img);
}
///////////////////////////////////////////////////////////////////
//
// main()
//
///////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    Parameters params = parse_params(argc, argv);

    // test_Vec3();
    // test_Camera();
    auto t_start = std::chrono::high_resolution_clock::now();
    test_scene5(params);
    auto t_end = std::chrono::high_resolution_clock::now();
    std::cout << "Elpased time (seconds) = " << std::chrono::duration_cast<std::chrono::seconds>(t_end-t_start).count() << std::endl;
    return 0;
}