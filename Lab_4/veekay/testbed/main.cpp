#include <cstdint>
#include <climits>
#include <cstring>
#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>

#include <veekay/veekay.hpp>

#include <vulkan/vulkan_core.h>
#include <imgui.h>
#include <lodepng.h>

namespace {

constexpr uint32_t max_models = 1024;

// ============================================================================
// СТРУКТУРЫ ДАННЫХ
// ============================================================================

// Вершина содержит геометрические данные: позицию, нормаль и текстурные координаты
struct Vertex {
    veekay::vec3 position;
    veekay::vec3 normal;
    veekay::vec2 uv;
};

// Направленный свет (один глобальный источник, например, солнце)
struct DirectionalLight {
    veekay::vec3 direction;     // направление света
    float _pad0;
    veekay::vec3 ambient;       // фоновое освещение
    float _pad1;
    veekay::vec3 diffuse;       // рассеянное освещение
    float _pad2;
    veekay::vec3 specular;      // отраженное освещение (блики)
    float _pad3;
};

// Точечный источник света (например, лампочка)
// Затухает по закону обратных квадратов расстояния
struct PointLight {
    veekay::vec3 position;      // позиция источника
    float _pad0;
    veekay::vec3 ambient;
    float _pad1;
    veekay::vec3 diffuse;
    float _pad2;
    veekay::vec3 specular;
    float constant;             // коэффициенты затухания
    float linear;
    float quadratic;
    float _pad3;
};

// Прожектор (направленный конус света)
// Имеет позицию, направление и угол раскрытия с гладкими краями
struct SpotLight {
    veekay::vec3 position;      // позиция прожектора
    float _pad0;
    veekay::vec3 direction;     // направление луча
    float _pad1;
    veekay::vec3 ambient;
    float _pad2;
    veekay::vec3 diffuse;
    float _pad3;
    veekay::vec3 specular;
    float constant;             // коэффициенты затухания
    float linear;
    float quadratic;
    float cutOff;               // внутренний угол конуса (cos)
    float outerCutOff;          // внешний угол для плавного перехода (cos)
    float _pad4[3];
};

// Uniform-буфер для данных сцены (передается в шейдеры)
struct SceneUniforms {
    veekay::mat4 view_projection;       // матрица камеры
    veekay::vec3 view_position;         // позиция камеры
    float _pad0;
    DirectionalLight directional_light; // направленный свет
    uint32_t point_light_count;         // количество точечных источников
    uint32_t spot_light_count;          // количество прожекторов
    float _pad1[2];
    veekay::mat4 light_space_matrix;    // матрица для shadow mapping (offset 160, кратен 16)
};

// Uniform-буфер для данных модели (отдельный для каждого объекта)
struct ModelUniforms {
    veekay::mat4 model;                 // матрица трансформации модели
    veekay::vec3 albedo_color;          // диффузный цвет
    float shininess;                    // степень блеска (1-256)
    veekay::vec3 specular_color;        // цвет бликов
    float _pad0;
};

// Материал объекта (свойства поверхности + текстура)
struct Material {
    veekay::vec3 albedo;                // основной цвет (диффузный компонент)
    veekay::vec3 specular;              // цвет отражения (specular компонент)
    float shininess;                    // блеск (чем выше, тем острее блики)
    
    // Текстурирование
    veekay::graphics::Texture* texture; // текстура материала
    VkSampler sampler;                  // сэмплер для текстуры
    VkDescriptorSet descriptor_set;     // набор дескрипторов для этого материала
};

// Меш (геометрия объекта): буферы вершин и индексов
struct Mesh {
    veekay::graphics::Buffer* vertex_buffer;
    veekay::graphics::Buffer* index_buffer;
    uint32_t indices;                   // количество индексов для отрисовки
};

// Трансформация объекта в пространстве
struct Transform {
    veekay::vec3 position = {};
    veekay::vec3 scale = {1.0f, 1.0f, 1.0f};
    veekay::vec3 rotation = {};         // углы Эйлера в градусах

    veekay::mat4 matrix() const;        // вычисляет итоговую матрицу трансформации
};

// 3D-модель: геометрия + трансформация + материал
struct Model {
    Mesh mesh;
    Transform transform;
    Material material;
};

// Камера с Look-At матрицей (положение + ориентация через yaw/pitch)
struct Camera {
    constexpr static float default_fov = 60.0f;
    constexpr static float default_near_plane = 0.01f;
    constexpr static float default_far_plane = 100.0f;

    veekay::vec3 position = {};
    float yaw = 0.0f;                   // поворот по горизонтали (градусы)
    float pitch = 0.0f;                 // наклон вверх/вниз (градусы)

    float fov = default_fov;
    float near_plane = default_near_plane;
    float far_plane = default_far_plane;

    veekay::vec3 front() const;         // направление взгляда камеры
    veekay::vec3 right() const;         // правый вектор камеры

    veekay::mat4 view() const;          // матрица вида
    veekay::mat4 view_projection(float aspect_ratio) const;
};

// ============================================================================
// ГЛОБАЛЬНЫЕ ОБЪЕКТЫ СЦЕНЫ
// ============================================================================

inline namespace {
    // Камера начинается над сценой, смотрит вниз
    Camera camera{
        .position = {0.4f, -4.20f, 8.0f},
        .yaw = 0.0f,
        .pitch = 20.0f,
    };

    std::vector<Model> models;          // все 3D-объекты в сцене
}

// ============================================================================
// VULKAN ОБЪЕКТЫ
// ============================================================================

inline namespace {
    VkShaderModule vertex_shader_module;
    VkShaderModule fragment_shader_module;

    VkDescriptorPool descriptor_pool;
    VkDescriptorSetLayout descriptor_set_layout;

    VkPipelineLayout pipeline_layout;
    VkPipeline pipeline;

    // Uniform-буферы для передачи данных в шейдеры
    veekay::graphics::Buffer* scene_uniforms_buffer;
    veekay::graphics::Buffer* model_uniforms_buffer;

    // Storage-буферы для динамических массивов источников света
    veekay::graphics::Buffer* point_lights_buffer;
    veekay::graphics::Buffer* spot_lights_buffer;
    
    constexpr uint32_t max_point_lights = 16;
    constexpr uint32_t max_spot_lights = 16;
    
    // Массивы источников света (на CPU стороне)
    DirectionalLight directional_light{
        .direction = {0.0f, 1.0f, 0.0f},
        .ambient = {0.05f, 0.05f, 0.05f},
        .diffuse = {0.8f, 0.8f, 0.8f},
        .specular = {1.0f, 1.0f, 1.0f},
    };
    bool directional_light_enabled = false;
    
    std::vector<PointLight> point_lights;
    std::vector<SpotLight> spot_lights;
    
    // Тип источника света для теней
    enum class ShadowLightType { Directional, Point, Spot };
    ShadowLightType shadow_light_type = ShadowLightType::Point;
    int shadow_light_index = 0;  // Индекс выбранного источника

    // Предзагруженная геометрия
    Mesh plane_mesh;
    Mesh cube_mesh;
    Mesh sphere_mesh;

    // Заглушка для текстур (розовая шахматная доска)
    veekay::graphics::Texture* missing_texture;
    VkSampler missing_texture_sampler;
    
    // Загруженные текстуры для разных материалов
    veekay::graphics::Texture* cube_texture;
    VkSampler cube_sampler;
    
    veekay::graphics::Texture* sphere_texture;
    VkSampler sphere_sampler;
    
    veekay::graphics::Texture* floor_texture;
    VkSampler floor_sampler;

    // ========================================================================
    // SHADOW MAPPING ОБЪЕКТЫ
    // ========================================================================

    // Параметры shadow map
    constexpr uint32_t shadow_map_size = 2048;  // Разрешение карты теней (чем больше - тем четче тени)

    // Текстура глубины для shadow map
    VkImage shadow_map_image;
    VkImageView shadow_map_view;
    VkDeviceMemory shadow_map_memory;

    // Сэмплер с поддержкой сравнения для PCF
    VkSampler shadow_map_sampler;

    // Шейдеры для shadow pass
    VkShaderModule shadow_vertex_shader_module;
    VkShaderModule shadow_fragment_shader_module;

    // Pipeline для shadow pass (отрисовка в depth buffer)
    VkPipelineLayout shadow_pipeline_layout;
    VkPipeline shadow_pipeline;

    // Descriptor set layout и pool для shadow pass
    VkDescriptorSetLayout shadow_descriptor_set_layout;
    VkDescriptorSet shadow_descriptor_set;

    // Render pass и framebuffer для shadow pass
    VkRenderPass shadow_render_pass;
    VkFramebuffer shadow_framebuffer;

    // Буфер для матрицы light space
    veekay::graphics::Buffer* light_space_buffer;
    
    // Расширенный descriptor pool для множества наборов дескрипторов
    constexpr uint32_t max_descriptor_sets = 32;
}

// ============================================================================
// ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
// ============================================================================

// Конвертация градусов в радианы
float toRadians(float degrees) {
    return degrees * float(M_PI) / 180.0f;
}

// Вычисление матрицы трансформации: Translation * Rotation * Scale
veekay::mat4 Transform::matrix() const {
    // Матрица масштабирования
    veekay::mat4 s = veekay::mat4::identity();
    s.elements[0][0] = scale.x;
    s.elements[1][1] = scale.y;
    s.elements[2][2] = scale.z;
    
    // Углы Эйлера в радианах
    float rx = toRadians(rotation.x);
    float ry = toRadians(rotation.y);
    float rz = toRadians(rotation.z);
    
    // Матрица вращения вокруг оси X
    veekay::mat4 rot_x = veekay::mat4::identity();
    rot_x.elements[1][1] = cosf(rx);
    rot_x.elements[1][2] = -sinf(rx);
    rot_x.elements[2][1] = sinf(rx);
    rot_x.elements[2][2] = cosf(rx);
    
    // Матрица вращения вокруг оси Y
    veekay::mat4 rot_y = veekay::mat4::identity();
    rot_y.elements[0][0] = cosf(ry);
    rot_y.elements[0][2] = sinf(ry);
    rot_y.elements[2][0] = -sinf(ry);
    rot_y.elements[2][2] = cosf(ry);
    
    // Матрица вращения вокруг оси Z
    veekay::mat4 rot_z = veekay::mat4::identity();
    rot_z.elements[0][0] = cosf(rz);
    rot_z.elements[0][1] = -sinf(rz);
    rot_z.elements[1][0] = sinf(rz);
    rot_z.elements[1][1] = cosf(rz);
    
    // Матрица перемещения
    veekay::mat4 t = veekay::mat4::translation(position);
    
    // Порядок: Translation * RotationZ * RotationY * RotationX * Scale
    // ВАЖНО: veekay::mat4::operator* выполняет умножение в обратном порядке (B * A)
    // Поэтому пишем в обратном порядке: s * rot * t -> T * R * S
    return s * (rot_x * (rot_y * (rot_z * t)));
}

// Вычисление направления взгляда камеры из углов yaw/pitch
veekay::vec3 Camera::front() const {
    float yaw_rad = toRadians(yaw);
    float pitch_rad = toRadians(pitch);
    
    veekay::vec3 direction;
    direction.x = cosf(pitch_rad) * sinf(yaw_rad);
    direction.y = -sinf(pitch_rad);  // инвертируем для Y-down системы координат
    direction.z = cosf(pitch_rad) * cosf(yaw_rad);
    
    // Нормализация вектора
    float length = sqrtf(direction.x * direction.x + 
                        direction.y * direction.y + 
                        direction.z * direction.z);
    if (length > 0.0f) {
        direction.x /= length;
        direction.y /= length;
        direction.z /= length;
    }
    
    return direction;
}

// Вычисление правого вектора камеры (перпендикулярно направлению взгляда)
veekay::vec3 Camera::right() const {
    veekay::vec3 f = front();
    veekay::vec3 world_up = {0.0f, -1.0f, 0.0f};  // Y-down система
    
    // Векторное произведение: world_up × front
    veekay::vec3 right_vec;
    right_vec.x = world_up.y * f.z - world_up.z * f.y;
    right_vec.y = world_up.z * f.x - world_up.x * f.z;
    right_vec.z = world_up.x * f.y - world_up.y * f.x;
    
    // Нормализация
    float length = sqrtf(right_vec.x * right_vec.x + 
                        right_vec.y * right_vec.y + 
                        right_vec.z * right_vec.z);
    if (length > 0.0f) {
        right_vec.x /= length;
        right_vec.y /= length;
        right_vec.z /= length;
    }
    
    return right_vec;
}

// Построение Look-At матрицы вида
veekay::mat4 Camera::view() const {
    veekay::vec3 f = front();
    veekay::vec3 r = right();
    
    // Вычисляем вектор up: right × front
    veekay::vec3 u;
    u.x = r.y * f.z - r.z * f.y;
    u.y = r.z * f.x - r.x * f.z;
    u.z = r.x * f.y - r.y * f.x;

    veekay::mat4 result = veekay::mat4::identity();
    
    // Заполняем матрицу вида
    result.elements[0][0] = r.x;
    result.elements[1][0] = r.y;
    result.elements[2][0] = r.z;
    
    result.elements[0][1] = u.x;
    result.elements[1][1] = u.y;
    result.elements[2][1] = u.z;
    
    result.elements[0][2] = -f.x;
    result.elements[1][2] = -f.y;
    result.elements[2][2] = -f.z;
    
    // Скалярное произведение для смещения
    result.elements[3][0] = -(r.x * position.x + r.y * position.y + r.z * position.z);
    result.elements[3][1] = -(u.x * position.x + u.y * position.y + u.z * position.z);
    result.elements[3][2] = (f.x * position.x + f.y * position.y + f.z * position.z);
    
    return result;
}

// Комбинированная матрица: вид × проекция
veekay::mat4 Camera::view_projection(float aspect_ratio) const {
    auto projection = veekay::mat4::projection(fov, aspect_ratio, near_plane, far_plane);
    return view() * projection;
}

// Загрузка скомпилированного SPIR-V шейдера из файла
VkShaderModule loadShaderModule(const char* path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    size_t size = file.tellg();
    std::vector<uint32_t> buffer(size / sizeof(uint32_t));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(buffer.data()), size);
    file.close();

    VkShaderModuleCreateInfo info{
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = size,
        .pCode = buffer.data(),
    };

    VkShaderModule result;
    if (vkCreateShaderModule(veekay::app.vk_device, &
                             info, nullptr, &result) != VK_SUCCESS) {
        return nullptr;
    }

    return result;
}

// Загрузка текстуры из PNG файла с использованием lodepng
veekay::graphics::Texture* loadTexture(VkCommandBuffer cmd, const char* path) {
    std::vector<unsigned char> image;
    unsigned width, height;
    
    // Загружаем PNG файл
    unsigned error = lodepng::decode(image, width, height, path);
    
    if (error) {
        std::cerr << "Failed to load texture from " << path << ": " 
                  << lodepng_error_text(error) << "\n";
        return nullptr;
    }
    
    std::cout << "Loaded texture: " << path << " (" << width << "x" << height << ")\n";
    
    // Создаем текстуру из загруженных пикселей
    // RGBA формат, 8 бит на канал
    return new veekay::graphics::Texture(
        cmd, 
        width, 
        height,
        VK_FORMAT_R8G8B8A8_UNORM,  // RGBA 8-bit unsigned normalized
        image.data()
    );
}

// Создание сэмплера с указанными параметрами фильтрации
VkSampler createSampler(VkFilter filter = VK_FILTER_LINEAR, 
                       VkSamplerAddressMode addressMode = VK_SAMPLER_ADDRESS_MODE_REPEAT) {
    VkDevice& device = veekay::app.vk_device;
    
    VkSamplerCreateInfo info{
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter = filter,           // Увеличение текселя (при приближении)
        .minFilter = filter,           // Уменьшение текселя (при удалении)
        .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,  // Линейная интерполяция между mip-уровнями
        .addressModeU = addressMode,   // Повтор/зажим по оси U
        .addressModeV = addressMode,   // Повтор/зажим по оси V
        .addressModeW = addressMode,   // Повтор/зажим по оси W
        .mipLodBias = 0.0f,
        .anisotropyEnable = VK_TRUE,   // Включаем анизотропную фильтрацию
        .maxAnisotropy = 16.0f,        // Максимальная степень анизотропии
        .minLod = 0.0f,
        .maxLod = VK_LOD_CLAMP_NONE,   // Без ограничения на mip-уровни
    };
    
    VkSampler sampler;
    if (vkCreateSampler(device, &info, nullptr, &sampler) != VK_SUCCESS) {
        std::cerr << "Failed to create Vulkan sampler\n";
        return VK_NULL_HANDLE;
    }
    
    return sampler;
}

// ============================================================================
// ФУНКЦИЯ ДЛЯ ВЫЧИСЛЕНИЯ МАТРИЦЫ LIGHT SPACE
// ============================================================================
veekay::mat4 calculateLightSpaceMatrix(const veekay::vec3& light_position) {
    // Параметры ортографической проекции
    float left = -10.0f;
    float right = 10.0f;
    float bottom = -10.0f;
    float top = 10.0f;
    float near_plane = 0.1f;
    float far_plane = 20.0f;
    
    // Строим простую lookAt матрицу
    veekay::vec3 eye = light_position;
    veekay::vec3 center = {0.0f, 0.0f, 0.0f};  // Смотрим на центр сцены
    veekay::vec3 up = {0.0f, 0.0f, 1.0f};      // Z-axis как up (т.к. свет вдоль Y)
    
    // Forward = от камеры к цели (нормализованный)
    veekay::vec3 f = {
        center.x - eye.x,
        center.y - eye.y,
        center.z - eye.z
    };
    float f_len = std::sqrt(f.x*f.x + f.y*f.y + f.z*f.z);
    f.x /= f_len; f.y /= f_len; f.z /= f_len;
    
    // Right = forward × up
    veekay::vec3 r = {
        f.y * up.z - f.z * up.y,
        f.z * up.x - f.x * up.z,
        f.x * up.y - f.y * up.x
    };
    float r_len = std::sqrt(r.x*r.x + r.y*r.y + r.z*r.z);
    r.x /= r_len; r.y /= r_len; r.z /= r_len;
    
    // Up = right × forward
    veekay::vec3 u = {
        r.y * f.z - r.z * f.y,
        r.z * f.x - r.x * f.z,
        r.x * f.y - r.y * f.x
    };
    
    // View matrix (стандартная правосторонняя, камера смотрит в -Z)
    veekay::mat4 view = veekay::mat4::identity();
    view.elements[0][0] = r.x;
    view.elements[1][0] = r.y;
    view.elements[2][0] = r.z;
    view.elements[3][0] = -(r.x * eye.x + r.y * eye.y + r.z * eye.z);
    
    view.elements[0][1] = u.x;
    view.elements[1][1] = u.y;
    view.elements[2][1] = u.z;
    view.elements[3][1] = -(u.x * eye.x + u.y * eye.y + u.z * eye.z);
    
    view.elements[0][2] = -f.x;  // Камера смотрит в -Z
    view.elements[1][2] = -f.y;
    view.elements[2][2] = -f.z;
    view.elements[3][2] = (f.x * eye.x + f.y * eye.y + f.z * eye.z);
    
    // Orthographic projection для Vulkan [0,1] depth
    // Z_view отрицательный для объектов перед камерой: [-far, -near]
    // Преобразование: Z_ndc = -(Z_view + near) / (far - near)
    // Это маппит Z_view=-near → 0 и Z_view=-far → 1
    veekay::mat4 proj = veekay::mat4::identity();
    proj.elements[0][0] = 2.0f / (right - left);
    proj.elements[1][1] = 2.0f / (top - bottom);
    proj.elements[2][2] = -1.0f / (far_plane - near_plane);
    proj.elements[3][0] = -(right + left) / (right - left);
    proj.elements[3][1] = -(top + bottom) / (top - bottom);
    proj.elements[3][2] = -near_plane / (far_plane - near_plane);
    
    // ВАЖНО: veekay::mat4::operator* выполняет умножение в обратном порядке (B * A)
    // Поэтому чтобы получить Proj * View, нужно написать view * proj
    return view * proj;
}

// ============================================================================
// ИНИЦИАЛИЗАЦИЯ
// ============================================================================

void initialize(VkCommandBuffer cmd) {
    VkDevice& device = veekay::app.vk_device;

    // ========================================================================
    // Построение графического пайплайна
    // ========================================================================
    {
        // Загрузка вершинного и фрагментного шейдеров
        vertex_shader_module = loadShaderModule("./shaders/shader.vert.spv");
        if (!vertex_shader_module) {
            std::cerr << "Failed to load Vulkan vertex shader from file\n";
            veekay::app.running = false;
            return;
        }

        fragment_shader_module = loadShaderModule("./shaders/shader.frag.spv");
        if (!fragment_shader_module) {
            std::cerr << "Failed to load Vulkan fragment shader from file\n";
            veekay::app.running = false;
            return;
        }

        VkPipelineShaderStageCreateInfo stage_infos[2];

        stage_infos[0] = VkPipelineShaderStageCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_VERTEX_BIT,
            .module = vertex_shader_module,
            .pName = "main",
        };

        stage_infos[1] = VkPipelineShaderStageCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
            .module = fragment_shader_module,
            .pName = "main",
        };

        // Описание формата вершин (размер структуры Vertex)
        VkVertexInputBindingDescription buffer_binding{
            .binding = 0,
            .stride = sizeof(Vertex),
            .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
        };

        // Атрибуты вершин: позиция, нормаль, UV
        VkVertexInputAttributeDescription attributes[] = {
            {
                .location = 0,
                .binding = 0,
                .format = VK_FORMAT_R32G32B32_SFLOAT,
                .offset = offsetof(Vertex, position),
            },
            {
                .location = 1,
                .binding = 0,
                .format = VK_FORMAT_R32G32B32_SFLOAT,
                .offset = offsetof(Vertex, normal),
            },
            {
                .location = 2,
                .binding = 0,
                .format = VK_FORMAT_R32G32_SFLOAT,
                .offset = offsetof(Vertex, uv),
            },
        };

        VkPipelineVertexInputStateCreateInfo input_state_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            .vertexBindingDescriptionCount = 1,
            .pVertexBindingDescriptions = &buffer_binding,
            .vertexAttributeDescriptionCount = sizeof(attributes) / sizeof(attributes[0]),
            .pVertexAttributeDescriptions = attributes,
        };

        // Интерпретация вершин как списка треугольников
        VkPipelineInputAssemblyStateCreateInfo assembly_state_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        };

        // Настройки растеризации: отсечение задних граней, заполнение треугольников
        VkPipelineRasterizationStateCreateInfo raster_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            .polygonMode = VK_POLYGON_MODE_FILL,
            .cullMode = VK_CULL_MODE_NONE,  // ИЗМЕНЕНО: отключаем отсечение граней
            .frontFace = VK_FRONT_FACE_CLOCKWISE,
            .lineWidth = 1.0f,
        };

        VkPipelineMultisampleStateCreateInfo sample_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
            .sampleShadingEnable = false,
            .minSampleShading = 1.0f,
        };

        VkViewport viewport{
            .x = 0.0f,
            .y = 0.0f,
            .width = static_cast<float>(veekay::app.window_width),
            .height = static_cast<float>(veekay::app.window_height),
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        };

        VkRect2D scissor{
            .offset = {0, 0},
            .extent = {veekay::app.window_width, veekay::app.window_height},
        };

        VkPipelineViewportStateCreateInfo viewport_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            .viewportCount = 1,
            .pViewports = &viewport,
            .scissorCount = 1,
            .pScissors = &scissor,
        };

        // Включение depth-тестирования
        VkPipelineDepthStencilStateCreateInfo depth_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
            .depthTestEnable = true,
            .depthWriteEnable = true,  // ИЗМЕНЕНО: отключаем запись глубины для прозрачности
            .depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
        };

        VkPipelineColorBlendAttachmentState attachment_info{
            .blendEnable = VK_TRUE,  // ВКЛЮЧАЕМ смешивание
            .srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA,
            .dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
            .colorBlendOp = VK_BLEND_OP_ADD,
            .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
            .dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
            .alphaBlendOp = VK_BLEND_OP_ADD,
            .colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
                            VK_COLOR_COMPONENT_G_BIT |
                            VK_COLOR_COMPONENT_B_BIT |
                            VK_COLOR_COMPONENT_A_BIT,
        };

        VkPipelineColorBlendStateCreateInfo blend_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            .logicOpEnable = false,
            .logicOp = VK_LOGIC_OP_COPY,
            .attachmentCount = 1,
            .pAttachments = &attachment_info
        };

        // Создание descriptor pool для хранения descriptor sets
        {
            VkDescriptorPoolSize pools[] = {
                {
                    .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                    .descriptorCount = max_descriptor_sets * 2,  // Увеличиваем количество
                },
                {
                    .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
                    .descriptorCount = max_descriptor_sets * 2,
                },
                {
                    .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    .descriptorCount = max_descriptor_sets * 3,  // Увеличено: текстура + shadow map + shadow map raw
                },
                {
                    .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .descriptorCount = max_descriptor_sets * 2,
                }
            };
            
            VkDescriptorPoolCreateInfo info{
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                .maxSets = max_descriptor_sets,  // Множество наборов дескрипторов
                .poolSizeCount = sizeof(pools) / sizeof(pools[0]),
                .pPoolSizes = pools,
            };

            if (vkCreateDescriptorPool(device, &info, nullptr,
                                       &descriptor_pool) != VK_SUCCESS) {
                std::cerr << "Failed to create Vulkan descriptor pool\n";
                veekay::app.running = false;
                return;
            }
        }

        // Описание layout для descriptor set
        // Binding 0: SceneUniforms (uniform buffer)
        // Binding 1: ModelUniforms (dynamic uniform buffer)
        // Binding 2: PointLights (storage buffer)
        // Binding 3: SpotLights (storage buffer)
        // Binding 4: Texture + Sampler (combined image sampler)

        {
            VkDescriptorSetLayoutBinding bindings[] = {
                {
                    .binding = 0,
                    .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                },
                {
                    .binding = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                },
                {
                    .binding = 2,
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
                },
                {
                    .binding = 3,
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
                },
                {
                    .binding = 4,  // Текстура материала + сэмплер
                    .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
                },
                {
                    .binding = 5,  // Shadow Map + сэмплер с PCF
                    .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
                },
                {
                    .binding = 6,  // Shadow Map raw (для чтения глубины без сравнения)
                    .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
                },
            };

            VkDescriptorSetLayoutCreateInfo info{
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                .bindingCount = sizeof(bindings) / sizeof(bindings[0]),
                .pBindings = bindings,
            };

            if (vkCreateDescriptorSetLayout(device, &info, nullptr,
                                            &descriptor_set_layout) != VK_SUCCESS) {
                std::cerr << "Failed to create Vulkan descriptor set layout\n";
                veekay::app.running = false;
                return;
            }
        }

        // УДАЛЯЕМ старое выделение единственного descriptor_set
        // Теперь каждый материал будет иметь свой набор дескрипторов

        // Выделение descriptor set из pool
        // {
        //     VkDescriptorSetAllocateInfo info{
        //         .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        //         .descriptorPool = descriptor_pool,
        //         .descriptorSetCount = 1,
        //         .pSetLayouts = &descriptor_set_layout,
        //     };

        //     if (vkAllocateDescriptorSets(device, &info, &descriptor_set) != VK_SUCCESS) {
        //         std::cerr << "Failed to create Vulkan descriptor set\n";
        //         veekay::app.running = false;
        //         return;
        //     }
        // }

        // Создание pipeline layout
        VkPipelineLayoutCreateInfo layout_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 1,
            .pSetLayouts = &descriptor_set_layout,
        };

        if (vkCreatePipelineLayout(device, &layout_info,
                                   nullptr, &pipeline_layout) != VK_SUCCESS) {
            std::cerr << "Failed to create Vulkan pipeline layout\n";
            veekay::app.running = false;
            return;
        }
        
        // Финальная сборка graphics pipeline
        VkGraphicsPipelineCreateInfo info{
            .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            .stageCount = 2,
            .pStages = stage_infos,
            .pVertexInputState = &input_state_info,
            .pInputAssemblyState = &assembly_state_info,
            .pViewportState = &viewport_info,
            .pRasterizationState = &raster_info,
            .pMultisampleState = &sample_info,
            .pDepthStencilState = &depth_info,
            .pColorBlendState = &blend_info,
            .layout = pipeline_layout,
            .renderPass = veekay::app.vk_render_pass,
        };

        if (vkCreateGraphicsPipelines(device, nullptr,
                                      1, &info, nullptr, &pipeline) != VK_SUCCESS) {
            std::cerr << "Failed to create Vulkan pipeline\n";
            veekay::app.running = false;
            return;
        }
    }

    // ========================================================================
    // Создание буферов для uniform/storage данных
    // ========================================================================

    scene_uniforms_buffer = new veekay::graphics::Buffer(
        sizeof(SceneUniforms),
        nullptr,
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

    model_uniforms_buffer = new veekay::graphics::Buffer(
        max_models * veekay::graphics::Buffer::structureAlignment(sizeof(ModelUniforms)),
        nullptr,
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

    point_lights_buffer = new veekay::graphics::Buffer(
        max_point_lights * sizeof(PointLight),
        nullptr,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    spot_lights_buffer = new veekay::graphics::Buffer(
        max_spot_lights * sizeof(SpotLight),
        nullptr,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    // ========================================================================
    // Инициализация источников света
    // ========================================================================

    // Добавляем один точечный источник света СВЕРХУ (теперь используется для теней)
    point_lights.clear();
    point_lights.push_back(PointLight{
        .position = {6.0f, -4.0f, 5.0f},  // Свет сверху (-Y = вверх в этой системе)
        .ambient = {0.2f, 0.2f, 0.2f},
        .diffuse = {1.0f, 1.0f, 1.0f},    // Яркий белый свет
        .specular = {1.0f, 1.0f, 1.0f},
        .constant = 0.6f,
        .linear = 0.014f,                  // Слабое затухание (освещает далеко)
        .quadratic = 0.0007f,
    });

    // ВРЕМЕННО ОТКЛЮЧЕНО: Добавляем один прожектор (направлен сверху вниз на пол)
    spot_lights.clear();
    /*
    spot_lights.push_back(SpotLight{
        .position = {0.0f, -3.0f, 0.0f},
        .direction = {0.0f, 1.0f, 0.0f},      // направлен к Y=0 (к полу)
        .ambient = {0.1f, 0.1f, 0.1f},
        .diffuse = {0.8f, 0.8f, 0.8f},
        .specular = {0.5f, 0.5f, 0.5f},
        .constant = 1.0f,
        .linear = 0.045f,
        .quadratic = 0.0075f,
        .cutOff = cosf(toRadians(25.0f)),
        .outerCutOff = cosf(toRadians(35.0f)),
    });
    */

    // ========================================================================
    // Загрузка текстур из файлов
    // ========================================================================
    {
        // Заглушка (уже создана в предыдущем коде)
        VkSamplerCreateInfo info{
            .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
            .magFilter = VK_FILTER_NEAREST,
            .minFilter = VK_FILTER_NEAREST,
            .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        };

        if (vkCreateSampler(device, &info, nullptr, &missing_texture_sampler) != VK_SUCCESS) {
            std::cerr << "Failed to create Vulkan texture sampler\n";
            veekay::app.running = false;
            return;
        }

        uint32_t pixels[] = {
            0xff000000, 0xffff00ff,
            0xffff00ff, 0xff000000,
        };

        missing_texture = new veekay::graphics::Texture(cmd, 2, 2,
                                                        VK_FORMAT_B8G8R8A8_UNORM,
                                                        pixels);
        
        // Загружаем текстуры для разных объектов
        // Поместите файлы в папку ./textures/
        cube_texture = loadTexture(cmd, "./textures/cube.png");
        if (!cube_texture) cube_texture = missing_texture;
        cube_sampler = createSampler(VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_REPEAT);
        
        sphere_texture = loadTexture(cmd, "./textures/sphere.png");
        if (!sphere_texture) sphere_texture = missing_texture;
        sphere_sampler = createSampler(VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_REPEAT);
        
        floor_texture = loadTexture(cmd, "./textures/floor.png");
        if (!floor_texture) floor_texture = missing_texture;
        floor_sampler = createSampler(VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_REPEAT);
    }

    // ========================================================================
    // Создание геометрии: плоскость (пол)
    // ========================================================================
    {
        std::vector<Vertex> vertices = {
            // Верхняя сторона (видна снизу, нормали вверх)
            {{-5.0f, 0.0f, -5.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
            {{5.0f, 0.0f, -5.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}},
            {{5.0f, 0.0f, 5.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f}},
            {{-5.0f, 0.0f, 5.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f}},
            
            // Нижняя сторона (видна сверху, нормали вниз)
            {{-5.0f, 0.0f, -5.0f}, {0.0f, -1.0f, 0.0f}, {0.0f, 0.0f}},
            {{5.0f, 0.0f, -5.0f}, {0.0f, -1.0f, 0.0f}, {1.0f, 0.0f}},
            {{5.0f, 0.0f, 5.0f}, {0.0f, -1.0f, 0.0f}, {1.0f, 1.0f}},
            {{-5.0f, 0.0f, 5.0f}, {0.0f, -1.0f, 0.0f}, {0.0f, 1.0f}},
        };

        std::vector<uint32_t> indices = {
            0, 1, 2, 2, 3, 0,        // верхняя сторона
            4, 6, 5, 6, 4, 7         // нижняя сторона
        };

        plane_mesh.vertex_buffer = new veekay::graphics::Buffer(
            vertices.size() * sizeof(Vertex), vertices.data(),
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

        plane_mesh.index_buffer = new veekay::graphics::Buffer(
            indices.size() * sizeof(uint32_t), indices.data(),
            VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

        plane_mesh.indices = uint32_t(indices.size());
    }

    // ========================================================================
    // Создание геометрии: куб
    // ========================================================================
    {
        std::vector<Vertex> vertices = {
            // Передняя грань (-Z)
            {{-0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {0.0f, 0.0f}},
            {{+0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {1.0f, 0.0f}},
            {{+0.5f, +0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {1.0f, 1.0f}},
            {{-0.5f, +0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {0.0f, 1.0f}},

            // Правая грань (+X)
            {{+0.5f, -0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},
            {{+0.5f, -0.5f, +0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
            {{+0.5f, +0.5f, +0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 1.0f}},
            {{+0.5f, +0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f}},

            // Задняя грань (+Z)
            {{+0.5f, -0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f}},
            {{-0.5f, -0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f}},
            {{-0.5f, +0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f}},
            {{+0.5f, +0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}},

            // Левая грань (-X)
            {{-0.5f, -0.5f, +0.5f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},
            {{-0.5f, -0.5f, -0.5f}, {-1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
            {{-0.5f, +0.5f, -0.5f}, {-1.0f, 0.0f, 0.0f}, {1.0f, 1.0f}},
            {{-0.5f, +0.5f, +0.5f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 1.0f}},

            // Нижняя грань (-Y)
            {{-0.5f, -0.5f, +0.5f}, {0.0f, -1.0f, 0.0f}, {0.0f, 0.0f}},
            {{+0.5f, -0.5f, +0.5f}, {0.0f, -1.0f, 0.0f}, {1.0f, 0.0f}},
            {{+0.5f, -0.5f, -0.5f}, {0.0f, -1.0f, 0.0f}, {1.0f, 1.0f}},
            {{-0.5f, -0.5f, -0.5f}, {0.0f, -1.0f, 0.0f}, {0.0f, 1.0f}},

            // Верхняя грань (+Y)
            {{-0.5f, +0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
            {{+0.5f, +0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}},
            {{+0.5f, +0.5f, +0.5f}, {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f}},
            {{-0.5f, +0.5f, +0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f}},
        };

        std::vector<uint32_t> indices = {
            0, 1, 2, 2, 3, 0,
            4, 5, 6, 6, 7, 4,
            8, 9, 10, 10, 11, 8,
            12, 13, 14, 14, 15, 12,
            16, 17, 18, 18, 19, 16,
            20, 21, 22, 22, 23, 20,
        };

        cube_mesh.vertex_buffer = new veekay::graphics::Buffer(
            vertices.size() * sizeof(Vertex), vertices.data(),
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

        cube_mesh.index_buffer = new veekay::graphics::Buffer(
            indices.size() * sizeof(uint32_t), indices.data(),
            VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

        cube_mesh.indices = uint32_t(indices.size());
    }

    // ========================================================================
    // Создание геометрии: сфера (параметризация широта-долгота)
    // ========================================================================
    {
        const float radius = 0.5f;
        const int sectors = 32;  // количество вертикальных сегментов
        const int stacks = 16;   // количество горизонтальных сегментов

        std::vector<Vertex> verts;
        std::vector<uint32_t> inds;

        // Генерация вершин
        for (int i = 0; i <= stacks; ++i) {
            float stack_angle = float(M_PI / 2.0f) - float(i) * float(M_PI) / float(stacks);
            float xy = radius * cosf(stack_angle);
            float z = radius * sinf(stack_angle);

            for (int j = 0; j <= sectors; ++j) {
                float sector_angle = float(j) * 2.0f * float(M_PI) / float(sectors);
                float x = xy * cosf(sector_angle);
                float y = xy * sinf(sector_angle);

                veekay::vec3 pos = { x, z, y };
                
                // Нормаль = нормализованная позиция
                veekay::vec3 normal = pos;
                float length = sqrtf(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
                if (length > 0.0f) {
                    normal.x /= length;
                    normal.y /= length;
                    normal.z /= length;
                }
                
                veekay::vec2 uv = { float(j) / float(sectors), float(i) / float(stacks) };

                verts.push_back({ pos, normal, uv });
            }
        }

        // Генерация индексов
        for (int i = 0; i < stacks; ++i) {
            int k1 = i * (sectors + 1);
            int k2 = k1 + sectors + 1;

            for (int j = 0; j < sectors; ++j, ++k1, ++k2) {
                if (i != 0) {
                    inds.push_back(uint32_t(k1));
                    inds.push_back(uint32_t(k2));
                    inds.push_back(uint32_t(k1 + 1));
                }
                if (i != (stacks - 1)) {
                    inds.push_back(uint32_t(k1 + 1));
                    inds.push_back(uint32_t(k2));
                    inds.push_back(uint32_t(k2 + 1));
                }
            }
        }

        sphere_mesh.vertex_buffer = new veekay::graphics::Buffer(
            verts.size() * sizeof(Vertex), verts.data(),
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

        sphere_mesh.index_buffer = new veekay::graphics::Buffer(
            inds.size() * sizeof(uint32_t), inds.data(),
            VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

        sphere_mesh.indices = uint32_t(inds.size());
    }

    // ========================================================================
    // Настройка начальной сцены: куб, сфера и пол
    // ========================================================================
    models.clear();

    // Статичный куб - СТЕКЛО из Minecraft
    models.emplace_back(Model{
        .mesh = cube_mesh,
        .transform = Transform{
            .position = {0.0f, -0.5f, 0.0f},
        },
        .material = Material{
            .albedo = veekay::vec3{0.9f, 0.95f, 1.0f},   // ИЗМЕНЕНО: слегка голубоватый оттенок стекла
            .specular = veekay::vec3{1.0f, 1.0f, 1.0f},  // ИЗМЕНЕНО: яркие белые блики (стекло отражает свет)
            .shininess = 128.0f,                          // ИЗМЕНЕНО: очень острые блики (гладкая поверхность)
            .texture = cube_texture,
            .sampler = cube_sampler,
        }
    });

    // Анимированная сфера - ДЕРЕВО из Minecraft
    models.emplace_back(Model{
        .mesh = sphere_mesh,
        .transform = Transform{
            .position = {2.0f, -0.5f, 0.0f},
            .scale = {0.6f, 0.6f, 0.6f}
        },
        .material = Material{
            .albedo = veekay::vec3{1.0f, 1.0f, 1.0f}, // чистый белый (не тонируем текстуру)
            .specular = veekay::vec3{0.2f, 0.2f, 0.2f}, // дерево почти не блестит
            .shininess = 4.0f,  // очень матовый
            .texture = sphere_texture,
            .sampler = sphere_sampler,
        }
    });

    // Пол - ЗЕМЛЯ из Minecraft
    models.emplace_back(Model{
        .mesh = plane_mesh,
        .transform = Transform{
            .position = {0.0f, 0.0f, 0.0f},
            .scale = {1.0f, 1.0f, 1.0f}
        },
        .material = Material{
            .albedo = veekay::vec3{0.95f, 0.9f, 0.85f}, // светло-коричневый оттенок для земли
            .specular = veekay::vec3{0.1f, 0.1f, 0.1f}, // земля не блестит
            .shininess = 2.0f,                           // максимально матовый
            .texture = floor_texture,
            .sampler = floor_sampler,
        }
    });

    // ========================================================================
    // ИНИЦИАЛИЗАЦИЯ SHADOW MAPPING
    // ========================================================================
    {
        // Создание depth image для shadow map
        VkImageCreateInfo image_info{
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .imageType = VK_IMAGE_TYPE_2D,
            .format = VK_FORMAT_D32_SFLOAT,  // 32-битная глубина с плавающей точкой
            .extent = {
                .width = shadow_map_size,
                .height = shadow_map_size,
                .depth = 1,
            },
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .tiling = VK_IMAGE_TILING_OPTIMAL,
            .usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        };
        
        if (vkCreateImage(device, &image_info, nullptr, &shadow_map_image) != VK_SUCCESS) {
            std::cerr << "Failed to create shadow map image\n";
            veekay::app.running = false;
            return;
        }
        
        // Выделение памяти для image
        VkMemoryRequirements mem_requirements;
        vkGetImageMemoryRequirements(device, shadow_map_image, &mem_requirements);
        
        VkPhysicalDeviceMemoryProperties mem_properties;
        vkGetPhysicalDeviceMemoryProperties(veekay::app.vk_physical_device, &mem_properties);
        
        uint32_t memory_type_index = UINT32_MAX;
        for (uint32_t i = 0; i < mem_properties.memoryTypeCount; ++i) {
            if ((mem_requirements.memoryTypeBits & (1 << i)) &&
                (mem_properties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
                memory_type_index = i;
                break;
            }
        }
        
        VkMemoryAllocateInfo alloc_info{
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize = mem_requirements.size,
            .memoryTypeIndex = memory_type_index,
        };
        
        if (vkAllocateMemory(device, &alloc_info, nullptr, &shadow_map_memory) != VK_SUCCESS) {
            std::cerr << "Failed to allocate shadow map memory\n";
            veekay::app.running = false;
            return;
        }
        
        vkBindImageMemory(device, shadow_map_image, shadow_map_memory, 0);
        
        // Создание image view для shadow map
        VkImageViewCreateInfo view_info{
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .image = shadow_map_image,
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = VK_FORMAT_D32_SFLOAT,
            .subresourceRange = {
                .aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
        };
        
        if (vkCreateImageView(device, &view_info, nullptr, &shadow_map_view) != VK_SUCCESS) {
            std::cerr << "Failed to create shadow map image view\n";
            veekay::app.running = false;
            return;
        }
        
        // Создание сэмплера с поддержкой сравнения (для PCF)
        VkSamplerCreateInfo sampler_info{
            .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
            .magFilter = VK_FILTER_LINEAR,
            .minFilter = VK_FILTER_LINEAR,
            .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
            .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
            .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
            .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
            .mipLodBias = 0.0f,
            .anisotropyEnable = VK_FALSE,
            .compareEnable = VK_TRUE,  // ВАЖНО: включаем сравнение для shadow mapping
            .compareOp = VK_COMPARE_OP_LESS_OR_EQUAL,  // Пиксель в тени, если его глубина больше
            .minLod = 0.0f,
            .maxLod = 0.0f,
            .borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE,  // За границами shadow map - нет тени
            .unnormalizedCoordinates = VK_FALSE,
        };
        
        if (vkCreateSampler(device, &sampler_info, nullptr, &shadow_map_sampler) != VK_SUCCESS) {
            std::cerr << "Failed to create shadow map sampler\n";
            veekay::app.running = false;
            return;
        }
        
        // Создание буфера для light space matrix
        light_space_buffer = new veekay::graphics::Buffer(
            sizeof(veekay::mat4),
            nullptr,
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
        
        // Создание Render Pass для shadow mapping
        VkAttachmentDescription depth_attachment{
            .format = VK_FORMAT_D32_SFLOAT,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
        };

        VkAttachmentReference depth_ref{
            .attachment = 0,
            .layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        };

        VkSubpassDescription subpass{
            .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
            .colorAttachmentCount = 0,
            .pColorAttachments = nullptr,
            .pDepthStencilAttachment = &depth_ref,
        };

        VkSubpassDependency dependency{
            .srcSubpass = VK_SUBPASS_EXTERNAL,
            .dstSubpass = 0,
            .srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
            .srcAccessMask = VK_ACCESS_SHADER_READ_BIT,
            .dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            .dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT,
        };

        VkRenderPassCreateInfo render_pass_info{
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
            .attachmentCount = 1,
            .pAttachments = &depth_attachment,
            .subpassCount = 1,
            .pSubpasses = &subpass,
            .dependencyCount = 1,
            .pDependencies = &dependency,
        };

        if (vkCreateRenderPass(device, &render_pass_info, nullptr, &shadow_render_pass) != VK_SUCCESS) {
            std::cerr << "Failed to create shadow render pass\n";
            veekay::app.running = false;
            return;
        }

        // Создание Framebuffer для shadow map
        VkFramebufferCreateInfo framebuffer_info{
            .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
            .renderPass = shadow_render_pass,
            .attachmentCount = 1,
            .pAttachments = &shadow_map_view,
            .width = shadow_map_size,
            .height = shadow_map_size,
            .layers = 1,
        };

        if (vkCreateFramebuffer(device, &framebuffer_info, nullptr, &shadow_framebuffer) != VK_SUCCESS) {
            std::cerr << "Failed to create shadow framebuffer\n";
            veekay::app.running = false;
            return;
        }
        
        std::cout << "Shadow mapping initialized successfully\n";
    }

    // ========================================================================
    // СОЗДАНИЕ SHADOW PIPELINE
    // ========================================================================
    {
        // Загрузка shadow шейдеров
        shadow_vertex_shader_module = loadShaderModule("./shaders/shadow.vert.spv");
        if (!shadow_vertex_shader_module) {
            std::cerr << "Failed to load shadow vertex shader\n";
            veekay::app.running = false;
            return;
        }

        shadow_fragment_shader_module = loadShaderModule("./shaders/shadow.frag.spv");
        if (!shadow_fragment_shader_module) {
            std::cerr << "Failed to load shadow fragment shader\n";
            veekay::app.running = false;
            return;
        }

        VkPipelineShaderStageCreateInfo shadow_stages[2];
        shadow_stages[0] = VkPipelineShaderStageCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_VERTEX_BIT,
            .module = shadow_vertex_shader_module,
            .pName = "main",
        };

        shadow_stages[1] = VkPipelineShaderStageCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
            .module = shadow_fragment_shader_module,
            .pName = "main",
        };

        // Вершинный формат для shadow pass (только позиция)
        VkVertexInputBindingDescription shadow_binding{
            .binding = 0,
            .stride = sizeof(Vertex),
            .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
        };

        VkVertexInputAttributeDescription shadow_attribute{
            .location = 0,
            .binding = 0,
            .format = VK_FORMAT_R32G32B32_SFLOAT,
            .offset = offsetof(Vertex, position),
        };

        VkPipelineVertexInputStateCreateInfo shadow_input{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            .vertexBindingDescriptionCount = 1,
            .pVertexBindingDescriptions = &shadow_binding,
            .vertexAttributeDescriptionCount = 1,
            .pVertexAttributeDescriptions = &shadow_attribute,
        };

        VkPipelineInputAssemblyStateCreateInfo shadow_assembly{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        };

        // Viewport для shadow map
        VkViewport shadow_viewport{
            .x = 0.0f,
            .y = 0.0f,
            .width = static_cast<float>(shadow_map_size),
            .height = static_cast<float>(shadow_map_size),
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        };

        VkRect2D shadow_scissor{
            .offset = {0, 0},
            .extent = {shadow_map_size, shadow_map_size},
        };

        VkPipelineViewportStateCreateInfo shadow_viewport_state{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            .viewportCount = 1,
            .pViewports = &shadow_viewport,
            .scissorCount = 1,
            .pScissors = &shadow_scissor,
        };

        // Растеризация для shadow pass
        VkPipelineRasterizationStateCreateInfo shadow_raster{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            .depthClampEnable = VK_FALSE,
            .rasterizerDiscardEnable = VK_FALSE,
            .polygonMode = VK_POLYGON_MODE_FILL,
            .cullMode = VK_CULL_MODE_NONE,  // Отключаем culling для корректных теней
            .frontFace = VK_FRONT_FACE_CLOCKWISE,
            .depthBiasEnable = VK_TRUE,  // ВАЖНО: включаем depth bias для борьбы с shadow acne
            .depthBiasConstantFactor = 1.25f,
            .depthBiasClamp = 0.0f,
            .depthBiasSlopeFactor = 1.75f,
            .lineWidth = 1.0f,
        };

        VkPipelineMultisampleStateCreateInfo shadow_multisample{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
            .sampleShadingEnable = VK_FALSE,
        };

        // Depth test для shadow pass (записываем глубину)
        VkPipelineDepthStencilStateCreateInfo shadow_depth{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
            .depthTestEnable = VK_TRUE,
            .depthWriteEnable = VK_TRUE,
            .depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
            .depthBoundsTestEnable = VK_FALSE,
            .stencilTestEnable = VK_FALSE,
        };

        // Descriptor set layout для shadow pass
        VkDescriptorSetLayoutBinding shadow_bindings[] = {
            {
                .binding = 0,
                .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
            },
            {
                .binding = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
            },
        };

        VkDescriptorSetLayoutCreateInfo shadow_layout_info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = 2,
            .pBindings = shadow_bindings,
        };

        if (vkCreateDescriptorSetLayout(device, &shadow_layout_info, nullptr,
                                        &shadow_descriptor_set_layout) != VK_SUCCESS) {
            std::cerr << "Failed to create shadow descriptor set layout\n";
            veekay::app.running = false;
            return;
        }

        // Выделение shadow descriptor set
        VkDescriptorSetAllocateInfo shadow_alloc{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = descriptor_pool,
            .descriptorSetCount = 1,
            .pSetLayouts = &shadow_descriptor_set_layout,
        };

        if (vkAllocateDescriptorSets(device, &shadow_alloc, &shadow_descriptor_set) != VK_SUCCESS) {
            std::cerr << "Failed to allocate shadow descriptor set\n";
            veekay::app.running = false;
            return;
        }

        // Привязка буферов к shadow descriptor set
        VkDescriptorBufferInfo shadow_buffer_infos[] = {
            {
                .buffer = light_space_buffer->buffer,
                .offset = 0,
                .range = sizeof(veekay::mat4),
            },
            {
                .buffer = model_uniforms_buffer->buffer,
                .offset = 0,
                .range = sizeof(ModelUniforms),
            },
        };

        VkWriteDescriptorSet shadow_writes[] = {
            {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = shadow_descriptor_set,
                .dstBinding = 0,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                .pBufferInfo = &shadow_buffer_infos[0],
            },
            {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = shadow_descriptor_set,
                .dstBinding = 1,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
                .pBufferInfo = &shadow_buffer_infos[1],
            },
        };

        vkUpdateDescriptorSets(device, 2, shadow_writes, 0, nullptr);

        // Создание shadow pipeline layout
        VkPipelineLayoutCreateInfo shadow_pipeline_layout_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 1,
            .pSetLayouts = &shadow_descriptor_set_layout,
        };

        if (vkCreatePipelineLayout(device, &shadow_pipeline_layout_info,
                                   nullptr, &shadow_pipeline_layout) != VK_SUCCESS) {
            std::cerr << "Failed to create shadow pipeline layout\n";
            veekay::app.running = false;
            return;
        }

        // Создание shadow pipeline (используем обычный Render Pass)
        VkGraphicsPipelineCreateInfo shadow_pipeline_info{
            .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            .stageCount = 2,
            .pStages = shadow_stages,
            .pVertexInputState = &shadow_input,
            .pInputAssemblyState = &shadow_assembly,
            .pViewportState = &shadow_viewport_state,
            .pRasterizationState = &shadow_raster,
            .pMultisampleState = &shadow_multisample,
            .pDepthStencilState = &shadow_depth,
            .pColorBlendState = nullptr,  // Нет color blending
            .layout = shadow_pipeline_layout,
            .renderPass = shadow_render_pass,  // ИСПРАВЛЕНО: используем render pass
            .subpass = 0,
        };

        if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1,
                                      &shadow_pipeline_info, nullptr, &shadow_pipeline) != VK_SUCCESS) {
            std::cerr << "Failed to create shadow pipeline\n";
            veekay::app.running = false;
            return;
        }

        std::cout << "Shadow pipeline created successfully\n";
    }

    // ========================================================================
    // Создание наборов дескрипторов для каждого материала (ПОСЛЕ shadow map)
    // ========================================================================
    for (Model& model : models) {
        Material& mat = model.material;
        
        // Выделяем новый descriptor set для этого материала
        VkDescriptorSetAllocateInfo alloc_info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = descriptor_pool,
            .descriptorSetCount = 1,
            .pSetLayouts = &descriptor_set_layout,
        };

        if (vkAllocateDescriptorSets(device, &alloc_info, &mat.descriptor_set) != VK_SUCCESS) {
            std::cerr << "Failed to allocate descriptor set for material\n";
            veekay::app.running = false;
            return;
        }

        // Привязываем буферы и текстуру к descriptor set
        VkDescriptorBufferInfo buffer_infos[] = {
            {
                .buffer = scene_uniforms_buffer->buffer,
                .offset = 0,
                .range = sizeof(SceneUniforms),
            },
            {
                .buffer = model_uniforms_buffer->buffer,
                .offset = 0,
                .range = sizeof(ModelUniforms),
            },
            {
                .buffer = point_lights_buffer->buffer,
                .offset = 0,
                .range = max_point_lights * sizeof(PointLight),
            },
            {
                .buffer = spot_lights_buffer->buffer,
                .offset = 0,
                .range = max_spot_lights * sizeof(SpotLight),
            },
        };

        VkDescriptorImageInfo image_info{
            .sampler = mat.sampler,
            .imageView = mat.texture->view,
            .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        };

        VkDescriptorImageInfo shadow_image_info{
            .sampler = shadow_map_sampler,
            .imageView = shadow_map_view,
            .imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
        };
        
        // Shadow Map как обычная текстура (для чтения размера в шейдере)
        VkDescriptorImageInfo shadow_image_info_raw{
            .sampler = missing_texture_sampler,  // Обычный sampler без сравнения
            .imageView = shadow_map_view,
            .imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
        };

        VkWriteDescriptorSet write_infos[] = {
            {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = mat.descriptor_set,
                .dstBinding = 0,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                .pBufferInfo = &buffer_infos[0],
            },
            {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = mat.descriptor_set,
                .dstBinding = 1,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
                .pBufferInfo = &buffer_infos[1],
            },
            {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = mat.descriptor_set,
                .dstBinding = 2,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pBufferInfo = &buffer_infos[2],
            },
            {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = mat.descriptor_set,
                .dstBinding = 3,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pBufferInfo = &buffer_infos[3],
            },
            {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = mat.descriptor_set,
                .dstBinding = 4,  // Текстура материала
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .pImageInfo = &image_info,
            },
            {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = mat.descriptor_set,
                .dstBinding = 5,  // Shadow Map
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .pImageInfo = &shadow_image_info,
            },
            {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = mat.descriptor_set,
                .dstBinding = 6,  // Shadow Map raw
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .pImageInfo = &shadow_image_info_raw,
            },
        };

        vkUpdateDescriptorSets(device, sizeof(write_infos) / sizeof(write_infos[0]),
                               write_infos, 0, nullptr);
    }
}

// ============================================================================
// ОЧИСТКА РЕСУРСОВ
// ============================================================================
void shutdown() {
    VkDevice& device = veekay::app.vk_device;

    // Очищаем Shadow Pipeline ресурсы
    if (shadow_pipeline) {
        vkDestroyPipeline(device, shadow_pipeline, nullptr);
    }
    if (shadow_pipeline_layout) {
        vkDestroyPipelineLayout(device, shadow_pipeline_layout, nullptr);
    }
    if (shadow_descriptor_set_layout) {
        vkDestroyDescriptorSetLayout(device, shadow_descriptor_set_layout, nullptr);
    }
    if (shadow_framebuffer) {
        vkDestroyFramebuffer(device, shadow_framebuffer, nullptr);
    }
    if (shadow_render_pass) {
        vkDestroyRenderPass(device, shadow_render_pass, nullptr);
    }
    if (shadow_fragment_shader_module) {
        vkDestroyShaderModule(device, shadow_fragment_shader_module, nullptr);
    }
    if (shadow_vertex_shader_module) {
        vkDestroyShaderModule(device, shadow_vertex_shader_module, nullptr);
    }

    // Очищаем Shadow Mapping ресурсы
    if (light_space_buffer) {
        delete light_space_buffer;
    }
    if (shadow_map_sampler) {
        vkDestroySampler(device, shadow_map_sampler, nullptr);
    }
    if (shadow_map_view) {
        vkDestroyImageView(device, shadow_map_view, nullptr);
    }
    if (shadow_map_image) {
        vkDestroyImage(device, shadow_map_image, nullptr);
    }
    if (shadow_map_memory) {
        vkFreeMemory(device, shadow_map_memory, nullptr);
    }

    // Очищаем загруженные текстуры (проверяем, что они не являются заглушкой)
    if (floor_texture && floor_texture != missing_texture) {
        delete floor_texture;
    }
    if (sphere_texture && sphere_texture != missing_texture) {
        delete sphere_texture;
    }
    if (cube_texture && cube_texture != missing_texture) {
        delete cube_texture;
    }
    
    // Удаляем сэмплеры (проверяем, что они не совпадают с заглушкой)
    if (floor_sampler && floor_sampler != missing_texture_sampler) {
        vkDestroySampler(device, floor_sampler, nullptr);
    }
    if (sphere_sampler && sphere_sampler != missing_texture_sampler) {
        vkDestroySampler(device, sphere_sampler, nullptr);
    }
    if (cube_sampler && cube_sampler != missing_texture_sampler) {
        vkDestroySampler(device, cube_sampler, nullptr);
    }
    
    // Удаляем заглушку ОДИН РАЗ (ИСПРАВЛЕНО: убрали дубликат)
    if (missing_texture_sampler) {
        vkDestroySampler(device, missing_texture_sampler, nullptr);
    }
    if (missing_texture) {
        delete missing_texture;
    }

    // Удаляем меши
    delete sphere_mesh.index_buffer;
    delete sphere_mesh.vertex_buffer;
    
    delete cube_mesh.index_buffer;
    delete cube_mesh.vertex_buffer;

    delete plane_mesh.index_buffer;
    delete plane_mesh.vertex_buffer;

    // Удаляем буферы
    delete spot_lights_buffer;
    delete point_lights_buffer;
    delete model_uniforms_buffer;
    delete scene_uniforms_buffer;

    // Удаляем Vulkan объекты дескрипторов и пайплайна
    vkDestroyDescriptorSetLayout(device, descriptor_set_layout, nullptr);
    vkDestroyDescriptorPool(device, descriptor_pool, nullptr);

    vkDestroyPipeline(device, pipeline, nullptr);
    vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
    vkDestroyShaderModule(device, fragment_shader_module, nullptr);
    vkDestroyShaderModule(device, vertex_shader_module, nullptr);
}


// ============================================================================
// ОБНОВЛЕНИЕ ЛОГИКИ И UI
// ============================================================================
// Эта функция вызывается каждый кадр и отвечает за:
// - Отрисовку интерфейса ImGui для управления освещением и материалами
// - Обработку пользовательского ввода (клавиатура + мышь)
// - Обновление позиций объектов (анимация)
// - Подготовку данных для отправки в шейдеры
void update(double time) {
    // ========================================================================
    // UI: ГЛАВНАЯ ПАНЕЛЬ УПРАВЛЕНИЯ С ВКЛАДКАМИ
    // ========================================================================
    ImGui::SetNextWindowSize(ImVec2(500, 700), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(20, 20), ImGuiCond_FirstUseEver);
    
    ImGui::Begin("Scene Control Panel", nullptr, ImGuiWindowFlags_MenuBar);
    
    // Меню-бар с информацией о приложении
    if (ImGui::BeginMenuBar()) {
        ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.3f, 1.0f), "Vulkan Lighting Demo");
        ImGui::EndMenuBar();
    }
    
    // Создаем систему вкладок
    if (ImGui::BeginTabBar("MainTabBar", ImGuiTabBarFlags_None)) {
        
        // ====================================================================
        // ВКЛАДКА 1: КАМЕРА
        // ====================================================================
        if (ImGui::BeginTabItem("Camera")) {
            ImGui::Spacing();
            ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.2f, 1.0f), "Camera Information");
            ImGui::Separator();
            ImGui::Spacing();
            
            // Информация о положении
            ImGui::Text("Position:");
            ImGui::Indent();
            ImGui::Text("X: %.2f", camera.position.x);
            ImGui::Text("Y: %.2f", camera.position.y);
            ImGui::Text("Z: %.2f", camera.position.z);
            ImGui::Unindent();
            
            ImGui::Spacing();
            ImGui::Text("Rotation:");
            ImGui::Indent();
            ImGui::Text("Yaw:   %.1f°", camera.yaw);
            ImGui::Text("Pitch: %.1f°", camera.pitch);
            ImGui::Unindent();
            
            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();
            
            // Параметры камеры (редактируемые)
            ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.2f, 1.0f), "Camera Settings");
            ImGui::Separator();
            ImGui::Spacing();
            
            ImGui::SliderFloat("FOV", &camera.fov, 30.0f, 120.0f, "%.0f°");
            ImGui::DragFloat("Near Plane", &camera.near_plane, 0.01f, 0.001f, 10.0f, "%.3f");
            ImGui::DragFloat("Far Plane", &camera.far_plane, 1.0f, 1.0f, 1000.0f, "%.1f");
            
            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();
            
            // Управление
            ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.2f, 1.0f), "Controls");
            ImGui::Separator();
            ImGui::Spacing();
            
            ImGui::BulletText("WASD - Move horizontally");
            ImGui::BulletText("Q/E - Move up/down");
            ImGui::BulletText("Arrow Keys - Rotate camera");
            ImGui::BulletText("Right Mouse - Look around");
            
            ImGui::Spacing();
            
            // Кнопка сброса камеры
            if (ImGui::Button("Reset Camera", ImVec2(-1, 0))) {
                camera.position = {0.4f, -4.20f, 8.0f};
                camera.yaw = 0.0f;
                camera.pitch = 20.0f;
                camera.fov = Camera::default_fov;
                camera.near_plane = Camera::default_near_plane;
                camera.far_plane = Camera::default_far_plane;
            }
            
            ImGui::EndTabItem();
        }
        
        // ====================================================================
        // ВКЛАДКА 2: МАТЕРИАЛЫ
        // ====================================================================
        if (ImGui::BeginTabItem("Materials")) {
            ImGui::Spacing();
            ImGui::TextColored(ImVec4(0.5f, 0.8f, 1.0f, 1.0f), "Object Materials");
            ImGui::Separator();
            ImGui::Spacing();
            
            for (size_t i = 0; i < models.size(); ++i) {
                ImGui::PushID(int(i));
                
                // Присваиваем понятные имена объектам
                std::string name;
                const char* icon = "";
                if (i == 0) { name = "Cube"; icon = "[C]"; }
                else if (i == 1) { name = "Sphere"; icon = "[S]"; }
                else if (i == 2) { name = "Floor"; icon = "[F]"; }
                else { name = "Object " + std::to_string(i); icon = "[?]"; }
                
                // Используем CollapsingHeader для каждого объекта
                if (ImGui::CollapsingHeader((std::string(icon) + " " + name).c_str())) {
                    ImGui::Indent();
                    
                    ImGui::Text("Colors:");
                    ImGui::ColorEdit3(("Albedo##" + name).c_str(), &models[i].material.albedo.x);
                    ImGui::ColorEdit3(("Specular##" + name).c_str(), &models[i].material.specular.x);
                    
                    ImGui::Spacing();
                    ImGui::Text("Properties:");
                    ImGui::SliderFloat(("Shininess##" + name).c_str(), &models[i].material.shininess, 1.0f, 256.0f, "%.1f");
                    
                    // Пресеты материалов
                    ImGui::Spacing();
                    ImGui::Text("Presets:");
                    if (ImGui::Button(("Gold##" + name).c_str())) {
                        models[i].material.albedo = {1.0f, 0.765f, 0.336f};
                        models[i].material.specular = {1.0f, 0.9f, 0.5f};
                        models[i].material.shininess = 128.0f;
                    }
                    ImGui::SameLine();
                    if (ImGui::Button(("Silver##" + name).c_str())) {
                        models[i].material.albedo = {0.75f, 0.75f, 0.75f};
                        models[i].material.specular = {1.0f, 1.0f, 1.0f};
                        models[i].material.shininess = 128.0f;
                    }
                    ImGui::SameLine();
                    if (ImGui::Button(("Plastic##" + name).c_str())) {
                        models[i].material.albedo = {0.5f, 0.5f, 0.5f};
                        models[i].material.specular = {0.3f, 0.3f, 0.3f};
                        models[i].material.shininess = 32.0f;
                    }
                    
                    ImGui::Unindent();
                    ImGui::Spacing();
                }
                
                ImGui::PopID();
            }
            
            ImGui::EndTabItem();
        }
        
        // ====================================================================
        // ВКЛАДКА 3: ОСВЕЩЕНИЕ
        // ====================================================================
        if (ImGui::BeginTabItem("Lighting")) {
            ImGui::Spacing();
            ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.3f, 1.0f), "Light Sources");
            ImGui::Separator();
            ImGui::Spacing();
            
            // =============================================================
            // НАПРАВЛЕННЫЙ СВЕТ
            // =============================================================
            ImGui::PushStyleColor(ImGuiCol_Header, ImVec4(0.3f, 0.5f, 0.8f, 0.8f));
            if (ImGui::CollapsingHeader("Directional Light (Sun)")) {
                ImGui::Indent();
                
                ImGui::Checkbox("Enable##DirLight", &directional_light_enabled);
                
                if (directional_light_enabled) {
                    ImGui::Spacing();
                    ImGui::DragFloat3("Direction", &directional_light.direction.x, 0.01f, -1.0f, 1.0f);
                    ImGui::ColorEdit3("Ambient##Dir", &directional_light.ambient.x);
                    ImGui::ColorEdit3("Diffuse##Dir", &directional_light.diffuse.x);
                    ImGui::ColorEdit3("Specular##Dir", &directional_light.specular.x);
                    
                    ImGui::Spacing();
                    if (ImGui::Button("Use for Shadows##DirShadow", ImVec2(-1, 0))) {
                        shadow_light_type = ShadowLightType::Directional;
                        shadow_light_index = 0;
                    }
                }
                
                ImGui::Unindent();
            }
            ImGui::PopStyleColor();
            
            ImGui::Spacing();
            
            // =============================================================
            // ТОЧЕЧНЫЙ ИСТОЧНИК СВЕТА
            // =============================================================
            ImGui::PushStyleColor(ImGuiCol_Header, ImVec4(0.8f, 0.5f, 0.2f, 0.8f));
            if (ImGui::CollapsingHeader("Point Light", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::Indent();
                
                if (point_lights.empty()) {
                    ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f), "Point light is disabled");
                    ImGui::Spacing();
                    if (ImGui::Button("Enable Point Light", ImVec2(-1, 0))) {
                        point_lights.push_back(PointLight{
                            .position = {0.0f, -2.0f, 0.0f},
                            .ambient = {0.1f, 0.1f, 0.1f},
                            .diffuse = {1.0f, 1.0f, 1.0f},
                            .specular = {1.0f, 1.0f, 1.0f},
                            .constant = 1.0f,
                            .linear = 0.09f,
                            .quadratic = 0.032f,
                        });
                    }
                } else {
                    // Есть ровно один источник
                    PointLight& light = point_lights[0];
                    
                    ImGui::DragFloat3("Position##PointPos", &light.position.x, 0.1f);
                    ImGui::ColorEdit3("Ambient##PointAmb", &light.ambient.x);
                    ImGui::ColorEdit3("Diffuse##PointDiff", &light.diffuse.x);
                    ImGui::ColorEdit3("Specular##PointSpec", &light.specular.x);
                    
                    ImGui::Text("Attenuation:");
                    ImGui::Indent();
                    ImGui::DragFloat("Constant##PointConst", &light.constant, 0.01f, 0.0f, 10.0f);
                    ImGui::DragFloat("Linear##PointLin", &light.linear, 0.01f, 0.0f, 1.0f);
                    ImGui::DragFloat("Quadratic##PointQuad", &light.quadratic, 0.001f, 0.0f, 1.0f);
                    ImGui::Unindent();
                    
                    ImGui::Spacing();
                    if (ImGui::Button("Use for Shadows##PointShadow", ImVec2(-1, 0))) {
                        shadow_light_type = ShadowLightType::Point;
                        shadow_light_index = 0;
                    }
                    
                    ImGui::Spacing();
                    ImGui::Separator();
                    ImGui::Spacing();
                    
                    if (ImGui::Button("Disable Point Light", ImVec2(-1, 0))) {
                        point_lights.clear();
                        // Переключаемся на directional если был активен point
                        if (shadow_light_type == ShadowLightType::Point) {
                            shadow_light_type = ShadowLightType::Directional;
                        }
                    }
                }
                
                ImGui::Unindent();
            }
            ImGui::PopStyleColor();
            
            ImGui::Spacing();
            
            // =============================================================
            // ПРОЖЕКТОР
            // =============================================================
            ImGui::PushStyleColor(ImGuiCol_Header, ImVec4(0.5f, 0.8f, 0.3f, 0.8f));
            if (ImGui::CollapsingHeader("Spot Light")) {
                ImGui::Indent();
                
                if (spot_lights.empty()) {
                    ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Spot light is disabled");
                    ImGui::Spacing();
                    if (ImGui::Button("Enable Spot Light", ImVec2(-1, 0))) {
                        spot_lights.push_back(SpotLight{
                            .position = {0.0f, -3.0f, 0.0f},
                            .direction = {0.0f, 1.0f, 0.0f},
                            .ambient = {0.05f, 0.05f, 0.05f},
                            .diffuse = {1.0f, 1.0f, 1.0f},
                            .specular = {1.0f, 1.0f, 1.0f},
                            .constant = 1.0f,
                            .linear = 0.09f,
                            .quadratic = 0.032f,
                            .cutOff = cosf(toRadians(12.5f)),
                            .outerCutOff = cosf(toRadians(17.5f)),
                        });
                    }
                } else {
                    // Есть ровно один прожектор
                    SpotLight& light = spot_lights[0];
                    
                    ImGui::DragFloat3("Position##SpotPos", &light.position.x, 0.1f);
                    ImGui::DragFloat3("Direction##SpotDir", &light.direction.x, 0.01f, -1.0f, 1.0f);
                    ImGui::ColorEdit3("Ambient##SpotAmb", &light.ambient.x);
                    ImGui::ColorEdit3("Diffuse##SpotDiff", &light.diffuse.x);
                    ImGui::ColorEdit3("Specular##SpotSpec", &light.specular.x);
                    
                    ImGui::Text("Attenuation:");
                    ImGui::Indent();
                    ImGui::DragFloat("Constant##SpotConst", &light.constant, 0.01f, 0.0f, 10.0f);
                    ImGui::DragFloat("Linear##SpotLin", &light.linear, 0.01f, 0.0f, 1.0f);
                    ImGui::DragFloat("Quadratic##SpotQuad", &light.quadratic, 0.001f, 0.0f, 1.0f);
                    ImGui::Unindent();
                    
                    ImGui::Text("Cone:");
                    ImGui::Indent();
                    float inner_angle = acosf(light.cutOff) * 180.0f / M_PI;
                    float outer_angle = acosf(light.outerCutOff) * 180.0f / M_PI;
                    
                    if (ImGui::SliderFloat("Inner Angle##SpotInner", &inner_angle, 0.0f, 90.0f, "%.1f°")) {
                        light.cutOff = cosf(toRadians(inner_angle));
                    }
                    if (ImGui::SliderFloat("Outer Angle##SpotOuter", &outer_angle, 0.0f, 90.0f, "%.1f°")) {
                        light.outerCutOff = cosf(toRadians(outer_angle));
                    }
                    ImGui::Unindent();
                    
                    ImGui::Spacing();
                    // SpotLight тени не поддерживаются (требуют перспективную проекцию)
                    ImGui::BeginDisabled();
                    ImGui::Button("Use for Shadows##SpotShadow", ImVec2(-1, 0));
                    ImGui::EndDisabled();
                    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
                        ImGui::SetTooltip("Spot Light shadows are not supported\n(require perspective projection)");
                    }
                    
                    ImGui::Spacing();
                    ImGui::Separator();
                    ImGui::Spacing();
                    
                    if (ImGui::Button("Disable Spot Light", ImVec2(-1, 0))) {
                        spot_lights.clear();
                        // Переключаемся на directional если был активен spot
                        if (shadow_light_type == ShadowLightType::Spot) {
                            shadow_light_type = ShadowLightType::Directional;
                        }
                    }
                }
                
                ImGui::Unindent();
            }
            ImGui::PopStyleColor();
            
            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();
            
            // Информация о тенях
            ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.5f, 1.0f), "Shadow Settings");
            ImGui::Separator();
            ImGui::Spacing();
            
            ImGui::Text("Current shadow source:");
            ImGui::Indent();
            if (shadow_light_type == ShadowLightType::Directional) {
                ImGui::BulletText("Directional Light");
            } else if (shadow_light_type == ShadowLightType::Point) {
                ImGui::BulletText("Point Light %d", shadow_light_index);
            } else if (shadow_light_type == ShadowLightType::Spot) {
                ImGui::BulletText("Spot Light %d", shadow_light_index);
            }
            ImGui::Unindent();
            
            ImGui::EndTabItem();
        }
        
        ImGui::EndTabBar();
    }
    
    ImGui::End();

    // ========================================================================
    // Управление камерой с клавиатуры
    // ========================================================================
    using namespace veekay::input;

    const float move_speed = 0.1f;      // Скорость перемещения камеры
    const float rotate_speed = 2.0f;    // Скорость вращения камеры

    // Вычисляем векторы направления для перемещения камеры
    veekay::vec3 camera_front = camera.front();  // Направление взгляда
    veekay::vec3 camera_right = camera.right();  // Правый вектор
    veekay::vec3 camera_up = {0.0f, -1.0f, 0.0f}; // Вектор "вверх" (Y-down координаты)

    // Движение вперед/назад (W/S)
    if (keyboard::isKeyDown(keyboard::Key::w))
        camera.position -= camera_front * move_speed;

    if (keyboard::isKeyDown(keyboard::Key::s))
        camera.position += camera_front * move_speed;

    // Движение влево/вправо (A/D)
    if (keyboard::isKeyDown(keyboard::Key::d))
        camera.position += camera_right * move_speed;

    if (keyboard::isKeyDown(keyboard::Key::a))
        camera.position -= camera_right * move_speed;

    // Движение вверх/вниз (Q/E)
    if (keyboard::isKeyDown(keyboard::Key::q))
        camera.position += camera_up * move_speed;

    if (keyboard::isKeyDown(keyboard::Key::e))
        camera.position -= camera_up * move_speed;

    // Поворот камеры стрелками клавиатуры
    if (keyboard::isKeyDown(keyboard::Key::left))
        camera.yaw -= rotate_speed;

    if (keyboard::isKeyDown(keyboard::Key::right))
        camera.yaw += rotate_speed;

    if (keyboard::isKeyDown(keyboard::Key::up))
        camera.pitch -= rotate_speed;

    if (keyboard::isKeyDown(keyboard::Key::down))
        camera.pitch += rotate_speed;

    // Ограничиваем угол наклона, чтобы не "перевернуть" камеру
    if (camera.pitch > 89.0f) camera.pitch = 89.0f;
    if (camera.pitch < -89.0f) camera.pitch = -89.0f;

    // ========================================================================
    // Управление камерой мышью (при удержании правой кнопки)
    // ========================================================================
    static bool mouse_control_active = false;
    
    if (mouse::isButtonDown(mouse::Button::right)) {
        if (!mouse_control_active) {
            mouse_control_active = true;
        }
        
        // Получаем смещение курсора с прошлого кадра
        auto delta = mouse::cursorDelta();
        const float mouse_sensitivity = 0.15f;  // Чувствительность мыши
        
        // Обновляем углы камеры на основе движения мыши
        camera.yaw += delta.x * mouse_sensitivity;
        camera.pitch -= delta.y * mouse_sensitivity;  // Инвертируем Y для интуитивного управления
        
        // Ограничиваем pitch
        if (camera.pitch > 89.0f) camera.pitch = 89.0f;
        if (camera.pitch < -89.0f) camera.pitch = -89.0f;
    } else {
        mouse_control_active = false;
    }

    // ========================================================================
    // Анимация: вращение сферы по круговой орбите
    // ========================================================================
    const float orbit_radius = 2.0f;   // Радиус орбиты
    const float orbit_speed = 1.0f;    // Скорость вращения (рад/с)
    
    if (models.size() >= 2) {
        Model& sphere = models[1];
        float angle = float(time * orbit_speed);  // Текущий угол поворота
        float x = orbit_radius * cosf(angle);
        float z = orbit_radius * sinf(angle);
        sphere.transform.position = { x, -0.5f, z };
    }

    // ========================================================================
    // Подготовка данных для отправки в шейдеры
    // ========================================================================
    
    // Вычисляем соотношение сторон экрана для матрицы проекции
    float aspect_ratio = float(veekay::app.window_width) / float(veekay::app.window_height);
    
    // Вычисляем light space matrix для выбранного источника света
    veekay::vec3 light_pos;
    if (shadow_light_type == ShadowLightType::Directional) {
        // Для направленного света используем позицию на расстоянии вдоль направления
        veekay::vec3 dir = directional_light.direction;
        float length = sqrtf(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z);
        if (length > 0.001f) {
            dir.x /= length; dir.y /= length; dir.z /= length;
        }
        light_pos = {-dir.x * 10.0f, -dir.y * 10.0f, -dir.z * 10.0f};
    } else if (shadow_light_type == ShadowLightType::Point) {
        if (!point_lights.empty() && shadow_light_index < int(point_lights.size())) {
            light_pos = point_lights[shadow_light_index].position;
        } else {
            light_pos = {0.0f, -5.0f, 0.0f};
        }
    } else { // Spot
        if (!spot_lights.empty() && shadow_light_index < int(spot_lights.size())) {
            light_pos = spot_lights[shadow_light_index].position;
        } else {
            light_pos = {0.0f, -5.0f, 0.0f};
        }
    }
    
    veekay::mat4 light_space_matrix = calculateLightSpaceMatrix(light_pos);
    
    // Формируем uniform-буфер с данными сцены
    SceneUniforms scene_uniforms{
        .view_projection = camera.view_projection(aspect_ratio),
        .view_position = camera.position,
        .directional_light = directional_light_enabled ? directional_light : DirectionalLight{
            .direction = {0.0f, 0.0f, 0.0f},
            .ambient = {0.0f, 0.0f, 0.0f},
            .diffuse = {0.0f, 0.0f, 0.0f},
            .specular = {0.0f, 0.0f, 0.0f},
        },
        .point_light_count = static_cast<uint32_t>(point_lights.size()),
        .spot_light_count = static_cast<uint32_t>(spot_lights.size()),
        .light_space_matrix = light_space_matrix,
    };

    // Копируем массивы источников света в GPU буферы
    if (!point_lights.empty()) {
        memcpy(point_lights_buffer->mapped_region, point_lights.data(), 
               point_lights.size() * sizeof(PointLight));
    }
    
    if (!spot_lights.empty()) {
        memcpy(spot_lights_buffer->mapped_region, spot_lights.data(), 
               spot_lights.size() * sizeof(SpotLight));
    }

    // Формируем uniform-буферы для каждой модели
    std::vector<ModelUniforms> model_uniforms(models.size());
    for (size_t i = 0, n = models.size(); i < n; ++i) {
        const Model& model = models[i];
        ModelUniforms& uniforms = model_uniforms[i];

        uniforms.model = model.transform.matrix();       // Матрица трансформации
        uniforms.albedo_color = model.material.albedo;   // Основной цвет
        uniforms.shininess = model.material.shininess;   // Степень блеска
        uniforms.specular_color = model.material.specular; // Цвет бликов
    }

    // Копируем данные сцены в GPU буфер
    *(SceneUniforms*)scene_uniforms_buffer->mapped_region = scene_uniforms;

    // Копируем light space matrix для shadow pass
    *(veekay::mat4*)light_space_buffer->mapped_region = light_space_matrix;

    // Копируем данные моделей с учетом выравнивания (alignment)
    const size_t alignment =
        veekay::graphics::Buffer::structureAlignment(sizeof(ModelUniforms));

    for (size_t i = 0, n = model_uniforms.size(); i < n; ++i) {
        const ModelUniforms& uniforms = model_uniforms[i];

        // Вычисляем смещение с учетом выравнивания памяти GPU
        char* const pointer = static_cast<char*>(model_uniforms_buffer->mapped_region) + i * alignment;
        *reinterpret_cast<ModelUniforms*>(pointer) = uniforms;
    }
}

// ============================================================================
// ОТРИСОВКА КАДРА
// ============================================================================
// Эта функция записывает команды рендеринга в Vulkan command buffer
// Вызывается каждый кадр после update()
void render(VkCommandBuffer cmd, VkFramebuffer framebuffer) {
    // Статическая переменная для отслеживания первого кадра
    static bool first_frame = true;
    
    // Очищаем command buffer от предыдущих команд
    vkResetCommandBuffer(cmd, 0);

    // Начинаем запись команд рендеринга
    {
        VkCommandBufferBeginInfo info{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        };
        vkBeginCommandBuffer(cmd, &info);
    }

    VkDeviceSize zero_offset = 0;
    const size_t model_uniforms_alignment =
        veekay::graphics::Buffer::structureAlignment(sizeof(ModelUniforms));

    // ========================================================================
    // ПРОХОД 1: SHADOW PASS (рендеринг в depth map)
    // ========================================================================
    {
        // Переводим shadow map в состояние для записи
        // В первом кадре layout = UNDEFINED, в последующих = DEPTH_STENCIL_READ_ONLY_OPTIMAL
        VkImageMemoryBarrier barrier{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .srcAccessMask = first_frame ? static_cast<VkAccessFlags>(0) : static_cast<VkAccessFlags>(VK_ACCESS_SHADER_READ_BIT),
            .dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            .oldLayout = first_frame ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
            .newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = shadow_map_image,
            .subresourceRange = {
                .aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
        };

        vkCmdPipelineBarrier(cmd,
            first_frame ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT : VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
            0, 0, nullptr, 0, nullptr, 1, &barrier);

        // Начинаем shadow render pass
        VkClearValue clear_depth{.depthStencil = {1.0f, 0}};
        
        VkRenderPassBeginInfo shadow_pass_info{
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            .renderPass = shadow_render_pass,
            .framebuffer = shadow_framebuffer,
            .renderArea = {
                .offset = {0, 0},
                .extent = {shadow_map_size, shadow_map_size},
            },
            .clearValueCount = 1,
            .pClearValues = &clear_depth,
        };

        vkCmdBeginRenderPass(cmd, &shadow_pass_info, VK_SUBPASS_CONTENTS_INLINE);

        // Привязываем shadow pipeline
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, shadow_pipeline);

        // Рендерим все модели в shadow map
        VkBuffer current_vertex_buffer = VK_NULL_HANDLE;
        VkBuffer current_index_buffer = VK_NULL_HANDLE;

        for (size_t i = 0; i < models.size(); ++i) {
            const Model& model = models[i];
            const Mesh& mesh = model.mesh;

            if (current_vertex_buffer != mesh.vertex_buffer->buffer) {
                current_vertex_buffer = mesh.vertex_buffer->buffer;
                vkCmdBindVertexBuffers(cmd, 0, 1, &current_vertex_buffer, &zero_offset);
            }

            if (current_index_buffer != mesh.index_buffer->buffer) {
                current_index_buffer = mesh.index_buffer->buffer;
                vkCmdBindIndexBuffer(cmd, current_index_buffer, zero_offset, VK_INDEX_TYPE_UINT32);
            }

            // Привязываем descriptor set для shadow pass
            uint32_t offset = i * model_uniforms_alignment;
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, shadow_pipeline_layout,
                                    0, 1, &shadow_descriptor_set, 1, &offset);

            // Рисуем геометрию
            vkCmdDrawIndexed(cmd, mesh.indices, 1, 0, 0, 0);
        }

        // Render pass автоматически переводит изображение из DEPTH_STENCIL_ATTACHMENT_OPTIMAL
        // в DEPTH_STENCIL_READ_ONLY_OPTIMAL (указано в finalLayout)
        vkCmdEndRenderPass(cmd);
        
        // Дополнительный барьер для синхронизации доступа между проходами
        VkImageMemoryBarrier read_barrier{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
            .newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = shadow_map_image,
            .subresourceRange = {
                .aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
        };

        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &read_barrier);
    }

    // ========================================================================
    // ПРОХОД 2: MAIN PASS (основной рендеринг с тенями)
    // ========================================================================
    {
        // Значение для очистки буфера цвета (темно-серый фон)
        VkClearValue clear_color{.color = {{0.1f, 0.1f, 0.1f, 1.0f}}};
        
        // Значение для очистки буфера глубины (максимальная глубина)
        VkClearValue clear_depth{.depthStencil = {1.0f, 0}};

        VkClearValue clear_values[] = {clear_color, clear_depth};

        VkRenderPassBeginInfo info{
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            .renderPass = veekay::app.vk_render_pass,
            .framebuffer = framebuffer,
            .renderArea = {
                .extent = {
                    veekay::app.window_width,
                    veekay::app.window_height
                },
            },
            .clearValueCount = 2,
            .pClearValues = clear_values,
        };

        vkCmdBeginRenderPass(cmd, &info, VK_SUBPASS_CONTENTS_INLINE);

        // Привязываем основной graphics pipeline
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

        // Создаем порядок рендеринга для прозрачности
        std::vector<size_t> render_order(models.size());
        for (size_t i = 0; i < models.size(); ++i) {
            render_order[i] = i;
        }
        
        // Сортируем от дальних к ближним
        std::sort(render_order.begin(), render_order.end(), 
            [](size_t a, size_t b) {
                veekay::vec3 delta_a = models[a].transform.position - camera.position;
                float dist_sq_a = delta_a.x * delta_a.x + delta_a.y * delta_a.y + delta_a.z * delta_a.z;
                
                veekay::vec3 delta_b = models[b].transform.position - camera.position;
                float dist_sq_b = delta_b.x * delta_b.x + delta_b.y * delta_b.y + delta_b.z * delta_b.z;
                
                return dist_sq_a > dist_sq_b;
            });

        VkBuffer current_vertex_buffer = VK_NULL_HANDLE;
        VkBuffer current_index_buffer = VK_NULL_HANDLE;

        // Рендерим все модели
        for (size_t idx = 0; idx < render_order.size(); ++idx) {
            size_t i = render_order[idx];
            const Model& model = models[i];
            const Mesh& mesh = model.mesh;

            if (current_vertex_buffer != mesh.vertex_buffer->buffer) {
                current_vertex_buffer = mesh.vertex_buffer->buffer;
                vkCmdBindVertexBuffers(cmd, 0, 1, &current_vertex_buffer, &zero_offset);
            }

            if (current_index_buffer != mesh.index_buffer->buffer) {
                current_index_buffer = mesh.index_buffer->buffer;
                vkCmdBindIndexBuffer(cmd, current_index_buffer, zero_offset, VK_INDEX_TYPE_UINT32);
            }

            // Привязываем descriptor set материала (включает shadow map)
            uint32_t offset = i * model_uniforms_alignment;
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
                                    0, 1, &model.material.descriptor_set, 1, &offset);

            // Отрисовка геометрии
            vkCmdDrawIndexed(cmd, mesh.indices, 1, 0, 0, 0);
        }

        vkCmdEndRenderPass(cmd);
    }

    // Завершаем запись команд в command buffer
    if (vkEndCommandBuffer(cmd) != VK_SUCCESS) {
        std::cerr << "Failed to end command buffer recording\n";
        return;
    }
    
    // После первого кадра переключаем флаг
    first_frame = false;
}

} // namespace

// ============================================================================
// ТОЧКА ВХОДА В ПРОГРАММУ
// ============================================================================
// Запускает главный цикл приложения с передачей callback-функций
int main() {
    // Передаем в движок veekay три callback-функции:
    // - init: вызывается один раз при запуске для инициализации Vulkan
    // - shutdown: вызывается при завершении для освобождения ресурсов
    // - update: вызывается каждый кадр для обновления логики и UI
    // - render: вызывается каждый кадр для записи команд рендеринга
    return veekay::run({
        .init = initialize,
        .shutdown = shutdown,
        .update = update,
        .render = render,
    });
}
