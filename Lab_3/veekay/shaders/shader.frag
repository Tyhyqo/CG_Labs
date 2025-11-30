#version 450

// ============================================================================
// ВХОДНЫЕ ДАННЫЕ ИЗ ВЕРШИННОГО ШЕЙДЕРА
// ============================================================================
// Интерполированные значения для каждого фрагмента (пикселя)
layout(location = 0) in vec3 frag_position;  // Позиция фрагмента в мировых координатах
layout(location = 1) in vec3 frag_normal;    // Нормаль поверхности (интерполированная)
layout(location = 2) in vec2 frag_uv;        // Текстурные координаты (не используются в этой лабе)

// ============================================================================
// ВЫХОДНЫЕ ДАННЫЕ
// ============================================================================
layout(location = 0) out vec4 out_color;     // Итоговый цвет пикселя (RGBA)

// ============================================================================
// СТРУКТУРЫ ИСТОЧНИКОВ СВЕТА
// ============================================================================
// Должны полностью совпадать с определениями в C++ коде!

// Направленный свет (глобальное освещение, например, солнце)
// Не имеет позиции, только направление
struct DirectionalLight {
    vec3 direction;   // Направление света (откуда светит)
    float _pad0;      // Выравнивание для std140 layout
    vec3 ambient;     // Фоновая составляющая (цвет окружающего света)
    float _pad1;
    vec3 diffuse;     // Диффузная составляющая (основной цвет света)
    float _pad2;
    vec3 specular;    // Зеркальная составляющая (цвет бликов)
    float _pad3;
};

// Точечный источник света (лампочка)
// Свет затухает с расстоянием по квадратичному закону
struct PointLight {
    vec3 position;    // Позиция источника в мировых координатах
    float _pad0;
    vec3 ambient;     // Фоновая составляющая
    float _pad1;
    vec3 diffuse;     // Диффузная составляющая
    float _pad2;
    vec3 specular;    // Зеркальная составляющая
    float constant;   // Константа затухания (обычно 1.0)
    float linear;     // Линейная составляющая затухания
    float quadratic;  // Квадратичная составляющая затухания
    float _pad3;
};

// Прожектор (направленный конус света)
// Свет затухает с расстоянием И по углу отклонения от оси
struct SpotLight {
    vec3 position;     // Позиция прожектора
    float _pad0;
    vec3 direction;    // Направление луча прожектора
    float _pad1;
    vec3 ambient;      // Фоновая составляющая
    float _pad2;
    vec3 diffuse;      // Диффузная составляющая
    float _pad3;
    vec3 specular;     // Зеркальная составляющая
    float constant;    // Константа затухания
    float linear;      // Линейная составляющая затухания
    float quadratic;   // Квадратичная составляющая затухания
    float cutOff;      // Косинус внутреннего угла конуса (полная яркость)
    float outerCutOff; // Косинус внешнего угла конуса (начало затухания)
    float _pad4[3];
};

// ============================================================================
// UNIFORM-БУФЕРЫ (ДАННЫЕ СЦЕНЫ И МАТЕРИАЛА)
// ============================================================================

// Глобальные данные сцены (общие для всех объектов)
layout(binding = 0) uniform SceneUniforms {
    mat4 view_projection;                  // Матрица вид-проекция (для отсечения)
    vec3 view_position;                    // Позиция камеры (для расчета бликов)
    float _pad0;
    DirectionalLight directional_light;    // Единственный глобальный направленный свет
    uint point_light_count;                // Количество активных точечных источников
    uint spot_light_count;                 // Количество активных прожекторов
    float _pad1;
    float _pad2;
} scene;

// Данные материала текущего объекта (индивидуальные для каждой модели)
layout(binding = 1) uniform ModelUniforms {
    mat4 model;                // Матрица трансформации модели (не используется во фрагментном шейдере)
    vec3 albedo_color;         // Основной цвет поверхности (диффузный компонент)
    float shininess;           // Степень блеска (от 1 до 256, чем больше - тем меньше блик)
    vec3 specular_color;       // Цвет зеркального отражения (обычно белый или цвет металла)
    float _pad0;
} material;

// ============================================================================
// STORAGE-БУФЕРЫ (ДИНАМИЧЕСКИЕ МАССИВЫ ИСТОЧНИКОВ СВЕТА)
// ============================================================================

// Массив всех точечных источников света в сцене
layout(binding = 2) readonly buffer PointLightsBuffer {
    PointLight point_lights[];  // Динамический массив (размер = point_light_count)
};

// Массив всех прожекторов в сцене
layout(binding = 3) readonly buffer SpotLightsBuffer {
    SpotLight spot_lights[];    // Динамический массив (размер = spot_light_count)
};

// НОВОЕ: Текстура материала + сэмплер
layout(binding = 4) uniform sampler2D material_texture;

// ============================================================================
// РАСЧЕТ НАПРАВЛЕННОГО СВЕТА (модель Блинн-Фонга)
// ============================================================================
// Направленный свет не зависит от расстояния, только от ориентации поверхности
vec3 calcDirectionalLight(DirectionalLight light, vec3 normal, vec3 view_dir) {
    // Нормализуем направление света (инвертируем, т.к. храним "откуда", а нужно "куда")
    vec3 light_dir = normalize(-light.direction);
    
    // AMBIENT (фоновое освещение) - не зависит от углов, всегда присутствует
    vec3 ambient = light.ambient * material.albedo_color;
    
    // DIFFUSE (диффузное освещение) - зависит от угла между нормалью и светом
    // Формула Ламберта: интенсивность = cos(угол) = dot(N, L)
    float diff = max(dot(normal, light_dir), 0.0);  // Отсекаем отрицательные значения
    vec3 diffuse = light.diffuse * diff * material.albedo_color;
    
    // SPECULAR (зеркальное отражение) - модель Блинн-Фонга
    // Используем halfway вектор (биссектрису между направлением света и взгляда)
    vec3 halfway_dir = normalize(light_dir + view_dir);
    float spec = pow(max(dot(normal, halfway_dir), 0.0), material.shininess);
    vec3 specular = light.specular * spec * material.specular_color;
    
    // Суммируем все три компонента освещения
    return ambient + diffuse + specular;
}

// ============================================================================
// РАСЧЕТ ТОЧЕЧНОГО ИСТОЧНИКА СВЕТА (с затуханием по расстоянию)
// ============================================================================
vec3 calcPointLight(PointLight light, vec3 normal, vec3 frag_pos, vec3 view_dir) {
    // Направление от фрагмента к источнику света
    vec3 light_dir = normalize(light.position - frag_pos);
    
    // Расстояние от фрагмента до источника света
    float distance = length(light.position - frag_pos);
    
    // ЗАТУХАНИЕ: формула Кука-Торренса (обратно пропорционально квадрату расстояния)
    // attenuation = 1 / (Kc + Kl*d + Kq*d²)
    // Kc - константа (базовая яркость), Kl - линейное затухание, Kq - квадратичное
    float attenuation = 1.0 / (light.constant + light.linear * distance + 
                               light.quadratic * (distance * distance));
    
    // AMBIENT (фоновое освещение)
    vec3 ambient = light.ambient * material.albedo_color;
    
    // DIFFUSE (диффузное освещение по закону Ламберта)
    float diff = max(dot(normal, light_dir), 0.0);
    vec3 diffuse = light.diffuse * diff * material.albedo_color;
    
    // SPECULAR (зеркальное отражение по модели Блинн-Фонга)
    vec3 halfway_dir = normalize(light_dir + view_dir);
    float spec = pow(max(dot(normal, halfway_dir), 0.0), material.shininess);
    vec3 specular = light.specular * spec * material.specular_color;
    
    // Применяем затухание ко всем компонентам
    ambient *= attenuation;
    diffuse *= attenuation;
    specular *= attenuation;
    
    return ambient + diffuse + specular;
}

// ============================================================================
// РАСЧЕТ ПРОЖЕКТОРА (с затуханием по расстоянию и углу)
// ============================================================================
vec3 calcSpotLight(SpotLight light, vec3 normal, vec3 frag_pos, vec3 view_dir) {
    // Направление от фрагмента к прожектору
    vec3 light_dir = normalize(light.position - frag_pos);
    
    // УГЛОВОЕ ЗАТУХАНИЕ: проверяем, попадает ли фрагмент в конус света
    // theta - косинус угла между направлением луча и направлением на фрагмент
    float theta = dot(light_dir, normalize(-light.direction));
    
    // ПЛАВНЫЕ КРАЯ: линейная интерполяция между cutOff (полная яркость) и outerCutOff (темнота)
    // epsilon - ширина "размытой" зоны на краю конуса
    float epsilon = light.cutOff - light.outerCutOff;
    // intensity = 1.0 внутри cutOff, плавно падает до 0.0 за пределами outerCutOff
    float intensity = clamp((theta - light.outerCutOff) / epsilon, 0.0, 1.0);
    
    // ЗАТУХАНИЕ ПО РАССТОЯНИЮ (аналогично точечному источнику)
    float distance = length(light.position - frag_pos);
    float attenuation = 1.0 / (light.constant + light.linear * distance + 
                               light.quadratic * (distance * distance));
    
    // AMBIENT (фоновое освещение)
    vec3 ambient = light.ambient * material.albedo_color;
    
    // DIFFUSE (диффузное освещение)
    float diff = max(dot(normal, light_dir), 0.0);
    vec3 diffuse = light.diffuse * diff * material.albedo_color;
    
    // SPECULAR (зеркальное отражение)
    vec3 halfway_dir = normalize(light_dir + view_dir);
    float spec = pow(max(dot(normal, halfway_dir), 0.0), material.shininess);
    vec3 specular = light.specular * spec * material.specular_color;
    
    // Применяем УГЛОВОЕ затухание к diffuse и specular (ambient не затухает по углу)
    diffuse *= intensity * attenuation;
    specular *= intensity * attenuation;
    // Ambient затухает только по расстоянию
    ambient *= attenuation;
    
    return ambient + diffuse + specular;
}

// ============================================================================
// ГЛАВНАЯ ФУНКЦИЯ ФРАГМЕНТНОГО ШЕЙДЕРА
// ============================================================================
// Вычисляет итоговый цвет пикселя с учетом всех источников света
void main() {
    // Нормализуем нормаль (интерполяция могла изменить её длину)
    vec3 normal = normalize(frag_normal);
    
    // Направление от фрагмента к камере (для расчета бликов)
    vec3 view_dir = normalize(scene.view_position - frag_position);
    
    // СЭМПЛИРУЕМ ТЕКСТУРУ: получаем цвет из текстуры по UV-координатам
    vec4 tex_color = texture(material_texture, frag_uv);
    
    // СМЕШИВАЕМ цвет материала с цветом текстуры
    // Используем умножение для модуляции (тонирования текстуры)
    vec3 final_albedo = material.albedo_color * tex_color.rgb;
    
    // НАЧИНАЕМ с направленного света (используем final_albedo вместо material.albedo_color)
    // Пересчитываем ambient компонент
    vec3 dir_ambient = scene.directional_light.ambient * final_albedo;
    
    // Направление света
    vec3 light_dir = normalize(-scene.directional_light.direction);
    
    // DIFFUSE компонент
    float diff = max(dot(normal, light_dir), 0.0);
    vec3 dir_diffuse = scene.directional_light.diffuse * diff * final_albedo;
    
    // SPECULAR компонент (не меняется, используется specular_color из материала)
    vec3 halfway_dir = normalize(light_dir + view_dir);
    float spec = pow(max(dot(normal, halfway_dir), 0.0), material.shininess);
    vec3 dir_specular = scene.directional_light.specular * spec * material.specular_color;
    
    // В конце расчета specular
    float fresnel = pow(1.0 - max(dot(normal, view_dir), 0.0), 5.0);
    dir_specular *= (1.0 + fresnel * 2.0); // Усиливаем блики под углом

    vec3 result = dir_ambient + dir_diffuse + dir_specular;
    
    // ДОБАВЛЯЕМ освещение от всех точечных источников
    if (scene.point_light_count > 0u) {
        for (uint i = 0u; i < scene.point_light_count; ++i) {
            PointLight light = point_lights[i];
            
            // Направление от фрагмента к источнику света
            vec3 point_light_dir = normalize(light.position - frag_position);
            
            // Расстояние до источника
            float distance = length(light.position - frag_position);
            
            // Затухание
            float attenuation = 1.0 / (light.constant + light.linear * distance + 
                                       light.quadratic * (distance * distance));
            
            // AMBIENT
            vec3 point_ambient = light.ambient * final_albedo;
            
            // DIFFUSE
            float point_diff = max(dot(normal, point_light_dir), 0.0);
            vec3 point_diffuse = light.diffuse * point_diff * final_albedo;
            
            // SPECULAR
            vec3 point_halfway = normalize(point_light_dir + view_dir);
            float point_spec = pow(max(dot(normal, point_halfway), 0.0), material.shininess);
            vec3 point_specular = light.specular * point_spec * material.specular_color;
            
            // Применяем затухание
            point_ambient *= attenuation;
            point_diffuse *= attenuation;
            point_specular *= attenuation;
            
            result += point_ambient + point_diffuse + point_specular;
        }
    }
    
    // ДОБАВЛЯЕМ освещение от всех прожекторов
    if (scene.spot_light_count > 0u) {
        for (uint i = 0u; i < scene.spot_light_count; ++i) {
            SpotLight light = spot_lights[i];
            
            // Направление от фрагмента к прожектору
            vec3 spot_light_dir = normalize(light.position - frag_position);
            
            // УГЛОВОЕ ЗАТУХАНИЕ
            float theta = dot(spot_light_dir, normalize(-light.direction));
            float epsilon = light.cutOff - light.outerCutOff;
            float intensity = clamp((theta - light.outerCutOff) / epsilon, 0.0, 1.0);
            
            // ЗАТУХАНИЕ ПО РАССТОЯНИЮ
            float distance = length(light.position - frag_position);
            float attenuation = 1.0 / (light.constant + light.linear * distance + 
                                       light.quadratic * (distance * distance));
            
            // AMBIENT
            vec3 spot_ambient = light.ambient * final_albedo;
            
            // DIFFUSE
            float spot_diff = max(dot(normal, spot_light_dir), 0.0);
            vec3 spot_diffuse = light.diffuse * spot_diff * final_albedo;
            
            // SPECULAR
            vec3 spot_halfway = normalize(spot_light_dir + view_dir);
            float spot_spec = pow(max(dot(normal, spot_halfway), 0.0), material.shininess);
            vec3 spot_specular = light.specular * spot_spec * material.specular_color;
            
            // Применяем УГЛОВОЕ затухание к diffuse и specular
            spot_diffuse *= intensity * attenuation;
            spot_specular *= intensity * attenuation;
            // Ambient затухает только по расстоянию
            spot_ambient *= attenuation;
            
            result += spot_ambient + spot_diffuse + spot_specular;
        }
    }
    
    // Учитываем альфа-канал текстуры
    out_color = vec4(result, tex_color.a);
}