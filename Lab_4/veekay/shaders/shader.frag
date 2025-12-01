#version 450

// ============================================================================
// ВХОДНЫЕ ДАННЫЕ ИЗ ВЕРШИННОГО ШЕЙДЕРА
// ============================================================================
// Интерполированные значения для каждого фрагмента (пикселя)
layout(location = 0) in vec3 frag_position;  // Позиция фрагмента в мировых координатах
layout(location = 1) in vec3 frag_normal;    // Нормаль поверхности (интерполированная)
layout(location = 2) in vec2 frag_uv;        // Текстурные координаты (не используются в этой лабе)
layout(location = 3) in vec4 frag_position_light_space;  // Позиция в пространстве света

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
    mat4 light_space_matrix;               // Матрица для shadow mapping
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

// НОВОЕ: Shadow Map + сэмплер с PCF
layout(binding = 5) uniform sampler2DShadow shadow_map;

// DEBUG: Shadow Map как обычная текстура для чтения глубины
layout(binding = 6) uniform sampler2D shadow_map_raw;

// ============================================================================
// РАСЧЕТ ТЕНИ (SHADOW MAPPING с PCF)
// ============================================================================
// Возвращает значение от 0.0 (полная тень) до 1.0 (нет тени)
float calculateShadow(vec4 frag_pos_light_space) {
    // Перспективное деление (переход из clip space в NDC)
    vec3 proj_coords = frag_pos_light_space.xyz / frag_pos_light_space.w;

    // Преобразуем X,Y из NDC [-1,1] в текстурные координаты [0,1].
    // Для глубины (Z) используем значение, которое формируется projection матрицей света.
    proj_coords.xy = proj_coords.xy * 0.5 + 0.5;
    
    // Если фрагмент за пределами light frustum, он не в тени
    if (proj_coords.z > 1.0 || proj_coords.z < 0.0 ||
        proj_coords.x < 0.0 || proj_coords.x > 1.0 ||
        proj_coords.y < 0.0 || proj_coords.y > 1.0) {
        return 1.0;  // Не в тени (за пределами карты теней)
    }
    
    // Bias для борьбы с shadow acne (артефактами самозатенения)
    // ИСПРАВЛЕНО: увеличен bias для лучшей борьбы с артефактами
    float bias = 0.005;
    
    // DEBUG: простая проверка без PCF
    // Раскомментируйте для отладки
    //float closest_depth = texture(shadow_map_raw, proj_coords.xy).r;
    //return (proj_coords.z - bias) > closest_depth ? 0.0 : 1.0;
    
    // PCF (Percentage Closer Filtering) - сглаживание теней
    // Сэмплируем несколько соседних текселей и усредняем результат
    float shadow = 0.0;
    // Используем raw shadow map (sampler2D) для определения размера текстуры
    vec2 texel_size = 1.0 / vec2(textureSize(shadow_map_raw, 0));
    
    // 3x3 kernel для PCF
    for(int x = -1; x <= 1; ++x) {
        for(int y = -1; y <= 1; ++y) {
            vec2 offset = vec2(x, y) * texel_size;
            // textureProj автоматически сравнивает глубину благодаря sampler2DShadow
            shadow += texture(shadow_map, vec3(proj_coords.xy + offset, proj_coords.z - bias));
        }
    }
    shadow /= 9.0;  // Усредняем по 9 сэмплам
    
    return shadow;
}

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
    // DEBUG: Визуализация light space координат
    // Раскомментируйте для отладки
    /*
    vec3 proj_coords = frag_position_light_space.xyz / frag_position_light_space.w;
    proj_coords = proj_coords * 0.5 + 0.5;
    
    // Цветовая визуализация глубины для лучшего понимания:
    // Синий = очень близко (0.0 - 0.3)
    // Зеленый = средняя дистанция (0.3 - 0.7)  
    // Красный = далеко (0.7 - 1.0)
    // Белый = за пределами far plane (>1.0)
    float depth = proj_coords.z;
    vec3 color;
    if (depth < 0.3) {
        color = vec3(0, 0, 1); // Синий
    } else if (depth < 0.7) {
        color = vec3(0, 1, 0); // Зеленый
    } else if (depth < 1.0) {
        color = vec3(1, 0, 0); // Красный
    } else {
        color = vec3(1, 1, 1); // Белый = ЗА ПРЕДЕЛАМИ!
    }
    out_color = vec4(color, 1.0);
    return;
    */
    
    // Нормализуем нормаль (интерполяция могла изменить её длину)
    vec3 normal = normalize(frag_normal);
    
    // Направление от фрагмента к камере (для расчета бликов)
    vec3 view_dir = normalize(scene.view_position - frag_position);
    
    // Начинаем с нулевого освещения (полная темнота)
    vec3 result = vec3(0.0);
    
    // ========================================================================
    // РАСЧЕТ ТЕНИ (ЕДИНАЯ ДЛЯ ВСЕХ ИСТОЧНИКОВ СВЕТА)
    // ========================================================================
    // Вычисляем тень один раз для всех источников света
    // sampler2DShadow возвращает 1.0 когда НЕТ тени (depth test passed)
    // и 0.0 когда ЕСТЬ тень (depth test failed)
    float shadow = calculateShadow(frag_position_light_space);
    
    // ========================================================================
    // НАПРАВЛЕННЫЙ СВЕТ (солнце) С ТЕНЯМИ
    // ========================================================================
    // Ambient освещение (фоновое) - всегда видно, не затеняется
    vec3 ambient = scene.directional_light.ambient * material.albedo_color;
    result += ambient;
    
    // Diffuse и Specular освещение - затеняются
    // Вычисляем только диффузную и зеркальную составляющие (без ambient)
    vec3 light_dir = normalize(-scene.directional_light.direction);
    float diff = max(dot(normal, light_dir), 0.0);
    vec3 diffuse = scene.directional_light.diffuse * diff * material.albedo_color;
    
    vec3 halfway_dir = normalize(light_dir + view_dir);
    float spec = pow(max(dot(normal, halfway_dir), 0.0), material.shininess);
    vec3 specular = scene.directional_light.specular * spec * material.specular_color;
    
    // Применяем тень к diffuse и specular (shadow=1.0 => полное освещение, shadow=0.0 => полная тень)
    result += (diffuse + specular) * shadow;
    
    // ========================================================================
    // ТОЧЕЧНЫЕ ИСТОЧНИКИ СВЕТА (лампочки) С ТЕНЯМИ
    // ========================================================================
    // Суммируем вклад всех активных точечных источников
    // ВАЖНО: ambient не затеняется, только diffuse и specular!
    for (uint i = 0u; i < scene.point_light_count; ++i) {
        PointLight light = point_lights[i];
        
        // Направление от фрагмента к источнику света
        vec3 light_dir = normalize(light.position - frag_position);
        
        // Расстояние и затухание
        float distance = length(light.position - frag_position);
        float attenuation = 1.0 / (light.constant + light.linear * distance + 
                                   light.quadratic * (distance * distance));
        
        // AMBIENT - не затеняется, всегда видно
        vec3 ambient = light.ambient * material.albedo_color * attenuation;
        result += ambient;
        
        // DIFFUSE - затеняется
        float diff = max(dot(normal, light_dir), 0.0);
        vec3 diffuse = light.diffuse * diff * material.albedo_color * attenuation;
        
        // SPECULAR - затеняется
        vec3 halfway_dir = normalize(light_dir + view_dir);
        float spec = pow(max(dot(normal, halfway_dir), 0.0), material.shininess);
        vec3 specular = light.specular * spec * material.specular_color * attenuation;
        
        // Применяем тень только к diffuse и specular
        result += (diffuse + specular) * shadow;
    }
    
    // ========================================================================
    // ПРОЖЕКТОРЫ (направленные конусы света) С ТЕНЯМИ
    // ========================================================================
    // Суммируем вклад всех активных прожекторов с правильным применением теней
    for (uint i = 0u; i < scene.spot_light_count; ++i) {
        SpotLight light = spot_lights[i];
        
        // Направление от фрагмента к прожектору
        vec3 light_dir = normalize(light.position - frag_position);
        
        // Угловое затухание (конус света)
        float theta = dot(light_dir, normalize(-light.direction));
        float epsilon = light.cutOff - light.outerCutOff;
        float intensity = clamp((theta - light.outerCutOff) / epsilon, 0.0, 1.0);
        
        // Затухание по расстоянию
        float distance = length(light.position - frag_position);
        float attenuation = 1.0 / (light.constant + light.linear * distance + 
                                   light.quadratic * (distance * distance));
        
        // AMBIENT - не затеняется
        vec3 ambient = light.ambient * material.albedo_color * attenuation;
        result += ambient;
        
        // DIFFUSE - затеняется
        float diff = max(dot(normal, light_dir), 0.0);
        vec3 diffuse = light.diffuse * diff * material.albedo_color * intensity * attenuation;
        
        // SPECULAR - затеняется
        vec3 halfway_dir = normalize(light_dir + view_dir);
        float spec = pow(max(dot(normal, halfway_dir), 0.0), material.shininess);
        vec3 specular = light.specular * spec * material.specular_color * intensity * attenuation;
        
        // Применяем тень только к diffuse и specular
        result += (diffuse + specular) * shadow;
    }
    
    // ========================================================================
    // ПРИМЕНЕНИЕ ТЕКСТУРЫ
    // ========================================================================
    // Умножаем результат освещения на цвет из текстуры
    vec3 texture_color = texture(material_texture, frag_uv).rgb;
    result *= texture_color;
    
    // ========================================================================
    // ФИНАЛЬНЫЙ ВЫВОД
    // ========================================================================
    // Записываем итоговый цвет с полной непрозрачностью (alpha = 1.0)
    out_color = vec4(result, 1.0);
}