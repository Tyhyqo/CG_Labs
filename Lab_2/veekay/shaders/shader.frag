#version 450

layout(location = 0) in vec3 frag_position;
layout(location = 1) in vec3 frag_normal;
layout(location = 2) in vec2 frag_uv;
layout(location = 3) in vec3 frag_albedo;
layout(location = 4) in vec3 frag_specular;
layout(location = 5) in float frag_shininess;

layout(location = 0) out vec4 out_color;

// DirectionalLight structure
struct DirectionalLight {
    vec3 direction;
    float _pad0;
    vec3 ambient;
    float _pad1;
    vec3 diffuse;
    float _pad2;
    vec3 specular;
    float _pad3;
};

// PointLight structure
struct PointLight {
    vec3 position;
    float _pad0;
    vec3 ambient;
    float _pad1;
    vec3 diffuse;
    float _pad2;
    vec3 specular;
    float constant;
    float linear;
    float quadratic;
    float _pad3;
};

// SpotLight structure
struct SpotLight {
    vec3 position;
    float _pad0;
    vec3 direction;
    float _pad1;
    vec3 ambient;
    float _pad2;
    vec3 diffuse;
    float _pad3;
    vec3 specular;
    float constant;
    float linear;
    float quadratic;
    float cutOff;
    float outerCutOff;
    float _pad4[3];
};

layout(binding = 0) uniform SceneUniforms {
    mat4 view_projection;
    vec3 view_position;
    float _pad0;
    DirectionalLight directional_light;
} scene;

layout(binding = 2) readonly buffer PointLightsBuffer {
    PointLight point_lights[];
};

layout(binding = 3) readonly buffer SpotLightsBuffer {
    SpotLight spot_lights[];
};

// Blinn-Phong для направленного света
vec3 calcDirectionalLight(DirectionalLight light, vec3 normal, vec3 view_dir) {
    vec3 light_dir = normalize(-light.direction);
    
    // Ambient
    vec3 ambient = light.ambient * frag_albedo;
    
    // Diffuse
    float diff = max(dot(normal, light_dir), 0.0);
    vec3 diffuse = light.diffuse * diff * frag_albedo;
    
    // Specular (Blinn-Phong)
    vec3 halfway_dir = normalize(light_dir + view_dir);
    float spec = pow(max(dot(normal, halfway_dir), 0.0), frag_shininess);
    vec3 specular = light.specular * spec * frag_specular;
    
    return ambient + diffuse + specular;
}

// Blinn-Phong для точечного света
vec3 calcPointLight(PointLight light, vec3 normal, vec3 frag_pos, vec3 view_dir) {
    vec3 light_dir = normalize(light.position - frag_pos);
    
    // Attenuation (затухание по закону обратных квадратов)
    float distance = length(light.position - frag_pos);
    float attenuation = 1.0 / (light.constant + light.linear * distance + 
                               light.quadratic * (distance * distance));
    
    // Ambient
    vec3 ambient = light.ambient * frag_albedo;
    
    // Diffuse
    float diff = max(dot(normal, light_dir), 0.0);
    vec3 diffuse = light.diffuse * diff * frag_albedo;
    
    // Specular (Blinn-Phong)
    vec3 halfway_dir = normalize(light_dir + view_dir);
    float spec = pow(max(dot(normal, halfway_dir), 0.0), frag_shininess);
    vec3 specular = light.specular * spec * frag_specular;
    
    ambient *= attenuation;
    diffuse *= attenuation;
    specular *= attenuation;
    
    return ambient + diffuse + specular;
}

// Blinn-Phong для прожектора с гладкими краями
vec3 calcSpotLight(SpotLight light, vec3 normal, vec3 frag_pos, vec3 view_dir) {
    vec3 light_dir = normalize(light.position - frag_pos);
    
    // Spotlight intensity (smooth edges - гладкие края)
    float theta = dot(light_dir, normalize(-light.direction));
    float epsilon = light.cutOff - light.outerCutOff;
    float intensity = clamp((theta - light.outerCutOff) / epsilon, 0.0, 1.0);
    
    // Attenuation (затухание по закону обратных квадратов)
    float distance = length(light.position - frag_pos);
    float attenuation = 1.0 / (light.constant + light.linear * distance + 
                               light.quadratic * (distance * distance));
    
    // Ambient
    vec3 ambient = light.ambient * frag_albedo;
    
    // Diffuse
    float diff = max(dot(normal, light_dir), 0.0);
    vec3 diffuse = light.diffuse * diff * frag_albedo;
    
    // Specular (Blinn-Phong)
    vec3 halfway_dir = normalize(light_dir + view_dir);
    float spec = pow(max(dot(normal, halfway_dir), 0.0), frag_shininess);
    vec3 specular = light.specular * spec * frag_specular;
    
    diffuse *= intensity * attenuation;
    specular *= intensity * attenuation;
    ambient *= attenuation;
    
    return ambient + diffuse + specular;
}

void main() {
    // Нормализуем нормаль (может прийти ненормализованной после интерполяции)
    vec3 normal = normalize(frag_normal);
    vec3 view_dir = normalize(scene.view_position - frag_position);
    
    // Directional light
    vec3 result = calcDirectionalLight(scene.directional_light, normal, view_dir);
    
    // Point lights (обрабатываем ПЕРВЫЙ)
    if (point_lights.length() > 0) {
        result += calcPointLight(point_lights[0], normal, frag_position, view_dir);
    }
    
    // Spot lights (обрабатываем ПЕРВЫЙ)
    if (spot_lights.length() > 0) {
        result += calcSpotLight(spot_lights[0], normal, frag_position, view_dir);
    }
    
    out_color = vec4(result, 1.0);
}
