#version 450

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_uv;

layout(binding = 0) uniform SceneUniforms {
    mat4 view_projection;
    vec3 view_position;
    // DirectionalLight данные
} scene;

layout(binding = 1) uniform ModelUniforms {
    mat4 model;
    vec3 albedo_color;
    float shininess;
    vec3 specular_color;
} model;

layout(location = 0) out vec3 frag_position;
layout(location = 1) out vec3 frag_normal;
layout(location = 2) out vec2 frag_uv;
layout(location = 3) out vec3 frag_albedo;
layout(location = 4) out vec3 frag_specular;
layout(location = 5) out float frag_shininess;

void main() {
    vec4 world_position = model.model * vec4(in_position, 1.0);
    gl_Position = scene.view_projection * world_position;
    
    frag_position = world_position.xyz;
    frag_normal = mat3(model.model) * in_normal;
    frag_uv = in_uv;
    frag_albedo = model.albedo_color;
    frag_specular = model.specular_color;
    frag_shininess = model.shininess;
}
