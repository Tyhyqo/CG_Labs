#include <cstdint>
#include <climits>
#include <cstring>
#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>

#include <veekay/veekay.hpp>

#include <vulkan/vulkan_core.h>
#include <imgui.h>
#include <lodepng.h>

namespace {

constexpr uint32_t max_models = 1024;

struct Vertex {
	veekay::vec3 position;
	veekay::vec3 normal;
	veekay::vec2 uv;
	// NOTE: You can add more attributes
};

// Направленный свет (один глобальный)
struct DirectionalLight {
    veekay::vec3 direction;
    float _pad0;
    veekay::vec3 ambient;
    float _pad1;
    veekay::vec3 diffuse;
    float _pad2;
    veekay::vec3 specular;
    float _pad3;
};

// Точечный источник света
struct PointLight {
    veekay::vec3 position;
    float _pad0;
    veekay::vec3 ambient;
    float _pad1;
    veekay::vec3 diffuse;
    float _pad2;
    veekay::vec3 specular;
    float constant;   // затухание
    float linear;
    float quadratic;
    float _pad3;
};

// Прожектор
struct SpotLight {
    veekay::vec3 position;
    float _pad0;
    veekay::vec3 direction;
    float _pad1;
    veekay::vec3 ambient;
    float _pad2;
    veekay::vec3 diffuse;
    float _pad3;
    veekay::vec3 specular;
    float constant;
    float linear;
    float quadratic;
    float cutOff;        // внутренний угол (cos)
    float outerCutOff;   // внешний угол (cos) для гладких краев
    float _pad4[3];
};

struct SceneUniforms {
	veekay::mat4 view_projection;
	veekay::vec3 view_position;  // позиция камеры для specular
	float _pad0;
	DirectionalLight directional_light;
};

struct ModelUniforms {
	veekay::mat4 model;
	veekay::vec3 albedo_color;
    float shininess;           // добавлено
    veekay::vec3 specular_color; // добавлено
	float _pad0;
};

struct Material {
    veekay::vec3 albedo;      // диффузный цвет
    veekay::vec3 specular;    // цвет бликов
    float shininess;          // блеск (показатель степени)
};

struct Mesh {
	veekay::graphics::Buffer* vertex_buffer;
	veekay::graphics::Buffer* index_buffer;
	uint32_t indices;
};

struct Transform {
	veekay::vec3 position = {};
	veekay::vec3 scale = {1.0f, 1.0f, 1.0f};
	veekay::vec3 rotation = {};

	// NOTE: Model matrix (translation, rotation and scaling)
	veekay::mat4 matrix() const;
};

struct Model {
	Mesh mesh;
	Transform transform;
	Material material;  // заменили albedo_color на material
};

struct Camera {
	constexpr static float default_fov = 60.0f;
	constexpr static float default_near_plane = 0.01f;
	constexpr static float default_far_plane = 100.0f;

	veekay::vec3 position = {};
	veekay::vec3 rotation = {};

	float fov = default_fov;
	float near_plane = default_near_plane;
	float far_plane = default_far_plane;

	// NOTE: View matrix of camera (inverse of a transform)
	veekay::mat4 view() const;

	// NOTE: View and projection composition
	veekay::mat4 view_projection(float aspect_ratio) const;
};

// NOTE: Scene objects
inline namespace {
	Camera camera{
		.position = {0.0f, -0.5f, -3.0f}
	};

	std::vector<Model> models;
}

// NOTE: Vulkan objects
inline namespace {
    VkShaderModule vertex_shader_module;
    VkShaderModule fragment_shader_module;

    VkDescriptorPool descriptor_pool;
    VkDescriptorSetLayout descriptor_set_layout;
    VkDescriptorSet descriptor_set;

    VkPipelineLayout pipeline_layout;
    VkPipeline pipeline;

    veekay::graphics::Buffer* scene_uniforms_buffer;
    veekay::graphics::Buffer* model_uniforms_buffer;

    // Добавьте эти строки:
    veekay::graphics::Buffer* point_lights_buffer;
    veekay::graphics::Buffer* spot_lights_buffer;
    
    constexpr uint32_t max_point_lights = 16;
    constexpr uint32_t max_spot_lights = 16;
    
    std::vector<PointLight> point_lights;
    std::vector<SpotLight> spot_lights;

    Mesh plane_mesh;
    Mesh cube_mesh;
    Mesh sphere_mesh;

	veekay::graphics::Texture* missing_texture;
	VkSampler missing_texture_sampler;

	veekay::graphics::Texture* texture;
	VkSampler texture_sampler;
}

float toRadians(float degrees) {
	return degrees * float(M_PI) / 180.0f;
}

veekay::mat4 Transform::matrix() const {
	// TODO: Scaling and rotation

	auto t = veekay::mat4::translation(position);

	return t;
}

veekay::mat4 Camera::view() const {
	// TODO: Rotation

	auto t = veekay::mat4::translation(-position);

	return t;
}

veekay::mat4 Camera::view_projection(float aspect_ratio) const {
	auto projection = veekay::mat4::projection(fov, aspect_ratio, near_plane, far_plane);

	return view() * projection;
}

// NOTE: Loads shader byte code from file
// NOTE: Your shaders are compiled via CMake with this code too, look it up
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

void initialize(VkCommandBuffer cmd) {
	VkDevice& device = veekay::app.vk_device;
	VkPhysicalDevice& physical_device = veekay::app.vk_physical_device;

	{ // NOTE: Build graphics pipeline
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

		// NOTE: Vertex shader stage
		stage_infos[0] = VkPipelineShaderStageCreateInfo{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_VERTEX_BIT,
			.module = vertex_shader_module,
			.pName = "main",
		};

		// NOTE: Fragment shader stage
		stage_infos[1] = VkPipelineShaderStageCreateInfo{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = fragment_shader_module,
			.pName = "main",
		};

		// NOTE: How many bytes does a vertex take?
		VkVertexInputBindingDescription buffer_binding{
			.binding = 0,
			.stride = sizeof(Vertex),
			.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
		};

		// NOTE: Declare vertex attributes
		VkVertexInputAttributeDescription attributes[] = {
			{
				.location = 0, // NOTE: First attribute
				.binding = 0, // NOTE: First vertex buffer
				.format = VK_FORMAT_R32G32B32_SFLOAT, // NOTE: 3-component vector of floats
				.offset = offsetof(Vertex, position), // NOTE: Offset of "position" field in a Vertex struct
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

		// NOTE: Describe inputs
		VkPipelineVertexInputStateCreateInfo input_state_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
			.vertexBindingDescriptionCount = 1,
			.pVertexBindingDescriptions = &buffer_binding,
			.vertexAttributeDescriptionCount = sizeof(attributes) / sizeof(attributes[0]),
			.pVertexAttributeDescriptions = attributes,
		};

		// NOTE: Every three vertices make up a triangle,
		//       so our vertex buffer contains a "list of triangles"
		VkPipelineInputAssemblyStateCreateInfo assembly_state_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
			.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
		};

		// NOTE: Declare clockwise triangle order as front-facing
		//       Discard triangles that are facing away
		//       Fill triangles, don't draw lines instaed
		VkPipelineRasterizationStateCreateInfo raster_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
			.polygonMode = VK_POLYGON_MODE_FILL,
			.cullMode = VK_CULL_MODE_BACK_BIT,
			.frontFace = VK_FRONT_FACE_CLOCKWISE,
			.lineWidth = 1.0f,
		};

		// NOTE: Use 1 sample per pixel
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

		// NOTE: Let rasterizer draw on the entire window
		VkPipelineViewportStateCreateInfo viewport_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,

			.viewportCount = 1,
			.pViewports = &viewport,

			.scissorCount = 1,
			.pScissors = &scissor,
		};

		// NOTE: Let rasterizer perform depth-testing and overwrite depth values on condition pass
		VkPipelineDepthStencilStateCreateInfo depth_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
			.depthTestEnable = true,
			.depthWriteEnable = true,
			.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
		};

		// NOTE: Let fragment shader write all the color channels
		VkPipelineColorBlendAttachmentState attachment_info{
			.colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
			                  VK_COLOR_COMPONENT_G_BIT |
			                  VK_COLOR_COMPONENT_B_BIT |
			                  VK_COLOR_COMPONENT_A_BIT,
		};

		// NOTE: Let rasterizer just copy resulting pixels onto a buffer, don't blend
		VkPipelineColorBlendStateCreateInfo blend_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,

			.logicOpEnable = false,
			.logicOp = VK_LOGIC_OP_COPY,

			.attachmentCount = 1,
			.pAttachments = &attachment_info
		};

		{
			VkDescriptorPoolSize pools[] = {
				{
					.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
					.descriptorCount = 8,
				},
				{
					.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
					.descriptorCount = 8,
				},
				{
					.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
					.descriptorCount = 8,
				},
				// Добавьте:
				{
					.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
					.descriptorCount = 8,
				}
			};
			
			VkDescriptorPoolCreateInfo info{
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
				.maxSets = 1,
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

		// NOTE: Descriptor set layout specification
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
				// Добавьте эти два binding:
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

		{
			VkDescriptorSetAllocateInfo info{
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
				.descriptorPool = descriptor_pool,
				.descriptorSetCount = 1,
				.pSetLayouts = &descriptor_set_layout,
			};

			if (vkAllocateDescriptorSets(device, &info, &descriptor_set) != VK_SUCCESS) {
				std::cerr << "Failed to create Vulkan descriptor set\n";
				veekay::app.running = false;
				return;
			}
		}

		// NOTE: Declare external data sources, only push constants this time
		VkPipelineLayoutCreateInfo layout_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
			.setLayoutCount = 1,
			.pSetLayouts = &descriptor_set_layout,
		};

		// NOTE: Create pipeline layout
		if (vkCreatePipelineLayout(device, &layout_info,
		                           nullptr, &pipeline_layout) != VK_SUCCESS) {
			std::cerr << "Failed to create Vulkan pipeline layout\n";
			veekay::app.running = false;
			return;
		}
		
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

		// NOTE: Create graphics pipeline
		if (vkCreateGraphicsPipelines(device, nullptr,
		                              1, &info, nullptr, &pipeline) != VK_SUCCESS) {
			std::cerr << "Failed to create Vulkan pipeline\n";
			veekay::app.running = false;
			return;
		}
	}

	scene_uniforms_buffer = new veekay::graphics::Buffer(
		sizeof(SceneUniforms),
		nullptr,
		VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

	model_uniforms_buffer = new veekay::graphics::Buffer(
		max_models * veekay::graphics::Buffer::structureAlignment(sizeof(ModelUniforms)),
		nullptr,
		VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

    // Добавьте создание буферов для источников света:
    point_lights_buffer = new veekay::graphics::Buffer(
        max_point_lights * sizeof(PointLight),
        nullptr,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    spot_lights_buffer = new veekay::graphics::Buffer(
        max_spot_lights * sizeof(SpotLight),
        nullptr,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    // Инициализация источников света (пример)
    point_lights.clear();
    point_lights.push_back(PointLight{
        .position = {2.0f, -2.0f, 2.0f},
        .ambient = {0.1f, 0.1f, 0.1f},
        .diffuse = {1.0f, 0.8f, 0.6f},
        .specular = {1.0f, 1.0f, 1.0f},
        .constant = 1.0f,
        .linear = 0.09f,
        .quadratic = 0.032f,
    });

    spot_lights.clear();
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

    // NOTE: This texture and sampler is used when texture could not be loaded
	{
		VkSamplerCreateInfo info{
			.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
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
	}

	{
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
			// Добавьте:
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

		VkWriteDescriptorSet write_infos[] = {
			{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = descriptor_set,
				.dstBinding = 0,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
				.pBufferInfo = &buffer_infos[0],
			},
			{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = descriptor_set,
				.dstBinding = 1,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
				.pBufferInfo = &buffer_infos[1],
			},
			// Добавьте:
			{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = descriptor_set,
				.dstBinding = 2,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
				.pBufferInfo = &buffer_infos[2],
			},
			{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = descriptor_set,
				.dstBinding = 3,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
				.pBufferInfo = &buffer_infos[3],
			},
		};

		vkUpdateDescriptorSets(device, sizeof(write_infos) / sizeof(write_infos[0]),
		                       write_infos, 0, nullptr);
	}

	// NOTE: Plane mesh initialization
	{
		// (v0)------(v1)
		//  |  \       |
		//  |   `--,   |
		//  |       \  |
		// (v3)------(v2)
		std::vector<Vertex> vertices = {
			{{-5.0f, 0.0f, 5.0f}, {0.0f, -1.0f, 0.0f}, {0.0f, 0.0f}},
			{{5.0f, 0.0f, 5.0f}, {0.0f, -1.0f, 0.0f}, {1.0f, 0.0f}},
			{{5.0f, 0.0f, -5.0f}, {0.0f, -1.0f, 0.0f}, {1.0f, 1.0f}},
			{{-5.0f, 0.0f, -5.0f}, {0.0f, -1.0f, 0.0f}, {0.0f, 1.0f}},
		};

		std::vector<uint32_t> indices = {
			0, 1, 2, 2, 3, 0
		};

		plane_mesh.vertex_buffer = new veekay::graphics::Buffer(
			vertices.size() * sizeof(Vertex), vertices.data(),
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

		plane_mesh.index_buffer = new veekay::graphics::Buffer(
			indices.size() * sizeof(uint32_t), indices.data(),
			VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

		plane_mesh.indices = uint32_t(indices.size());
	}

	// NOTE: Cube mesh initialization
	{
		std::vector<Vertex> vertices = {
			{{-0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {0.0f, 0.0f}},
			{{+0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {1.0f, 0.0f}},
			{{+0.5f, +0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {1.0f, 1.0f}},
			{{-0.5f, +0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {0.0f, 1.0f}},

			{{+0.5f, -0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},
			{{+0.5f, -0.5f, +0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
			{{+0.5f, +0.5f, +0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 1.0f}},
			{{+0.5f, +0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f}},

			{{+0.5f, -0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f}},
			{{-0.5f, -0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f}},
			{{-0.5f, +0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f}},
			{{+0.5f, +0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}},

			{{-0.5f, -0.5f, +0.5f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},
			{{-0.5f, -0.5f, -0.5f}, {-1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
			{{-0.5f, +0.5f, -0.5f}, {-1.0f, 0.0f, 0.0f}, {1.0f, 1.0f}},
			{{-0.5f, +0.5f, +0.5f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 1.0f}},

			{{-0.5f, -0.5f, +0.5f}, {0.0f, -1.0f, 0.0f}, {0.0f, 0.0f}},
			{{+0.5f, -0.5f, +0.5f}, {0.0f, -1.0f, 0.0f}, {1.0f, 0.0f}},
			{{+0.5f, -0.5f, -0.5f}, {0.0f, -1.0f, 0.0f}, {1.0f, 1.0f}},
			{{-0.5f, -0.5f, -0.5f}, {0.0f, -1.0f, 0.0f}, {0.0f, 1.0f}},

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

	// NOTE: Sphere mesh initialization (latitude-longitude parametrization)
    {
        const float radius = 0.5f;
        const int sectors = 32; // longitude divisions
        const int stacks = 16;  // latitude divisions

        std::vector<Vertex> verts;
        std::vector<uint32_t> inds;

        // Generate vertices
        for (int i = 0; i <= stacks; ++i) {
            float stack_angle = float(M_PI / 2.0f) - float(i) * float(M_PI) / float(stacks);
            float xy = radius * cosf(stack_angle);
            float z = radius * sinf(stack_angle);

            for (int j = 0; j <= sectors; ++j) {
                float sector_angle = float(j) * 2.0f * float(M_PI) / float(sectors);
                float x = xy * cosf(sector_angle);
                float y = xy * sinf(sector_angle);

                veekay::vec3 pos = { x, z, y };
                veekay::vec3 normal = pos; // normalized in shader
                veekay::vec2 uv = { float(j) / float(sectors), float(i) / float(stacks) };

                verts.push_back({ pos, normal, uv });
            }
        }

        // Generate indices
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

	// NOTE: Setup scene - static cube + orbiting sphere
    models.clear();

    models.emplace_back(Model{
        .mesh = cube_mesh,
        .transform = Transform{
            .position = {0.0f, -0.5f, 0.0f},
        },
        .material = Material{
            .albedo = veekay::vec3{1.0f, 0.5f, 0.2f},
            .specular = veekay::vec3{1.0f, 1.0f, 1.0f},
            .shininess = 32.0f,
        }
    });

    models.emplace_back(Model{
        .mesh = sphere_mesh,
        .transform = Transform{
            .position = {2.0f, -0.5f, 0.0f},
            .scale = {0.6f, 0.6f, 0.6f}
        },
        .material = Material{
            .albedo = veekay::vec3{0.2f, 0.6f, 1.0f},
            .specular = veekay::vec3{1.0f, 1.0f, 1.0f},
            .shininess = 32.0f,
        }
    });

    // Добавляем пол для демонстрации освещения
    models.emplace_back(Model{
        .mesh = plane_mesh,
        .transform = Transform{
            .position = {0.0f, 0.0f, 0.0f},
            .scale = {2.0f, 1.0f, 2.0f}
        },
        .material = Material{
            .albedo = veekay::vec3{0.8f, 0.8f, 0.8f},
            .specular = veekay::vec3{0.5f, 0.5f, 0.5f},
            .shininess = 16.0f,
        }
    });
}

// NOTE: Destroy resources here, do not cause leaks in your program!
void shutdown() {
    VkDevice& device = veekay::app.vk_device;

    vkDestroySampler(device, missing_texture_sampler, nullptr);
    delete missing_texture;

    delete cube_mesh.index_buffer;
    delete cube_mesh.vertex_buffer;

    delete plane_mesh.index_buffer;
    delete plane_mesh.vertex_buffer;
    
    delete sphere_mesh.index_buffer;  // добавьте
    delete sphere_mesh.vertex_buffer;  // добавьте

    delete spot_lights_buffer;  // добавьте
    delete point_lights_buffer;  // добавьте
    delete model_uniforms_buffer;
    delete scene_uniforms_buffer;

	vkDestroyDescriptorSetLayout(device, descriptor_set_layout, nullptr);
	vkDestroyDescriptorPool(device, descriptor_pool, nullptr);

	vkDestroyPipeline(device, pipeline, nullptr);
	vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
	vkDestroyShaderModule(device, fragment_shader_module, nullptr);
	vkDestroyShaderModule(device, vertex_shader_module, nullptr);
}

void update(double time) {
    ImGui::Begin("Lighting Controls");
    
    // Управление направленным светом
    if (ImGui::CollapsingHeader("Directional Light", ImGuiTreeNodeFlags_DefaultOpen)) {
        static float dir_direction[3] = {-0.2f, 1.0f, -0.3f};
        static float dir_ambient[3] = {0.2f, 0.2f, 0.2f};
        static float dir_diffuse[3] = {0.5f, 0.5f, 0.5f};
        static float dir_specular[3] = {1.0f, 1.0f, 1.0f};
        
        ImGui::DragFloat3("Direction", dir_direction, 0.01f, -1.0f, 1.0f);
        ImGui::ColorEdit3("Ambient", dir_ambient);
        ImGui::ColorEdit3("Diffuse", dir_diffuse);
        ImGui::ColorEdit3("Specular", dir_specular);
    }
    
    // Управление точечными источниками
    if (ImGui::CollapsingHeader("Point Lights", ImGuiTreeNodeFlags_DefaultOpen)) {
        for (size_t i = 0; i < point_lights.size(); ++i) {
            ImGui::PushID(int(i));
            if (ImGui::TreeNode(("Point Light " + std::to_string(i)).c_str())) {
                ImGui::DragFloat3("Position", &point_lights[i].position.x, 0.1f);
                ImGui::ColorEdit3("Ambient", &point_lights[i].ambient.x);
                ImGui::ColorEdit3("Diffuse", &point_lights[i].diffuse.x);
                ImGui::ColorEdit3("Specular", &point_lights[i].specular.x);
                ImGui::DragFloat("Constant", &point_lights[i].constant, 0.01f, 0.0f, 10.0f);
                ImGui::DragFloat("Linear", &point_lights[i].linear, 0.01f, 0.0f, 1.0f);
                ImGui::DragFloat("Quadratic", &point_lights[i].quadratic, 0.001f, 0.0f, 1.0f);
                ImGui::TreePop();
            }
            ImGui::PopID();
        }
        if (ImGui::Button("Add Point Light") && point_lights.size() < max_point_lights) {
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
        ImGui::SameLine();
        if (ImGui::Button("Remove Point Light") && !point_lights.empty()) {
            point_lights.pop_back();
        }
    }
    
    // Управление прожекторами
    if (ImGui::CollapsingHeader("Spot Lights", ImGuiTreeNodeFlags_DefaultOpen)) {
        for (size_t i = 0; i < spot_lights.size(); ++i) {
            ImGui::PushID(int(100 + i));
            if (ImGui::TreeNode(("Spot Light " + std::to_string(i)).c_str())) {
                ImGui::DragFloat3("Position", &spot_lights[i].position.x, 0.1f);
                ImGui::DragFloat3("Direction", &spot_lights[i].direction.x, 0.01f, -1.0f, 1.0f);
                ImGui::ColorEdit3("Ambient", &spot_lights[i].ambient.x);
                ImGui::ColorEdit3("Diffuse", &spot_lights[i].diffuse.x);
                ImGui::ColorEdit3("Specular", &spot_lights[i].specular.x);
                ImGui::DragFloat("Constant", &spot_lights[i].constant, 0.01f, 0.0f, 10.0f);
                ImGui::DragFloat("Linear", &spot_lights[i].linear, 0.01f, 0.0f, 1.0f);
                ImGui::DragFloat("Quadratic", &spot_lights[i].quadratic, 0.001f, 0.0f, 1.0f);
                
                float inner_angle = acosf(spot_lights[i].cutOff) * 180.0f / M_PI;
                float outer_angle = acosf(spot_lights[i].outerCutOff) * 180.0f / M_PI;
                
                if (ImGui::DragFloat("Inner Angle", &inner_angle, 0.5f, 0.0f, 90.0f)) {
                    spot_lights[i].cutOff = cosf(toRadians(inner_angle));
                }
                if (ImGui::DragFloat("Outer Angle", &outer_angle, 0.5f, 0.0f, 90.0f)) {
                    spot_lights[i].outerCutOff = cosf(toRadians(outer_angle));
                }
                
                ImGui::TreePop();
            }
            ImGui::PopID();
        }
        if (ImGui::Button("Add Spot Light") && spot_lights.size() < max_spot_lights) {
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
        ImGui::SameLine();
        if (ImGui::Button("Remove Spot Light") && !spot_lights.empty()) {
            spot_lights.pop_back();
        }
    }
    
    ImGui::End();
    
    // Остальной код update() без изменений...
    ImGui::Begin("Camera Controls");
    ImGui::Text("Movement: WASD, Q/Z (Up/Down)");
    ImGui::Text("Position: %.2f, %.2f, %.2f", camera.position.x, camera.position.y, camera.position.z);
    ImGui::End();

    // Управление камерой
    using namespace veekay::input;

    veekay::vec3 right = {1.0f, 0.0f, 0.0f};
    veekay::vec3 up = {0.0f, -1.0f, 0.0f};
    veekay::vec3 front = {0.0f, 0.0f, 1.0f};

    if (keyboard::isKeyDown(keyboard::Key::w))
        camera.position += front * 0.1f;

    if (keyboard::isKeyDown(keyboard::Key::s))
        camera.position -= front * 0.1f;

    if (keyboard::isKeyDown(keyboard::Key::d))
        camera.position += right * 0.1f;

    if (keyboard::isKeyDown(keyboard::Key::a))
        camera.position -= right * 0.1f;

    if (keyboard::isKeyDown(keyboard::Key::q))
        camera.position += up * 0.1f;

    if (keyboard::isKeyDown(keyboard::Key::z))
        camera.position -= up * 0.1f;

    // Анимация сферы
    const float orbit_radius = 2.0f;
    const float orbit_speed = 1.0f;
    if (models.size() >= 2) {
        Model& sphere = models[1];
        float angle = float(time * orbit_speed);
        float x = orbit_radius * cosf(angle);
        float z = orbit_radius * sinf(angle);
        sphere.transform.position = { x, -0.5f, z };
    }

    float aspect_ratio = float(veekay::app.window_width) / float(veekay::app.window_height);
    
    // Используем значения из UI для направленного света
    static float dir_direction[3] = {-0.2f, 1.0f, -0.3f};
    static float dir_ambient[3] = {0.2f, 0.2f, 0.2f};
    static float dir_diffuse[3] = {0.5f, 0.5f, 0.5f};
    static float dir_specular[3] = {1.0f, 1.0f, 1.0f};
    
    SceneUniforms scene_uniforms{
        .view_projection = camera.view_projection(aspect_ratio),
        .view_position = camera.position,
        .directional_light = DirectionalLight{
            .direction = {dir_direction[0], dir_direction[1], dir_direction[2]},
            .ambient = {dir_ambient[0], dir_ambient[1], dir_ambient[2]},
            .diffuse = {dir_diffuse[0], dir_diffuse[1], dir_diffuse[2]},
            .specular = {dir_specular[0], dir_specular[1], dir_specular[2]},
        }
    };

    // Обновляем буферы источников света
    if (!point_lights.empty()) {
        memcpy(point_lights_buffer->mapped_region, point_lights.data(), 
               point_lights.size() * sizeof(PointLight));
    }
    
    if (!spot_lights.empty()) {
        memcpy(spot_lights_buffer->mapped_region, spot_lights.data(), 
               spot_lights.size() * sizeof(SpotLight));
    }

    std::vector<ModelUniforms> model_uniforms(models.size());
    for (size_t i = 0, n = models.size(); i < n; ++i) {
        const Model& model = models[i];
        ModelUniforms& uniforms = model_uniforms[i];

        uniforms.model = model.transform.matrix();
        uniforms.albedo_color = model.material.albedo;
        uniforms.shininess = model.material.shininess;
        uniforms.specular_color = model.material.specular;
    }

    *(SceneUniforms*)scene_uniforms_buffer->mapped_region = scene_uniforms;

    const size_t alignment =
        veekay::graphics::Buffer::structureAlignment(sizeof(ModelUniforms));

    for (size_t i = 0, n = model_uniforms.size(); i < n; ++i) {
        const ModelUniforms& uniforms = model_uniforms[i];

        char* const pointer = static_cast<char*>(model_uniforms_buffer->mapped_region) + i * alignment;
        *reinterpret_cast<ModelUniforms*>(pointer) = uniforms;
    }
}

void render(VkCommandBuffer cmd, VkFramebuffer framebuffer) {
	vkResetCommandBuffer(cmd, 0);

	{ // NOTE: Start recording rendering commands
		VkCommandBufferBeginInfo info{
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
			.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
		};

		vkBeginCommandBuffer(cmd, &info);
	}

	{ // NOTE: Use current swapchain framebuffer and clear it
		VkClearValue clear_color{.color = {{0.1f, 0.1f, 0.1f, 1.0f}}};
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
	}

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
	VkDeviceSize zero_offset = 0;

	VkBuffer current_vertex_buffer = VK_NULL_HANDLE;
	VkBuffer current_index_buffer = VK_NULL_HANDLE;

	const size_t model_uniorms_alignment =
		veekay::graphics::Buffer::structureAlignment(sizeof(ModelUniforms));

	for (size_t i = 0, n = models.size(); i < n; ++i) {
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

		uint32_t offset = i * model_uniorms_alignment;
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
		                    0, 1, &descriptor_set, 1, &offset);

		vkCmdDrawIndexed(cmd, mesh.indices, 1, 0, 0, 0);
	}

	vkCmdEndRenderPass(cmd);
	vkEndCommandBuffer(cmd);
}

} // namespace

int main() {
	return veekay::run({
		.init = initialize,
		.shutdown = shutdown,
		.update = update,
		.render = render,
	});
}
