#![allow(unused_variables)]

use std::sync::Arc;
use std::time::Instant;

#[allow(unused_imports)]
#[macro_use]
extern crate vulkano;
#[macro_use]
extern crate vulkano_shader_derive;
extern crate image;

use vulkano::instance::Instance;
use vulkano::instance::InstanceExtensions;

use vulkano::instance::PhysicalDevice;

use vulkano::device::Device;
use vulkano::device::DeviceExtensions;
use vulkano::device::Queue;
use vulkano::instance::Features;

use vulkano::buffer::BufferUsage;
use vulkano::buffer::CpuAccessibleBuffer;

use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::command_buffer::CommandBuffer;

use vulkano::pipeline::ComputePipeline;
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;

use vulkano::sync::GpuFuture;

use vulkano::format::Format;
use vulkano::image::Dimensions;
use vulkano::image::StorageImage;

use image::{ImageBuffer, Rgba};

mod utils;
use utils::print_elapsed_and_reset;

#[allow(dead_code)]
struct MyStruct {
    a: u32,
    b: bool,
}

// http://vulkano.rs/guide/initialization
// http://vulkano.rs/guide/device-creation
fn initialize() -> (Arc<Device>, Arc<Queue>) {
    let instance = Instance::new(None, &InstanceExtensions::none(), None)
        .expect("failed to create instance");

    let physical = PhysicalDevice::enumerate(&instance).next().expect("no device available");

    for family in physical.queue_families() {
        println!("Found a queue family with {:?} queue(s)", family.queues_count());
    }

    let queue_family = physical.queue_families()
        .find(|&q| q.supports_graphics())
        .expect("couldn't find a graphical queue family");

    let (device, mut queues) = {
        Device::new(physical, &Features::none(), &DeviceExtensions::none(),
                    [(queue_family, 0.5)].iter().cloned()).expect("failed to create device")
    };

    let queue = queues.next().unwrap();

    (device, queue)
}

#[allow(dead_code)]
// http://vulkano.rs/guide/buffer-creation
fn buffer_creation(device: Arc<Device>) {
    let data = 12;
    let _buffer = CpuAccessibleBuffer::from_data(device.clone(), BufferUsage::all(), data)
        .expect("failed to create buffer");

    let iter = (0 .. 128).map(|_| 5u8);
    let buffer = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(),
                                                iter).unwrap();

    let mut content = buffer.write().unwrap();
    // this time `content` derefs to `[u8]`
    content[12] = 83;
    content[7] = 3;

    let data = MyStruct { a: 5, b: true };
    let buffer = CpuAccessibleBuffer::from_data(device.clone(), BufferUsage::all(),
                                                data).unwrap();

    let mut content = buffer.write().unwrap();
    // `content` implements `DerefMut` whose target is of type `MyStruct` (the content of the buffer)
    content.a *= 2;
    content.b = false;
}

#[allow(dead_code)]
// http://vulkano.rs/guide/example-operation
fn example_operation_copy(device: Arc<Device>, queue: Arc<Queue>) {
    let source_content = 0 .. 64;
    let source = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(),
                                                source_content).expect("failed to create buffer");

    let dest_content = (0 .. 64).map(|_| 0);
    let dest = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(),
                                              dest_content).expect("failed to create buffer");

    let command_buffer = AutoCommandBufferBuilder::new(device.clone(), queue.family()).unwrap()
        .copy_buffer(source.clone(), dest.clone()).unwrap()
        .build().unwrap();

    let finished = command_buffer.execute(queue.clone()).unwrap();

    finished.then_signal_fence_and_flush().unwrap()
        .wait(None).unwrap();

    let src_content = source.read().unwrap();
    let dest_content = dest.read().unwrap();
    assert_eq!(&*src_content, &*dest_content);
}

#[allow(dead_code)]
// http://vulkano.rs/guide/compute-intro + following
fn compute_operations(device: Arc<Device>, queue: Arc<Queue>) {
    let timer = &mut Instant::now();

    let data_iter = 0 .. 65536;
    let data_buffer = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(),
                                                     data_iter).expect("failed to create buffer");

    let shader = cs::Shader::load(device.clone())
        .expect("failed to create shader module");

    let compute_pipeline = Arc::new(ComputePipeline::new(device.clone(), &shader.main_entry_point(), &())
        .expect("failed to create compute pipeline"));

    let set = Arc::new(PersistentDescriptorSet::start(compute_pipeline.clone(), 0)
        .add_buffer(data_buffer.clone()).unwrap()
        .build().unwrap()
    );

    let command_buffer = AutoCommandBufferBuilder::new(device.clone(), queue.family()).unwrap()
        .dispatch([1024, 1, 1], compute_pipeline.clone(), set.clone(), ()).unwrap()
        .build().unwrap();

    let finished = command_buffer.execute(queue.clone()).unwrap();
    print_elapsed_and_reset("set up compute and execute", timer);

    finished.then_signal_fence_and_flush().unwrap()
        .wait(None).unwrap();
    print_elapsed_and_reset("wait for compute execution", timer);

    let content = data_buffer.read().unwrap();
    for (n, val) in content.iter().enumerate() {
        assert_eq!(*val, n as u32 * 12);
    }
    print_elapsed_and_reset("read back result", timer);
    println!("Everything succeeded!");
}

#[allow(dead_code)]
mod cs {
    #[derive(VulkanoShader)]
    #[ty = "compute"]
    #[src = "
#version 450

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer Data {
    uint data[];
} buf;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    buf.data[idx] *= 12;
}"
    ]
    struct Dummy;
}

#[allow(dead_code)]
// http://vulkano.rs/guide/image-creation + following
fn image_creation(device: Arc<Device>, queue: Arc<Queue>) {
    let image = StorageImage::new(device.clone(), Dimensions::Dim2d { width: 1024, height: 1024 },
                                  Format::R8G8B8A8Unorm, Some(queue.family())).unwrap();

    use vulkano::format::ClearValue;

    let buf = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(),
                                         (0 .. 1024 * 1024 * 4).map(|_| 0u8))
                                         .expect("failed to create buffer");

    let command_buffer = AutoCommandBufferBuilder::new(device.clone(), queue.family()).unwrap()
        .clear_color_image(image.clone(), ClearValue::Float([0.0, 0.0, 1.0, 1.0])).unwrap()
        .copy_image_to_buffer(image.clone(), buf.clone()).unwrap()
        .build().unwrap();

    let finished = command_buffer.execute(queue.clone()).unwrap();
    finished.then_signal_fence_and_flush().unwrap()
        .wait(None).unwrap();

    let buffer_content = buf.read().unwrap();
    let image = ImageBuffer::<Rgba<u8>, _>::from_raw(1024, 1024, &buffer_content[..]).unwrap();
    image.save("image.png").unwrap();
}

#[allow(dead_code)]
mod mandelbrot {
    #[derive(VulkanoShader)]
    #[ty = "compute"]
    #[src = "
#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0, rgba8) uniform writeonly image2D img;

void main() {
    vec2 norm_coordinates = (gl_GlobalInvocationID.xy + vec2(0.5)) / vec2(imageSize(img));
    vec2 c = (norm_coordinates - vec2(0.5)) * 2.0 - vec2(1.0, 0.0);

    vec2 z = vec2(0.0, 0.0);
    float i;
    for (i = 0.0; i < 1.0; i += 0.005) {
        z = vec2(
            z.x * z.x - z.y * z.y + c.x,
            z.y * z.x + z.x * z.y + c.y
        );

        if (length(z) > 4.0) {
            break;
        }
    }

    vec4 to_write = vec4(vec3(i), 1.0);
    imageStore(img, ivec2(gl_GlobalInvocationID.xy), to_write);
}"
    ]
    struct Dummy;
}
#[allow(dead_code)]
// http://vulkano.rs/guide/mandelbrot
fn mandelbrot(device: Arc<Device>, queue: Arc<Queue>) {
    let image = StorageImage::new(device.clone(), Dimensions::Dim2d { width: 1024, height: 1024 },
                              Format::R8G8B8A8Unorm, Some(queue.family())).unwrap();

    let shader = mandelbrot::Shader::load(device.clone())
        .expect("failed to create shader module");

    let compute_pipeline = Arc::new(ComputePipeline::new(device.clone(), &shader.main_entry_point(), &())
        .expect("failed to create compute pipeline"));

    let set = Arc::new(PersistentDescriptorSet::start(compute_pipeline.clone(), 0)
        .add_image(image.clone()).unwrap()
        .build().unwrap()
    );

    let buf = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(),
                                             (0 .. 1024 * 1024 * 4).map(|_| 0u8))
                                             .expect("failed to create buffer");

    let command_buffer = AutoCommandBufferBuilder::new(device.clone(), queue.family()).unwrap()
        .dispatch([1024 / 8, 1024 / 8, 1], compute_pipeline.clone(), set.clone(), ()).unwrap()
        .copy_image_to_buffer(image.clone(), buf.clone()).unwrap()
        .build().unwrap();

    let finished = command_buffer.execute(queue.clone()).unwrap();
    finished.then_signal_fence_and_flush().unwrap()
        .wait(None).unwrap();

    let buffer_content = buf.read().unwrap();
    let image = ImageBuffer::<Rgba<u8>, _>::from_raw(1024, 1024, &buffer_content[..]).unwrap();
    image.save("image.png").unwrap();
}

#[derive(Copy, Clone)]
struct Vertex {
    position: [f32; 2],
}

impl_vertex!(Vertex, position);

#[allow(dead_code)]
mod vs {
    #[derive(VulkanoShader)]
    #[ty = "vertex"]
    #[src = "
#version 450

layout(location = 0) in vec2 position;

void main() {
    gl_Position = vec4(position, 0.0, 1.0);
}
"]
    struct Dummy;
}

#[allow(dead_code)]
mod fs {
    #[derive(VulkanoShader)]
    #[ty = "fragment"]
    #[src = "
#version 450

layout(location = 0) out vec4 f_color;

void main() {
    f_color = vec4(1.0, 0.0, 0.0, 1.0);
}
"]
    struct Dummy;
}

fn main() {
    let (device, queue) = initialize();

    // buffer_creation(device.clone());

    // example_operation_copy(device.clone(), queue.clone());

    // compute_operations(device.clone(), queue.clone());

    // image_creation(device.clone(), queue.clone());

    // mandelbrot(device.clone(), queue.clone());


    // http://vulkano.rs/guide/vertex-input
    let vertex1 = Vertex { position: [-0.5, -0.5] };
    let vertex2 = Vertex { position: [ 0.0,  0.5] };
    let vertex3 = Vertex { position: [ 0.5, -0.25] };

    let vertex_buffer = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(),
        vec![vertex1, vertex2, vertex3].into_iter()).unwrap();

    let render_pass = Arc::new(single_pass_renderpass!(device.clone(),
        attachments: {
            color: {
                load: Clear,
                store: Store,
                format: Format::R8G8B8A8Unorm,
                samples: 1,
            }
        },
        pass: {
            color: [color],
            depth_stencil: {}
        }
    ).unwrap());

    let image = StorageImage::new(device.clone(), Dimensions::Dim2d { width: 1024, height: 1024 },
        Format::R8G8B8A8Unorm, Some(queue.family())).unwrap();

    use vulkano::framebuffer::Framebuffer;

    let framebuffer = Arc::new(Framebuffer::start(render_pass.clone())
        .add(image.clone()).unwrap()
        .build().unwrap());

    let vs = vs::Shader::load(device.clone()).expect("failed to create shader module");
    let fs = fs::Shader::load(device.clone()).expect("failed to create shader module");

    use vulkano::pipeline::GraphicsPipeline;
    use vulkano::framebuffer::Subpass;

    let pipeline = Arc::new(GraphicsPipeline::start()
        // Defines what kind of vertex input is expected.
        .vertex_input_single_buffer::<Vertex>()
        // The vertex shader.
        .vertex_shader(vs.main_entry_point(), ())
        // Defines the viewport (explanations below).
        .viewports_dynamic_scissors_irrelevant(1)
        // The fragment shader.
        .fragment_shader(fs.main_entry_point(), ())
        // This graphics pipeline object concerns the first pass of the render pass.
        .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
        // Now that everything is specified, we call `build`.
        .build(device.clone())
        .unwrap());

    use vulkano::command_buffer::DynamicState;
    use vulkano::pipeline::viewport::Viewport;

    let dynamic_state = DynamicState {
        viewports: Some(vec![Viewport {
            origin: [0.0, 0.0],
            dimensions: [1024.0, 1024.0],
            depth_range: 0.0 .. 1.0,
        }]),
        .. DynamicState::none()
    };

    let buf = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(),
                                        (0 .. 1024 * 1024 * 4).map(|_| 0u8))
                                        .expect("failed to create buffer");

    let command_buffer = AutoCommandBufferBuilder::primary_one_time_submit(device.clone(), queue.family()).unwrap()
        .begin_render_pass(framebuffer.clone(), false, vec![[0.0, 0.0, 1.0, 1.0].into()])
        .unwrap()

        .draw(pipeline.clone(), &dynamic_state, vertex_buffer.clone(), (), ())
        .unwrap()

        .end_render_pass()
        .unwrap()
        .copy_image_to_buffer(image.clone(), buf.clone())
        .unwrap()

        .build()
        .unwrap();

    let finished = command_buffer.execute(queue.clone()).unwrap();
    finished.then_signal_fence_and_flush().unwrap()
        .wait(None).unwrap();

    let buffer_content = buf.read().unwrap();
    let image = ImageBuffer::<Rgba<u8>, _>::from_raw(1024, 1024, &buffer_content[..]).unwrap();
    image.save("triangle.png").unwrap();
}
