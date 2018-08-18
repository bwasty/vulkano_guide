#![allow(unused_variables)]

use std::sync::Arc;
use std::time::Instant;

#[allow(unused_imports)]
#[macro_use]
extern crate vulkano;

#[macro_use]
extern crate vulkano_shader_derive;

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

use vulkano::sync::GpuFuture;

mod utils;
use utils::print_elapsed;

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

fn main() {
    let (device, queue) = initialize();

    buffer_creation(device.clone());

    example_operation_copy(device.clone(), queue.clone());

    // http://vulkano.rs/guide/compute-intro
    let timer = Instant::now();

    let data_iter = 0 .. 65536;
    let data_buffer = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(),
                                                     data_iter).expect("failed to create buffer");

    let shader = cs::Shader::load(device.clone())
        .expect("failed to create shader module");


    use vulkano::pipeline::ComputePipeline;

    let compute_pipeline = Arc::new(ComputePipeline::new(device.clone(), &shader.main_entry_point(), &())
        .expect("failed to create compute pipeline"));


    use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;

    let set = Arc::new(PersistentDescriptorSet::start(compute_pipeline.clone(), 0)
        .add_buffer(data_buffer.clone()).unwrap()
        .build().unwrap()
    );

    let command_buffer = AutoCommandBufferBuilder::new(device.clone(), queue.family()).unwrap()
        .dispatch([1024, 1, 1], compute_pipeline.clone(), set.clone(), ()).unwrap()
        .build().unwrap();

    let finished = command_buffer.execute(queue.clone()).unwrap();
    print_elapsed("set up compute and execute", timer);

    let timer = Instant::now();
    finished.then_signal_fence_and_flush().unwrap()
        .wait(None).unwrap();
    print_elapsed("wait for compute execution", timer);
    let timer = Instant::now();

    let content = data_buffer.read().unwrap();
    for (n, val) in content.iter().enumerate() {
        assert_eq!(*val, n as u32 * 12);
    }
    print_elapsed("read back result", timer);

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
