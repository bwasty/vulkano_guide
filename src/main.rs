#![allow(unused_variables)]

use std::sync::Arc;

#[macro_use]
extern crate vulkano;

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

fn main() {
    let (device, queue) = initialize();

    buffer_creation(device.clone());

    // http://vulkano.rs/guide/example-operation
    let source_content = 0 .. 64;
    let source = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(),
                                                source_content).expect("failed to create buffer");

    let dest_content = (0 .. 64).map(|_| 0);
    let dest = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(),
                                              dest_content).expect("failed to create buffer");

    let command_buffer = AutoCommandBufferBuilder::new(device.clone(), queue.family()).unwrap()
        .copy_buffer(source.clone(), dest.clone()).unwrap()
        .build().unwrap();

    use vulkano::command_buffer::CommandBuffer;
    let finished = command_buffer.execute(queue.clone()).unwrap();

    use vulkano::sync::GpuFuture;

    finished.then_signal_fence_and_flush().unwrap()
        .wait(None).unwrap();

    let src_content = source.read().unwrap();
    let dest_content = dest.read().unwrap();
    assert_eq!(&*src_content, &*dest_content);
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
