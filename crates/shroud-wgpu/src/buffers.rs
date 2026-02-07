//! GPU buffer allocation, data staging, and asynchronous readback.

use wgpu::util::DeviceExt;
use wgpu::{Buffer, BufferDescriptor, BufferUsages, Device, Queue};

pub struct GpuBufferHelper;

impl GpuBufferHelper {
    /// Creates a GPU storage buffer pre-initialized with host data.
    pub fn create_storage_buffer_init<T: bytemuck::Pod>(
        device: &Device,
        label: &str,
        data: &[T],
        writable: bool,
    ) -> Buffer {
        let mut usages = BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST;
        if writable {
            usages |= BufferUsages::STORAGE;
        }

        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(data),
            usage: usages,
        })
    }

    /// Creates an empty storage buffer with a given byte size.
    pub fn create_storage_buffer(device: &Device, label: &str, size_bytes: u64) -> Buffer {
        device.create_buffer(&BufferDescriptor {
            label: Some(label),
            size: size_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Reads back data from a GPU buffer to host memory asynchronously.
    pub async fn readback_buffer<T: bytemuck::Pod>(
        device: &Device,
        queue: &Queue,
        source: &Buffer,
        size_bytes: u64,
    ) -> Result<Vec<T>, String> {
        let staging_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("staging_readback"),
            size: size_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("readback_encoder"),
        });

        encoder.copy_buffer_to_buffer(source, 0, &staging_buffer, 0, size_bytes);
        queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |res| {
            sender.send(res).expect("Failed to send map result");
        });

        device.poll(wgpu::Maintain::Wait);

        match receiver.recv() {
            Ok(Ok(())) => {
                let mapped_data = buffer_slice.get_mapped_range();
                let result = bytemuck::cast_slice(&mapped_data).to_vec();
                drop(mapped_data);
                staging_buffer.unmap();
                Ok(result)
            }
            Ok(Err(e)) => Err(format!("Buffer mapping failed: {e:?}")),
            Err(e) => Err(format!("Channel receive error: {e}")),
        }
    }
}
