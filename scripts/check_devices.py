import torch

def check_cuda_devices():
    if torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
        print(f"CUDA is available. {num_devices} device(s) detected:")
        for i in range(num_devices):
            device_name = torch.cuda.get_device_name(i)
            print(f"  - Device {i}: {device_name}")
    else:
        print("CUDA is not available.")

if __name__ == "__main__":
    check_cuda_devices()
