import subprocess
import sys

# def uninstall_package(package_name):
#     try:
#         subprocess.check_call([sys.executable, '-m', 'pip', 'uninstall', '-y', package_name])
#         print(f"Successfully uninstalled {package_name}")
#     except subprocess.CalledProcessError as e:
#         print(f"Error uninstalling {package_name}: {e}")
#     except FileNotFoundError:
#         print("pip command not found. Make sure pip is installed and in your PATH.")

# if __name__ == "__main__":
#     packages_to_uninstall = ['torch', 'torchvision', 'torchaudio']
#     for package in packages_to_uninstall:
#         uninstall_package(package)


# import subprocess
# import sys

# def install_pytorch_cuda():
#     cuda_version = "cu118"
#     install_command = [
#         sys.executable,
#         '-m',
#         'pip',
#         'install',
#         'torch',
#         'torchvision',
#         'torchaudio',
#         '--index-url',
#         f'https://download.pytorch.org/whl/{cuda_version}'
#     ]

#     print(f"Attempting to install PyTorch with CUDA {cuda_version}...")
#     try:
#         subprocess.check_call(install_command)
#         print(f"Successfully installed PyTorch with CUDA {cuda_version}. You can now run your script.")
#     except subprocess.CalledProcessError as e:
#         print(f"Error installing PyTorch: {e}")
#         print("Please check the error message and your internet connection.")
#         print("You might need to manually run the command in your terminal:")
#         print(f"`{' '.join(install_command)}`")
#     except FileNotFoundError:
#         print("pip command not found. Make sure pip is installed and in your PATH.")

# if __name__ == "__main__":
#     install_pytorch_cuda()

import torch

print(f"Number of CUDA devices available: {torch.cuda.device_count()}")
print(f"Current CUDA device: {torch.cuda.current_device()}")
print(f"Name of the CUDA device: {torch.cuda.get_device_name(0)}")
device = torch.device("cuda")
print("device:", device)