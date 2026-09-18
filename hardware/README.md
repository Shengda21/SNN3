# Recorded hardware and environment

`initial_probe.txt` preserves the original device and filesystem probe from the rented instance. `gpus.csv` preserves its two reported GPU identities. The software observations and the rental description are distinguished below.

| Attribute | Available evidence | Interpretation |
|---|---|---|
| GPU device name | Original `nvidia-smi`: two devices named `NVIDIA GeForce RTX 4090 D` | Software-reported identity |
| Visible memory | `49140 MiB` for each device | Memory exposed to the instance, not a verified retail board specification |
| Power limit | `425.00 W` for each device | Reported configured limit, not measured energy use |
| Driver | `580.76.05` | Recorded driver version |
| Driver CUDA capability | `nvidia-smi` prints CUDA `13.0` | Driver capability display; experiments used PyTorch CUDA `12.4` |
| GPU UUIDs | Preserved in `gpus.csv` | Original device identifiers used for record linkage |
| Compute mode / MIG | `Default` / `N/A` in the initial probe | Does not establish physical exclusivity or the provider's virtualization implementation |
| Training runtime | Final E0 record and run configurations: Python `3.12.3`, PyTorch `2.6.0+cu124`, CUDA `12.4` | Used by the reported new-run results |
| CPU thread setting | Language run configurations: `cpu_threads: 4` per process | Observed program setting, not a hardware inventory |
| Rental CPU and memory | User-supplied rental specification: `40 vCPU Intel Xeon Platinum 8481C`, `180 GB` RAM | Provider/rental description relayed by the user; not independently captured by `lscpu` or a hardware inventory |
| Rental GPU product | User-supplied rental specification: `vGPU-48GB-425W (48GB) × 2` | Provider/rental product description |

The retained records do not determine the physical board construction, whether memory was modified, the virtualization technology, the number of physical host GPUs, GPU partitioning, or exclusive use by this tenant. No `lscpu`, `nvidia-smi -q`, hypervisor inventory, or provider topology record is present in this archive. The filesystem output shows the container-visible mounts but cannot settle those hardware questions. No retrospective hardware observations have been invented.

The initial probe contains no running GPU processes at that instant. It does not prove that the device was physically exclusive for the entire experiment. During the recorded latency phase the authors stopped their other training and transfer activity, as specified in the protocol and execution records.
