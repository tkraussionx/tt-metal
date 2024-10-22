import ttnn

type_size_in_bytes = {
    ttnn.bfloat4_b: 0.5,
    ttnn.bfloat8_b: 1,
    ttnn.bfloat16: 2,
    ttnn.uint32: 4,
    ttnn.int32: 4,
}


class DramUsage:
    dram_usage_backup_filename = "dram_usage_backup.txt"
    _instance = None
    total_usage_per_device = 0

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DramUsage, cls).__new__(cls)
            cls._instance.dram_usages = []
            cls._instance.dram_usages_per_device = []
            cls._instance.filename = cls.dram_usage_backup_filename
            with open(cls._instance.filename, "w") as f:
                f.write("DRAM Usage(MB)\n")
        return cls._instance

    def append(self, shape, dram_usage, dram_usage_per_device):
        # Append dram_usage to list and save to file
        self.dram_usages.append(dram_usage)
        self.dram_usages_per_device.append(dram_usage_per_device)
        self.total_usage_per_device += dram_usage_per_device
        with open(self.filename, "a") as f:
            f.write(f"{shape}\t{dram_usage_per_device/2**20}\t{self.total_usage_per_device/2**20}\n")
        with open(self.filename[:4] + "_per_device.txt", "a") as f:
            f.write(f"{dram_usage_per_device}\n")

    def save(self, file_path):
        # create per_device file_path
        per_device_file_path = file_path[:4] + "_per_device.txt"
        assert file_path != DramUsage.dram_usage_backup_filename[:4] + "_per_device.txt", "Cannot save to backup file"
        with open(file_path, "w") as f:
            for dram_usage in self.dram_usages:
                f.write(f"{dram_usage}\n")
        with open(per_device_file_path, "w") as f:
            for dram_usage_per_device in self.dram_usages_per_device:
                f.write(f"{dram_usage_per_device}\n")


# Create singleton object for DramUsage
global_dram_usage = DramUsage()


def calculate_dram_usage(shape, dtype, mesh_mapper, mesh_device):
    if mesh_device is None:
        num_devices = 8
    else:
        num_devices = mesh_device.get_num_devices()
    total_dram_usage = 0
    total_dram_usage_per_device = 0
    input_size = 1
    for dim in shape:
        input_size *= dim

    size_in_bytes = input_size * type_size_in_bytes[dtype]
    total_dram_usage = size_in_bytes
    total_dram_usage_per_device = size_in_bytes

    if isinstance(mesh_mapper, ttnn.ReplicateTensorToMesh):
        total_dram_usage = size_in_bytes * num_devices
    elif isinstance(mesh_mapper, ttnn.ShardTensorToMesh):
        total_dram_usage_per_device = size_in_bytes / num_devices
    else:
        raise ValueError(f"Unsupported mesh_mapper {mesh_mapper}")

    print_dram_usage(shape, dtype, mesh_mapper, total_dram_usage, total_dram_usage_per_device)

    return total_dram_usage, total_dram_usage_per_device


def print_dram_usage(shape, dtype, mesh_mapper, dram_usage, dram_usage_per_device):
    print(
        f"Shape: {shape}, dtype: {dtype}, mesh_mapper: {mesh_mapper}\n"
        + f"DRAM Usage: {dram_usage} bytes, {dram_usage_per_device} bytes per device"
    )


def as_tensor(*args, **kwargs):
    # calculate dram_usage
    dram_usage, dram_usage_per_device = calculate_dram_usage(
        args[0].shape,
        kwargs["dtype"],
        kwargs["mesh_mapper"],
        kwargs["device"],
    )
    global_dram_usage.append(args[0].shape, dram_usage, dram_usage_per_device)
    return ttnn.as_tensor(*args, **kwargs)


def from_torch(*args, **kwargs):
    if "device" not in kwargs:
        kwargs["device"] = None
    # calculate dram_usage
    dram_usage, dram_usage_per_device = calculate_dram_usage(
        args[0].shape,
        kwargs["dtype"],
        kwargs["mesh_mapper"],
        kwargs["device"],
    )
    global_dram_usage.append(args[0].shape, dram_usage, dram_usage_per_device)
    return ttnn.from_torch(*args, **kwargs)
