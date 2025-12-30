import h5py

try:
    with h5py.File('/home/zty/Desktop/LSY/catkin_ws_v1/src/VINS-Mono/vins_estimator/src/netvlad_initial/EuRoC/euroc_global_netmap.h5', 'r') as file:
        print("File opened successfully!")
        
        def print_group_items(group, depth=0):
            for key in group:
                item = group[key]
                indent = "  " * depth
                if isinstance(item, h5py.Group):
                    print(f"{indent}Group: {key}")
                    print_group_items(item, depth + 1) 
                elif isinstance(item, h5py.Dataset):
                    print(f"{indent}Dataset: {key}")
                    print(f"    Shape: {item.shape}, Type: {item.dtype}")
                    if item.shape[0] < 10:  
                        print(f"    Data (first 10 values): {item[:10]}")
                    else:
                        print(f"    Data (first 10 values): {item[:10]}")

        print_group_items(file)
except Exception as e:
    print(f"Failed to open file: {e}")
