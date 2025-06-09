# export model
from utils import add_pypath
import os
import argparse
import lightning
import torch
from internal.utils.gaussian_utils import GaussianPlyUtils
from internal.utils.gaussian_model_loader import GaussianModelLoader
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("input")
parser.add_argument("--output", "-o", required=False, default=None)
parser.add_argument("--colored", "-c", action="store_true", default=False)
args = parser.parse_args()



# search input file
print("Searching checkpoint file...")

gs_path = args.input
# check if input is a directory
if os.path.isdir(args.input) is True:
    # check if seganygs exists
    segany_file = os.path.join(args.input, "seganygs")

    if os.path.exists(segany_file):
        #read the path in the file
        with open(segany_file, "r") as f:
            lines = f.readlines()
            gs_path = lines[0].strip()
    else:
        pass

load_file = GaussianModelLoader.search_load_file(gs_path)
assert load_file.endswith(".ckpt"), f"Not a valid ckpt file can be found in '{args.input}'"

# auto select output path if not provided
if args.output is None:
    args.output = load_file[:load_file.rfind(".")] + ".ply"
    # if provided input path is a directory, write output file to `PROVIDED_PATH/point_cloud/iteration_.../point_cloud.ply`
    if os.path.isdir(args.input) is True:
        try:
            iteration = load_file[load_file.rfind("=") + 1:load_file.rfind(".")]
            if len(iteration) > 0:
                args.output = os.path.join(args.input, "point_cloud", f"iteration_{iteration}", "point_cloud.ply")
        except:
            pass
else:
    # add ply if not provided
    if args.output.endswith(".ply") is False:
        args.output += ".ply"

assert os.path.exists(args.output) is False, f"Output file already exists, please remove it first: '{args.output}'"

print(f"Loading checkpoint '{load_file}'...")
ckpt = torch.load(load_file)
print("Converting...")
model = GaussianPlyUtils.load_from_state_dict(ckpt["state_dict"]).to_ply_format()

model.save_to_ply(args.output, args.colored)
print(f"Saved to '{args.output}'")



# check for segments directory
segments_dir = os.path.join(args.input, "segments")



if os.path.exists(segments_dir):
    seg_assignments = np.full(len(model.xyz), -1, dtype=np.int32)
    # check if the segments directory is empty
    seg_id = 0
    seg_name_to_id = {}
    id_to_name = []
    sensors = {}

    # check all the saved segments
    for f in os.listdir(segments_dir):
        if not f.endswith(".npz"):
            continue
        segment = np.load(os.path.join(segments_dir, f), allow_pickle=True)
        name = f[:f.rfind(".")]
        print(f"Loading segment '{name}'...")
        # check keys 
        transform = segment["transform"]
        mask = segment["mask"]

        print(f"Segment '{name}' has {mask.sum()} points, with center at {np.mean(model.xyz[mask], axis=0)}")
        coordinates = segment["coordinates"]
        seg_name_to_id[name] = seg_id
        id_to_name.append(name)

        if len(transform.shape) == 2:
            sensors[name] = {
                "transform": transform,
                "coordinates": coordinates,
            }
            print("with estimated center at", transform[:3, 3])
        

        
        seg_assignments[mask] = seg_id
        seg_id += 1

    # sort transforms and coordinates by name 
    transforms = None
    coordinates = None

    if sensors:

        # sort the keys by name
        sorted_keys = sorted(sensors.keys())
        print(sorted_keys)
        
        transforms = np.array([sensors[k]["transform"] for k in sorted_keys])
        coordinates = np.array([sensors[k]["coordinates"] for k in sorted_keys])
    
    # save to npz
    seg_save_to = args.output[:args.output.rfind(".")] + "_segments.npz"
    
    
    np.savez_compressed(seg_save_to,
                        seg_ids=seg_assignments,
                        transforms=transforms,
                        coordinates=coordinates,
                        id_to_name=id_to_name,)
    
    print("分割信息保存成功至", seg_save_to)
