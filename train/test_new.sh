source /home/aiops/wangzh/miniconda3/bin/activate
conda activate depth

# python train/test_new.py --task "non_spatial"

# python train/test_new.py --task "object_orientation"

# python train/test_new.py --task "relative_depth"

# python train/test_new.py --task "relative_size"

# python train/test_new.py --task "relative_spatial_position"


python train/test_new.py --task "none_spatial"

python train/test_new.py --task "orientation"

python train/test_new.py --task "relative_depth"

python train/test_new.py --task "relative_size"

python train/test_new.py --task "spatial_relation"



# python train/test_open.py --task "none_spatial"

# python train/test_open.py --task "object_orientation"

# python train/test_open.py --task "relative_depth"

# python train/test_open.py --task "relative_size"

# python train/test_open.py --task "relative_spatial_position"


# python train/test_open.py --task "none_spatial"

# python train/test_open.py --task "orientation"

# python train/test_open.py --task "relative_depth"

# python train/test_open.py --task "relative_size"

# python train/test_open.py --task "spatial_relation"