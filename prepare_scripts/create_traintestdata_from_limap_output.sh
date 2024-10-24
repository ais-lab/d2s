directories=("chess" "fire" "heads" "office" "redkitchen" "stairs")
target_folder="/home/pc1/Desktop/7scenes_sfmGT"
sfm=("sfm_sift_full" "sfm_superpoint+superglue" "sfm_superpoint+superglue+depth")
limap_output="/home/pc1/Desktop/limap/tmp/7scenes"
sfm_full="/home/pc1/Desktop/datasets/imgs_datasets/7scenes/7scenes_sfmGT_triangulated"
listfile_full=("images.bin" "cameras.bin" "points3D.bin" "list_test.txt")
list_limap_out=("images.bin" "cameras.bin" "points3D.bin")
if [ ! -d "$target_folder" ]; then 
	mkdir -p "$target_folder"
fi
for scene in "${directories[@]}"
do 
	dir="$target_folder/$scene"
	if [ ! -d "$dir" ]; then
		mkdir -p "$dir"
	fi
	for sfm_name in "${sfm[@]}"
	do
		echo "Created: $dir/$sfm_name"
		if [ ! -d "$dir/$sfm_name" ]; then 
			mkdir -p "$dir/$sfm_name"
		fi
		if [ "$sfm_name" == "sfm_sift_full" ]; then 
			for file_full in "${listfile_full[@]}"
			do 
				cp "$sfm_full/$scene/$file_full" "$dir/$sfm_name"
			done 
        else
            for file in "${list_limap_out[@]}"
			do 
				cp "$limap_output/$scene/"localization"/$sfm_name/$file" "$dir/$sfm_name"
			done
		fi
	done
done
