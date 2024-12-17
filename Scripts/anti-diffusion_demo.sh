export DEFENSE=anti-diffusion
export KEY=sks
export DATASET=demo
export IMAGES_ROOT=Datasets/${DATASET}/
export MODEL_PATH=stabilityai/stable-diffusion-2-1-base
export SAVE_DIR=Outputs/Defense/${DATASET}_using_${DEFENSE}/
export CLASS_DIR="Dataset/class-person"
export DREAMBOOTH_OUTPUT_DIR=Outputs/Editing/dreambooth_test_${DATASET}_using_${DEFENSE}_${KEY}/

for dir in $(ls ${IMAGES_ROOT}); do
    python Main.py --images_root=${IMAGES_ROOT}${dir}/set_B --save_dir=$SAVE_DIR${dir} --diffusion_path=${MODEL_PATH}  --pgd_alpha=2e-3 --pgd_eps=1e-2 --epoches=5 --defense_method=${DEFENSE} --pgd_itrs=10
done